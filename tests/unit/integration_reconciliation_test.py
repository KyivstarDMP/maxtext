# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavioral checks at the upstream/input/metrics integration boundaries."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from maxtext.input_pipeline import data_processing_utils, grain_data_processing, input_pipeline_utils
from maxtext.input_pipeline.protos import example_pb2


def test_tokenizer_cache_distinguishes_effective_bos_eos_and_revision():
  config = SimpleNamespace(
      tokenizer_path="example/tokenizer",
      tokenizer_type="huggingface",
      add_bos=True,
      add_eos=False,
      hf_access_token=None,
      tokenizer_revision="revision-a",
  )
  data_processing_utils._build_tokenizer_cached.cache_clear()  # pylint: disable=protected-access
  try:
    with mock.patch.object(data_processing_utils.tokenizer, "build_tokenizer") as build:
      build.side_effect = lambda *args: SimpleNamespace(pad_id=0, identity=args)
      first, _ = data_processing_utils.get_tokenizer_and_pad_id(config)
      repeated, _ = data_processing_utils.get_tokenizer_and_pad_id(config, add_bos=True, add_eos=False)
      changed, _ = data_processing_utils.get_tokenizer_and_pad_id(config, add_bos=False, add_eos=True)
      config.tokenizer_revision = "revision-b"
      revised, _ = data_processing_utils.get_tokenizer_and_pad_id(config)
    assert first is repeated
    assert changed is not first and revised is not first
    assert build.call_count == 3
    assert changed.identity[2:4] == (False, True)
    assert first.identity[-1] == "revision-a"
    assert revised.identity[-1] == "revision-b"
  finally:
    data_processing_utils._build_tokenizer_cached.cache_clear()  # pylint: disable=protected-access


def test_record_alias_optional_tools_boolean_and_dataset_identity():
  record = example_pb2.Example()
  record.features.feature["text"].bytes_list.value.append(b'[{"role":"user","content":"Question"}]')
  record.features.feature["thinking"].int64_list.value.append(1)
  columns = ["messages", "tools", "thinking"]
  parsed = input_pipeline_utils.ParseFeatures(columns, tokenize=True).map(
      {"raw": record.SerializeToString(), "dataset_id": 2}
  )
  normalized = input_pipeline_utils.NormalizeFeatures(columns, True, scalar_bool_columns=("thinking",)).map(parsed)
  assert normalized["messages"].startswith('[{"role":"user"')
  assert normalized["thinking"] is True
  assert normalized["dataset_id"] == 2
  assert "tools" not in normalized
  with pytest.raises(ValueError, match="required"):
    input_pipeline_utils.ParseFeatures(["required", "tools"], True).map(record.SerializeToString())


def test_numeric_record_carriers_remain_numeric():
  record = example_pb2.Example()
  record.features.feature["inputs"].int64_list.value.extend([1, 2])
  record.features.feature["weights"].float_list.value.extend([0.25, 0.75])
  parsed = input_pipeline_utils.ParseFeatures(["inputs", "weights"], tokenize=False).map(record.SerializeToString())
  np.testing.assert_array_equal(parsed["inputs"], [1, 2])
  np.testing.assert_allclose(parsed["weights"], [0.25, 0.75])


@pytest.mark.parametrize("file_type", ["parquet", "tfrecord", "mmap", "mmap_npy"])
def test_metrics_reject_sources_that_do_not_stamp_dataset_ids(file_type):
  config = SimpleNamespace(
      global_batch_size_to_load=1,
      per_dataset_metrics=True,
      use_multimodal=False,
      grain_file_type=file_type,
  )
  with pytest.raises(ValueError, match="stamps source dataset IDs"):
    grain_data_processing.make_grain_train_iterator(config, SimpleNamespace(size=1), [0])


@pytest.mark.parametrize(
    "objective,indexer,tiling,diagnostic",
    [
        ("block_diffusion", False, 1, "block diffusion"),
        ("causal_lm", True, 1, "indexer warm-up"),
    ],
)
def test_unsupported_metrics_fail_before_forward(objective, indexer, tiling, diagnostic):
  from maxtext.trainers.pre_train import train  # pylint: disable=import-outside-toplevel

  config = SimpleNamespace(
      training_objective=objective,
      per_dataset_metrics=True,
      use_indexer=indexer,
      indexer_sparse_training=False,
      num_vocab_tiling=tiling,
  )
  with pytest.raises(ValueError, match=diagnostic):
    train.loss_fn(object(), config, {}, None, None)


@pytest.mark.parametrize("missing", ["loss", "correct", "both"])
def test_missing_metric_numerators_do_not_become_successful_zeros(missing):
  from maxtext.trainers.pre_train import train  # pylint: disable=import-outside-toplevel

  config = SimpleNamespace(per_dataset_metrics=True, per_dataset_names="a,b")
  data = {"dataset_id": np.array([[1, 2]]), "targets_segmentation": np.array([[1, 1]])}
  loss = None if missing in ("loss", "both") else np.array([0.0, 1.0, 2.0])
  correct = None if missing in ("correct", "both") else np.array([0, 1, 0])
  with pytest.raises(ValueError, match="loss and correct-token sums"):
    train._assemble_per_dataset_aux(config, data, loss, correct)  # pylint: disable=protected-access


def test_dataset_logger_uses_token_weighted_sums_and_omits_absent_values():
  from maxtext.common.metric_logger import MetricLogger  # pylint: disable=import-outside-toplevel

  logger = MetricLogger.__new__(MetricLogger)
  logger.config = SimpleNamespace(per_dataset_log_period=2, log_period=1, steps=3, per_dataset_names="a,b,absent")
  logger._per_dataset_accum = None  # pylint: disable=protected-access
  first = {
      "scalar": {},
      "per_dataset": {
          "xent_sum_by_ds": np.array([0.0, 2.0, 9.0, 0.0]),
          "correct_by_ds": np.array([0, 1, 0, 0]),
          "token_count_by_ds": np.array([0, 1, 3, 0]),
      },
  }
  second = {
      "scalar": {},
      "per_dataset": {
          "xent_sum_by_ds": np.array([0.0, 18.0, 3.0, 0.0]),
          "correct_by_ds": np.array([0, 1, 0, 0]),
          "token_count_by_ds": np.array([0, 3, 1, 0]),
      },
  }
  logger._expand_per_dataset_train(first, 0)  # pylint: disable=protected-access
  assert not first["scalar"]
  logger._expand_per_dataset_train(second, 1)  # pylint: disable=protected-access
  assert second["scalar"] == {
      "per_dataset_train_tokens/a": 4.0,
      "per_dataset_train_loss/a": 5.0,
      "per_dataset_train_accuracy/a": 0.5,
      "per_dataset_train_tokens/b": 4.0,
      "per_dataset_train_loss/b": 3.0,
      "per_dataset_train_accuracy/b": 0.0,
      "per_dataset_train_tokens/absent": 0.0,
  }
  assert logger._per_dataset_accum is None  # pylint: disable=protected-access
