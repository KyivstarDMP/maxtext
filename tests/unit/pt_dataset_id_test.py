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

"""Source-ID propagation through the ordinary pretraining Grain pipeline (per_dataset_metrics)."""

import copy
from types import SimpleNamespace
from unittest import mock

import grain.python as grain
import numpy as np
import pytest

from maxtext.input_pipeline import data_processing_utils, grain_data_processing, grain_tokenizer, input_pipeline_utils
from maxtext.input_pipeline.protos import example_pb2

PACKED_KEYS = {
    "inputs",
    "targets",
    "inputs_position",
    "targets_position",
    "inputs_segmentation",
    "targets_segmentation",
}
# Source 1 owns tokens 10..29, source 2 owns 30..49, so ownership is observable from the token alone.
SOURCE_TOKENS = [(1, list(range(10, 21))), (2, list(range(30, 33))), (1, [22, 23]), (2, list(range(33, 43)))]


class _RangeTokenizer:
  """Whitespace integer tokenizer; keeps every source's token range disjoint."""

  bos_id = 2

  def encode(self, text):
    return [int(token) for token in text.split()] if text else []


def _owner(tokens):
  return np.where(tokens >= 30, 2, 1)


def _config(packing, truncate, workers=0):
  return SimpleNamespace(
      grain_file_type="arrayrecord",
      is_offline_distillation=False,
      use_truncation=truncate,
      max_target_length=8,
      elastic_enabled=False,
      global_batch_size_to_load=1,
      expansion_factor_real_data=1,
      packing=True,
      grain_packing_type=packing,
      max_segments_per_seq=None,
      grain_use_elastic_iterator=False,
      add_bos=False,
      per_dataset_names="first,second",
      grain_ram_budget_mb=1024,
      grain_worker_count=workers,
  )


def _records(tokenize, stamped):
  """Serialized records from both sources, wrapped like the mixture loader does when stamped."""
  records = []
  for source_id, tokens in SOURCE_TOKENS:
    if not tokenize:
      tokens = tokens[:6]  # Pre-tokenized records are not chunked; keep them within max_target_length.
    record = example_pb2.Example()
    if tokenize:
      record.features.feature["text"].bytes_list.value.append(" ".join(map(str, tokens)).encode())
    else:
      record.features.feature["text"].int64_list.value.extend(tokens)
    raw = record.SerializeToString()
    records.append({"raw": raw, "dataset_id": source_id} if stamped else raw)
  return records


def _pipeline(config, tokenize, stamped):
  dataset = grain.MapDataset.source(_records(tokenize, stamped)).to_iter_dataset()
  with mock.patch.object(data_processing_utils, "get_tokenizer_and_pad_id", return_value=(_RangeTokenizer(), 0)):
    return grain_data_processing.pretrain_preprocessing_pipeline(
        dataset, config, ["text"], tokenize, config.grain_worker_count, 1, stamp_dataset_id=stamped
    )


ROUTES = [(True, False), (True, True), (False, True)]  # chunk, trim, pre-tokenized
PACKERS = ["best_fit", "first_fit", "concat_then_split"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokenize,truncate", ROUTES)
@pytest.mark.parametrize("packing", PACKERS)
def test_every_loss_bearing_target_carries_its_source(tokenize, truncate, packing):
  batches = list(_pipeline(_config(packing, truncate), tokenize, stamped=True))
  checked = 0
  for batch in batches:
    assert set(batch) == PACKED_KEYS | {"dataset_id"}
    assert batch["dataset_id"].shape == batch["targets"].shape
    assert batch["dataset_id"].dtype == np.int32
    live = batch["targets_segmentation"] != 0
    np.testing.assert_array_equal(batch["dataset_id"][live], _owner(batch["targets"])[live])
    assert np.all(batch["dataset_id"][~live] == 0)
    checked += int(live.sum())
  assert checked > 0
  assert {1, 2} <= set(np.unique(np.concatenate([b["dataset_id"][b["targets_segmentation"] != 0] for b in batches])))


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokenize,truncate", ROUTES)
@pytest.mark.parametrize("packing", PACKERS)
def test_stamping_does_not_change_the_token_stream(tokenize, truncate, packing):
  config = _config(packing, truncate)
  stamped = list(_pipeline(config, tokenize, stamped=True))
  unstamped = list(_pipeline(config, tokenize, stamped=False))
  assert len(stamped) == len(unstamped)
  for with_ids, without_ids in zip(stamped, unstamped):
    assert set(without_ids) == PACKED_KEYS
    for key in PACKED_KEYS:
      np.testing.assert_array_equal(with_ids[key], without_ids[key])


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokenize,truncate", ROUTES)
@pytest.mark.parametrize("packing", PACKERS)
def test_stamped_iterator_restores_exactly(tokenize, truncate, packing):
  pipeline = _pipeline(_config(packing, truncate), tokenize, stamped=True)
  cursor = iter(pipeline)
  next(cursor)
  state = copy.deepcopy(cursor.get_state())
  tail = list(cursor)
  restored = iter(pipeline)
  restored.set_state(state)
  restored_tail = list(restored)
  assert len(tail) == len(restored_tail) > 0
  for expected, actual in zip(tail, restored_tail):
    assert expected.keys() == actual.keys()
    for key in expected:
      np.testing.assert_array_equal(expected[key], actual[key])


@pytest.mark.cpu_only
def test_source_ids_survive_worker_processes():
  batches = list(_pipeline(_config("best_fit", truncate=False, workers=1), True, stamped=True))
  assert batches
  for batch in batches:
    live = batch["targets_segmentation"] != 0
    np.testing.assert_array_equal(batch["dataset_id"][live], _owner(batch["targets"])[live])


def test_chunking_copies_only_the_source_id():
  chunker = grain_tokenizer.TokenizeAndChunk("text", 4, _RangeTokenizer())
  chunks = chunker.flat_map({"text": "10 11 12 13 14 15", "dataset_id": np.int32(2), "other": "dropped"})
  assert [sorted(chunk) for chunk in chunks] == [["dataset_id", "text"], ["dataset_id", "text"]]
  assert [int(chunk["dataset_id"]) for chunk in chunks] == [2, 2]
  assert not chunker.flat_map({"text": "", "dataset_id": np.int32(1)})
  unstamped = chunker.flat_map({"text": "10 11"})
  assert [sorted(chunk) for chunk in unstamped] == [["text"]]


@pytest.mark.parametrize(
    "element,diagnostic",
    [
        ({"text": np.arange(3)}, "stamped dataset_id"),
        ({"text": np.arange(3), "dataset_id": np.int32(0)}, "outside the configured sources"),
        ({"text": np.arange(3), "dataset_id": np.int32(3)}, "outside the configured sources"),
        ({"text": np.arange(3), "dataset_id": np.array([1, 1, 1])}, "one integer per record"),
        ({"text": np.arange(3), "dataset_id": 1.0}, "one integer per record"),
    ],
)
def test_source_id_expansion_rejects_missing_or_invalid_ids(element, diagnostic):
  with pytest.raises(ValueError, match=diagnostic):
    input_pipeline_utils.ExpandDatasetId("text", num_datasets=2).map(element)


def test_source_id_expansion_matches_token_length():
  expanded = input_pipeline_utils.ExpandDatasetId("text", num_datasets=2).map(
      {"text": np.arange(5), "dataset_id": np.int32(2)}
  )
  np.testing.assert_array_equal(expanded["dataset_id"], [2, 2, 2, 2, 2])
  assert expanded["dataset_id"].dtype == np.int32


@pytest.mark.parametrize(
    "overrides,diagnostic",
    [
        ({"grain_file_type": "parquet"}, "only for ArrayRecord"),
        ({"is_offline_distillation": True}, "offline distillation"),
    ],
)
def test_stamped_pretraining_rejects_unsupported_routes(overrides, diagnostic):
  config = _config("best_fit", truncate=False)
  for key, value in overrides.items():
    setattr(config, key, value)
  with pytest.raises(ValueError, match=diagnostic):
    grain_data_processing.pretrain_preprocessing_pipeline(
        grain.MapDataset.source([]).to_iter_dataset(), config, ["text"], True, 0, 1, stamp_dataset_id=True
    )
