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

"""Check unsupported SFT configurations at the entry points, before data I/O."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from maxtext.input_pipeline import grain_data_processing, input_pipeline_interface
from maxtext.trainers.post_train import hooks
from maxtext.trainers.post_train.sft import train_sft
from maxtext.trainers.pre_train import train


def _config(**overrides):
  """Small explicit configuration for entry-point guards."""
  values = dict(  # pylint: disable=use-dict-literal
      dataset_type="grain",
      use_sft=True,
      use_dpo=False,
      use_multimodal=False,
      packing=True,
      per_dataset_metrics=True,
      per_dataset_names="train",
      grain_train_files="train*",
      grain_train_mixture_config_path="",
      per_dataset_eval_names="eval",
      per_dataset_eval_files="eval*",
      expansion_factor_real_data=1,
      data_sharding=["data"],
      global_batch_size_to_load=1,
      global_batch_size_to_train_on=1,
      global_batch_size_to_load_eval=1,
      global_batch_size_to_eval_on=1,
      max_target_length=8,
      eval_interval=1,
      eval_steps=2,
      target_eval_loss=0.0,
      optimizer_memory_host_offload=False,
  )
  values.update(overrides)
  return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "overrides,diagnostic",
    [
        ({"packing": False}, "packing=true"),
        ({"expansion_factor_real_data": 2}, "placeholder hosts"),
        ({"eval_steps": 0}, "eval_steps > 0"),
        ({"eval_steps": -1}, "eval_steps > 0"),
        ({"target_eval_loss": 0.1}, "target_eval_loss > 0"),
        ({"sft_long_example_handling": "window", "grain_use_elastic_iterator": True}, "elastic"),
        (
            {"per_dataset_metrics": False, "dataset_type": "hf", "sft_long_example_handling": "window"},
            "only for the Grain",
        ),
    ],
)
def test_unsupported_configs_fail_before_data_access(overrides, diagnostic):
  with mock.patch.object(input_pipeline_interface, "get_process_loading_real_data") as access:
    with pytest.raises(ValueError, match=diagnostic):
      input_pipeline_interface.create_data_iterator(_config(**overrides), object())
  access.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"sft_chat_template_mode": "assistant_mask"},
        {"sft_enable_thinking": False},
        {"sft_enable_thinking_column": "thinking"},
        {"sft_preserve_thinking": "true"},
        {"sft_long_example_handling": "window"},
        {"sft_window_pin_leading_context": True},
        {"chat_template_path": "template.jinja"},
        {"chat_template_revision": "revision"},
        {"chat_template_sha256": "sha"},
        {"train_data_columns": ["messages", "tools"]},
        {"eval_data_columns": ["messages", "tools"]},
    ],
)
def test_multimodal_rejects_unapplied_text_options(overrides):
  config = _config(per_dataset_metrics=False, use_multimodal=True, **overrides)
  with mock.patch.object(input_pipeline_interface, "get_process_loading_real_data") as access:
    with pytest.raises(ValueError, match="Multimodal SFT does not apply"):
      input_pipeline_interface.create_data_iterator(config, object())
  access.assert_not_called()


@pytest.mark.parametrize(
    "overrides,named_eval",
    [
        ({}, True),
        ({"per_dataset_metrics": False, "packing": False, "eval_interval": 0}, False),
        ({"per_dataset_metrics": False, "use_multimodal": True}, False),
        ({"eval_interval": 0, "eval_steps": 0, "target_eval_loss": 1.0}, False),
    ],
)
def test_supported_configs_reach_iterator_builders(overrides, named_eval):
  config = _config(**overrides)
  with (
      mock.patch.object(input_pipeline_interface, "get_process_loading_real_data", return_value=[0]),
      mock.patch.object(input_pipeline_interface.jax, "process_index", return_value=0),
      mock.patch.object(input_pipeline_interface, "make_grain_train_iterator") as train_data,
      mock.patch.object(input_pipeline_interface, "make_grain_eval_iterator") as eval_data,
  ):
    train_iter, eval_iter = input_pipeline_interface.create_data_iterator(config, object())
  train_data.assert_called_once()
  assert train_iter is train_data.return_value
  if named_eval:
    assert eval_iter == {"eval": eval_data.return_value}
    assert eval_data.call_args.kwargs == {"eval_files_override": "eval*", "force_padding_batch": True}
  elif config.eval_interval <= 0:
    eval_data.assert_not_called()
    assert eval_iter is None
  else:
    assert eval_iter is eval_data.return_value
    assert not eval_data.call_args.kwargs


@pytest.mark.parametrize("overrides", [{"per_dataset_metrics": True}, {"per_dataset_eval_files": "eval*"}])
def test_tunix_rejects_metrics_and_named_eval_at_both_entry_points(overrides):
  config = _config(per_dataset_metrics=False, per_dataset_eval_files="")
  for name, value in overrides.items():
    setattr(config, name, value)
  with mock.patch.object(hooks, "create_data_iterator") as access:
    with pytest.raises(ValueError, match="Tunix post-training"):
      train_sft.validate_config(config)
    with pytest.raises(ValueError, match="Tunix post-training"):
      hooks.BaseDataHooks(config, object(), None)
  access.assert_not_called()


def test_tunix_defaults_still_reach_data_hooks():
  config = _config(per_dataset_metrics=False, per_dataset_eval_files="")
  train_sft.validate_config(config)
  with (
      mock.patch.object(hooks, "create_data_iterator", return_value=(object(), None)) as access,
      mock.patch.object(hooks, "DataLoader"),
  ):
    hooks.BaseDataHooks(config, object(), None)
  access.assert_called_once()


def test_missing_training_dataset_id_fails_before_forward():
  model = mock.Mock()
  config = _config(use_indexer=False)
  with pytest.raises(ValueError, match="dataset_id in the training batch"):
    train.loss_fn(model, config, {}, None, None, is_train=True)
  model.assert_not_called()


def test_named_eval_config_does_not_require_an_unused_aggregate_file():
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel

  config = pyconfig.initialize(
      [None, "src/maxtext/configs/base.yml"],
      run_name="named_eval_config_test",
      base_output_directory="/tmp/named_eval_config_test",
      enable_checkpointing=False,
      enable_tensorboard=False,
      dataset_type="grain",
      grain_file_type="arrayrecord",
      grain_train_files="train*",
      grain_eval_files="",
      grain_train_mixture_config_path="",
      use_sft=True,
      packing=True,
      per_dataset_metrics=True,
      per_dataset_names="train",
      per_dataset_eval_names="eval",
      per_dataset_eval_files="eval*",
      eval_interval=1,
      eval_steps=2,
      log_config=False,
  )
  with (
      mock.patch.object(input_pipeline_interface, "get_process_loading_real_data", return_value=[0]),
      mock.patch.object(input_pipeline_interface.jax, "process_index", return_value=0),
      mock.patch.object(input_pipeline_interface, "make_grain_train_iterator"),
      mock.patch.object(input_pipeline_interface, "make_grain_eval_iterator") as eval_data,
  ):
    _, eval_iter = input_pipeline_interface.create_data_iterator(config, object())
  assert eval_iter == {"eval": eval_data.return_value}


@pytest.mark.parametrize("force_padding", [False, True])
@pytest.mark.parametrize("use_sft", [False, True])
def test_empty_host_padding_template_is_specific_to_named_eval(force_padding, use_sft):
  config = _config(
      use_sft=use_sft,
      per_dataset_metrics=False,
      grain_file_type="arrayrecord",
      data_shuffle_seed=0,
      mmap_npy_split="",
      grain_eval_files="eval*",
      grain_shuffle_buffer_size=0,
      grain_worker_count_eval=0,
      grain_num_threads_eval=1,
      grain_prefetch_buffer_size_eval=1,
      grain_data_source_max_workers=1,
      eval_data_columns=["messages"],
      tokenize_eval_data=True,
      grain_per_worker_buffer_size_eval=1,
      generate_padding_batch_eval=True,
      elastic_enabled=False,
      colocated_python_data_input=False,
  )
  with (
      mock.patch.object(grain_data_processing, "get_datasets", return_value=[]),
      mock.patch.object(grain_data_processing, "_get_pipeline_fn", return_value=lambda **kwargs: []),
      mock.patch.object(grain_data_processing.jax, "process_index", return_value=0),
      mock.patch.object(grain_data_processing.jax, "process_count", return_value=1),
  ):
    iterator = grain_data_processing.make_grain_eval_iterator(
        config, SimpleNamespace(size=1), [0], force_padding_batch=force_padding
    )
  if force_padding:
    padding = iterator._make_padding_batch()  # pylint: disable=protected-access
    assert set(padding) == {
        "inputs",
        "inputs_position",
        "inputs_segmentation",
        "targets",
        "targets_position",
        "targets_segmentation",
    }
    for value in padding.values():
      assert value.shape == (1, 8) and value.dtype == np.int32
      assert not value.any()
  else:
    assert iterator.padding_batch_template is None
    with pytest.raises(ValueError, match="no padding_batch_template"):
      iterator._make_padding_batch()  # pylint: disable=protected-access
    iterator.last_local_data = {"different_schema": np.ones((1, 3), dtype=np.float32)}
    padding = iterator._make_padding_batch()  # pylint: disable=protected-access
    assert set(padding) == {"different_schema"}
    np.testing.assert_array_equal(padding["different_schema"], np.zeros((1, 3), dtype=np.float32))


@pytest.mark.parametrize(
    "padding,epochs,hosts,warning",
    [(False, 1, 2, True), (True, 1, 2, False), (False, None, 2, False), (False, 1, 1, False)],
)
def test_finite_window_warning_does_not_consume_or_reset_iterator(padding, epochs, hosts, warning):
  config = _config(
      per_dataset_metrics=False,
      sft_long_example_handling="window",
      num_epoch=epochs,
      generate_padding_batch_train=padding,
      eval_interval=0,
  )
  with (
      mock.patch.object(input_pipeline_interface, "get_process_loading_real_data", return_value=list(range(hosts))),
      mock.patch.object(input_pipeline_interface.jax, "process_index", return_value=0),
      mock.patch.object(input_pipeline_interface, "make_grain_train_iterator") as build,
      mock.patch.object(input_pipeline_interface.max_logging, "warning") as warn,
  ):
    iterator, _ = input_pipeline_interface.create_data_iterator(config, object())
  assert iterator is build.return_value
  iterator.__next__.assert_not_called()
  iterator.__iter__.assert_not_called()
  iterator.reset.assert_not_called()
  assert warn.call_count == int(warning)
  if warning:
    assert "Startup does not measure this capacity" in warn.call_args.args[0]
    assert "does not provide coordinated exhaustion" in warn.call_args.args[0]
