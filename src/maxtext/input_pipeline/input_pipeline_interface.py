# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input pipeline"""
import functools
import json
import re

import jax
from jax.sharding import PartitionSpec as P

from maxtext.configs import pyconfig
from maxtext.input_pipeline.grain_data_processing import make_grain_train_iterator
from maxtext.input_pipeline.grain_data_processing import make_grain_eval_iterator
from maxtext.input_pipeline.hf_data_processing import make_hf_train_iterator
from maxtext.input_pipeline.hf_data_processing import make_hf_eval_iterator
from maxtext.input_pipeline.olmo_grain_data_processing import make_olmo_grain_train_iterator
from maxtext.input_pipeline.olmo_grain_data_processing import make_olmo_grain_eval_iterator
from maxtext.input_pipeline.synthetic_data_processing import SyntheticDataIterator
from maxtext.input_pipeline.synthetic_data_processing import PlaceHolderDataIterator
from maxtext.utils import max_logging
from maxtext.utils.sharding import remove_size_one_mesh_axis


def get_process_loading_real_data(
    data_sharding, global_batch_size_to_load, global_batch_size_to_train_on, max_target_length, mesh
):
  """Get list of processes loading data from GCS when expansion_factor_real_data != -1"""
  data_sharding_pspec = remove_size_one_mesh_axis(P(*data_sharding), mesh)
  sharding = jax.sharding.NamedSharding(mesh, data_sharding_pspec)
  devices_indices_map = sharding.devices_indices_map((global_batch_size_to_load, max_target_length))
  batch_cutoff = global_batch_size_to_train_on
  process_loading_real_data = set()
  for p, indices in devices_indices_map.items():
    if not indices[0].stop or indices[0].stop <= batch_cutoff:
      process_loading_real_data.add(p.process_index)
  return list(process_loading_real_data)


def create_process_specific_iterator(config: pyconfig.HyperParameters, mesh, process_indices, input_iterator):
  """
  If the current process's index is among the `process_indices`, a real
  data iterator is created. Otherwise, a placeholder iterator is returned.
  """
  if jax.process_index() in process_indices:
    iterator_fn = functools.partial(input_iterator, config, mesh, process_indices)
    output_iterator = iterator_fn()
  else:
    output_iterator = PlaceHolderDataIterator(config, mesh)
  return output_iterator


def _parse_dataset_names(value: str, field: str) -> list[str]:
  """Require unambiguous labels without changing the source order."""
  names = value.split(",")
  if any(not name or name != name.strip() for name in names):
    raise ValueError(f"{field} must contain nonempty names without surrounding whitespace.")
  if len(set(names)) != len(names):
    raise ValueError(f"{field} must contain unique names.")
  if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None for name in names):
    raise ValueError(f"{field} names must match [A-Za-z0-9][A-Za-z0-9_.-]*.")
  return names


def _validate_per_dataset_names(config: pyconfig.HyperParameters) -> None:
  """Validate source-to-label alignment before any dataset iterator is created."""
  names = _parse_dataset_names(config.per_dataset_names, "per_dataset_names")
  if config.grain_train_mixture_config_path:
    with open(config.grain_train_mixture_config_path, "r", encoding="utf-8") as mixture_file:
      mixture = json.load(mixture_file)
    if not isinstance(mixture, dict) or not mixture:
      raise ValueError("grain_train_mixture_config_path must contain a nonempty JSON object.")
    if names != list(mixture):
      raise ValueError("per_dataset_names must match the JSON training mixture keys in their original order.")
  else:
    sources = config.grain_train_files.split(";")
    if any(not source.strip() for source in sources):
      raise ValueError("Per-dataset metrics require nonempty grain_train_files components.")
    if len(names) != len(sources):
      raise ValueError(
          f"per_dataset_names ({len(names)}) must provide one name per training mixture component ({len(sources)}), "
          "in grain_train_files order."
      )

  if config.per_dataset_eval_names or config.per_dataset_eval_files:
    eval_names = _parse_dataset_names(config.per_dataset_eval_names, "per_dataset_eval_names")
    eval_files = config.per_dataset_eval_files.split(";")
    if any(not source.strip() for source in eval_files):
      raise ValueError("per_dataset_eval_files must contain nonempty file patterns.")
    if len(eval_names) != len(eval_files):
      raise ValueError("per_dataset_eval_names and per_dataset_eval_files must align one-to-one in order.")


def validate_post_train_data_options(config: pyconfig.HyperParameters) -> None:
  """Reject dataset attribution options unsupported by the Tunix consumers."""
  if (
      getattr(config, "per_dataset_metrics", False)
      or getattr(config, "per_dataset_eval_files", "")
      or getattr(config, "per_dataset_eval_names", "")
  ):
    raise ValueError(
        "Per-dataset metrics and named evaluation require maxtext.trainers.pre_train.train; "
        "Tunix post-training does not support these options."
    )


def _validate_sft_input_options(config: pyconfig.HyperParameters) -> None:
  """Reject text SFT settings that the selected input pipeline cannot apply."""
  if not getattr(config, "use_sft", False):
    return
  if getattr(config, "use_multimodal", False):
    text_defaults = {
        "sft_chat_template_mode": "segmented",
        "sft_enable_thinking": True,
        "sft_enable_thinking_column": "",
        "sft_preserve_thinking": "auto",
        "sft_long_example_handling": "truncate",
        "sft_window_pin_leading_context": False,
        "sft_window_overlap": 256,
        "sft_window_context_cap": -1,
        "sft_window_max_fan_out": 32,
        "sft_window_pinned_context_overflow": "error",
        "sft_window_pinned_context_warn_fraction": 0.5,
        "chat_template": "",
        "chat_template_path": "",
        "chat_template_revision": "",
        "chat_template_sha256": "",
    }
    unsupported = [key for key, default in text_defaults.items() if getattr(config, key, default) != default]
    for key in ("train_data_columns", "eval_data_columns"):
      if "tools" in getattr(config, key, ()):
        unsupported.append(key)
    if unsupported:
      raise ValueError(f"Multimodal SFT does not apply text chat-template options: {', '.join(unsupported)}.")
  if getattr(config, "sft_long_example_handling", "truncate") == "window":
    if config.dataset_type == "hf":
      raise ValueError(
          "sft_long_example_handling='window' is currently implemented only for the Grain SFT pipeline; "
          "dataset_type='hf' would otherwise silently use head-truncating SFTPromptMasking."
      )
    if getattr(config, "grain_use_elastic_iterator", False):
      raise ValueError("sft_long_example_handling='window' is not supported with grain_use_elastic_iterator=True.")


def create_data_iterator(config: pyconfig.HyperParameters, mesh):
  """Create train and eval data iterators given configs and mesh."""
  if config.per_dataset_metrics and config.dataset_type != "grain":
    raise ValueError("Per-dataset metrics require the Grain ArrayRecord pipeline.")
  if config.per_dataset_metrics and not config.use_sft:
    raise ValueError("Per-dataset metrics require use_sft=true; pretraining does not preserve dataset IDs.")
  if config.per_dataset_metrics:
    if not config.packing:
      raise ValueError("Per-dataset metrics require packing=true for token-aligned dataset IDs.")
    if config.expansion_factor_real_data > 1:
      raise ValueError("Per-dataset metrics do not support expansion_factor_real_data > 1 (placeholder hosts).")
    if config.eval_interval > 0 and config.per_dataset_eval_files:
      if config.eval_steps <= 0:
        raise ValueError("Named per-dataset evaluation requires eval_steps > 0 to bound its padding batches.")
      if config.target_eval_loss > 0:
        raise ValueError("target_eval_loss > 0 is not supported with named per-dataset evaluation.")
    _validate_per_dataset_names(config)
  _validate_sft_input_options(config)

  # Return synthetic dataset if selected
  if config.dataset_type == "synthetic":
    eval_iterator = SyntheticDataIterator(config, mesh) if config.eval_interval > 0 else None
    return SyntheticDataIterator(config, mesh), eval_iterator
  dataset_type_to_train_eval_iterator = {
      "grain": (make_grain_train_iterator, make_grain_eval_iterator),
      "hf": (make_hf_train_iterator, make_hf_eval_iterator),
      "olmo_grain": (make_olmo_grain_train_iterator, make_olmo_grain_eval_iterator),
  }
  if config.dataset_type in ("tfds", "c4_mlperf"):
    from maxtext.input_pipeline.tfds_data_processing import make_tfds_train_iterator, make_tfds_eval_iterator  # pylint: disable=import-outside-toplevel
    from maxtext.input_pipeline.tfds_data_processing_c4_mlperf import make_c4_mlperf_train_iterator, make_c4_mlperf_eval_iterator  # pylint: disable=import-outside-toplevel

    dataset_type_to_train_eval_iterator["tfds"] = (make_tfds_train_iterator, make_tfds_eval_iterator)
    dataset_type_to_train_eval_iterator["c4_mlperf"] = (make_c4_mlperf_train_iterator, make_c4_mlperf_eval_iterator)

  # Collect train and eval iterators
  if config.dataset_type in ["tfds", "grain", "hf", "c4_mlperf", "olmo_grain"]:
    if config.dataset_type == "c4_mlperf":
      assert config.packing, "c4_mlperf dataloader only works with packing. For padded version, use tfds dataloader"
    train_iterator, eval_iterator = dataset_type_to_train_eval_iterator[config.dataset_type]
  else:
    max_logging.log(
        f"WARNING: '{config.dataset_type}' is not a supported dataset type."
        "Using synthetic data. Please choose from 'tfds', 'grain', 'hf', or 'c4_mlperf'."
    )
    output_train_iterator, output_eval_iterator = SyntheticDataIterator(config, mesh), None
    return output_train_iterator, output_eval_iterator

  # Generate output train iterator
  process_indices_train = get_process_loading_real_data(
      config.data_sharding,
      config.global_batch_size_to_load,
      config.global_batch_size_to_train_on,
      config.max_target_length,
      mesh,
  )
  if (
      config.dataset_type == "grain"
      and getattr(config, "use_sft", False)
      and getattr(config, "sft_long_example_handling", "truncate") == "window"
      and getattr(config, "num_epoch", None) is not None
      and config.num_epoch > 0
      and not getattr(config, "generate_padding_batch_train", False)
      and len(process_indices_train) > 1
  ):
    max_logging.warning(
        "Finite multi-host SFT windowing without training padding can exhaust one host before others. "
        "Independently preflight the minimum available batches across data-loading hosts after rendering, "
        "windowing, packing and resume, and ensure it covers the planned batch calls. Startup does not "
        "measure this capacity; this warning does not provide coordinated exhaustion."
    )
  output_train_iterator = create_process_specific_iterator(config, mesh, process_indices_train, train_iterator)
  if config.expansion_factor_real_data > 1:  # assert number of hosts loading real data
    assert len(process_indices_train) == jax.process_count() // config.expansion_factor_real_data

  # Generate output eval iterator
  output_eval_iterator = None
  if config.eval_interval > 0:
    process_indices_eval = get_process_loading_real_data(
        config.data_sharding,
        config.global_batch_size_to_load_eval,
        config.global_batch_size_to_eval_on,
        config.max_target_length,
        mesh,
    )

    if config.expansion_factor_real_data > 1:
      assert len(process_indices_eval) == jax.process_count() // config.expansion_factor_real_data
    if config.per_dataset_metrics and config.per_dataset_eval_files:
      # Option B: one single-dataset eval iterator per component, keyed by name. jit_eval_step is
      # shape-based, so a dict here is fine; the eval loop runs one pass per entry.
      names = config.per_dataset_eval_names.split(",")
      globs = config.per_dataset_eval_files.split(";")
      # force_padding_batch=True (per-dataset iterators ONLY): these small single-dataset splits do not
      # divide evenly across hosts, so without padding a short host raises StopIteration and exits the eval
      # loop early -> unequal jit_eval_step launch counts -> E0200 SPMD desync. Padding keeps every host in
      # lockstep for a fixed eval_steps launches. The aggregate else-branch below is left untouched (keeps
      # config.generate_padding_batch_eval).
      output_eval_iterator = {
          name: create_process_specific_iterator(
              config,
              mesh,
              process_indices_eval,
              functools.partial(eval_iterator, eval_files_override=glob, force_padding_batch=True),
          )
          for name, glob in zip(names, globs)
      }
    else:
      output_eval_iterator = create_process_specific_iterator(config, mesh, process_indices_eval, eval_iterator)
  return output_train_iterator, output_eval_iterator
