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

"""Offline checks that explicit tokenizer and template pins are applied or rejected."""

# pylint: disable=protected-access

import hashlib
from types import SimpleNamespace
from unittest import mock

import pytest

from maxtext.input_pipeline import grain_data_processing, hf_data_processing, input_pipeline_utils, tokenizer
from maxtext.input_pipeline import instruction_data_processing as templates


REVISION = "01234567" * 5


def test_hf_unequal_conversational_columns_fail_before_losing_tool_results():
  row = {
      "prompt": [{"role": "user", "content": "question"}],
      "completion": [
          {"role": "assistant", "content": "call"},
          {"role": "tool", "content": "result"},
          {"role": "assistant", "content": "answer"},
      ],
  }
  with pytest.raises(ValueError, match="Use one ordered messages column"):
    input_pipeline_utils.combine_columns(row, ["prompt", "completion"], "messages")
  assert "messages" not in row


@pytest.mark.parametrize("kind", ["sentencepiece", "tiktoken"])
def test_local_tokenizer_backend_rejects_unused_revision(kind):
  with pytest.raises(ValueError, match="tokenizer_revision requires tokenizer_type=huggingface"):
    tokenizer.build_tokenizer("not-read", kind, False, False, None, REVISION)


def test_tfds_tokenizer_helper_forwards_revision_to_the_loader():
  with mock.patch.object(tokenizer, "build_tokenizer") as build:
    actual = input_pipeline_utils.get_tokenizer("example/tokenizer", "huggingface", False, True, None, REVISION)
  build.assert_called_once_with("example/tokenizer", "huggingface", False, True, None, REVISION)
  assert actual is build.return_value


@pytest.mark.parametrize("pin", [{"chat_template_revision": REVISION}, {"chat_template_sha256": "a" * 64}])
@pytest.mark.parametrize("inline,path", [(None, ""), ("inline", ""), ("inline", "hf://example/model/chat.jinja")])
def test_grain_and_hf_reject_template_pins_that_are_not_selected(pin, inline, path):
  config = SimpleNamespace(chat_template=inline, chat_template_path=path, **pin)
  with mock.patch.object(templates, "load_chat_template_from_file") as load:
    with pytest.raises(ValueError, match="require a selected chat_template_path"):
      grain_data_processing._configure_sft_chat_template(config, ["messages"], object(), tokenize=True)
    with pytest.raises(ValueError, match="require a selected chat_template_path"):
      hf_data_processing.preprocessing_pipeline(
          dataloading_host_index=0,
          dataloading_host_count=1,
          global_mesh=object(),
          dataset=object(),
          config=config,
          data_column_names=["messages"],
          tokenize=True,
          tokenizer_path="not-read",
          hf_access_token=None,
          global_batch_size=1,
          max_target_length=8,
          shuffle=False,
          data_shuffle_seed=0,
          use_sft=True,
          chat_template=inline,
          chat_template_path=path,
          **pin,
      )
  load.assert_not_called()


def test_local_template_revision_is_rejected_even_when_the_file_exists(tmp_path):
  path = tmp_path / "template.jinja"
  path.write_text("template")
  with pytest.raises(ValueError, match="only to a Hub template path"):
    templates.load_chat_template_from_file(str(path), revision=REVISION)


def test_missing_pinned_template_fails_before_an_rl_tokenizer_is_mutated(tmp_path):
  model_tokenizer = SimpleNamespace(chat_template=None)
  config = SimpleNamespace(chat_template_path=str(tmp_path / "missing.jinja"), chat_template_sha256="a" * 64)
  with pytest.raises(FileNotFoundError, match="Pinned chat template file does not exist"):
    templates.configure_tokenizer_chat_template(model_tokenizer, config)
  assert model_tokenizer.chat_template is None


@pytest.mark.parametrize("pin", [{"chat_template_revision": REVISION}, {"chat_template_sha256": "a" * 64}])
def test_rl_existing_template_cannot_silently_ignore_a_pin(pin):
  model_tokenizer = SimpleNamespace(chat_template="existing")
  config = SimpleNamespace(chat_template_path="hf://example/model/chat.jinja", **pin)
  with pytest.raises(ValueError, match="pins are unused"):
    templates.configure_tokenizer_chat_template(model_tokenizer, config)
  assert model_tokenizer.chat_template == "existing"


def test_rl_loads_selected_hub_template_with_revision_and_hash(tmp_path):
  template = "{% for message in messages %}{{ message.content }}{% endfor %}"
  path = tmp_path / "chat.jinja"
  path.write_text(template)
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  config = SimpleNamespace(
      chat_template_path="hf://example/model/chat.jinja",
      chat_template_revision=REVISION,
      chat_template_sha256=digest,
      hf_access_token=None,
  )
  model_tokenizer = SimpleNamespace(chat_template=None)
  with mock.patch.object(templates, "hf_hub_download", return_value=str(path)) as download:
    templates.configure_tokenizer_chat_template(model_tokenizer, config)
  download.assert_called_once_with(repo_id="example/model", filename="chat.jinja", revision=REVISION, token=None)
  assert model_tokenizer.chat_template == template


@pytest.mark.parametrize("existing,inline,expected", [("existing", "inline", "existing"), (None, "inline", "inline")])
def test_rl_unpinned_template_precedence_is_preserved(existing, inline, expected):
  model_tokenizer = SimpleNamespace(chat_template=existing)
  templates.configure_tokenizer_chat_template(model_tokenizer, SimpleNamespace(chat_template=inline))
  assert model_tokenizer.chat_template == expected


@pytest.mark.parametrize("trainer", ["dpo", "distillation"])
def test_pad_id_loaders_forward_revision_before_model_setup(trainer):
  from contextlib import nullcontext  # pylint: disable=import-outside-toplevel
  from maxtext.trainers.post_train.dpo import train_dpo  # pylint: disable=import-outside-toplevel
  from maxtext.trainers.post_train.distillation import train_distill  # pylint: disable=import-outside-toplevel

  config = SimpleNamespace(
      tokenizer_path="example/model",
      tokenizer_type="huggingface",
      add_bos=False,
      add_eos=False,
      hf_access_token=None,
      tokenizer_revision=REVISION,
  )
  with mock.patch.object(tokenizer, "build_tokenizer", side_effect=RuntimeError("reached tokenizer load")) as build:
    with pytest.raises(RuntimeError, match="reached tokenizer load"):
      if trainer == "dpo":
        with (
            mock.patch.object(train_dpo, "get_tunix_config"),
            mock.patch.object(train_dpo, "maybe_record_goodput", return_value=nullcontext()),
        ):
          train_dpo.setup_trainer_state(config)
      else:
        train_distill.build_training_components(config, object())
  build.assert_called_once()
  assert build.call_args.kwargs["tokenizer_revision"] == REVISION


def test_hf_vision_tokenizer_load_honors_revision():
  config = SimpleNamespace(
      elastic_enabled=False,
      use_tunix_gradient_accumulation=False,
      enable_data_shuffling=False,
      num_epoch=1,
      image_placeholder="<image>",
      model_name="example",
      tokenizer_path="example/model",
      hf_access_token=None,
      tokenizer_revision=REVISION,
  )
  dataset = mock.MagicMock()
  dataset.features = {"query": object(), "response": object(), "images": object()}
  dataset.select_columns.return_value = dataset
  dataset.map.return_value = dataset
  with mock.patch.object(
      hf_data_processing.transformers.AutoTokenizer, "from_pretrained", side_effect=RuntimeError("reached tokenizer load")
  ) as load:
    with pytest.raises(RuntimeError, match="reached tokenizer load"):
      hf_data_processing.vision_sft_preprocessing_pipeline(
          dataset, config, 0, 1, SimpleNamespace(size=1), ["query", "response"], "images", 1
      )
  load.assert_called_once()
  assert load.call_args.kwargs["revision"] == REVISION
