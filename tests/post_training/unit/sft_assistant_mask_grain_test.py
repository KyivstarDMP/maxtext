# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grain wiring tests for chat-template assistant masks."""

# pylint: disable=missing-function-docstring,protected-access

import copy
import json
from types import SimpleNamespace

import pytest

from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_utils


pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
MESSAGES = [
    {"role": "developer", "content": "Follow the tool result."},
    {"role": "user", "content": "Look up the value."},
    {
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
    },
    {"role": "tool", "name": "lookup", "content": "result"},
    {"role": "assistant", "content": "answer"},
]
INPUT_IDS = [1, 2, 101, 102, 3, 4, 201, 202]
ASSISTANT_MASK = [0, 0, 1, 1, 0, 0, 1, 1]


class _AssistantMaskTokenizer:
  """Return a deterministic token stream and ownership mask."""

  name_or_path = "synthetic-assistant-mask-tokenizer"
  chat_template = "{% generation %}assistant{% endgeneration %}"

  def apply_chat_template(self, messages, **kwargs):
    assert messages == MESSAGES
    assert kwargs == {
        "add_generation_prompt": False,
        "tokenize": True,
        "return_dict": True,
        "return_assistant_tokens_mask": True,
        "enable_thinking": True,
        "preserve_thinking": True,
        "tools": TOOLS,
    }
    return {"input_ids": list(INPUT_IDS), "assistant_masks": list(ASSISTANT_MASK)}


def _config(**overrides):
  values = {
      "chat_template": "",
      "chat_template_path": "",
      "sft_chat_template_mode": "segmented",
      "sft_train_on_completion_only": True,
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def test_grain_formatter_emits_one_token_stream_with_template_owned_runs():
  formatted = grain_data_processing._format_chat_template_grain(
      {"messages": json.dumps(MESSAGES), "tools": json.dumps(TOOLS)},
      data_columns=["messages", "tools"],
      tokenizer_model=_AssistantMaskTokenizer(),
      chat_template_mode="assistant_mask",
  )

  assert [token for run in formatted["messages"] for token in run] == INPUT_IDS
  assert formatted["is_prompt"] == [True, False, True, False]


def test_grain_formatter_defaults_to_segmented_mode(monkeypatch):
  expected = {"messages": ["segment"], "is_prompt": [False]}

  def _segmented_formatter(element, **kwargs):
    assert kwargs["data_column_name"] == "messages"
    element.update(expected)
    return element

  monkeypatch.setattr(input_pipeline_utils, "apply_chat_template", _segmented_formatter)
  monkeypatch.setattr(
      input_pipeline_utils,
      "apply_chat_template_with_assistant_mask",
      lambda *args, **kwargs: pytest.fail("assistant-mask formatter must remain opt-in"),
  )

  formatted = grain_data_processing._format_chat_template_grain(
      {"messages": copy.deepcopy(MESSAGES[:2])},
      data_columns=["messages"],
      tokenizer_model=object(),
  )

  assert formatted == expected


def test_grain_loads_generation_marked_template_from_configured_path(monkeypatch):
  template = "{% generation %}{{ messages[-1]['content'] }}{% endgeneration %}"
  monkeypatch.setattr(
      grain_data_processing.instruction_data_processing,
      "load_chat_template_from_file",
      lambda path: template if path == "/tmp/training-template.jinja" else None,
  )
  tokenizer = SimpleNamespace(chat_template="inference-template")

  mode = grain_data_processing._configure_sft_chat_template(
      _config(chat_template_path="/tmp/training-template.jinja", sft_chat_template_mode="assistant_mask"),
      ["messages"],
      tokenizer,
      True,
  )

  assert mode == "assistant_mask"
  assert tokenizer.chat_template == template


def test_default_mode_keeps_tokenizer_template():
  tokenizer = SimpleNamespace(chat_template="inference-template")

  mode = grain_data_processing._configure_sft_chat_template(_config(), ["messages"], tokenizer, True)

  assert mode == "segmented"
  assert tokenizer.chat_template == "inference-template"


@pytest.mark.parametrize(
    ("overrides", "tokenize", "match"),
    [
        ({"sft_chat_template_mode": "unknown"}, True, "must be 'segmented' or 'assistant_mask'"),
        ({"sft_chat_template_mode": "assistant_mask"}, False, "requires tokenize=True"),
        (
            {"sft_chat_template_mode": "assistant_mask", "sft_train_on_completion_only": False},
            True,
            "requires sft_train_on_completion_only=True",
        ),
        ({"sft_chat_template_mode": "assistant_mask"}, True, "requires an active chat template"),
    ],
)
def test_assistant_mask_mode_rejects_missing_contract(overrides, tokenize, match):
  tokenizer = SimpleNamespace(chat_template="inference-template")

  with pytest.raises(ValueError, match=match):
    grain_data_processing._configure_sft_chat_template(
        _config(**overrides),
        ["messages"],
        tokenizer,
        tokenize,
    )


def test_missing_template_path_fails_before_iteration(monkeypatch):
  monkeypatch.setattr(
      grain_data_processing.instruction_data_processing,
      "load_chat_template_from_file",
      lambda path: None,
  )

  with pytest.raises(ValueError, match="Unable to load SFT chat template"):
    grain_data_processing._configure_sft_chat_template(
        _config(chat_template_path="/tmp/missing-template.jinja"),
        ["messages"],
        SimpleNamespace(chat_template="inference-template"),
        True,
    )


def test_hf_sft_pipeline_rejects_assistant_mask_mode():
  with pytest.raises(ValueError, match="supported only by the Grain SFT pipeline"):
    hf_data_processing.preprocessing_pipeline(
        dataloading_host_index=0,
        dataloading_host_count=1,
        global_mesh=None,
        dataset=None,
        config=None,
        data_column_names=["messages"],
        tokenize=True,
        tokenizer_path="unused",
        hf_access_token=None,
        global_batch_size=1,
        max_target_length=8,
        shuffle=False,
        data_shuffle_seed=0,
        use_sft=True,
        sft_chat_template_mode="assistant_mask",
    )
