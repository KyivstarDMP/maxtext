# Copyright 2026 Google LLC
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

"""Offline BPE regressions for segmented boundaries and consistent tools/pin kwargs."""

import copy
import json

import pytest
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from maxtext.input_pipeline import input_pipeline_utils as utils


pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]


def _tokenizer(rewrite_context=False):
  """Construct a real BPE tokenizer and generation-marked template without downloads."""
  tokens = [
      "[UNK]",
      "[PAD]",
      "[B]",
      "[D]",
      "[U]",
      "[A]",
      "[E]",
      "[T]",
      "u",
      "v",
      "h",
      "e",
      "l",
      "o",
      "he",
      "hel",  # codespell:ignore hel
      "hell",
      "hello",
  ]
  backend = Tokenizer(
      models.BPE(
          vocab={token: index for index, token in enumerate(tokens)},
          merges=[("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")],  # codespell:ignore hel
          unk_token="[UNK]",
      )
  )
  backend.decoder = decoders.Fuse()
  tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", bos_token="[B]")
  tokenizer.add_special_tokens({"additional_special_tokens": ["[D]", "[U]", "[A]", "[E]", "[T]"]})
  tokenizer.chat_template = (
      "[B]{% if tools is defined and tools is not none %}[T]{% endif %}"
      "{% for message in messages %}"
      "{% if message.role == 'developer' and message.content %}[D]{{ message.content }}[E]"
      "{% elif message.role == 'user' %}[U]"
      + (
          "{% if messages[-1].role == 'assistant' %}v{% else %}{{ message.content }}{% endif %}"
          if rewrite_context
          else "{{ message.content }}"
      )
      + "[E]{% elif message.role == 'assistant' %}[A]{% generation %}{{ message.content }}[E]{% endgeneration %}"
      "{% endif %}{% endfor %}{% if add_generation_prompt %}[A]h{% endif %}"
  )
  return tokenizer


def test_bpe_assistant_prefill_merge_keeps_full_ids_and_completion_targets():
  tokenizer = _tokenizer()
  messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "hello"}]
  formatted = utils.apply_chat_template({"messages": messages}, tokenizer, "messages")
  runs = formatted[utils.SFT_SEGMENT_IDS_KEY]
  # The prompt's 'h' merges into the single 'hello' token in the completed render.
  # This divergence is after the complete user context, and must remain supported.
  assert runs == [[2, 4, 8, 6, 5], [17, 6]]
  assert formatted["is_prompt"] == [True, False]
  assert sum(runs, []) == tokenizer.apply_chat_template(
      messages, tokenize=True, add_generation_prompt=False, return_dict=False
  )
  row = utils.SFTPromptMasking("messages", completion_only=True, max_target_length=32, unk_id=1).map(
      {"messages": runs, "is_prompt": formatted["is_prompt"]}
  )
  assert row["inputs"].tolist() == [2, 4, 8, 6, 5, 17, 6]
  assert row["targets"].tolist() == [1, 1, 1, 1, 1, 17, 6]


def test_bpe_context_rewrite_raises_instead_of_supervising_user_tokens():
  tokenizer = _tokenizer(rewrite_context=True)
  messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "hello"}]
  with pytest.raises(ValueError, match="changes established context tokens"):
    utils.apply_chat_template({"messages": messages}, tokenizer, "messages")


@pytest.mark.parametrize("mode", ["segmented", "assistant_mask"])
@pytest.mark.parametrize(
    "tools", [None, [], "[]", [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]]
)
def test_empty_tools_are_absent_consistently_for_full_render_and_pin(mode, tools):
  tokenizer = _tokenizer()
  messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "hello"}]
  example = {"messages": copy.deepcopy(messages)}
  if tools is not None:
    example["tools"] = tools
  formatter = utils.apply_chat_template if mode == "segmented" else utils.apply_chat_template_with_assistant_mask
  formatted = formatter(example, tokenizer, "messages", "tools", pin_leading_context=True)
  nonempty = bool(json.loads(tools) if isinstance(tools, str) else tools)
  expected_pin = [2, 7] if nonempty else [2]
  assert formatted[utils.SFT_PINNED_CONTEXT_IDS_KEY] == expected_pin
  runs = formatted[utils.SFT_SEGMENT_IDS_KEY] if mode == "segmented" else formatted["messages"]
  assert sum(runs, []) == expected_pin + [4, 8, 6, 5, 17, 6]
  assert formatted["is_prompt"] == [True, False]
