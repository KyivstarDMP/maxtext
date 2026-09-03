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

"""Regression tests for retaining role=tool results as masked SFT context."""

import copy

import pytest

from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline.input_pipeline_utils import (
    SFT_PINNED_CONTEXT_IDS_KEY,
    SFTPromptMasking,
    apply_chat_template,
)


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
SENTINEL = "TOOL_RESULT_SENTINEL"


class _PrefixStableToolTokenizer:
  """Small character tokenizer with a tool-response opener seam."""

  name_or_path = "prefix-stable-tool-tokenizer"

  def __init__(self):
    self.enable_thinking_calls = []

  def _render(self, messages, add_generation_prompt, tools, enable_thinking):
    """Render a deterministic chat stream for delta and masking assertions."""
    assert tools == TOOLS
    self.enable_thinking_calls.append(enable_thinking)

    rendered = "<B>"
    loop_messages = messages
    if tools or (messages and messages[0]["role"] in ("system", "developer")):
      rendered += "<D>"
      if messages and messages[0]["role"] in ("system", "developer"):
        rendered += messages[0]["content"]
        loop_messages = messages[1:]
      if tools:
        rendered += "<TOOLS>"
      rendered += "</D>"

    for idx, message in enumerate(loop_messages):
      role = message["role"]
      if role == "user":
        rendered += f"<U>{message['content']}</U>"
      elif role == "assistant":
        rendered += "<A>"
        if message.get("tool_calls"):
          rendered += "<CALL>"
          # Both target tokenizer families end a terminal call-only render with
          # their function-response opener. When tools follow, the first tool
          # message emits that same opener at the identical stream position.
          if idx == len(loop_messages) - 1:
            rendered += "<R>"
        else:
          rendered += f"{message.get('content', '')}</A>"
      elif role == "tool":
        rendered += f"<R>{message['content']}</R>"
      else:
        raise ValueError(f"Unsupported role in test tokenizer: {role}")

    if add_generation_prompt:
      rendered += "<A>"
    return rendered

  def apply_chat_template(
      self,
      messages,
      *,
      add_generation_prompt,
      tokenize,
      enable_thinking,
      tools=None,
  ):
    """Expose the subset of the Hugging Face chat-template API used by MaxText."""
    rendered = self._render(messages, add_generation_prompt, tools, enable_thinking)
    return self.encode(rendered) if tokenize else rendered

  @staticmethod
  def encode(text):
    return [ord(char) + 1 for char in text]

  @staticmethod
  def decode(token_ids, skip_special_tokens=False):
    del skip_special_tokens
    return "".join(chr(int(token_id) - 1) for token_id in token_ids)


class _NonPrefixStableToolTokenizer(_PrefixStableToolTokenizer):
  name_or_path = "non-prefix-stable-tool-tokenizer"

  def _render(self, messages, add_generation_prompt, tools, enable_thinking):
    rendered = super()._render(messages, add_generation_prompt, tools, enable_thinking)
    if any(message["role"] == "tool" for message in messages):
      rendered = rendered.replace("<CALL>", "<CHANGED_CALL>", 1)
    return rendered


class _DivergentGenerationPromptTokenizer(_PrefixStableToolTokenizer):
  """Mimic a speculative thinking prefix absent from a content-only assistant render."""

  name_or_path = "divergent-generation-prompt-tokenizer"

  def _render(self, messages, add_generation_prompt, tools, enable_thinking):
    rendered = super()._render(messages, add_generation_prompt, tools, enable_thinking)
    if add_generation_prompt and messages[-1]["role"] == "tool":
      rendered += "<SPECULATIVE_THINKING_PREFIX>"
    return rendered


class _ToolIgnoringTokenizer(_PrefixStableToolTokenizer):
  """Mimic a template that ignores role=tool after an unstructured call string."""

  name_or_path = "tool-ignoring-tokenizer"

  def _render(self, messages, add_generation_prompt, tools, enable_thinking):
    rendered = super()._render(messages, add_generation_prompt, tools, enable_thinking)
    for message in messages:
      if message["role"] == "tool":
        rendered = rendered.replace(f"<R>{message['content']}</R>", "")
    return rendered


class _NonPrefixStableLeadingTokenizer(_PrefixStableToolTokenizer):
  name_or_path = "non-prefix-stable-leading-tokenizer"

  def _render(self, messages, add_generation_prompt, tools, enable_thinking):
    rendered = super()._render(messages, add_generation_prompt, tools, enable_thinking)
    if len(messages) == 1 and messages[0]["role"] == "developer":
      rendered = rendered.replace("<D>", "<CHANGED_LEADING>", 1)
    return rendered


class _RoundTripDriftLeadingTokenizer(_PrefixStableToolTokenizer):
  """Keep template tokenization stable but alter later segment encoding."""

  name_or_path = "round-trip-drift-leading-tokenizer"

  @staticmethod
  def _raw_encode(text):
    return [ord(char) + 1 for char in text]

  def apply_chat_template(
      self,
      messages,
      *,
      add_generation_prompt,
      tokenize,
      enable_thinking,
      tools=None,
  ):
    rendered = self._render(messages, add_generation_prompt, tools, enable_thinking)
    return self._raw_encode(rendered) if tokenize else rendered

  def encode(self, text):  # pylint: disable=arguments-differ
    if "<U>" in text:
      text = text.replace("<D>", "<ROUND_TRIP_CHANGED>", 1)
    return self._raw_encode(text)


def _tool_call():
  return {
      "role": "assistant",
      "content": "",
      "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
  }


def _standard_round(tool_messages=None):
  if tool_messages is None:
    tool_messages = [{"role": "tool", "name": "lookup", "content": SENTINEL}]
  return [
      {"role": "system", "content": "Follow the tool result."},
      {"role": "user", "content": "Look it up."},
      _tool_call(),
      *tool_messages,
      {"role": "assistant", "content": "The lookup succeeded."},
  ]


def _format(messages, tokenizer=None, enable_thinking=True):
  tokenizer = tokenizer or _PrefixStableToolTokenizer()
  return tokenizer, apply_chat_template(
      {"messages": copy.deepcopy(messages), "tools": copy.deepcopy(TOOLS)},
      tokenizer,
      "messages",
      "tools",
      enable_thinking=enable_thinking,
  )


def _format_with_pin(messages, tokenizer=None, enable_thinking=True):
  tokenizer = tokenizer or _PrefixStableToolTokenizer()
  return tokenizer, apply_chat_template(
      {"messages": copy.deepcopy(messages), "tools": copy.deepcopy(TOOLS)},
      tokenizer,
      "messages",
      "tools",
      pin_leading_context=True,
      enable_thinking=enable_thinking,
  )


def _tokenized_segments(tokenizer, formatted):
  return [tokenizer.encode(segment) for segment in formatted["messages"]]


def test_tool_result_is_emitted_once_as_masked_context_and_stream_is_token_exact():
  messages = _standard_round()
  tokenizer, formatted = _format(messages)

  assert formatted["is_prompt"] == [True, False, True, False]
  assert SENTINEL in formatted["messages"][2]
  assert "The lookup succeeded." in formatted["messages"][3]

  segment_ids = _tokenized_segments(tokenizer, formatted)
  assembled_ids = [token_id for segment in segment_ids for token_id in segment]
  canonical_ids = tokenizer.apply_chat_template(
      messages,
      tools=TOOLS,
      add_generation_prompt=False,
      tokenize=True,
      enable_thinking=True,
  )
  assert assembled_ids == canonical_ids
  assert tokenizer.decode(assembled_ids).count(SENTINEL) == 1

  masked = SFTPromptMasking("messages", completion_only=True, max_target_length=4096, unk_id=0).map(
      {"messages": segment_ids, "is_prompt": formatted["is_prompt"]}
  )
  input_text = tokenizer.decode(masked["inputs"])
  loss_text = tokenizer.decode(masked["targets"][masked["targets"] != 0])
  assert input_text.count(SENTINEL) == 1
  assert SENTINEL not in loss_text
  assert "<CALL><R>" in loss_text  # The model-emitted call-turn stop/EOM token is a loss target.
  assert "The lookup succeeded." in loss_text


def test_segmented_tool_round_passes_one_thinking_value_to_every_internal_render():
  tokenizer, _ = _format(_standard_round(), enable_thinking=False)

  assert tokenizer.enable_thinking_calls
  assert set(tokenizer.enable_thinking_calls) == {False}


def test_leading_developer_is_not_replayed_and_pin_is_exact():
  messages = [
      {"role": "developer", "content": "Follow the declared tools."},
      {"role": "user", "content": "First question."},
      {"role": "assistant", "content": "First answer."},
      {"role": "user", "content": "Second question."},
      {"role": "assistant", "content": "Second answer."},
  ]
  tokenizer, formatted = _format_with_pin(messages)
  segment_ids = _tokenized_segments(tokenizer, formatted)
  assembled_text = tokenizer.decode([token for segment in segment_ids for token in segment])
  pinned_text = tokenizer.decode(formatted[SFT_PINNED_CONTEXT_IDS_KEY])

  assert formatted["is_prompt"] == [True, False, True, False]
  assert pinned_text == "<B><D>Follow the declared tools.<TOOLS></D>"
  assert formatted["messages"][0].startswith(pinned_text)
  assert assembled_text.count("Follow the declared tools.") == 1
  assert formatted["messages"][0].count("<TOOLS>") == 1
  assert formatted["messages"][2].count("<TOOLS>") == 1


def test_tools_only_conversation_pins_generated_developer_block():
  messages = [
      {"role": "user", "content": "Use a tool."},
      {"role": "assistant", "content": "Done."},
  ]
  tokenizer, formatted = _format_with_pin(messages)

  assert tokenizer.decode(formatted[SFT_PINNED_CONTEXT_IDS_KEY]) == "<B><D><TOOLS></D>"


def test_pinned_context_prefix_mismatch_has_bounded_diagnostics():
  messages = [
      {"role": "developer", "content": "Instructions must not be dumped in full."},
      {"role": "user", "content": "Use a tool."},
      {"role": "assistant", "content": "Done."},
  ]
  with pytest.raises(ValueError) as exc_info:
    _format_with_pin(messages, _NonPrefixStableLeadingTokenizer())

  message = str(exc_info.value)
  assert "pinned-context mismatch" in message
  assert "Tokenizer: non-prefix-stable-leading-tokenizer" in message
  assert "revision: unknown" in message
  assert "divergence offset:" in message
  assert len(message) < 1500


def test_pinned_context_decode_encode_mismatch_fails_before_windowing():
  messages = [
      {"role": "developer", "content": "Instructions must remain token-exact."},
      {"role": "user", "content": "Use a tool."},
      {"role": "assistant", "content": "Done."},
  ]
  tokenizer, formatted = _format_with_pin(messages, _RoundTripDriftLeadingTokenizer())

  with pytest.raises(ValueError) as exc_info:
    grain_data_processing._tokenize_sft_chunks(formatted, "messages", tokenizer)  # pylint: disable=protected-access

  message = str(exc_info.value)
  assert "pinned-context mismatch" in message
  assert "decode/encode boundary" in message
  assert "Tokenizer: round-trip-drift-leading-tokenizer" in message
  assert "Instructions must remain" not in message
  assert len(message) < 1500


def test_developer_message_after_index_zero_is_rejected():
  messages = [
      {"role": "user", "content": "First question."},
      {"role": "developer", "content": "Too late."},
      {"role": "assistant", "content": "Answer."},
  ]
  with pytest.raises(ValueError, match="'developer' message found at index 1"):
    _format(messages)


def test_consecutive_tool_results_are_emitted_once_and_in_order():
  tool_messages = [
      {"role": "tool", "name": "lookup", "content": "FIRST_TOOL_RESULT"},
      {"role": "tool", "name": "lookup", "content": "SECOND_TOOL_RESULT"},
  ]
  messages = _standard_round(tool_messages)
  tokenizer, formatted = _format(messages)
  assembled_ids = [token_id for segment in _tokenized_segments(tokenizer, formatted) for token_id in segment]
  assembled_text = tokenizer.decode(assembled_ids)

  assert formatted["is_prompt"] == [True, False, True, False]
  assert assembled_text.count("FIRST_TOOL_RESULT") == 1
  assert assembled_text.count("SECOND_TOOL_RESULT") == 1
  assert assembled_text.index("FIRST_TOOL_RESULT") < assembled_text.index("SECOND_TOOL_RESULT")
  assert assembled_ids == tokenizer.apply_chat_template(
      messages,
      tools=TOOLS,
      add_generation_prompt=False,
      tokenize=True,
      enable_thinking=True,
  )


def test_two_tool_call_rounds_without_leading_reseed_mask_each_result_once():
  first_sentinel = "FIRST_ROUND_TOOL_RESULT"
  second_sentinel = "SECOND_ROUND_TOOL_RESULT"
  leading_messages = [{"role": "system", "content": "Follow every tool result."}]
  first_round = [
      {"role": "user", "content": "First lookup."},
      _tool_call(),
      {"role": "tool", "name": "lookup", "content": first_sentinel},
      {"role": "assistant", "content": "First answer."},
  ]
  second_round = [
      {"role": "user", "content": "Second lookup."},
      _tool_call(),
      {"role": "tool", "name": "lookup", "content": second_sentinel},
      {"role": "assistant", "content": "Second answer."},
  ]
  messages = leading_messages + first_round + second_round
  tokenizer, formatted = _format(messages)

  assert formatted["is_prompt"] == [True, False, True, False, True, False, True, False]
  segment_ids = _tokenized_segments(tokenizer, formatted)
  first_round_ids = [token_id for segment in segment_ids[:4] for token_id in segment]
  second_round_ids = [token_id for segment in segment_ids[4:] for token_id in segment]
  for round_messages, assembled_ids, sentinel in (
      (leading_messages + first_round, first_round_ids, first_sentinel),
      (second_round, second_round_ids, second_sentinel),
  ):
    canonical_ids = tokenizer.apply_chat_template(
        round_messages,
        tools=TOOLS,
        add_generation_prompt=False,
        tokenize=True,
        enable_thinking=True,
    )
    assert assembled_ids == canonical_ids
    assert tokenizer.decode(assembled_ids).count(sentinel) == 1

  masked = SFTPromptMasking("messages", completion_only=True, max_target_length=4096, unk_id=0).map(
      {"messages": segment_ids, "is_prompt": formatted["is_prompt"]}
  )
  input_text = tokenizer.decode(masked["inputs"])
  loss_text = tokenizer.decode(masked["targets"][masked["targets"] != 0])
  assert input_text.count(first_sentinel) == 1
  assert input_text.count(second_sentinel) == 1
  assert first_sentinel not in loss_text
  assert second_sentinel not in loss_text


def test_noncanonical_generation_prompt_suffix_is_not_added_to_assembled_stream():
  messages = _standard_round()
  tokenizer, formatted = _format(messages, _DivergentGenerationPromptTokenizer())
  assembled_ids = [token_id for segment in _tokenized_segments(tokenizer, formatted) for token_id in segment]

  assert "<SPECULATIVE_THINKING_PREFIX>" not in formatted["messages"][2]
  assert assembled_ids == tokenizer.apply_chat_template(
      messages,
      tools=TOOLS,
      add_generation_prompt=False,
      tokenize=True,
      enable_thinking=True,
  )


def test_tool_to_user_adjacency_does_not_emit_a_second_tool_segment():
  messages = [
      {"role": "system", "content": "Follow the tool result."},
      {"role": "user", "content": "Look it up."},
      _tool_call(),
      {"role": "tool", "name": "lookup", "content": SENTINEL},
      {"role": "user", "content": "Now explain it."},
      {"role": "assistant", "content": "The lookup succeeded."},
  ]
  tokenizer, formatted = _format(messages)
  assembled_text = tokenizer.decode(
      [token_id for segment in _tokenized_segments(tokenizer, formatted) for token_id in segment]
  )

  assert formatted["is_prompt"] == [True, False, True, False]
  assert assembled_text.count(SENTINEL) == 1
  assert sum(SENTINEL in segment for segment in formatted["messages"]) == 1


def test_prefix_mismatch_error_has_bounded_structural_diagnostics():
  with pytest.raises(ValueError) as exc_info:
    _format(_standard_round(), _NonPrefixStableToolTokenizer())

  message = str(exc_info.value)
  assert "baseline render is not an exact token prefix" in message
  assert "Tokenizer: non-prefix-stable-tool-tokenizer" in message
  assert "Trailing tool messages: 1" in message
  assert "divergence offset:" in message
  assert "Baseline divergence window" in message
  assert "Superset divergence window" in message
  assert SENTINEL not in message


def test_template_that_ignores_tool_message_fails_instead_of_silently_dropping_it():
  messages = [
      {"role": "system", "content": "Follow the tool result."},
      {"role": "user", "content": "Look it up."},
      {"role": "assistant", "content": "[lookup()]"},
      {"role": "tool", "name": "lookup", "content": SENTINEL},
      {"role": "assistant", "content": "The lookup succeeded."},
  ]
  with pytest.raises(ValueError, match="emitted no tool-result context tokens") as exc_info:
    _format(messages, _ToolIgnoringTokenizer())

  assert "Tokenizer: tool-ignoring-tokenizer" in str(exc_info.value)
  assert SENTINEL not in str(exc_info.value)
