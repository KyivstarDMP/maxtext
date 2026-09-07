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

# pylint: disable=protected-access

import copy
import json
import os

import pytest

from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import input_pipeline_utils
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
  return grain_data_processing._tokenize_sft_chunks(copy.deepcopy(formatted), "messages", tokenizer)["messages"]


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


def test_pinned_context_uses_original_ids_despite_decode_encode_drift():
  messages = [
      {"role": "developer", "content": "Instructions must remain token-exact."},
      {"role": "user", "content": "Use a tool."},
      {"role": "assistant", "content": "Done."},
  ]
  tokenizer, formatted = _format_with_pin(messages, _RoundTripDriftLeadingTokenizer())

  tokenized = grain_data_processing._tokenize_sft_chunks(copy.deepcopy(formatted), "messages", tokenizer)
  assert tokenized["messages"][0][: len(formatted[SFT_PINNED_CONTEXT_IDS_KEY])] == formatted[SFT_PINNED_CONTEXT_IDS_KEY]
  assert input_pipeline_utils.SFT_SEGMENT_IDS_KEY not in tokenized

  # Legacy callers without the side column retain the fallback pin guard.
  formatted.pop(input_pipeline_utils.SFT_SEGMENT_IDS_KEY)

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


def test_tool_to_user_adjacency_preserves_the_complete_stream_and_ownership():
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

  assert formatted["is_prompt"] == [True, False, True, True, False]
  assert assembled_text.count(SENTINEL) == 1
  assert sum(SENTINEL in segment for segment in formatted["messages"]) == 1
  assert assembled_text == tokenizer.decode(
      tokenizer.apply_chat_template(
          messages,
          tools=TOOLS,
          add_generation_prompt=False,
          tokenize=True,
          enable_thinking=True,
      )
  )
  masked = SFTPromptMasking("messages", completion_only=True, max_target_length=4096, unk_id=0).map(
      {"messages": _tokenized_segments(tokenizer, formatted), "is_prompt": formatted["is_prompt"]}
  )
  assert tokenizer.decode(masked["targets"][masked["targets"] != 0]) == "<CALL><R>The lookup succeeded.</A>"


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


class _DummyPrefixTokenizer(_PrefixStableToolTokenizer):
  """Model word-initial whitespace added by standalone encoding."""

  def encode(self, text):  # pylint: disable=arguments-differ
    return super().encode((" " if text and text[0].isalpha() else "") + text)


def test_original_segment_ids_bypass_dummy_prefix_encoding():
  messages = _standard_round()
  tokenizer, formatted = _format(messages, _DummyPrefixTokenizer())
  assert tokenizer.encode(formatted["messages"][2]) != formatted[input_pipeline_utils.SFT_SEGMENT_IDS_KEY][2]
  assembled = [i for segment in _tokenized_segments(tokenizer, formatted) for i in segment]
  assert assembled == tokenizer.apply_chat_template(
      messages,
      tools=TOOLS,
      add_generation_prompt=False,
      tokenize=True,
      enable_thinking=True,
  )


@pytest.mark.parametrize(
    "bad_ids", [[], [[1]], [[1], [], [2], [3]], [[1], [True], [2], [3]], [[1], [1.5], [2], [3]], [[1], [-1], [2], [3]]]
)
def test_invalid_segment_side_column_is_rejected(bad_ids):
  tokenizer, formatted = _format(_standard_round())
  formatted[input_pipeline_utils.SFT_SEGMENT_IDS_KEY] = bad_ids
  with pytest.raises(ValueError, match="sft_segment_ids"):
    grain_data_processing._tokenize_sft_chunks(formatted, "messages", tokenizer)


def test_carried_ids_still_validate_the_leading_pin():
  tokenizer, formatted = _format_with_pin(_standard_round())
  formatted[input_pipeline_utils.SFT_SEGMENT_IDS_KEY][0][0] += 1
  with pytest.raises(ValueError, match="pinned-context mismatch"):
    grain_data_processing._tokenize_sft_chunks(formatted, "messages", tokenizer)


def test_hf_batched_map_carries_unequal_segment_counts_without_encoding():
  import datasets  # pylint: disable=import-outside-toplevel

  tok, first = _format(_standard_round())
  _, second = _format([{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}])
  key = input_pipeline_utils.SFT_SEGMENT_IDS_KEY
  rows = [{k: row[k] for k in ("messages", "is_prompt", key)} for row in (first, second)]

  def never_encode(*args, **kwargs):
    raise AssertionError("HF must consume the original IDs")

  dataset = (
      datasets.Dataset.from_list(rows)
      .map(
          input_pipeline_utils.tokenization,
          batched=True,
          batch_size=2,
          fn_kwargs={"hf_tokenizer": never_encode, "truncation": False, "max_length": 4096, "column_names": ["messages"]},
      )
      .remove_columns([key])
  )
  assert dataset[0]["messages"] == first[key]
  assert dataset[1]["messages"] == second[key]
  assert key not in dataset.column_names
  assert len(dataset[0]["messages"]) != len(dataset[1]["messages"])
  assert tok.decode(dataset[0]["messages"][2]).startswith(SENTINEL)


def test_string_only_formatter_does_not_expose_side_column():
  tok = _PrefixStableToolTokenizer()
  result = apply_chat_template(
      {"messages": _standard_round(), "tools": TOOLS},
      tok,
      "messages",
      "tools",
      return_segment_ids=False,
  )
  assert input_pipeline_utils.SFT_SEGMENT_IDS_KEY not in result
  assert all(isinstance(chunk, str) for chunk in result["messages"])


def test_interrupted_tool_round_then_second_call_is_token_exact():
  messages = _standard_round()[:-1] + [
      {"role": "user", "content": "Check again."},
      _tool_call(),
      {"role": "tool", "content": "SECOND_RESULT", "name": "lookup"},
      {"role": "assistant", "content": "Both checked."},
  ]
  tok, result = _format(messages)
  assert [i for segment in _tokenized_segments(tok, result) for i in segment] == tok.apply_chat_template(
      messages,
      tools=TOOLS,
      add_generation_prompt=False,
      tokenize=True,
      enable_thinking=True,
  )
  masked = SFTPromptMasking("messages", completion_only=True, max_target_length=4096, unk_id=0).map(
      {"messages": _tokenized_segments(tok, result), "is_prompt": result["is_prompt"]}
  )
  assert tok.decode(masked["targets"][masked["targets"] != 0]) == "<CALL><R><CALL><R>Both checked.</A>"


class _UserSpeculativeTokenizer(_PrefixStableToolTokenizer):
  """Keep the existing no-think speculative prefix at each user boundary."""

  def _render(self, messages, add_generation_prompt, tools, enable_thinking):
    text = super()._render(messages, add_generation_prompt, tools, enable_thinking)
    return text + ("§SPEC§" if add_generation_prompt and not enable_thinking else "")


def test_no_think_user_prompts_keep_both_speculative_insertions():
  messages = _standard_round()[:-1] + [
      {"role": "user", "content": "Next question."},
      {"role": "assistant", "content": "Answer."},
  ]
  tok, result = _format(messages, _UserSpeculativeTokenizer(), enable_thinking=False)
  expected = (
      "<B><D>Follow the tool result.<TOOLS></D><U>Look it up.</U><A>§SPEC§<CALL><R>"
      + SENTINEL
      + "</R><U>Next question.</U><A>§SPEC§Answer.</A>"
  )
  assert [i for segment in _tokenized_segments(tok, result) for i in segment] == tok.encode(expected)
  assert result["is_prompt"] == [True, False, True, True, False]
  # Direct tool->assistant continuation still trims speculative prompt tokens.
  _, direct = _format(_standard_round(), _UserSpeculativeTokenizer(), enable_thinking=False)
  assert "§SPEC§" in direct["messages"][0]
  assert all("§SPEC§" not in text for text in direct["messages"][1:])


class _BodyDroppingTokenizer(_PrefixStableToolTokenizer):
  """Retain response delimiters while dropping all or later result bodies."""

  def __init__(self, keep_first=False):
    super().__init__()
    self.keep_first = keep_first

  def _render(self, messages, add_generation_prompt, tools, enable_thinking):
    messages = copy.deepcopy(messages)
    result_count = 0
    for message in messages:
      if message["role"] == "tool":
        if not self.keep_first or result_count:
          message["content"] = ""
        result_count += 1
    return super()._render(messages, add_generation_prompt, tools, enable_thinking)


@pytest.mark.parametrize("next_role", ["assistant", "user"])
@pytest.mark.parametrize("keep_first", [False, True])
def test_missing_tool_bodies_fail_at_both_segmented_boundaries(next_role, keep_first):
  results = [{"role": "tool", "content": body} for body in (SENTINEL, "SECOND_PRIVATE_BODY")]
  messages = _standard_round(results)
  if next_role == "user":
    messages.insert(-1, {"role": "user", "content": "Continue."})
  with pytest.raises(ValueError, match="dropped or altered a tool result body") as exc_info:
    _format(messages, _BodyDroppingTokenizer(keep_first))
  diagnostic = str(exc_info.value)
  assert f"result index/count: {2 if keep_first else 1}/2" in diagnostic
  assert "body length:" in diagnostic
  assert len(diagnostic) < 1000
  assert all(result["content"] not in diagnostic for result in results)


@pytest.mark.parametrize("next_role", ["assistant", "user"])
def test_identical_tool_bodies_need_two_distinct_occurrences(next_role):
  messages = _standard_round([{"role": "tool", "content": SENTINEL} for _ in range(2)])
  if next_role == "user":
    messages.insert(-1, {"role": "user", "content": "Continue."})
  _, result = _format(messages)
  assert sum(text.count(SENTINEL) for text in result["messages"]) == 2
  with pytest.raises(ValueError, match="result index/count: 2/2"):
    _format(messages, _BodyDroppingTokenizer(keep_first=True))


class _JsonEscapedBodyTokenizer(_PrefixStableToolTokenizer):

  def _render(self, messages, add_generation_prompt, tools, enable_thinking):
    messages = copy.deepcopy(messages)
    for message in messages:
      if message["role"] == "tool":
        message["content"] = json.dumps(message["content"].strip())[1:-1]
    return super()._render(messages, add_generation_prompt, tools, enable_thinking)


@pytest.mark.parametrize("escaped", [False, True])
def test_whitespace_and_unicode_string_bodies_accept_supported_serialization(escaped):
  body = '  Kyiv: "ясно"\nnext line  '
  tokenizer = _JsonEscapedBodyTokenizer() if escaped else _PrefixStableToolTokenizer()
  _format(_standard_round([{"role": "tool", "content": body}]), tokenizer)


def test_structured_tool_bodies_are_skipped_with_one_warning_per_worker(monkeypatch):
  monkeypatch.setattr(input_pipeline_utils, "_TOOL_BODY_WARNING_PIDS", set())
  messages = _standard_round([{"role": "tool", "content": {"value": 1}}, {"role": "tool", "content": [2]}])
  with pytest.warns(UserWarning, match="Skipping non-string tool result bodies") as captured:
    _format(messages)
    _format(messages)
  assert len(captured) == 1


@pytest.mark.parametrize(
    "bodies,runs,valid",
    [
        (["one", "two"], ["one then two"], True),
        (["one", "two"], ["two then one"], False),
        (["same", "same"], ["same", "same"], True),
        (["same", "same"], ["same"], False),
        (["result"], ["res", "ult"], False),
        (["  ", "result"], ["result"], True),
    ],
)
def test_body_presence_search_consumes_occurrences_in_stream_order(bodies, runs, valid):
  args = (_PrefixStableToolTokenizer(), [{"content": body} for body in bodies], runs)
  if valid:
    input_pipeline_utils.validate_tool_result_bodies(*args, mode="test", roles=["tool"] * len(bodies))
  else:
    with pytest.raises(ValueError, match="dropped or altered a tool result body"):
      input_pipeline_utils.validate_tool_result_bodies(*args, mode="test", roles=["tool"] * len(bodies))


@pytest.mark.skipif(os.environ.get("MAXTEXT_RUN_NETWORK_TESTS") != "1", reason="network integration is opt-in")
@pytest.mark.parametrize("next_role", ["assistant", "user"])
def test_public_smol_template_preserves_string_tool_bodies(next_role):
  from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

  tokenizer = AutoTokenizer.from_pretrained(
      "HuggingFaceTB/SmolLM3-3B",
      revision="a07cc9a04f16550a088caea529712d1d335b0ac1",
      add_bos_token=False,
      add_eos_token=False,
  )
  messages = _standard_round(
      [
          {"role": "tool", "name": "lookup", "content": '  First "quoted" result  '},
          {"role": "tool", "name": "lookup", "content": "Kyiv: ясно\nSecond line"},
      ]
  )
  if next_role == "user":
    messages.insert(-1, {"role": "user", "content": "Now explain it."})
  _, formatted = _format(messages, tokenizer)
  expected = input_pipeline_utils.extract_token_ids(
      tokenizer.apply_chat_template(
          messages,
          tools=TOOLS,
          tokenize=True,
          add_generation_prompt=False,
          enable_thinking=True,
      )
  )
  assert [token for run in _tokenized_segments(tokenizer, formatted) for token in run] == expected
  canonical = input_pipeline_utils.apply_chat_template_with_assistant_mask(
      {"messages": copy.deepcopy(messages), "tools": TOOLS},
      tokenizer,
      "messages",
      "tools",
  )
  assert [token for run in canonical["messages"] for token in run] == expected
