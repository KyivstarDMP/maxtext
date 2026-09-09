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

"""Token-native SFT mask-adapter tests with no tokenizer, Grain reader, or trainer."""

import pytest

from maxtext.input_pipeline.input_pipeline_utils import (
    SFT_PINNED_CONTEXT_IDS_KEY,
    SFTPromptMasking,
    SFTPromptMaskingWindows,
    split_sft_token_stream_by_assistant_mask,
)

pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]

PAD = 0


def _loss_tokens(records):
  """Return the non-masked targets in training order."""
  return [int(token) for record in records for token in record["targets"] if token != PAD]


def _count_subsequence(tokens, subsequence):
  return sum(
      tokens[index : index + len(subsequence)] == subsequence for index in range(len(tokens) - len(subsequence) + 1)
  )


def test_splitter_preserves_canonical_ids_and_coalesces_equal_mask_runs():
  input_ids = [11, 12, 101, 102, 21, 22, 201, 202, 23, 301]
  assistant_mask = [False, False, 1, 1, 0, 0, True, True, 0, 1]

  token_runs, is_prompt = split_sft_token_stream_by_assistant_mask(input_ids, assistant_mask)

  assert token_runs == [[11, 12], [101, 102], [21, 22], [201, 202], [23], [301]]
  assert is_prompt == [True, False, True, False, True, False]
  assert [token for run in token_runs for token in run] == input_ids


@pytest.mark.parametrize(
    ("input_ids", "assistant_mask", "error_type", "match"),
    [
        ([], [], ValueError, "at least one token"),
        ([1, 2], [0], ValueError, "identical lengths"),
        ([1, 2], [0, 2], ValueError, "only integer/bool 0 or 1"),
        ([1, 2, 3], [0, 0, 0], ValueError, "no loss-bearing tokens"),
        ([1, object()], [0, 1], TypeError, "integer token IDs"),
    ],
)
def test_splitter_rejects_malformed_token_streams(input_ids, assistant_mask, error_type, match):
  with pytest.raises(error_type, match=match):
    split_sft_token_stream_by_assistant_mask(input_ids, assistant_mask)


def test_short_canonical_stream_reaches_existing_masking_contract_token_exact():
  """Alternating template mask spans need no downstream masking changes."""
  input_ids = [1, 10, 11, 101, 102, 20, 21, 201, 202, 22, 301, 106]
  assistant_mask = [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1]
  token_runs, is_prompt = split_sft_token_stream_by_assistant_mask(input_ids, assistant_mask)

  record = SFTPromptMasking(
      "text",
      completion_only=True,
      max_target_length=32,
      unk_id=PAD,
  ).map({"text": token_runs, "is_prompt": is_prompt})

  assert record["inputs"].tolist() == input_ids
  assert record["targets"].tolist() == [
      token_id if is_assistant else PAD for token_id, is_assistant in zip(input_ids, assistant_mask, strict=True)
  ]
  assert _loss_tokens([record]) == [
      token_id for token_id, is_assistant in zip(input_ids, assistant_mask, strict=True) if is_assistant
  ]


def test_long_canonical_stream_preserves_pin_and_loss_tiling_through_windows():
  pinned = [1, 2, 3]
  leading_context = pinned + list(range(10, 26))
  tool_call = [101, 102, 50]
  tool_result = list(range(201, 209))
  first_answer = list(range(301, 325)) + [106]
  later_user = list(range(401, 407))
  second_answer = list(range(501, 511)) + [106]
  input_ids = leading_context + tool_call + tool_result + first_answer + later_user + second_answer
  assistant_mask = (
      [0] * len(leading_context)
      + [1] * len(tool_call)
      + [0] * len(tool_result)
      + [1] * len(first_answer)
      + [0] * len(later_user)
      + [1] * len(second_answer)
  )
  expected_loss = [token_id for token_id, is_assistant in zip(input_ids, assistant_mask, strict=True) if is_assistant]
  token_runs, is_prompt = split_sft_token_stream_by_assistant_mask(input_ids, assistant_mask)
  element = {"text": token_runs, "is_prompt": is_prompt, SFT_PINNED_CONTEXT_IDS_KEY: pinned}

  pinned_records = SFTPromptMaskingWindows(
      "text",
      completion_only=True,
      max_target_length=20,
      unk_id=PAD,
      overlap=2,
      pin_leading_context=True,
  ).flat_map(element)
  unpinned_records = SFTPromptMaskingWindows(
      "text",
      completion_only=True,
      max_target_length=20,
      unk_id=PAD,
      overlap=2,
      pin_leading_context=False,
  ).flat_map(element)

  assert len(pinned_records) > 1
  assert all(len(record["inputs"]) <= 20 for record in pinned_records)
  assert _loss_tokens(pinned_records) == expected_loss
  assert _loss_tokens(unpinned_records) == expected_loss
  for record in pinned_records:
    inputs = record["inputs"].tolist()
    targets = record["targets"].tolist()
    assert inputs[: len(pinned)] == pinned
    assert targets[: len(pinned)] == [PAD] * len(pinned)
    assert _count_subsequence(inputs, pinned) == 1
