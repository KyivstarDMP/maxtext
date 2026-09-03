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

"""Pure-logic tests for SFTPromptMaskingWindows (no tokenizer / no GCS).

Validates that long SFT examples keep the turn terminator (e.g. <end_of_turn>) in the loss by
windowing instead of head-truncating, and that the fast path matches SFTPromptMasking exactly.
"""
import pytest

pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]

import numpy as np

from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline.input_pipeline_utils import (
    SFT_PINNED_CONTEXT_IDS_KEY,
    SFTPromptMasking,
    SFTPromptMaskingWindows,
    get_sft_window_geometry,
)

PAD = 0
EOT = 129  # stand-in for <end_of_turn>


def _loss_tokens(record):
  """Tokens that actually contribute to the loss = targets != pad."""
  t = record["targets"]
  return [int(x) for x in t[t != PAD]]


def _count_subsequence(tokens, subsequence):
  return sum(
      tokens[index : index + len(subsequence)] == subsequence for index in range(len(tokens) - len(subsequence) + 1)
  )


def test_short_example_matches_sftpromptmasking():
  """An example that fits in max_target_length yields one record, identical to SFTPromptMasking."""
  length = 20
  element = {
      "text": [[1, 2, 3], [101, 102, 103, EOT]],
      "is_prompt": [True, False],
      SFT_PINNED_CONTEXT_IDS_KEY: [1, 2],
  }
  windows = SFTPromptMaskingWindows(
      "text",
      completion_only=True,
      max_target_length=length,
      unk_id=PAD,
      pin_leading_context=True,
  )
  recs = windows.flat_map({k: list(v) for k, v in element.items()})

  baseline = SFTPromptMasking("text", completion_only=True, max_target_length=length, unk_id=PAD)
  expected = baseline.map({k: list(v) for k, v in element.items()})

  assert len(recs) == 1
  assert np.array_equal(recs[0]["inputs"], expected["inputs"])
  assert np.array_equal(recs[0]["targets"], expected["targets"])


def test_long_prefix_keeps_exact_pin_and_freshest_tail_masked():
  """A real left cut keeps one pin plus the newest tail, all as masked context."""
  length = 20
  pinned = [1, 2, 3]
  prompt = pinned + list(range(10, 22))
  completion = list(range(100, 115)) + [EOT]
  windows = SFTPromptMaskingWindows(
      "text",
      completion_only=True,
      max_target_length=length,
      unk_id=PAD,
      overlap=2,
      pin_leading_context=True,
  )
  recs = windows.flat_map(
      {
          "text": [prompt, completion],
          "is_prompt": [True, False],
          SFT_PINNED_CONTEXT_IDS_KEY: pinned,
      }
  )

  geometry = get_sft_window_geometry(length, overlap=2, context_cap=-1)
  expected_context = pinned + prompt[-(geometry.context_cap - len(pinned)) :]
  for record in recs:
    inputs = [int(token) for token in record["inputs"]]
    targets = [int(token) for token in record["targets"]]
    assert inputs[: geometry.context_cap] == expected_context
    assert targets[: geometry.context_cap] == [PAD] * geometry.context_cap
    assert _count_subsequence(inputs, pinned) == 1
  assert [token for record in recs for token in _loss_tokens(record)] == completion


def test_later_completion_window_keeps_pin_and_all_completion_loss_once():
  """Pinning also protects later assistant turns after the accumulated conversation exceeds the cap."""
  length = 16
  pinned = [1, 2]
  completion_one = [101, 102, EOT]
  completion_two = [201, 202, 203, 204, 205, EOT]
  segments = [
      pinned + [3, 4, 5, 6],
      completion_one,
      [7, 8, 9, 10],
      completion_two,
  ]
  windows = SFTPromptMaskingWindows(
      "text",
      completion_only=True,
      max_target_length=length,
      unk_id=PAD,
      overlap=2,
      pin_leading_context=True,
  )
  recs = windows.flat_map(
      {
          "text": segments,
          "is_prompt": [True, False, True, False],
          SFT_PINNED_CONTEXT_IDS_KEY: pinned,
      }
  )

  later_records = [record for record in recs if completion_two[0] in _loss_tokens(record)]
  assert later_records
  for record in later_records:
    assert [int(token) for token in record["inputs"][: len(pinned)]] == pinned
    assert [int(token) for token in record["targets"][: len(pinned)]] == [PAD] * len(pinned)
  assert [token for record in recs for token in _loss_tokens(record)] == completion_one + completion_two


def test_empty_pin_is_byte_identical_to_existing_tail_context_behavior():
  """Enabling the feature without a pinnable leading block does not alter existing windows."""
  element = {
      "text": [list(range(1, 16)), list(range(100, 120))],
      "is_prompt": [True, False],
      SFT_PINNED_CONTEXT_IDS_KEY: [],
  }
  baseline = SFTPromptMaskingWindows("text", completion_only=True, max_target_length=20, unk_id=PAD, overlap=2).flat_map(
      element
  )
  pinned = SFTPromptMaskingWindows(
      "text",
      completion_only=True,
      max_target_length=20,
      unk_id=PAD,
      overlap=2,
      pin_leading_context=True,
  ).flat_map(element)

  assert len(pinned) == len(baseline)
  for actual, expected in zip(pinned, baseline, strict=True):
    assert np.array_equal(actual["inputs"], expected["inputs"])
    assert np.array_equal(actual["targets"], expected["targets"])


def test_pin_at_effective_cap_raises_before_silently_losing_tail_context():
  """A complete leading block that fills the cap must receive a data/config disposition."""
  pinned = list(range(1, 11))
  windows = SFTPromptMaskingWindows(
      "text",
      completion_only=True,
      max_target_length=20,
      unk_id=PAD,
      overlap=2,
      pin_leading_context=True,
  )
  with pytest.raises(ValueError, match="leaves no recent-conversation tail budget"):
    windows.flat_map(
        {
            "text": [pinned + [11, 12], list(range(100, 116))],
            "is_prompt": [True, False],
            SFT_PINNED_CONTEXT_IDS_KEY: pinned,
        }
    )


def test_large_but_valid_pin_warns_without_changing_tokens(monkeypatch):
  """A near-degenerate pin is allowed, counted, and warned without changing construction."""
  warnings = []
  monkeypatch.setattr(input_pipeline_utils.max_logging, "warning", warnings.append)
  pinned = [1, 2, 3, 4, 5, 6]
  element = {
      "text": [pinned + list(range(10, 19)), list(range(100, 116))],
      "is_prompt": [True, False],
      SFT_PINNED_CONTEXT_IDS_KEY: pinned,
  }
  windows = SFTPromptMaskingWindows(
      "text",
      completion_only=True,
      max_target_length=20,
      unk_id=PAD,
      overlap=2,
      pin_leading_context=True,
      pinned_context_warn_fraction=0.5,
  )
  recs = windows.flat_map(element)

  assert windows.pinned_context_warning_count == 1
  assert len(warnings) == 1
  assert "pinned_tokens=6" in warnings[0]
  assert "remaining_tail_budget=4" in warnings[0]
  assert [token for record in recs for token in _loss_tokens(record)] == element["text"][1]


def test_pin_must_match_accumulated_prefix_at_a_real_left_cut():
  windows = SFTPromptMaskingWindows(
      "text",
      completion_only=True,
      max_target_length=20,
      unk_id=PAD,
      overlap=2,
      pin_leading_context=True,
  )
  with pytest.raises(ValueError, match="not an exact token prefix"):
    windows.flat_map(
        {
            "text": [list(range(1, 16)), list(range(100, 116))],
            "is_prompt": [True, False],
            SFT_PINNED_CONTEXT_IDS_KEY: [999, 998],
        }
    )


@pytest.mark.parametrize(
    ("length", "overlap", "requested_cap", "expected"),
    [
        (20, 2, -1, (2, 2, 10)),
        (16, 256, -1, (2, 2, 8)),
        (8, 256, -1, (1, 1, 4)),
        (16, 0, 20, (0, 2, 14)),
        (16, 2, 1, (2, 2, 1)),
    ],
)
def test_shared_window_geometry_matches_transform_clamps(length, overlap, requested_cap, expected):
  geometry = get_sft_window_geometry(length, overlap, requested_cap)
  assert (geometry.overlap_cap, geometry.min_loss_room, geometry.context_cap) == expected


def test_truncate_baseline_drops_eot():
  """Sanity: the old 1:1 transform drops the terminator for an over-length example (the bug)."""
  length = 20
  prompt = [1, 2, 3]
  completion = list(range(100, 129)) + [EOT]  # 30 tokens, terminator last
  baseline = SFTPromptMasking("text", completion_only=True, max_target_length=length, unk_id=PAD)
  out = baseline.map({"text": [prompt, completion], "is_prompt": [True, False]})
  assert len(out["inputs"]) == length
  assert EOT not in set(int(x) for x in out["targets"])  # terminator truncated away → not in loss


def test_long_single_turn_keeps_eot_and_covers_completion_once():
  """A long single-turn example is split into windows; the terminator stays in the loss and every
  completion token is trained exactly once (disjoint loss), with the prompt pinned as context."""
  length = 20
  prompt = [1, 2, 3]
  completion = list(range(100, 129)) + [EOT]  # 30 tokens
  windows = SFTPromptMaskingWindows("text", completion_only=True, max_target_length=length, unk_id=PAD, overlap=2)
  recs = windows.flat_map({"text": [prompt, completion], "is_prompt": [True, False]})

  assert len(recs) >= 2
  for r in recs:
    assert len(r["inputs"]) == len(r["targets"]) <= length
    # prompt pinned (masked) as context in every window
    assert [int(x) for x in r["inputs"][: len(prompt)]] == prompt
    assert [int(x) for x in r["targets"][: len(prompt)]] == [PAD] * len(prompt)

  # disjoint + complete + ordered coverage of the completion
  covered = []
  for r in recs:
    covered += _loss_tokens(r)
  assert covered == completion

  # terminator is in the loss of the LAST window
  assert _loss_tokens(recs[-1])[-1] == EOT


def test_multi_turn_each_turn_terminator_in_loss():
  """A multi-turn over-length example keeps each assistant turn's terminator in the loss."""
  length = 16
  # system+user1 | answer1+EOT | user2 | answer2+EOT  (total > length)
  segments = [
      [1, 2, 3, 4, 5],  # prompt (system+user1)
      [201, 202, 203, EOT],  # completion 1
      [6, 7, 8],  # prompt (user2)
      [211, 212, 213, 214, 215, EOT],  # completion 2
  ]
  is_prompt = [True, False, True, False]
  windows = SFTPromptMaskingWindows("text", completion_only=True, max_target_length=length, unk_id=PAD, overlap=2)
  recs = windows.flat_map({"text": segments, "is_prompt": is_prompt})

  covered = []
  for r in recs:
    assert len(r["inputs"]) == len(r["targets"]) <= length
    covered += _loss_tokens(r)

  # both completions fully covered, each exactly once, terminators included
  assert covered == [201, 202, 203, EOT, 211, 212, 213, 214, 215, EOT]
  assert covered.count(EOT) == 2


def test_tool_result_prompt_is_masked_context_for_following_assistant_window():
  """A tool-result prompt immediately preceding an assistant completion remains visible but masked."""
  length = 16
  tool_result = [71, 72]
  final_completion = [211, 212, 213, 214, 215, EOT]
  segments = [
      [1, 2, 3, 4, 5],  # system + user prompt
      [201, 202, 203, EOT],  # assistant tool call (loss)
      tool_result,  # rendered tool result + assistant prefix (masked)
      final_completion,  # assistant response conditioned on the result (loss)
  ]
  windows = SFTPromptMaskingWindows(
      "text",
      completion_only=True,
      max_target_length=length,
      unk_id=PAD,
      overlap=2,
      context_cap=10,
  )
  recs = windows.flat_map({"text": segments, "is_prompt": [True, False, True, False]})

  final_records = [record for record in recs if final_completion[0] in _loss_tokens(record)]
  assert final_records
  for record in final_records:
    inputs = [int(token) for token in record["inputs"]]
    targets = [int(token) for token in record["targets"]]
    tool_start = next(i for i in range(len(inputs) - len(tool_result) + 1) if inputs[i : i + 2] == tool_result)
    assert targets[tool_start : tool_start + len(tool_result)] == [PAD] * len(tool_result)


def test_fan_out_cap_is_respected():
  """max_fan_out bounds the number of emitted records (runaway guard)."""
  length = 12
  prompt = [1, 2]
  completion = list(range(100, 200))  # very long
  windows = SFTPromptMaskingWindows(
      "text", completion_only=True, max_target_length=length, unk_id=PAD, overlap=1, max_fan_out=3
  )
  recs = windows.flat_map({"text": [prompt, completion], "is_prompt": [True, False]})
  assert len(recs) == 3
  for r in recs:
    assert len(r["inputs"]) <= length


if __name__ == "__main__":
  for name, fn in sorted(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"PASS {name}")
  print("ALL PASS")
