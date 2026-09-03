<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# SFT long-example windowing

The default SFT transform truncates the concatenated token stream at
`max_target_length`. For an over-length completion this can remove its final
tokens, including the turn terminator, from the loss. The Grain SFT pipeline can
instead split long examples into bounded loss-bearing windows.

```yaml
sft_train_on_completion_only: true
sft_long_example_handling: window
sft_window_overlap: 256
sft_window_context_cap: -1
sft_window_max_fan_out: 32
```

Examples that already fit produce the same single record as the ordinary
completion-only transform.

## Window geometry

Let `L` be `max_target_length`:

```text
overlap_cap = clamp(configured_overlap, 0, L // 8)
min_loss_room = max(1, L // 8)
requested_context = configured_context_cap, or L // 2 when it is not positive
context_cap = clamp(requested_context, 1, L - overlap_cap - min_loss_room)
```

For each completion run, a record contains:

```text
bounded earlier context, masked
+ overlap from the same completion's previous slice, masked
+ new completion slice, loss applied
```

New completion slices are disjoint, so every retained completion token receives
loss once. Overlap supplies local continuity but does not duplicate loss.

`sft_window_max_fan_out` is a hard guard. Reaching it logs and drops remaining
completion tokens, so preflight data at the intended geometry and choose the
limit deliberately.

## Leading-context pinning

Tail-only context can remove the start of a long conversation. Enable pinning
to preserve the tokenizer's exact canonical leading block when a real left cut
is required:

```yaml
sft_window_pin_leading_context: true
sft_window_pinned_context_overflow: error
sft_window_pinned_context_warn_fraction: 0.5
```

The leading block can contain BOS, the initial system/developer message, and
native tool declarations. MaxText obtains it from `apply_chat_template` and
requires it to be an exact token prefix of both the first prompt and the final
formatted stream. It does not reconstruct the block from decoded text.

When there is no explicit leading message, MaxText tries a tokenizer-generated
empty developer block only as a render probe. The probe is not inserted into
the source conversation. If that result is not an exact prefix, MaxText tries
an exact BOS-only prefix; otherwise it fails.

Pinning changes a record only when the accumulated prefix exceeds the effective
context cap. The replacement context is:

```text
canonical pin + freshest tail that fits the remaining context budget
```

The pin must be shorter than the context cap. Filling the cap would leave no
room for recent conversational context, so the configured `error` policy fails
instead of truncating instructions or tool schemas. A large valid pin emits
bounded warnings according to `sft_window_pinned_context_warn_fraction`.

## Supported paths

Windowing and leading-context pinning require tokenized, completion-only Grain
SFT. Both segmented and assistant-mask formatters emit the shared token-run
contract used by the window transform:

- segmented mode validates the pin across its later decode/tokenize boundary;
- assistant-mask mode carries tokenizer-produced IDs end to end and validates
  the pin against the full canonical stream.

The Hugging Face SFT pipeline does not implement leading-context pinning.

## Preflight invariants

At the final tokenizer revision and template, verify that:

- every record length is at most `max_target_length`;
- every record contains at least one new loss-bearing token;
- the concatenation of loss slices equals the retained completion tokens in
  order, including the final terminator;
- overlap and all earlier context are masked;
- every left-cut record begins with exactly one copy of the pin;
- pin overflow and fan-out dispositions are counted before training.

Windowing preserves token ownership; it does not decide which assistant spans
the template owns. See [canonical rendering](gemma4_sft_canonical_rendering.md)
for that contract.
