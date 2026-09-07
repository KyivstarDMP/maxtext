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

## Per-completion construction

The transform walks the formatted segments in order and keeps `prefix`, the
tokens of every prompt segment and every already-completed assistant segment
before the completion currently being processed. Prompt segments alone never
emit a record; they are accumulated into `prefix` for the next completion.

For each completion the context `ctx` is chosen **once** and then frozen across
all windows of that completion:

```text
if len(prefix) <= context_cap:
    ctx = prefix                                   # everything fits
elif pinning is disabled or the example has no pin:
    ctx = the newest context_cap tokens of prefix  # plain left cut
else:
    tail_budget = context_cap - len(pin)
    ctx = pin + the newest tail_budget tokens of prefix after the pin
```

`ctx` is recomputed only when the next completion starts, after the finished
completion and any following prompt segments have been appended to `prefix`.
It does not grow with every earlier slice of the same completion.

Within one completion the records are:

```text
inputs  = ctx + previous-slice overlap + new completion slice
targets = MASK(ctx + previous-slice overlap) + new completion slice
```

The overlap comes only from an earlier slice of the **same** completion. Earlier
assistant turns can appear in `ctx` as ordinary conversation history, but they
are never completion overlap. New slices are disjoint, so each retained
completion token receives loss exactly once, and the final slice carries the
completion's turn terminator.

`sft_window_max_fan_out` is counted per example across **all** of its
completion segments, not per completion. When the cap is reached the transform
returns the records already produced and logs how many trailing completion
tokens, potentially including a terminator, were dropped.

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

### What the pin does and does not protect

- The pin protects the leading instructions and native tool declarations. It
  does not protect earlier tool results or earlier assistant turns; those stay
  in the accumulated prefix and compete for the recent-history tail like any
  other context.
- The pin branch is used only when a real left cut is required. If the prefix
  fits under the cap, the leading block is present naturally and records are
  identical with pinning on or off.
- Segmented rendering passes the tool declarations to each independently
  rendered round, so later rounds can contain further copies of the
  declarations; assistant-mask rendering emits them once at the beginning.
  Pinning neither creates nor removes those copies.
- Pin-fraction warnings are rate-limited per transform instance, normally one
  Grain worker: the first three occurrences are logged, then every power of
  two, and the log line reports `warning_rows_seen_by_worker`. Treat that count
  as worker-local; it is not a corpus total and must not be summed as one.
- Warnings and runtime errors are a final safety net. Before enabling pinning
  on a full corpus, compute pin lengths against the same effective context-cap
  formula and disposition rows whose pin would leave no recent-history budget.

### Compact mental model

```text
ctx = full prefix,                          if it fits
    = newest context tail,                  if a left cut is needed without pinning
    = pin + newest history in the budget,   if a left cut is needed with pinning

window = frozen ctx + optional masked same-completion overlap + next disjoint loss slice
```

Invariants: every record is at most `max_target_length` tokens; context and
overlap are masked; every emitted completion token receives loss exactly once;
the pin changes only contexts that would otherwise left-cut it; the fan-out cap
is a runaway guard that drops and logs trailing tokens when reached.

## Supported paths

Windowing and leading-context pinning require tokenized, completion-only Grain
SFT. Both segmented and assistant-mask formatters emit the shared token-run
contract used by the window transform:

- segmented mode validates the pin against the carried IDs of segment zero
  before windowing; it does not re-encode the decoded chunk. Legacy callers
  without `sft_segment_ids` retain the encode-and-validate fallback;
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
