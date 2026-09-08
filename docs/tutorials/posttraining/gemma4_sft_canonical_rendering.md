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

# Canonical Gemma 4 SFT rendering and token ownership

MaxText has two ways to turn a conversation into completion-only SFT tokens.
The default `segmented` mode renders one conversational round at a time and
finds prompt/completion deltas. The opt-in `assistant_mask` mode renders the
complete conversation once and asks the tokenizer for a token-aligned
assistant ownership mask.

Canonical rendering is useful for templates whose earlier output can depend on
later messages. It also avoids a decode-and-tokenize round trip: the exact token
IDs returned by `apply_chat_template` become the training stream.

## Configuration

Canonical rendering is currently supported by the tokenized Grain SFT
pipeline. It requires completion-only training and a template containing Jinja
`{% generation %}` blocks.

```yaml
dataset_type: grain
tokenize_train_data: true
use_sft: true
sft_train_on_completion_only: true
sft_chat_template_mode: assistant_mask

tokenizer_type: huggingface
tokenizer_path: example-org/example-model
tokenizer_revision: <40_HEX_COMMIT>
chat_template_path: hf://example-org/example-model/templates/training.jinja
chat_template_revision: <40_HEX_COMMIT>
```

`tokenizer_revision` and `chat_template_revision` are passed separately from
the paths. Use immutable commit IDs when the run must be reproducible. An
optional `chat_template_sha256` can verify the exact downloaded bytes.

The Hugging Face SFT pipeline rejects `assistant_mask`; keep its default
`segmented` mode. It can still load a local or Hub-hosted template through
`chat_template_path`.

## Ownership contract

For the canonical stream, MaxText calls the tokenizer once with the preservation
policy resolved as described under Thinking mode:

```python
tokenizer.apply_chat_template(
    conversation,
    tools=tools,
    add_generation_prompt=False,
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=True,
    enable_thinking=enable_thinking,
    preserve_thinking=resolved_preserve_thinking,
)
```

The returned `input_ids` and `assistant_masks` must be non-empty,
one-dimensional, and equal in length. Mask values have this meaning:

| Assistant mask | MaxText run        | Completion-only target |
| -------------- | ------------------ | ---------------------- |
| `0`            | prompt/context     | masked                 |
| `1`            | model-owned output | loss applied           |

Adjacent tokens with equal ownership are coalesced into runs. Concatenating the
runs exactly reconstructs `input_ids`; no token is decoded or re-tokenized.
The pipeline rejects masks that contain no loss-bearing tokens.

The template, rather than a role-name heuristic, therefore owns delimiters,
reasoning scaffolds, calls, and stop tokens. Place `{% generation %}` markers
around every token span the model should learn to produce. Keep user messages,
tool results, and other supplied context outside those blocks.

## Template capability marker

A template that is unsafe under segmented rendering can declare:

```jinja
{# maxtext-template-capability: requires-assistant-mask #}
```

MaxText reads this comment from the template source and rejects any mode other
than `assistant_mask`. A generation block alone does not imply the capability;
templates may use generation blocks while remaining valid in segmented mode.

## Thinking mode

`sft_enable_thinking` supplies one boolean for every row. For a mixed dataset,
set `sft_enable_thinking_column` to the name of a required per-row boolean
column. Grain validates actual booleans and passes the selected value as
the row's `enable_thinking` and, by default in canonical mode,
`preserve_thinking`.

`sft_preserve_thinking` independently controls the template's historical
reasoning policy. It accepts `auto` (the default), `true`, or `false`:

| Setting          | Canonical (`assistant_mask`)           | Segmented                                  |
| ---------------- | -------------------------------------- | ------------------------------------------ |
| `auto`           | Pass the row's `enable_thinking` value | Omit the kwarg; use the template's default |
| `true` / `false` | Pass the explicit boolean              | Pass the explicit boolean                  |

MaxText resolves the policy separately for each row and uses it for every full,
prefix, suffix and leading-pin render. It does not change shared tokenizer
state. For example, `sft_enable_thinking=true` with
`sft_preserve_thinking=false` permits current reasoning while asking the
template not to preserve historical reasoning.

The template defines which reasoning this flag affects. A template may retain
historical tool-call reasoning while still removing reasoning from ordinary
historical answers. The flag does not make segmented and canonical token
streams equivalent.

In segmented mode, a reasoning-bearing call followed by a tool result and a
new user can fail the tool-to-user prefix check if the template removes that
call's now-historical reasoning. `auto` retains this fail-loud behavior for
templates whose default removes it. Explicit `true` can make that boundary
prefix-stable when the template supports preserving the call reasoning; this
is not a guarantee for every template or conversation shape.

Do not use the per-row column with the Hugging Face SFT pipeline; that pipeline
supports only the constant thinking setting. It supports
`sft_preserve_thinking` under the segmented policy above.

## Tools and tool results

Native tool declarations belong in the optional `tools` column. `role="tool"`
messages must remain masked context: later assistant tokens can attend to the
result, but the result itself should not receive completion loss. In canonical
mode the template enforces this by keeping result tokens outside generation
blocks.

Segmented mode now preserves this result context too. It compares tokenized
renders around the result boundary and fails if the template is not prefix
stable there. Canonical mode avoids that boundary inference because the
template returns the complete ownership mask directly.

## Verification

Test template changes at the token level. At minimum, verify that:

- concatenated runs equal the tokenizer's complete `input_ids`;
- targets select exactly the tokens where `assistant_masks` is `1`;
- tool results remain visible but masked;
- the final completion terminator receives loss;
- every supported thinking mode produces the intended ownership mask.

Decoded text is useful for review, but it is not enough to prove special-token
boundaries or loss ownership.

See also [the data contract](gemma4_sft_data_contract.md),
[serving frontiers](gemma4_sft_serving_frontiers.md), and
[long-example windowing](sft_long_example_windowing.md).
