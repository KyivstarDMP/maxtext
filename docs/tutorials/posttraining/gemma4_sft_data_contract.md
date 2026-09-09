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

# Gemma 4 SFT data contract

This contract describes the records consumed by MaxText's text SFT pipelines.
It separates source-data structure from template-specific rendering so both can
be validated before an accelerator job starts.

## Conversation columns

SFT accepts one of these conversational shapes after metadata columns are
removed:

- `messages`
- `messages` plus `tools`
- `prompt` plus `completion`
- `question` plus `answer`

`prompt`/`completion` and `question`/`answer` records are converted to one user
message followed by one assistant message. A `messages` record is a non-empty
ordered list of mappings.

```json
[
  {"role": "developer", "content": "Use the supplied context."},
  {"role": "user", "content": "Look up the value."},
  {
    "role": "assistant",
    "tool_calls": [
      {"type": "function", "function": {"name": "lookup", "arguments": {}}}
    ]
  },
  {"role": "tool", "name": "lookup", "content": "42"},
  {"role": "assistant", "content": "The value is 42."}
]
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
A `system` or `developer` message, when present, must be at index zero.
Assistant-mask mode permits an assistant tool-call message to omit `content`;
other messages require content expected by the selected template.

## Native tool declarations

The optional `tools` column contains declarations passed through the
tokenizer's `tools=` argument. It is separate from `role="tool"` messages:

- `tools` describes functions available to the model;
- an assistant `tool_calls` field is model output;
- a `role="tool"` message is externally supplied result context.

Declarations and result messages must use the schema expected by the chat
template. MaxText transports them but does not repair mismatched function names,
argument schemas, or result ordering.

## Thinking mode

Use one of two forms:

```yaml
sft_enable_thinking: true
sft_enable_thinking_column: ""
```

or, for mixed rows:

```yaml
train_data_columns: [messages, tools, enable_thinking]
sft_enable_thinking_column: enable_thinking
```

For Parquet and similar row sources, the per-row feature must be an actual
boolean. ArrayRecord/TensorFlow records carry one scalar `0` or `1` in a
single-element sequence, which MaxText normalizes to a boolean. Strings,
missing values, invalid integers, and non-scalar sequences are rejected. The
value applies to the entire conversation render; it is not changed between
assistant turns.

## Ownership is a template contract

For `sft_chat_template_mode=assistant_mask`, the training template must contain
Jinja `{% generation %}` blocks around model-owned spans. The resulting
assistant mask decides which exact token IDs receive loss. Roles alone do not
decide ownership.

A tool result should normally be rendered outside generation blocks. It then
appears in `inputs`, is masked in `targets`, and is available to the next
assistant response. Model-emitted call syntax and turn terminators belong inside
generation blocks when the model is expected to produce them.

If a template depends on later messages when rendering earlier messages, add
the `maxtext-template-capability: requires-assistant-mask` comment so MaxText
will reject the segmented path.

## Tool-result body validation

Ordered textual presence: for every `tool` message whose `content` is a string,
the stripped body (or its JSON-escaped form) occurs, in message order, inside
**some masked run** of the row, searched in stream order without stitching text
across run boundaries. All masked runs are eligible, including those before the
first loss-bearing run: prefix-expanded rows with `trainable:false` history
legitimately carry results there. Non-string bodies are skipped with one
warning per worker.

Stated non-guarantees (false passes that remain possible): any user or system
text quoting the result body (before or after the call) satisfies the check even
if the template drops the real result; a body that also appears in a loss-bearing
run (the check finds the masked copy and cannot tell which is the rendered
result); attribution when identical bodies repeat; preservation of structured
bodies; absence of extra template text. Positional proof would need
template-owned result-region boundaries, which the generic path does not have.
Use a template-specific oracle when that stronger guarantee is required.

The check is always on in both rendering modes. In segmented mode it searches
the emitted tool-result delta before an assistant or a user; in `assistant_mask`
mode it searches every decoded masked run of the full row. Each occurrence is
consumed before searching for the next body, so repeated identical bodies need
separate matches. Empty-after-strip strings are skipped. Supported string forms
are `body.strip()` verbatim or `json.dumps(body.strip())[1:-1]`; templates using
other serialization must be made compatible before using this check.

Validation runs during formatting, before truncation or windowing, and does not
prove that every later window retains each result. A miss raises a bounded
`ValueError` identifying the tokenizer, mode, roles, result index/count and body
length, without including the body. There is no startup probe: a streaming
dataset can encounter its first invalid tool row after training has progressed.
Validate the corpus with the intended template before submitting it.

## Long records

The default `sft_long_example_handling=truncate` keeps the head and can remove
the end of a long completion. Use `window` when every completion token,
including the terminator, must remain eligible for loss. If long left cuts must
retain the canonical leading instructions and native tool declarations, also
enable `sft_window_pin_leading_context`.

See [SFT long-example windowing](sft_long_example_windowing.md) for the exact
geometry and failure cases.

## Preflight checks

Before training, render representative and boundary-case rows with the exact
tokenizer revision and training template. Check:

- message roles and tool declarations are accepted;
- every row has at least one loss-bearing token;
- tool-result bytes occur in masked context before the dependent answer;
- thinking-mode values are valid and non-sticky across interleaved rows;
- over-length rows stay within the configured fan-out limit;
- pinned leading context is an exact prefix and is shorter than the effective
  context cap.

Record counts for rejected, shortened, split, or quarantined rows in the data
preparation layer. Silent row dropping inside the training transform changes
the mixture and is not a safe default.
