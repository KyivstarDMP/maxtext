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

# Chat templates for SFT

MaxText can use the chat template already stored in a Hugging Face tokenizer,
an inline `chat_template`, or a separate file selected by
`chat_template_path`.

Local and Hub files may use `.jinja`, `.jinja2`, `.j2`, or `.txt`, or be `.json`
objects containing a `chat_template` field. An existing file with an unsupported
extension raises an error naming that extension. A Hub-hosted file uses this
form:

```yaml
chat_template_path: hf://example-org/example-model/templates/training.jinja
chat_template_revision: <40_HEX_COMMIT>
```

Keep the revision in `chat_template_revision`, not after `@` in the URI. Branch
names can contain `/`, which makes an embedded revision ambiguous. Use an
immutable commit ID for reproducible runs. `chat_template_sha256` is an
optional check over the exact loaded file bytes.

Omitting `chat_template_revision` uses the Hub repository's default revision.
This remains supported for compatibility, but it is not reproducible and
should not be used for a pinned run.

The tokenizer has a separate pin:

```yaml
tokenizer_type: huggingface
tokenizer_path: example-org/example-model
tokenizer_revision: <40_HEX_COMMIT>
```

Separate fields allow a tokenizer and training template to come from different
repositories or commits when that is intentional.

Tokenizer revision plumbing currently covers the Grain/pre-train pipeline,
text-only Hugging Face SFT, and RL. Multimodal Hugging Face SFT and TFDS keep
their existing unpinned tokenizer loads.

## Training ownership

The Grain `assistant_mask` mode requires Jinja `{% generation %}` blocks. Tokens
inside the blocks receive completion loss; tokens outside them are masked
context. A minimal shape is:

```jinja
{% for message in messages %}
  {% if message['role'] == 'assistant' %}
    {% generation %}{{ message['content'] }}{% endgeneration %}
  {% else %}
    {{ message['content'] }}
  {% endif %}
{% endfor %}
```

Real templates must also render their required role headers, delimiters, stop
tokens, tools, and tool results. Put each token under the ownership of the actor
that supplies it; do not mark a tool result as model output merely because it
appears between two assistant segments.

A template that cannot be rendered safely by MaxText's segmented mode should
declare this non-rendering comment:

```jinja
{# maxtext-template-capability: requires-assistant-mask #}
```

MaxText then fails unless `sft_chat_template_mode=assistant_mask` is selected.

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

## Training and serving variants

A serving template defines the prefix used at an online generation frontier. A
training template additionally defines token ownership, and may validate fields
required by the training data contract. Keep separate named files when adding
generation blocks would affect serving behavior or when the training template
has stricter input requirements.

For paired variants, test that removing Jinja ownership markers leaves the
intended rendered token stream unchanged. Also compare the exact token prefix at
each serving frontier for multi-turn and tool-use examples.

See the
[canonical rendering guide](../../../../docs/tutorials/posttraining/gemma4_sft_canonical_rendering.md)
for pipeline constraints and required token-level checks.
