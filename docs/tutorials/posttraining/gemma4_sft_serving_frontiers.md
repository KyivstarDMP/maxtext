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

# Gemma 4 multi-turn SFT and serving frontiers

A multi-turn training record contains one serialized history and can supervise
several assistant spans. An online serving system instead renders a new prefix
at each generation frontier. Those views are identical only when later messages
do not change how earlier messages are serialized.

## Prefix stability

For a prefix-stable template, rendering through the first assistant response is
a token prefix of rendering the full conversation:

```text
render(user 1, assistant 1) <=prefix render(user 1, assistant 1, user 2, assistant 2)
```

Some templates inspect future messages to decide how to represent earlier
reasoning, tool calls, or adjacent assistant messages. For those templates the
relation may not hold. Repeated partial renders and longest-common-prefix deltas
can then move, duplicate, or omit tokens.

`sft_chat_template_mode=assistant_mask` removes this delta inference. It renders
the complete training conversation once and consumes token ownership from
Jinja generation blocks. This makes the training stream internally exact, but
it does not make one full-conversation stream identical to every earlier online
serving frontier.

## Choose an explicit policy

Use one of these data policies and document it:

1. **Canonical full-history training.** Keep one row and supervise all intended
   assistant spans when the serialized history seen by later spans matches the
   serving history.
2. **Frontier-expanded training.** Produce one derived row per selected serving
   frontier. Each row ends at that assistant response and supervises only the
   final physical model turn.
3. **Accepted history difference.** Keep one row when a small, measured
   difference is intentional and its effect is covered by evaluation.

Frontier expansion belongs in data preparation. It changes sample counts,
weights, sharding, and resumption behavior, so the training input pipeline
should not create it silently.

## Tool-use frontiers

A tool interaction commonly has two model-output segments separated by external
context:

```text
assistant call -> tool result -> assistant answer
```

The call and answer are model-owned. The tool result is masked context. A
template may represent the two assistant segments as one physical turn while
still using separate generation blocks around the two model-owned regions.

If an assistant message contains both visible content and tool calls, choose one
stable order for training and serving. Emitting the visible content before its
own calls aligns with the equivalent split form:

```text
assistant content -> assistant call -> tool result -> assistant answer
```

Whichever convention is chosen, parser behavior, stored conversation order,
the serving template, and the training template must agree.

## What to compare

For each supported conversation shape, compare the exact token prefix at every
loss position with the token prefix presented at the corresponding serving
frontier. Include:

- single-turn and multi-turn conversations;
- thinking enabled and disabled;
- calls with one and multiple results;
- a post-result answer;
- an assistant message containing both content and calls;
- histories where earlier reasoning is retained or removed.

Compare token IDs and ownership masks, not only decoded strings. If a frontier
is intentionally different, record the specific token difference and the data
policy that makes it acceptable.

See [canonical rendering](gemma4_sft_canonical_rendering.md) for the MaxText
configuration and [the data contract](gemma4_sft_data_contract.md) for source
row requirements.
