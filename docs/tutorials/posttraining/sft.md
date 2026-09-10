<!--
 Copyright 2023–2025 Google LLC

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

# SFT on single-host TPUs

Supervised fine-tuning (SFT) is a process where a pre-trained large language model is fine-tuned on a labeled dataset to adapt the model to perform better on specific tasks.

This tutorial demonstrates step-by-step instructions for setting up the environment and then training the model on a Hugging Face dataset using SFT.

We use [Tunix](https://github.com/google/tunix), a JAX-based library designed for post-training tasks, to perform SFT.

In this tutorial we use a single host TPU VM such as `v6e-8/v5p-8`. Let's get started!

## Install MaxText and Post-Training dependencies

For instructions on installing MaxText with post-training dependencies on your VM, please refer to the [official documentation](../../install_maxtext.md) and use the `maxtext[tpu-post-train]` installation path to include all necessary post-training dependencies.

> **Note:** If you have previously installed MaxText with a different option (e.g., `maxtext[tpu]`), we strongly recommend using a fresh virtual environment for `maxtext[tpu-post-train]` to avoid potential library version conflicts.

## Setup environment variables

Login to Hugging Face. Provide your access token when prompted:

```bash
hf auth login
```

Set up the following environment variables to configure your training run. Replace
placeholders with your actual values.

```bash
# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models.
export MODEL=<MODEL_NAME> # e.g., 'llama3.1-8b-Instruct'

# -- MaxText configuration --
# Use a GCS bucket you own to store logs and checkpoints. Ideally in the same
# region as your TPUs to minimize latency and costs.
# You can list your buckets and their locations in the
# [Cloud Console](https://console.cloud.google.com/storage/browser).
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET> # e.g., gs://my-bucket/maxtext-runs

# An arbitrary string to identify this specific run.
# We recommend to include the model, user, and timestamp.
# Note: Kubernetes requires workload names to be valid DNS labels (lowercase, no underscores or periods).
export RUN_NAME=<RUN_NAME>

export STEPS=<STEPS> # e.g., 1000
export PER_DEVICE_BATCH_SIZE=<BATCH_SIZE_PER_DEVICE> # e.g., 1

# -- Dataset configuration --
export DATASET_NAME=<DATASET_NAME> # e.g., HuggingFaceH4/ultrachat_200k
export TRAIN_SPLIT=<TRAIN_SPLIT> # e.g., train_sft
export TRAIN_DATA_COLUMNS=<DATA_COLUMNS> # e.g., ['messages']
```

## Get your model checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```sh
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting a Hugging Face checkpoint

Refer the steps in [Hugging Face to MaxText](hf-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```sh
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

> [!IMPORTANT]
> **Automatic `scan_layers` Resolution:**
> MaxText automatically loads `scan_layers` from the checkpoint's saved metadata when resuming (via `load_parameters_path`) if you do not explicitly specify it on the command-line.
>
> - You do not need to manually supply `scan_layers=False` (or `scan_layers=True`) when loading checkpoints; MaxText will configure this automatically.
> - If you do explicitly provide a `scan_layers` argument, it must match the checkpoint's saved setting or a `ValueError` mismatch error will be raised.
>   See the [Checkpoints concept guide](../../reference/core_concepts/checkpoints.md) for more details.

## Run SFT on Hugging Face Dataset

Now you are ready to run SFT using the following command:

```sh
python3 -m maxtext.trainers.post_train.sft.train_sft \
    run_name=${RUN_NAME?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    model_name=${MODEL?} \
    load_parameters_path=${MAXTEXT_CKPT_PATH?} \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE?} \
    steps=${STEPS?} \
    hf_path=${DATASET_NAME?} \
    train_split=${TRAIN_SPLIT?} \
    train_data_columns=${TRAIN_DATA_COLUMNS?} \
    profiler=xplane
```

Your fine-tuned model checkpoints will be saved here: `$BASE_OUTPUT_DIRECTORY/$RUN_NAME/checkpoints`.

## Dataset Customization & Chat Templates

Supervised Fine-Tuning in MaxText relies on tokenizing conversational datasets using chat templates. This requires the dataset structure and templates to be aligned.

### Supported Dataset Schemas

By default, MaxText SFT expects one of four conversational dataset structures:

- `["messages"]`: A single column containing a list of dictionaries with `role` and `content` (recommended).
- `["messages", "tools"]`: Messages plus native tool declarations passed to the chat template.
- `["prompt", "completion"]`: Separated prompt and completion columns.
- `["question", "answer"]`: Question and answer columns (e.g., math datasets).

During data processing, MaxText converts these into a unified `messages` schema (OpenAI-like format) before feeding it to the tokenizer:

```json
[
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi there!"}
]
```

### Custom Tokenizer Chat Templates

To customize the tokenizer's chat formatting (e.g., adding special tokens like `<start_of_turn>`, `<end_of_turn>`, etc.), you can provide a custom chat template using the `chat_template` or `chat_template_path` configs:

- **`chat_template`**: Use this config to specify a custom Jinja2 template string directly.
- **`chat_template_path`**: Path to a custom Jinja2 template file (e.g., `.jinja`), a JSON file containing the template, or an `hf://<org>/<repo>/<path>` Hub URI.
- **`tokenizer_revision`**: Optional Hugging Face revision passed when loading `tokenizer_path`.
- **`chat_template_revision`**: Optional Hugging Face revision used for an `hf://` template path.
- **`chat_template_sha256`**: Optional SHA-256 check over the exact loaded template bytes.
- **`use_chat_template=True`**: Enables chat template formatting.

Keep Hub revisions in the separate revision fields rather than embedding them in
the URI. For reproducible runs, use immutable commit IDs:

```yaml
tokenizer_type: huggingface
tokenizer_path: example-org/example-model
tokenizer_revision: <40_HEX_COMMIT>
chat_template_path: hf://example-org/example-model/templates/training.jinja
chat_template_revision: <40_HEX_COMMIT>
```

### Canonical token ownership for completion-only SFT

The tokenized Grain pipeline also supports an opt-in canonical mode. It renders
the full conversation once and uses Jinja `{% generation %}` blocks to decide
which exact tokens receive loss:

```yaml
dataset_type: grain
use_sft: true
tokenize_train_data: true
sft_train_on_completion_only: true
sft_chat_template_mode: assistant_mask
```

The default `segmented` mode remains available for compatible templates. Its
Grain and Hugging Face paths carry the original rendered token IDs through an
internal `sft_segment_ids` column, aligned with the decoded strings and
`is_prompt` flags. Tokenization consumes those IDs and removes the side column;
it does not encode the decoded chunks again. String-only (`tokenize=False`)
paths do not expose this column. Legacy callers without the column still encode
strings.

Segmented SFT rejects rows with no nonempty assistant completion segment, even
when `sft_train_on_completion_only=false`. This is an SFT input requirement,
independent of whether prompt tokens also receive loss. An empty-content
assistant remains valid when the template emits supervised closing tokens;
empty completions are also allowed alongside a nonempty completion in the row.
Trailing tool results must eventually be followed by an assistant or user so
their context can be emitted. A row ending in unemitted results raises instead
of silently discarding them. Terminal assistant calls without a recorded result
remain supported.

Segmented rendering restarts its round after an assistant followed by a new user.
That next round replays the template's BOS and leading system/developer/tools
context. Within one round, tool results are emitted once as a masked suffix.
At a tool-to-user boundary, the pending result suffix and the new user prompt
suffix are separate masked segments; earlier tokens are not replayed there.
This per-round contract differs from rendering one full multi-turn conversation
in `assistant_mask` mode.

For a user followed by an assistant, prompt emission waits for the completed
round. Both prompt and completion are slices of that full token render, split
at its common prefix with the generation prompt. Speculative prompt suffixes
absent from the actual response are therefore excluded, including after an
interrupted tool round. Original token IDs still flow downstream unchanged.
Previously emitted tool/history tokens must remain an exact prefix; a template
that changes them is rejected. Rows ending in an unemitted user message are
also rejected; the data producer must trim dangling tails before training.

`sft_preserve_thinking=auto` omits the preservation argument in segmented mode
and follows each row's thinking value in canonical mode. An explicit boolean
is passed to every render in either mode, including prefix and pin renders.
The template decides which historical reasoning it retains. A segmented
tool-to-user boundary can fail if adding the user makes earlier call reasoning
disappear; explicit preservation can resolve that seam for compatible
templates. See the thinking-mode contract below for the exact scope.

`assistant_mask` is Grain-only and requires a generation-marked template. For
the complete ownership, tools, thinking-mode, and validation contract, see
[Canonical Gemma 4 SFT rendering](gemma4_sft_canonical_rendering.md) and the
[Gemma 4 SFT data contract](gemma4_sft_data_contract.md).

For records longer than `max_target_length`, see
[SFT long-example windowing](sft_long_example_windowing.md). For the difference
between one canonical training history and online generation prefixes, see
[Gemma 4 multi-turn SFT and serving frontiers](gemma4_sft_serving_frontiers.md).

### Per-dataset metrics with the pre-training trainer

When training formatted SFT data through `maxtext.trainers.pre_train.train`,
`per_dataset_metrics=true` reports loss and next-token accuracy over the
supervised tokens. Linen and NNX support these metrics with and without
vocabulary tiling. Per-dataset loss includes the configured z-loss, matching
`learning/lm_loss`; this also applies to the Linen tiled path.

The tiled helpers return loss, z-loss, per-dataset loss sums, and correct-token
counts. With metrics disabled, both vectors are `None`. Without `dataset_id`
(as in separate evaluation passes), the helpers use two slots: an empty slot 0
and the batch totals in slot 1. With IDs, slot 0 is reserved for padding and the
remaining slots follow `per_dataset_names`. A dataset with supervised tokens
and no correct predictions has valid zero accuracy; a dataset with no
supervised tokens emits only its token count.

The NNX tiled path accumulates metrics from the logits already computed in
each chunk. It adds an argmax and segment reductions without an additional
output-head projection. Its reporting vectors do not contribute gradients or
add tensors to the backward residuals. Peak memory and throughput still need
measurement on the target hardware. Per-dataset metrics remain unsupported
for block diffusion and dense indexer warm-up and raise an error there.

### Advanced: Custom Dataset Formatter (e.g., ShareGPT)

If your dataset is in a format not natively supported—such as **ShareGPT** (which uses a `conversations` column with `from` and `value` keys)—you can write a custom Python formatting function to convert it on-the-fly.

#### 1. Write a custom formatting function

Create a Python file in your workspace (e.g., `src/maxtext/input_pipeline/custom_formatters.py`):

```python
def format_sharegpt(example):
    """Converts ShareGPT format (from/value) to standard messages (role/content)."""
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }

    messages = []
    for turn in example["conversations"]:
        role = role_map.get(turn["from"], "user")
        messages.append(
            {
                "role": role,
                "content": turn["value"],
            }
        )

    example["messages"] = messages
    return example
```

#### 2. Configure MaxText to use your formatter

When starting your SFT training, pass the following parameters:

- `train_data_columns`: Point to the original column name in the raw dataset (`"['conversations']"`).
- `formatting_func_path`: Point to the python import path of your formatting function (`"maxtext.input_pipeline.custom_formatters.format_sharegpt"`).

```sh
python3 -m maxtext.trainers.post_train.sft.train_sft \
    ... \
    train_data_columns="['conversations']" \
    formatting_func_path="maxtext.input_pipeline.custom_formatters.format_sharegpt"
```

### Runnable Example in the Codebase

For a complete, runnable SFT workflow that demonstrates how to configure the training loop and use a custom dataset formatter (`formatting_func_path` and `formatting_func_kwargs`), check out the [sft_qwen3_demo.ipynb](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/examples/sft_qwen3_demo.ipynb) Jupyter notebook.
