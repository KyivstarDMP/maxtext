# Copyright 2023–2025 Google LLC
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

"""Operations used by Grain"""

import dataclasses
import json
import warnings
from collections.abc import Mapping
from threading import current_thread
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
  import datasets
  import tensorflow as tf

import grain.python as grain
import numpy as np
from grain._src.python.dataset.sources.tfrecord_dataset import _TFRecordReader, _TFRecordDatasetIterator  # pylint: disable=protected-access
from grain.experimental import FlatMapTransform, TFRecordIterDataset
from jinja2.exceptions import TemplateError
from maxtext.input_pipeline.protos import example_pb2
from maxtext.input_pipeline import tokenizer
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import utils as mm_utils
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging

Features = dict[str, Any]
INPUT_TOKENS_KEY = "input_ids"
SFT_PINNED_CONTEXT_IDS_KEY = "sft_pinned_context_ids"
SFT_SEGMENT_IDS_KEY = "sft_segment_ids"

########## Functions used by TFDS pipeline


def normalize_features(x, column_name):
  return {"inputs": x[column_name], "targets": x[column_name]}


def get_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token=None):
  # Load tokenizer
  tokenizer_model = tokenizer.build_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token)
  return tokenizer_model


def truncate_to_max_allowable_length(x, max_length):
  return {k: v[:max_length] for k, v in x.items()}


def shift_data_by_truncation(x):
  x["inputs"] = x["inputs"][:-1]
  x["targets"] = x["targets"][1:]
  return x


def add_segmentation_and_position(x, data_columns, padding_token=0):
  import tensorflow as tf  # pylint: disable=import-outside-toplevel

  for data_column in data_columns:
    x[f"{data_column}_segmentation"] = tf.cast(x[data_column] != padding_token, tf.int32)
    x[f"{data_column}_position"] = tf.broadcast_to(
        tf.range(x[data_column].shape[-1], dtype=np.int32)[None, :], x[data_column].shape
    )
  return x


def TokenizeOp(tokenizer_model, features: Features, data_keys: Iterable[str] = ("inputs", "targets")) -> Features:
  """Op for tokenization"""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel

  def _process_string(string_tensor):
    # Extract string value and decode it if necessary
    string_value = string_tensor.numpy().decode("utf-8")
    # encode and extract the tokenized integers
    modified_string = tokenizer_model.encode(string_value)
    return [modified_string]

  for k in data_keys:
    features[k] = tf.py_function(_process_string, [features[k]], Tout=[tf.int32])[0]
  return features


########## Functions used by HF pipeline


def reformat_prompt(example, column, image_placeholder, model_name):
  """reformat prompt for multimodal SFT"""
  if isinstance(example["images"], list):
    num_images = len(example["images"])
  else:
    num_images = 1
  example[column] = mm_processor.reformat_prompt(example[column], image_placeholder, model_name, num_images)
  return example


def reformat_response(example, column, model_name):
  """reformat response for multimodal SFT"""
  example[column] = mm_processor.reformat_response(example[column][0], model_name)
  return example


def merge_image_columns(example, image_columns, max_num_images_per_example):
  """Merge multiple image columns into a single list of images."""
  images = []
  for col in image_columns:
    if isinstance(example[col], list):
      images.extend(example[col])
    else:
      images.append(example[col])

  example["images"] = images[:max_num_images_per_example] if max_num_images_per_example > 0 else images
  return example


def pre_process_image_sft(example, image_column, config):
  """pre-process image for multimodal SFT"""

  def _process_image_fn(image):
    if isinstance(image, list):
      image = [np.array(mm_utils.convert_to_RGB(img)) for img in image]
    else:
      image = np.array(mm_utils.convert_to_RGB(image))

    image = mm_processor.preprocess_image_for_training(image, config)
    return image

  example[image_column] = _process_image_fn(example[image_column])
  return example


def prepare_text_for_image_fusion(example, column_name, config):
  """prepare text for image fusion for multimodal SFT"""
  example[column_name] = mm_processor.prepare_text_for_image_fusion(
      tokens=example[column_name], config=config, processor_output=example["images"]
  )
  return example


def combine_columns(example, columns, data_column):
  """Combine columns such as 'prompt' and 'completion' for sft training"""
  assert len(columns) > 1
  combined = []
  for i in range(len(example[columns[0]])):
    for c in columns:
      combined.append(example[c][i])
  example[data_column] = combined
  return example


def is_conversational(features, data_columns):
  """Check if data is in a conversational format.
  Examples:

  features = {'prompt': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}],
              'completion': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}]}
  data_columns = ["prompt", "completion"]
  is_conversational(features, data_columns) return True.

  features = {'prompt': [Value(dtype='string', id=None)], 'completion': [Value(dtype='string', id=None)]}
  data_columns = ["prompt", "completion"]
  is_conversational(features, data_columns) returns False.
  """
  import datasets  # pylint: disable=import-outside-toplevel

  for column in data_columns:
    messages = features[column]
    if isinstance(messages, datasets.Sequence):
      if (
          # pyrefly: ignore[missing-attribute]
          isinstance(messages.feature, dict)
          and "role" in messages.feature
          and "content" in messages.feature  # pyrefly: ignore[missing-attribute]
      ):
        return True

  return False


def extract_token_ids(tokens):
  """Extracts token IDs from various tokenizer output formats.

  This helper function standardizes the extraction of tokenized integer IDs
  from common return types of Hugging Face tokenizers, including
  `BatchEncoding` objects, dictionaries, or simple lists.

  Args:
    tokens: The object containing token IDs. Supported types include:
      - A list of integers.
      - A dictionary containing the `INPUT_TOKENS_KEY`.
      - An object (e.g., `BatchEncoding`) with an attribute named `INPUT_TOKENS_KEY`.

  Returns:
    A list of integer token IDs.

  Raises:
    ValueError: If the input type is not supported or does not contain the expected key.
  """
  # attention masks in BatchEncoding are effectively ignored
  if hasattr(tokens, INPUT_TOKENS_KEY):
    return getattr(tokens, INPUT_TOKENS_KEY)
  elif isinstance(tokens, dict) and INPUT_TOKENS_KEY in tokens:
    return tokens[INPUT_TOKENS_KEY]
  elif isinstance(tokens, list):
    return tokens
  else:
    raise ValueError(f"Can't extract token_ids from type {type(tokens)}")


def verify_chat_template_generation_prompt_logic(tokenizer_model, enable_thinking=True):
  """Verifies the tokenizer's chat template for correct SFT loss masking.

  This function ensures that the tokens added by `add_generation_prompt=True`
  are identical to the tokens that begin an assistant's turn in a complete
  conversation, which is critical for masking prompt tokens during SFT loss
  calculation.

  Example of a mismatch:
    A `ValueError` is raised if the generation prompt and the actual
    assistant prefix do not match. For example:

    - `add_generation_prompt=True` on a user message produces a prompt ending in:
      `...<|im_start|>generation\n`
    - A full turn with an assistant message starts the reply with:
      `...<|im_start|>assistant\n...`

    This function would fail because the tokens for "generation" do not
    match the tokens for "assistant".

  Args:
    tokenizer_model: The Hugging Face tokenizer instance to verify.

  Raises:
    ValueError: If the `add_generation_prompt` tokens do not exactly
      match the beginning of an assistant message in the template.
  """
  dummy_msgs = [{"role": "system", "content": "System message"}, {"role": "user", "content": "Test message"}]

  try:
    prompt_wo_gen_tokens = tokenizer_model.apply_chat_template(
        dummy_msgs, add_generation_prompt=False, tokenize=True, enable_thinking=enable_thinking
    )
  except TemplateError:
    max_logging.info(
        "Tokenizer failed to apply chat template with 'system' role. "
        "Falling back to 'user' role only for chat template verification."
    )
    dummy_msgs.pop(0)
    prompt_wo_gen_tokens = tokenizer_model.apply_chat_template(
        dummy_msgs, add_generation_prompt=False, tokenize=True, enable_thinking=enable_thinking
    )
  prompt_wo_gen_ids = extract_token_ids(prompt_wo_gen_tokens)

  prompt_w_gen_tokens = tokenizer_model.apply_chat_template(
      dummy_msgs, add_generation_prompt=True, tokenize=True, enable_thinking=enable_thinking
  )
  prompt_w_gen_ids = extract_token_ids(prompt_w_gen_tokens)

  if prompt_w_gen_ids[: len(prompt_wo_gen_ids)] != prompt_wo_gen_ids:
    raise ValueError("Unable to extract generation prompt tokens.")
  # Extract the tokenized generation prompt (the expected assistant prefix)
  assistant_prefix = prompt_w_gen_ids[len(prompt_wo_gen_ids) :]
  full_turn_tokens = extract_token_ids(
      tokenizer_model.apply_chat_template(
          dummy_msgs + [{"role": "assistant", "content": "Dummy response"}],
          add_generation_prompt=False,
          tokenize=True,
          enable_thinking=enable_thinking,
      )
  )
  full_turn_ids = extract_token_ids(full_turn_tokens)
  # Extract the actual tokens that appear right after the user message in the full turn
  actual_prefix_in_full_turn = full_turn_ids[len(prompt_wo_gen_ids) : len(prompt_wo_gen_ids) + len(assistant_prefix)]

  if actual_prefix_in_full_turn != assistant_prefix:
    expected_str = tokenizer_model.decode(assistant_prefix)
    actual_str = tokenizer_model.decode(actual_prefix_in_full_turn)
    raise ValueError(
        "Chat template generation prompt mismatch!\n"
        f"Expected assistant prefix tokens: {assistant_prefix} ('{expected_str}')\n"
        f"Actual prefix tokens found: {actual_prefix_in_full_turn} ('{actual_str}')\n"
        "This means the tokenizer's chat template will break the sft masking logic."
    )


def _get_completion_in_chat_template(tokenizer_model, round_msgs, tools=None, enable_thinking=True):
  """
  Calculates the completion part of a conversation turn when formatted with a chat template.

  Uses the longest-common-prefix between the full conversation tokens and the
  generation-prompt tokens to locate where the completion starts.

  For most models (Llama, Qwen, …) the generation prompt is an exact prefix of the
  full conversation, so common_len == len(prompt_ids).

  For Gemma4, add_generation_prompt=True emits thinking-channel tokens
  (<|channel>thought\\n<channel|>) that diverge from the plain conversation
  at the model-turn boundary. The common prefix ends just before that
  divergence, and the completion correctly captures the thinking content
  and response tokens.

  Args:
    tokenizer_model: The tokenizer instance.
    round_msgs: Messages for the current conversational turn including the assistant response.

  Returns:
    The original completion token IDs from the full chat-template render.
  """
  tools_kwargs = {"tools": tools} if tools is not None else {}
  prompt_completion_tokens = tokenizer_model.apply_chat_template(
      round_msgs, add_generation_prompt=False, tokenize=True, enable_thinking=enable_thinking, **tools_kwargs
  )
  # include generation_prompt as part of the prompt tokens
  prompt_tokens = tokenizer_model.apply_chat_template(
      round_msgs[:-1], add_generation_prompt=True, tokenize=True, enable_thinking=enable_thinking, **tools_kwargs
  )

  prompt_completion_ids = extract_token_ids(prompt_completion_tokens)
  prompt_ids = extract_token_ids(prompt_tokens)

  # Walk forward until the two sequences diverge
  common_len = 0
  for full_id, prompt_id in zip(prompt_completion_ids, prompt_ids):
    if full_id == prompt_id:
      common_len += 1
    else:
      break

  if common_len == 0:
    raise ValueError(
        "Chat template generation prompt mismatch: no common prefix tokens found.\n"
        f"Full conversation tokens: {prompt_completion_ids} ('{tokenizer_model.decode(prompt_completion_ids)}')\n"
        f"Generation prompt tokens: {prompt_ids} ('{tokenizer_model.decode(prompt_ids)}')\n"
        "Cannot determine completion boundary."
    )

  completion_tokens = prompt_completion_ids[common_len:]
  return completion_tokens


def _render_suffix_ids(
    tokenizer_model,
    baseline_msgs,
    superset_msgs,
    *,
    baseline_gen,
    superset_gen,
    tools,
    enable_thinking,
    boundary_name,
):
  """Extract a token suffix only when the previous render remains an exact prefix."""
  kwargs = {"tools": tools} if tools is not None else {}
  baseline_ids = extract_token_ids(
      tokenizer_model.apply_chat_template(
          baseline_msgs,
          add_generation_prompt=baseline_gen,
          tokenize=True,
          enable_thinking=enable_thinking,
          **kwargs,
      )
  )
  superset_ids = extract_token_ids(
      tokenizer_model.apply_chat_template(
          superset_msgs,
          add_generation_prompt=superset_gen,
          tokenize=True,
          enable_thinking=enable_thinking,
          **kwargs,
      )
  )
  common_len = 0
  for left, right in zip(baseline_ids, superset_ids):
    if left != right:
      break
    common_len += 1
  if common_len != len(baseline_ids):
    # Structural diagnostics deliberately omit decoded windows and message bodies.
    raise ValueError(
        f"Chat template {boundary_name} mismatch: baseline render is not an exact token prefix. "
        f"Tokenizer: {getattr(tokenizer_model, 'name_or_path', type(tokenizer_model).__name__)}; "
        f"roles: {[message.get('role') for message in superset_msgs]}; "
        f"divergence offset: {common_len}; baseline tokens: {len(baseline_ids)}; "
        f"superset tokens: {len(superset_ids)}; "
        f"Trailing tool messages: {sum(message.get('role') == 'tool' for message in superset_msgs[len(baseline_msgs):])}; "
        f"Baseline divergence window: {baseline_ids[max(0, common_len - 16):common_len + 16]}; "
        f"Superset divergence window: {superset_ids[max(0, common_len - 16):common_len + 16]}."
    )
  return superset_ids, superset_ids[len(baseline_ids) :]


def _get_tool_results_and_completion_deltas(  # pylint: disable=too-many-locals
    tokenizer_model, round_msgs, assistant_message, tools=None, enable_thinking=True
):
  """Render trailing tool results and the assistant response as adjacent token deltas.

  ``round_msgs`` must be the live round state and end in one or more tool
  messages. The chat template render immediately before those tool messages is
  required to be an exact token prefix of the render including the tools and
  the next assistant generation prompt. Returning only the suffix preserves a
  single canonical token stream without replaying the preceding tool call.
  When a template uses the response opener as the assistant call turn's
  model-emitted stop/end-of-message token, that shared opener stays in the
  loss-applied call completion. Only the newly rendered tool-result suffix is
  returned as masked context.

  Args:
    tokenizer_model: The tokenizer instance.
    round_msgs: Live messages for the current round, ending in role=tool.
    assistant_message: The assistant message that follows the trailing tools.
    tools: Optional tool declarations passed to the chat template.

  Returns:
    A pair containing the masked tool-result/assistant-prefix token IDs and
    the loss-applied assistant-completion token IDs.

  Raises:
    ValueError: If no trailing tool messages exist, the template is not
      prefix-stable at the tool-result boundary, or the tool result is absent
      from the rendered context.
  """
  first_tool_idx = len(round_msgs)
  while first_tool_idx > 0 and round_msgs[first_tool_idx - 1]["role"] == "tool":
    first_tool_idx -= 1

  trailing_tool_count = len(round_msgs) - first_tool_idx
  if trailing_tool_count == 0:
    raise ValueError("Tool-result prompt extraction requires round_msgs to end with role='tool'.")
  if first_tool_idx == 0:
    raise ValueError("Tool-result prompt extraction requires context before the trailing tool message(s).")

  tokenizer_name = getattr(tokenizer_model, "name_or_path", type(tokenizer_model).__name__)
  roles = [message.get("role", "<missing>") for message in round_msgs]
  tools_kwargs = {"tools": tools} if tools is not None else {}
  superset_ids, suffix_ids = _render_suffix_ids(
      tokenizer_model,
      round_msgs[:first_tool_idx],
      round_msgs,
      baseline_gen=False,
      superset_gen=True,
      tools=tools,
      enable_thinking=enable_thinking,
      boundary_name="tool-result prompt",
  )
  baseline_ids = superset_ids[: len(superset_ids) - len(suffix_ids)]

  # Some templates emit speculative generation-prompt tokens which are not present when the
  # concrete assistant message has no corresponding content. Some templates add
  # an opening thinking-channel token after a tool result, but omits it from a full assistant
  # render with no reasoning field. Keep only the generation-prompt prefix shared by the real
  # assistant render so the carried delta does not add redundant tokens.
  tool_context_tokens = tokenizer_model.apply_chat_template(
      round_msgs,
      add_generation_prompt=False,
      tokenize=True,
      enable_thinking=enable_thinking,
      **tools_kwargs,
  )
  assistant_tokens = tokenizer_model.apply_chat_template(
      round_msgs + [assistant_message],
      add_generation_prompt=False,
      tokenize=True,
      enable_thinking=enable_thinking,
      **tools_kwargs,
  )
  tool_context_ids = extract_token_ids(tool_context_tokens)
  assistant_ids = extract_token_ids(assistant_tokens)

  if baseline_ids != tool_context_ids[: len(baseline_ids)]:
    raise ValueError(
        "Chat template tool-result context mismatch: adding the trailing tool message(s) changes tokens "
        "inside the preceding conversation. "
        f"Tokenizer: {tokenizer_name}; baseline tokens: {len(baseline_ids)}; "
        f"tool-context tokens: {len(tool_context_ids)}."
    )
  if len(tool_context_ids) == len(baseline_ids):
    raise ValueError(
        "Chat template emitted no tool-result context tokens. The role=tool message shape may be "
        "incompatible with this tokenizer template.\n"
        f"Tokenizer: {tokenizer_name}\n"
        f"Roles: {roles}\n"
        f"Trailing tool messages: {trailing_tool_count}\n"
        f"Baseline tokens: {len(baseline_ids)}; tool-context tokens: {len(tool_context_ids)}."
    )

  assistant_common_len = 0
  for prompt_id, assistant_id in zip(superset_ids, assistant_ids):
    if prompt_id != assistant_id:
      break
    assistant_common_len += 1

  if (
      tool_context_ids != superset_ids[: len(tool_context_ids)]
      or tool_context_ids != assistant_ids[: len(tool_context_ids)]
  ):
    raise ValueError(
        "Chat template tool-result context mismatch: rendering the following assistant changes tokens "
        "inside the tool-result context.\n"
        f"Tokenizer: {tokenizer_name}\n"
        f"Roles: {roles}\n"
        f"Trailing tool messages: {trailing_tool_count}\n"
        f"Baseline tokens: {len(baseline_ids)}; tool-context tokens: {len(tool_context_ids)}; "
        f"generation-prompt tokens: {len(superset_ids)}; assistant tokens: {len(assistant_ids)}; "
        f"assistant divergence offset: {assistant_common_len}"
    )
  if assistant_common_len < len(tool_context_ids):
    raise ValueError(
        "Chat template assistant generation prompt diverges before the complete tool-result context. "
        f"Tokenizer: {tokenizer_name}; tool-context tokens: {len(tool_context_ids)}; "
        f"assistant divergence offset: {assistant_common_len}."
    )

  delta_ids = superset_ids[len(baseline_ids) : assistant_common_len]
  completion_ids = assistant_ids[assistant_common_len:]
  return delta_ids, completion_ids


def validate_pinned_context_prefix(tokenizer_model, pinned_ids, prompt_ids, roles, boundary_name):
  """Require a pinned leading block to be an exact token prefix of a real prompt render."""
  common_len = 0
  for pinned_id, prompt_id in zip(pinned_ids, prompt_ids):
    if pinned_id != prompt_id:
      break
    common_len += 1

  if common_len == len(pinned_ids):
    return

  tokenizer_name = getattr(tokenizer_model, "name_or_path", type(tokenizer_model).__name__)
  tokenizer_revision = getattr(tokenizer_model, "_commit_hash", None)
  if tokenizer_revision is None:
    tokenizer_revision = getattr(tokenizer_model, "init_kwargs", {}).get("_commit_hash", "unknown")
  window_radius = 16
  start = max(0, common_len - window_radius)
  pinned_end = min(len(pinned_ids), common_len + window_radius)
  prompt_end = min(len(prompt_ids), common_len + window_radius)
  pinned_window = pinned_ids[start:pinned_end]
  prompt_window = prompt_ids[start:prompt_end]
  raise ValueError(
      "Chat template pinned-context mismatch: the canonical leading system/developer/tools block "
      f"is not an exact token prefix at the {boundary_name} boundary.\n"
      f"Tokenizer: {tokenizer_name}; revision: {tokenizer_revision}\n"
      f"Roles: {roles}\n"
      f"Pinned tokens: {len(pinned_ids)}; prompt tokens: {len(prompt_ids)}; divergence offset: {common_len}\n"
      f"Pinned divergence window [{start}:{pinned_end}]: {pinned_window} "
      f"('{tokenizer_model.decode(pinned_window, skip_special_tokens=False)}')\n"
      f"Prompt divergence window [{start}:{prompt_end}]: {prompt_window} "
      f"('{tokenizer_model.decode(prompt_window, skip_special_tokens=False)}')"
  )


def _get_pinned_context_ids(tokenizer_model, leading_message, first_prompt_messages, tools=None, enable_thinking=True):
  """Render only the canonical leading system/developer/native-tools block as token IDs.

  A tools-only conversation still has a tokenizer-generated leading developer
  block. Use an empty synthetic developer message only for rendering that
  block; it is not inserted into the source conversation. When neither an
  explicit leading message nor tools exist, retain only a tokenizer-generated
  leading block that is an exact prefix of the real first-user prompt. If the
  synthetic block is not a prefix, fall back to an exact BOS-only prefix.
  """
  tools_kwargs = {"tools": tools} if tools is not None else {}
  prompt_ids = extract_token_ids(
      tokenizer_model.apply_chat_template(
          first_prompt_messages,
          add_generation_prompt=True,
          tokenize=True,
          enable_thinking=enable_thinking,
          **tools_kwargs,
      )
  )

  if leading_message is not None or tools:
    pinned_messages = [leading_message or {"role": "developer", "content": ""}]
    pinned_ids = extract_token_ids(
        tokenizer_model.apply_chat_template(
            pinned_messages,
            add_generation_prompt=False,
            tokenize=True,
            enable_thinking=enable_thinking,
            **tools_kwargs,
        )
    )
  else:
    synthetic_ids = []
    try:
      synthetic_ids = extract_token_ids(
          tokenizer_model.apply_chat_template(
              [{"role": "developer", "content": ""}],
              add_generation_prompt=False,
              tokenize=True,
              enable_thinking=enable_thinking,
          )
      )
    except TemplateError:
      # Some templates reject developer/system messages. BOS remains a safe
      # candidate if and only if the real first-user render begins with it.
      pass

    if synthetic_ids and synthetic_ids == prompt_ids[: len(synthetic_ids)]:
      pinned_ids = synthetic_ids
    else:
      bos_token_id = getattr(tokenizer_model, "bos_token_id", None)
      bos_ids = [int(bos_token_id)] if bos_token_id is not None else []
      if bos_ids and bos_ids == prompt_ids[: len(bos_ids)]:
        pinned_ids = bos_ids
      else:
        tokenizer_name = getattr(tokenizer_model, "name_or_path", type(tokenizer_model).__name__)
        tokenizer_revision = getattr(tokenizer_model, "_commit_hash", None)
        if tokenizer_revision is None:
          tokenizer_revision = getattr(tokenizer_model, "init_kwargs", {}).get("_commit_hash", "unknown")
        raise ValueError(
            "Unable to derive a safe generated leading-context pin: neither the complete synthetic "
            "developer block nor BOS is an exact token prefix of the real first-user prompt. "
            f"Tokenizer: {tokenizer_name}; revision: {tokenizer_revision}; enable_thinking={enable_thinking}; "
            f"synthetic tokens: {len(synthetic_ids)}; BOS token: {bos_token_id}; prompt tokens: {len(prompt_ids)}; "
            f"prompt prefix IDs: {prompt_ids[:16]}"
        )

  # Do not recover a mismatch by taking len(pinned_ids) tokens from prompt_ids or by
  # pinning the complete first prompt. Without tokenizer-provided message spans, the
  # former can cut into the user message or omit part of a context-dependent leading
  # block, while the latter would replay the first user's task in later windows.
  validate_pinned_context_prefix(
      tokenizer_model,
      pinned_ids,
      prompt_ids,
      [message.get("role", "<missing>") for message in first_prompt_messages],
      "chat-template",
  )
  return pinned_ids


def apply_chat_template(
    example,
    tokenizer_model,
    data_column_name,
    tools_column_name=None,
    pin_leading_context=False,
    enable_thinking=True,
    return_segment_ids=True,
):
  """Formats conversational data by applying the tokenizer's chat template
  and identifying prompt/completion segments for SFT masking.

  Args:
    example: A dictionary containing conversational data. It is expected to have a key
      specified by `data_column_name` that holds a list of messages.
    tokenizer_model: The tokenizer instance associated with the language model,
      which contains the specific chat template.
    data_column_name: The name of the column in the `example` dictionary
      that contains the list of messages.
    tools_column_name: Optional column containing native tool declarations.
    pin_leading_context: Whether to emit the tokenized canonical leading
      system/developer/tools block for long-example windowing.
    return_segment_ids: Emit aligned original IDs for downstream tokenization.

  Returns:
    The modified `example` dictionary.
      - The `data_column_name` column will be updated to a list of
        messages, each formatted according to the tokenizer's chat template.
      - A new column "is_prompt" is added, where `True` indicates the
        tokens contain the system message, user message, and generation
        prompt (if applicable). `False` indicates the expected LLM
        completion, excluding the assistant's start tokens.
  """
  messages = []
  is_prompt = []
  segment_ids = []
  round_msgs = []
  emitted_len = 0
  leading_message = None
  pinned_context_ids = []
  pinned_context_rendered = False
  conversation = example[data_column_name]
  if isinstance(conversation, str):
    conversation = json.loads(conversation)
  tools = example.get(tools_column_name) if tools_column_name else None
  if isinstance(tools, str):
    tools = json.loads(tools)
  tools_kwargs = {"tools": tools} if tools is not None else {}

  def append_segment(ids, prompt):
    if not ids:
      return
    segment_ids.append(list(ids))
    messages.append(tokenizer_model.decode(ids, skip_special_tokens=False))
    is_prompt.append(prompt)

  try:
    for idx, message in enumerate(conversation):
      if message["role"] in ("system", "developer"):
        if idx != 0:
          raise ValueError(f"'{message['role']}' message found at index {idx}. It must be at index 0.")
        leading_message = message
        round_msgs.append(message)
      elif message["role"] == "user":
        if round_msgs and round_msgs[-1]["role"] == "tool":
          # Tools have accumulated but have not yet been emitted. Flush their
          # suffix before taking the user suffix, or their bodies would be lost.
          _, pending_ids = _render_suffix_ids(
              tokenizer_model,
              round_msgs[:emitted_len],
              round_msgs,
              baseline_gen=False,
              superset_gen=False,
              tools=tools,
              enable_thinking=enable_thinking,
              boundary_name="tool-result context",
          )
          append_segment(pending_ids, True)
          _, user_ids = _render_suffix_ids(
              tokenizer_model,
              round_msgs,
              round_msgs + [message],
              baseline_gen=False,
              superset_gen=True,
              tools=tools,
              enable_thinking=enable_thinking,
              boundary_name="tool-to-user prompt",
          )
          round_msgs.append(message)
          append_segment(user_ids, True)
          emitted_len = len(round_msgs)
          continue
        round_msgs.append(message)
        if pin_leading_context and not pinned_context_rendered:
          pinned_context_ids = _get_pinned_context_ids(
              tokenizer_model,
              leading_message,
              round_msgs,
              tools=tools,
              enable_thinking=enable_thinking,
          )
          pinned_context_rendered = True
        prompt_in_chat_template = tokenizer_model.apply_chat_template(
            round_msgs,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=enable_thinking,
            **tools_kwargs,
        )
        append_segment(extract_token_ids(prompt_in_chat_template), True)
        emitted_len = len(round_msgs)
      elif message["role"] == "tool":
        round_msgs.append(message)
      elif message["role"] == "assistant":
        if not round_msgs:
          raise ValueError(f"Assistant message at index {idx} with no preceding context.")
        if round_msgs[-1]["role"] == "tool":
          # Tool results condition this assistant response, so preserve their tokenizer-rendered
          # delta as prompt context. Marking it as a prompt keeps the result visible to attention
          # while excluding it from completion-only loss. Response openers remain in
          # the preceding loss-applied call completion because they are model-emitted stop/EOM
          # tokens; the resumed render reuses that token before appending the masked result body.
          tool_results_delta, completion = _get_tool_results_and_completion_deltas(
              tokenizer_model,
              round_msgs,
              message,
              tools=tools,
              enable_thinking=enable_thinking,
          )
          append_segment(tool_results_delta, True)
          round_msgs.append(message)
        else:
          round_msgs.append(message)
          completion = _get_completion_in_chat_template(
              tokenizer_model,
              round_msgs,
              tools=tools,
              enable_thinking=enable_thinking,
          )
        append_segment(completion, False)
        emitted_len = len(round_msgs)
        # Clear round only when the next message starts a new user turn or conversation ends
        # This preserves context for consecutive assistant/tool messages
        next_idx = idx + 1
        if next_idx >= len(conversation) or conversation[next_idx]["role"] == "user":
          round_msgs.clear()
          emitted_len = 0
      else:
        raise ValueError(f"Unsupported message role '{message['role']}' at index {idx}.")
  except ValueError as e:
    max_logging.log(f"Unable to apply chat template: {e}")
    raise e
  example["is_prompt"] = is_prompt
  example[data_column_name] = messages
  if return_segment_ids:
    example[SFT_SEGMENT_IDS_KEY] = segment_ids
  if pin_leading_context:
    example[SFT_PINNED_CONTEXT_IDS_KEY] = pinned_context_ids
  return example


def apply_chat_template_with_assistant_mask(
    example,
    tokenizer_model,
    data_column_name,
    tools_column_name=None,
    pin_leading_context=False,
    enable_thinking=True,
):
  """Render one canonical SFT token stream and use template-owned loss spans.

  This is the token-native alternative to :func:`apply_chat_template`'s
  segmented longest-common-prefix path. The tokenizer renders the complete
  conversation exactly once and returns an ``assistant_masks`` array produced
  by Jinja ``{% generation %}`` blocks. The aligned token IDs and mask are then
  converted into the existing MaxText ``token_runs`` / ``is_prompt`` contract
  without decoding or re-tokenizing.

  The template may use a per-message ``trainable`` boolean to suppress marker
  ownership for historical assistant turns in dataset-expanded prefixes.
  MaxText does not interpret that field itself; it consumes only the mask
  returned by Transformers.

  Args:
    example: A dictionary containing a complete conversational row.
    tokenizer_model: A Hugging Face tokenizer with a generation-marked chat
      template.
    data_column_name: Column containing the message list.
    tools_column_name: Optional column containing native tool declarations.
    pin_leading_context: Whether to emit the canonical leading
      system/developer/native-tools token prefix for long-example windowing.
  Returns:
    The modified example with token-ID runs in ``data_column_name``, matching
    ``is_prompt`` flags, and optionally ``sft_pinned_context_ids``.

  Raises:
    ValueError: If the row or tokenizer output violates the canonical
      token-stream/mask contract.
  """
  conversation = example[data_column_name]
  if isinstance(conversation, str):
    conversation = json.loads(conversation)
  if not isinstance(conversation, list) or not conversation:
    raise ValueError("Canonical assistant-mask SFT requires a non-empty message list.")

  leading_message = None
  first_user_index = None
  for index, message in enumerate(conversation):
    if not isinstance(message, Mapping) or "role" not in message:
      raise ValueError(f"SFT message at index {index} must be a mapping with a role.")
    role = message["role"]
    if role in ("system", "developer"):
      if index != 0:
        raise ValueError(f"'{role}' message found at index {index}. It must be at index 0.")
      leading_message = message
    elif role == "user" and first_user_index is None:
      first_user_index = index
    elif role not in ("user", "assistant", "tool"):
      raise ValueError(f"Unsupported message role '{role}' at index {index}.")

  tools = example.get(tools_column_name) if tools_column_name else None
  if isinstance(tools, str):
    tools = json.loads(tools)
  tools_kwargs = {"tools": tools} if tools is not None else {}

  try:
    encoded = tokenizer_model.apply_chat_template(
        conversation,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=enable_thinking,
        preserve_thinking=enable_thinking,
        **tools_kwargs,
    )
  except (TypeError, ValueError) as error:
    max_logging.log(f"Unable to apply canonical assistant-mask chat template: {error}")
    raise

  input_ids = extract_token_ids(encoded)
  if isinstance(input_ids, np.ndarray):
    input_ids = input_ids.tolist()
  if input_ids and isinstance(input_ids[0], (list, tuple, np.ndarray)):
    if len(input_ids) != 1:
      raise ValueError(f"Canonical assistant-mask SFT expected one token stream, got batch size {len(input_ids)}.")
    input_ids = list(input_ids[0])

  if isinstance(encoded, Mapping):
    assistant_mask = encoded.get("assistant_masks")
  else:
    assistant_mask = getattr(encoded, "assistant_masks", None)
  if assistant_mask is None:
    raise ValueError(
        "Tokenizer did not return assistant_masks. The selected chat template must contain "
        "{% generation %} blocks and the Transformers version must support return_assistant_tokens_mask."
    )
  if isinstance(assistant_mask, np.ndarray):
    assistant_mask = assistant_mask.tolist()
  if assistant_mask and isinstance(assistant_mask[0], (list, tuple, np.ndarray)):
    if len(assistant_mask) != 1:
      raise ValueError(f"Canonical assistant-mask SFT expected one ownership mask, got batch size {len(assistant_mask)}.")
    assistant_mask = list(assistant_mask[0])

  token_runs, is_prompt = split_sft_token_stream_by_assistant_mask(input_ids, assistant_mask)
  example[data_column_name] = token_runs
  example["is_prompt"] = is_prompt

  if pin_leading_context:
    if first_user_index is None:
      raise ValueError("SFT leading-context pinning requires at least one user message.")
    first_prompt_messages = conversation[: first_user_index + 1]
    pinned_context_ids = _get_pinned_context_ids(
        tokenizer_model,
        leading_message,
        first_prompt_messages,
        tools=tools,
        enable_thinking=enable_thinking,
    )
    validate_pinned_context_prefix(
        tokenizer_model,
        pinned_context_ids,
        [token_id for token_run in token_runs for token_id in token_run],
        [message.get("role", "<missing>") for message in conversation],
        "canonical-full-conversation",
    )
    example[SFT_PINNED_CONTEXT_IDS_KEY] = pinned_context_ids

  return example


def validate_sft_segment_ids(segment_ids, is_prompt, text_chunks):
  """Validate the aligned token side column without coercing malformed IDs."""
  if (
      not isinstance(segment_ids, list)
      or not segment_ids
      or len(segment_ids) != len(is_prompt)
      or len(segment_ids) != len(text_chunks)
  ):
    raise ValueError("sft_segment_ids must be non-empty and aligned with text chunks and is_prompt.")
  for ids in segment_ids:
    if (
        not isinstance(ids, list)
        or not ids
        or any(
            isinstance(token_id, (bool, np.bool_)) or not isinstance(token_id, (int, np.integer)) or token_id < 0
            for token_id in ids
        )
    ):
      raise ValueError("sft_segment_ids must contain non-empty lists of nonnegative integer token IDs.")
  return segment_ids


def tokenization(example, hf_tokenizer, truncation, max_length, column_names):
  """Tokenize a HuggingFace dataset"""
  if SFT_SEGMENT_IDS_KEY in example:
    if len(column_names) != 1:
      raise ValueError("sft_segment_ids requires exactly one conversational text column.")
    column_name = column_names[0]
    rows = example[SFT_SEGMENT_IDS_KEY]
    prompts = example["is_prompt"]
    texts = example[column_name]
    if len(rows) != len(prompts) or len(rows) != len(texts):
      raise ValueError("sft_segment_ids batch must align with text and is_prompt rows.")
    example[column_name] = [
        validate_sft_segment_ids(ids, flags, chunks) for ids, flags, chunks in zip(rows, prompts, texts)
    ]
    return example
  for column_name in column_names:
    if isinstance(example[column_name], list):
      example[column_name] = [
          hf_tokenizer(x, truncation=truncation, max_length=max_length)["input_ids"] for x in example[column_name]
      ]
    elif isinstance(example[column_name], str):
      example[column_name] = hf_tokenizer(example[column_name], truncation=truncation, max_length=max_length)["input_ids"]
  return example


def split_sft_token_stream_by_assistant_mask(input_ids, assistant_mask):
  """Convert one canonical token stream and binary assistant mask into SFT token runs.

  The returned ``token_runs`` / ``is_prompt`` pair is the existing input
  contract consumed by :class:`SFTPromptMasking` and
  :class:`SFTPromptMaskingWindows`. A mask value of 1 means the corresponding
  token is model-owned and loss-applied, so its run receives
  ``is_prompt=False``. A value of 0 means masked context and receives
  ``is_prompt=True``.

  This conversion is token-native by construction: it never decodes or
  re-tokenizes, and concatenating the returned runs exactly reconstructs the
  supplied token IDs. Adjacent spans with the same ownership are deliberately
  coalesced because the downstream masking transforms care about token
  ownership changes, not individual Jinja generation-block boundaries.

  Args:
    input_ids: One non-empty, one-dimensional token-ID sequence.
    assistant_mask: A binary sequence aligned one-to-one with ``input_ids``;
      1 selects loss and 0 selects masked context.

  Returns:
    A pair ``(token_runs, is_prompt)`` containing non-empty contiguous token
    runs and their existing MaxText prompt/completion flags.

  Raises:
    ValueError: If the stream is empty, lengths differ, a mask value is not an
      integer/bool 0 or 1, or the mask selects no loss-bearing tokens.
    TypeError: If a token ID cannot be converted to an integer.
  """
  try:
    token_ids = [int(token_id) for token_id in input_ids]
  except (TypeError, ValueError) as error:
    raise TypeError("input_ids must be a one-dimensional sequence of integer token IDs.") from error

  if not token_ids:
    raise ValueError("input_ids must contain at least one token.")

  mask_values = list(assistant_mask)
  if len(token_ids) != len(mask_values):
    raise ValueError(
        "input_ids and assistant_mask must have identical lengths: "
        f"input_ids={len(token_ids)}, assistant_mask={len(mask_values)}."
    )

  normalized_mask = []
  for index, mask_value in enumerate(mask_values):
    if isinstance(mask_value, (bool, np.bool_)):
      normalized_mask.append(int(mask_value))
    elif isinstance(mask_value, (int, np.integer)) and int(mask_value) in (0, 1):
      normalized_mask.append(int(mask_value))
    else:
      raise ValueError(
          "assistant_mask must contain only integer/bool 0 or 1 values; " f"got {mask_value!r} at index {index}."
      )

  if not any(normalized_mask):
    raise ValueError(
        "assistant_mask contains no loss-bearing tokens; a completion-only SFT example cannot be context-only."
    )

  token_runs = []
  is_prompt = []
  run_start = 0
  for index in range(1, len(token_ids)):
    if normalized_mask[index] == normalized_mask[run_start]:
      continue
    token_runs.append(token_ids[run_start:index])
    is_prompt.append(normalized_mask[run_start] == 0)
    run_start = index
  token_runs.append(token_ids[run_start:])
  is_prompt.append(normalized_mask[run_start] == 0)

  return token_runs, is_prompt


@dataclasses.dataclass
class SFTPromptMasking(grain.MapTransform):
  """Construct inputs and targets for SFT training. Concat prompt and completion to generate inputs.
  For targets, if train on completion only, the prompt will be masked by unk_id. Otherwise the same as inputs.
  """

  def __init__(self, text_column_name, completion_only, max_target_length, unk_id=0):
    self.text_column_name = text_column_name
    self.completion_only = completion_only
    self.max_target_length = max_target_length
    self.unk_id = unk_id

  def map(self, element):
    """
    Maps a single dataset element to an SFT training instance.
    It concatenates the prompt and completion to form the `inputs` sequence.
    For the `targets` sequence:
    - If `self.completion_only` is `True`, the prompt portion of the
      concatenated sequence is masked using `self.unk_id`.
    - If `self.completion_only` is `False`, the target sequence is
      identical to the input sequence.
    """
    inputs, targets = [], []
    for i, text in enumerate(element[self.text_column_name]):
      inputs += text
      targets += [self.unk_id] * len(text) if self.completion_only and element["is_prompt"][i] else text
    out = {
        "inputs": np.asarray(inputs[: self.max_target_length], dtype=np.int32),
        "targets": np.asarray(targets[: self.max_target_length], dtype=np.int32),
    }
    if "dataset_id" in element:
      out["dataset_id"] = np.full(len(out["inputs"]), np.int32(element["dataset_id"]), dtype=np.int32)
    return out


@dataclasses.dataclass(frozen=True)
class SFTWindowGeometry:
  """Effective bounded geometry shared by SFT windowing and preflight checks."""

  overlap_cap: int
  min_loss_room: int
  context_cap: int


def get_sft_window_geometry(max_target_length, overlap=256, context_cap=-1):
  """Return the exact clamped overlap, minimum loss room, and context cap used by SFT windows."""
  if max_target_length <= 0:
    raise ValueError(f"max_target_length must be positive, got {max_target_length}.")
  overlap_cap = max(0, min(overlap, max_target_length // 8))
  min_loss_room = max(1, max_target_length // 8)
  requested_cap = context_cap if (context_cap and context_cap > 0) else max_target_length // 2
  effective_context_cap = max(1, min(requested_cap, max_target_length - overlap_cap - min_loss_room))
  return SFTWindowGeometry(
      overlap_cap=overlap_cap,
      min_loss_room=min_loss_room,
      context_cap=effective_context_cap,
  )


@dataclasses.dataclass
class SFTPromptMaskingWindows(FlatMapTransform):
  """Construct SFT inputs/targets for completion-only training, splitting examples longer than
  ``max_target_length`` into multiple ``<= max_target_length`` records via a prompt-pinned sliding
  window instead of head-truncating them.

  Motivation: the 1:1 :class:`SFTPromptMasking` truncates the concatenated sequence with
  ``[:max_target_length]``. For any example longer than ``max_target_length`` this drops the tail —
  including the turn terminator (e.g. ``<end_of_turn>``) — from both ``inputs`` and ``targets``, so
  the stop token never enters the loss and the model is trained on a stop-less completion prefix.

  This transform instead emits, per completion segment of an over-length example, a sequence of
  windows. Each window is ``[ bounded conversation-prefix context (masked) ] +
  [ small completion overlap (masked) ] + [ a slice of new completion tokens (loss) ]``. By default,
  bounded context is the newest prefix tail. With leading-context pinning enabled, a real left cut
  instead keeps the canonical system/developer/native-tools block plus the newest tail that fits.
  The loss slices tile the completion with NO overlap, so every completion token — including the
  terminator in the final window — contributes to the loss exactly once. Examples that already fit
  yield a single record byte-identical to :class:`SFTPromptMasking`.

  Only completion-only SFT is supported (the context/overlap tokens are masked with ``unk_id``).
  """

  max_fan_out: int = 32

  def __init__(
      self,
      text_column_name,
      completion_only,
      max_target_length,
      unk_id=0,
      overlap=256,
      context_cap=-1,
      max_fan_out=32,
      pin_leading_context=False,
      pinned_context_overflow="error",
      pinned_context_warn_fraction=0.5,
  ):
    self.text_column_name = text_column_name
    self.completion_only = completion_only
    self.max_target_length = max_target_length
    self.unk_id = unk_id
    self.overlap = overlap
    self.context_cap = context_cap
    self.max_fan_out = max_fan_out
    self.pin_leading_context = pin_leading_context
    if pinned_context_overflow != "error":
      raise ValueError(
          "Only sft_window_pinned_context_overflow='error' is supported; " f"got {pinned_context_overflow!r}."
      )
    if not 0.0 < pinned_context_warn_fraction < 1.0:
      raise ValueError(
          "sft_window_pinned_context_warn_fraction must be between 0 and 1; " f"got {pinned_context_warn_fraction}."
      )
    self.pinned_context_overflow = pinned_context_overflow
    self.pinned_context_warn_fraction = pinned_context_warn_fraction
    # Grain workers own separate transform instances. This counter only rate-limits
    # warning logs within one instance; it is not a globally aggregatable row metric.
    self.pinned_context_warning_count = 0

  def _single_record(self, segments, is_prompt):
    """Fast path identical to SFTPromptMasking.map for examples that fit in max_target_length."""
    inputs, targets = [], []
    for seg, is_p in zip(segments, is_prompt):
      seg = list(seg)
      inputs += seg
      targets += [self.unk_id] * len(seg) if (self.completion_only and is_p) else seg
    return {
        "inputs": np.asarray(inputs, dtype=np.int32),
        "targets": np.asarray(targets, dtype=np.int32),
    }

  def _stamp_ds(self, records, element):
    """Attach a per-token dataset_id (constant across an example's fan-out windows)."""
    if "dataset_id" in element:
      ds_id = np.int32(element["dataset_id"])
      for r in records:
        r["dataset_id"] = np.full(len(r["inputs"]), ds_id, dtype=np.int32)
    return records

  def flat_map(self, element):
    """Split one over-length SFT example into bounded loss-bearing windows."""
    length = self.max_target_length
    segments = element[self.text_column_name]
    is_prompt = element["is_prompt"]

    total = sum(len(seg) for seg in segments)
    if total <= length:
      return self._stamp_ds([self._single_record(segments, is_prompt)], element)

    geometry = get_sft_window_geometry(length, self.overlap, self.context_cap)
    overlap_cap = geometry.overlap_cap
    cap = geometry.context_cap
    pinned = list(element.get(SFT_PINNED_CONTEXT_IDS_KEY, [])) if self.pin_leading_context else []

    records = []
    prefix = []  # all tokens of preceding segments, replayed (masked) as grounding context
    warned_for_element = False
    for seg, is_p in zip(segments, is_prompt):
      seg = list(seg)
      if self.completion_only and is_p:
        prefix += seg
        continue
      comp = seg
      if not comp:
        continue
      if len(prefix) <= cap or not pinned:
        ctx = list(prefix) if len(prefix) <= cap else prefix[-cap:]
      else:
        if prefix[: len(pinned)] != pinned:
          raise ValueError(
              "SFT pinned leading context is not an exact token prefix of the accumulated conversation. "
              f"Pinned tokens: {len(pinned)}; accumulated prefix tokens: {len(prefix)}."
          )
        if len(pinned) >= cap:
          # Overflow is intentionally fail-loud. Left-truncating can remove BOS,
          # opening delimiters, or the start of developer instructions; right-
          # truncating can cut tool schemas or their closing delimiters. Silently
          # dropping the example is also a dataset-policy decision, not a safe
          # transform default. Preflight and explicitly shorten/split the pin,
          # increase the context budget, or drop/quarantine the row with accounting.
          raise ValueError(
              "SFT pinned leading context leaves no recent-conversation tail budget. "
              f"Pinned tokens: {len(pinned)}; effective context cap: {cap}; "
              f"max_target_length: {length}; overlap cap: {overlap_cap}."
          )
        if not warned_for_element and len(pinned) > cap * self.pinned_context_warn_fraction:
          self.pinned_context_warning_count += 1
          warned_for_element = True
          warning_count = self.pinned_context_warning_count
          if warning_count <= 3 or warning_count & (warning_count - 1) == 0:
            max_logging.warning(
                "SFT pinned leading context consumes more than the configured warning fraction of the "
                f"effective context cap: pinned_tokens={len(pinned)}, effective_context_cap={cap}, "
                f"remaining_tail_budget={cap - len(pinned)}, warning_fraction="
                f"{self.pinned_context_warn_fraction}, warning_rows_seen_by_worker={warning_count}."
            )
        tail_budget = cap - len(pinned)
        tail_start = max(len(pinned), len(prefix) - tail_budget)
        ctx = pinned + prefix[tail_start:]
      i, n = 0, len(comp)
      while i < n:
        if len(records) >= self.max_fan_out:
          max_logging.log(
              f"SFTPromptMaskingWindows: hit max_fan_out={self.max_fan_out}; dropping {n - i} "
              "trailing completion token(s) (including the turn terminator) for one example."
          )
          return self._stamp_ds(records, element)
        overlap_tokens = comp[max(0, i - overlap_cap) : i]
        room = length - len(ctx) - len(overlap_tokens)
        loss_tokens = comp[i : i + room]
        n_mask = len(ctx) + len(overlap_tokens)
        records.append(
            {
                "inputs": np.asarray(ctx + overlap_tokens + loss_tokens, dtype=np.int32),
                "targets": np.asarray([self.unk_id] * n_mask + loss_tokens, dtype=np.int32),
            }
        )
        i += len(loss_tokens)
      prefix += comp

    return self._stamp_ds(records, element)


@dataclasses.dataclass
class SFTPromptMaskingVision(grain.MapTransform):
  """SFT prompt masking for multimodal"""

  def __init__(self, query_column, response_column, max_target_length, pad_id):
    self.query_column = query_column
    self.response_column = response_column
    self.max_target_length = max_target_length
    self.pad_id = pad_id

  def map(self, element):
    inputs = np.concatenate((element[self.query_column], element[self.response_column]))
    targets = np.concatenate((np.asarray([self.pad_id] * len(element[self.query_column])), element[self.response_column]))
    return {
        "inputs": np.asarray(inputs[: self.max_target_length], dtype=np.int32),
        "targets": np.asarray(targets[: self.max_target_length], dtype=np.int32),
        "images": element["images"],
    }


@dataclasses.dataclass
class HFNormalizeFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input"""

  def __init__(self, column_name):
    self.column_name = column_name

  def map(self, element):
    return {
        "inputs": np.asarray(element[self.column_name], dtype=np.int32),
        "targets": np.asarray(element[self.column_name], dtype=np.int32),
    }


class HFDataSource(grain.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""

  def __init__(
      self,
      dataset: "datasets.IterableDataset",
      dataloading_host_index: int,
      dataloading_host_count: int,
      num_threads: int,
      max_target_length: int,
      data_column_names: list[str],
  ):
    from datasets.distributed import split_dataset_by_node  # pylint: disable=import-outside-toplevel

    self._split_dataset_by_node = split_dataset_by_node
    self.dataset = dataset
    self.num_threads = num_threads
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.max_target_lenth = max_target_length
    self.data_column_names = data_column_names
    if hasattr(dataset, "n_shards"):
      self.n_shards = dataset.n_shards
    else:
      self.n_shards = 1
    self._check_shard_count()
    self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range(self.num_threads)]
    self.datasets = [self._split_dataset_by_node(dataset, world_size=self.n_shards, rank=x) for x in self.dataset_shards]
    self.data_iters = []

  def _check_shard_count(self):
    if self.n_shards < (self.dataloading_host_count * self.num_threads):
      warnings.warn(
          f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
          "smaller than number of host loading data. This is known to lead to inefficient dataloading. See"
          "github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#multihost-dataloading-best-practice"
      )
      self.n_shards = self.dataloading_host_count * self.num_threads

  def _update_shard(self, idx):
    """update shard"""
    new_shard = self.dataset_shards[idx] + self.dataloading_host_count * self.num_threads
    if new_shard < self.n_shards:
      max_logging.log(
          f"Updating host {self.dataloading_host_index} dataset {idx}, was on shard {self.dataset_shards[idx]}"
      )
      max_logging.log(f"New shard is {new_shard}")
      self.dataset_shards[idx] = new_shard
      self.datasets[idx] = self._split_dataset_by_node(
          self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx]
      )
      self.data_iters[idx] = iter(self.datasets[idx])
    else:
      raise StopIteration(f"Run out of shards on host {self.dataloading_host_index}, shard {new_shard} is not available")

  def __len__(self):
    """Return length of the HF dataset. Since HuggingFace IterableDataset does not have length,
    a fake length bigger than the dataset is returned"""
    return 10_000_000_000

  def __getitem__(self, index):
    """Since HuggingFace IterableDataset does not support random access by index.
    The next item in the iterator is returned."""
    if not self.data_iters:
      self.data_iters = [iter(x) for x in self.datasets]
    idx = int(current_thread().name.split("_")[1])

    while True:
      try:
        data = next(self.data_iters[idx])
        return data
      except StopIteration:
        self._update_shard(idx)


########## Functions used by Grain pipeline


class _GCSTFRecordReader(_TFRecordReader):
  """Extends Grain's _TFRecordReader to open TFRecord files from GCS via streaming BlobReader."""

  def __init__(self, path: str):
    # Skip parent __init__ (which calls open(path, "rb")) and open via GCS BlobReader instead.
    bucket_name, blob_name = gcs_utils.parse_gcs_bucket_and_prefix(path)
    self._reader = gcs_utils.storage.Client().bucket(bucket_name).blob(blob_name).open("rb")


class _GCSTFRecordDatasetIterator(_TFRecordDatasetIterator):
  """Extends Grain's _TFRecordDatasetIterator to use _GCSTFRecordReader for GCS paths."""

  def __init__(self, path: str):
    # Skip parent __init__ (which creates _TFRecordReader); use GCS-aware reader instead.
    grain.DatasetIterator.__init__(self)
    self._reader = _GCSTFRecordReader(path)


class GCSTFRecordIterDataset(TFRecordIterDataset):
  """Extends Grain's TFRecordIterDataset to support GCS paths."""

  def __iter__(self) -> grain.DatasetIterator:  # pylint: disable=non-iterator-returned
    return _GCSTFRecordDatasetIterator(self._path)


def make_tfrecord_iter_dataset(path: str):
  """Returns the appropriate TFRecordIterDataset for local or GCS paths."""
  if path.startswith("gs://"):
    return GCSTFRecordIterDataset(path)
  return TFRecordIterDataset(path)


def make_parquet_iter_dataset(path: str, hf_access_token: str | None = None):
  """Returns the appropriate ParquetIterDataset for local or HF paths."""
  if path.startswith("hf://"):
    from huggingface_hub import HfFileSystem  # pylint: disable=import-outside-toplevel

    return grain.experimental.ParquetIterDataset(path, filesystem=HfFileSystem(token=hf_access_token))
  return grain.experimental.ParquetIterDataset(path)


def compute_file_sharding(file_count, host_index, host_count):
  """Compute per-host file slicing and optional row-shard parameters.

  When file_count >= host_count, each host reads a disjoint subset of files via the
  standard `[host_index::host_count]` slice. When file_count < host_count, every file
  is replicated across `ceil(host_count/file_count)` hosts (or `floor` for files past
  the remainder), and those hosts shard records by index within the file. This bounds
  concurrent readers per file at `ceil(host_count/file_count)`

  Returns:
    file_slice (slice): file slice this host should keep — also the (start, step)
      pair `(host_index, host_count)` to feed `tf.distribute.InputContext` when applicable.
    files_per_host (int): files this host's slice contains per epoch.
    row_shard (tuple|None): `(row_shard_index, row_shard_count)` when host_count > file_count
      and the file's group has >1 reader; otherwise None.
  """
  if file_count >= host_count:
    return slice(host_index, None, host_count), max(file_count // host_count, 1), None
  file_idx = host_index % file_count
  row_shard_idx = host_index // file_count
  row_shard_count = (host_count // file_count) + (1 if file_idx < (host_count % file_count) else 0)
  row_shard = (row_shard_idx, row_shard_count) if row_shard_count > 1 else None
  return slice(file_idx, None, file_count), 1, row_shard


class _IndexShardDatasetIterator(grain.DatasetIterator):
  """Iterator that yields every nth element of its parent (round-robin by index)."""

  def __init__(self, parent: grain.DatasetIterator, host_index: int, host_count: int):
    super().__init__(parent)
    self._host_index = host_index
    self._host_count = host_count
    self._next_index = 0

  def __next__(self):
    while True:
      value = next(self._parent)
      current = self._next_index
      self._next_index += 1
      if current % self._host_count == self._host_index:
        return value

  def get_state(self):
    return {
        "next_index": self._next_index,
        "parent": self._parent.get_state(),
    }

  def set_state(self, state):
    self._next_index = state["next_index"]
    self._parent.set_state(state["parent"])


class IndexShardIterDataset(grain.IterDataset):
  """Shards an IterDataset across hosts by element index (host i keeps records where idx % N == i).

  Use when the upstream `IterDataset` order is deterministic and identical on every host;
  this guarantees disjoint, balanced slices without per-file sharding.
  """

  def __init__(self, parent: grain.IterDataset, host_index: int, host_count: int):
    super().__init__(parent)
    self._host_index = host_index
    self._host_count = host_count

  def __iter__(self) -> _IndexShardDatasetIterator:
    return _IndexShardDatasetIterator(
        self._parent.__iter__(),
        host_index=self._host_index,
        host_count=self._host_count,
    )


@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
  """Parse serialized tf.train.Example protos for arrayrecord/tfrecord datasets.

  Also validates that the stored field type matches `tokenize`: raises
  ValueError if `tokenize=True` but the column contains integers (pre-tokenized)
  or if `tokenize=False` but the column contains bytes (raw text).
  """

  def __init__(self, data_columns, tokenize):
    self.data_columns = list(data_columns)
    self.tokenize = tokenize

  # Columns that may legitimately be absent from a record (e.g. datasets without
  # function-calling data). Missing optional columns are skipped, not an error,
  # so a single mixture can blend tools/non-tools datasets.
  OPTIONAL_COLUMNS = frozenset({"tools"})

  def map(self, element):
    """Parse a serialized tf.train.Example proto and extract features."""
    dataset_id = None
    if isinstance(element, dict) and "raw" in element:  # per_dataset_metrics: stamped upstream
      dataset_id, element = element["dataset_id"], element["raw"]
    example = example_pb2.Example()
    example.ParseFromString(element)
    features = example.features.feature

    missing = [c for c in self.data_columns if c not in features and c not in self.OPTIONAL_COLUMNS]
    if missing:
      raise ValueError(
          f"Column {missing} not found in dataset. Available columns: {sorted(features.keys())}. "
          "Please set train_data_columns or eval_data_columns accordingly."
      )

    parsed = {}
    for col in self.data_columns:
      if col in features:
        f = features[col]

        # Dynamically check proto field type instead of relying on the tokenize flag
        if len(f.float_list.value) > 0:
          parsed[col] = np.array(f.float_list.value, dtype=np.float32)
        elif len(f.int64_list.value) > 0:
          parsed[col] = np.array(f.int64_list.value, dtype=np.int32)
        elif len(f.bytes_list.value) > 0:
          parsed[col] = np.array(f.bytes_list.value, dtype=object)
        else:
          parsed[col] = np.array([])

    # Reshape the flattened arrays back to 2D [seq_len, top_k]
    seq_len = len(parsed.get("inputs", []))
    if seq_len > 0:
      if "top_k_logits" in parsed and len(parsed["top_k_logits"]) > 0:
        parsed["top_k_logits"] = parsed["top_k_logits"].reshape(seq_len, -1)
      if "top_k_indices" in parsed and len(parsed["top_k_indices"]) > 0:
        parsed["top_k_indices"] = parsed["top_k_indices"].reshape(seq_len, -1)

    if dataset_id is not None:
      parsed["dataset_id"] = np.int32(dataset_id)
    return parsed


@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
  """Normalize text feature keys."""

  def __init__(self, column_names, tokenize, scalar_bool_columns=()):
    self.column_names = column_names
    self.tokenize = tokenize
    self.scalar_bool_columns = frozenset(scalar_bool_columns)

  # Columns that may legitimately be absent from a record (e.g. datasets
  # without function-calling data). Missing optional columns are skipped
  # instead of raising, so a single mixture can blend tools/non-tools datasets.
  OPTIONAL_COLUMNS = frozenset({"tools"})

  def map(self, element):
    """Normalize feature keys, skipping optional columns (e.g. `tools`) absent from a record."""
    out = {}
    for col in self.column_names:
      if col not in element:
        if col in self.OPTIONAL_COLUMNS:
          continue  # e.g. a dataset that has no `tools` column
        raise KeyError(f"Required column '{col}' missing from record. Present columns: {sorted(element.keys())}")
      if col in self.scalar_bool_columns:
        value = element[col]
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 1:
          raise ValueError(f"Boolean metadata column '{col}' must contain exactly one scalar 0/1 value.")
        scalar = value[0]
        if not isinstance(scalar, (int, np.integer, bool, np.bool_)) or int(scalar) not in (0, 1):
          raise ValueError(f"Boolean metadata column '{col}' must contain exactly one scalar 0/1 value.")
        out[col] = bool(int(scalar))
      else:
        out[col] = element[col][0].decode() if self.tokenize else element[col]
    if "dataset_id" in element:
      out["dataset_id"] = element["dataset_id"]
    return out


@dataclasses.dataclass
class KeepFeatures(grain.MapTransform):
  """Filter dataset elements to specified features for parquet and other non-proto formats.

  Retains only the keys present in `feature_names`. Validates the stored value
  type against `tokenize`: raises ValueError if `tokenize=True` but a column
  contains integer data (pre-tokenized), or if `tokenize=False` but a column
  contains string/bytes data (raw text).
  """

  def __init__(self, feature_names: list[str], tokenize: bool = True, scalar_bool_columns=()):
    self.feature_names = feature_names
    self.tokenize = tokenize
    self.scalar_bool_columns = frozenset(scalar_bool_columns)

  # See ParseFeatures.OPTIONAL_COLUMNS — absent optional columns are skipped, not an error.
  OPTIONAL_COLUMNS = frozenset({"tools"})

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    """Applies the feature filtering to the input element."""
    missing = [n for n in self.feature_names if n not in element and n not in self.OPTIONAL_COLUMNS]
    if missing:
      raise ValueError(
          f"Column {missing} not found in dataset. Available columns: {sorted(element.keys())}. "
          "Please set train_data_columns or eval_data_columns accordingly."
      )
    filtered = {k: v for k, v in element.items() if k in self.feature_names}
    for col, val in filtered.items():
      if col in self.scalar_bool_columns:
        if type(val) is bool:  # pylint: disable=unidiomatic-typecheck
          continue
        if isinstance(val, np.bool_):
          filtered[col] = bool(val)
          continue
        raise ValueError(f"Boolean metadata column '{col}' must contain an actual boolean, got {type(val).__name__}.")
      if self.tokenize:
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.integer):
          raise ValueError(
              f"tokenize_data=True but column '{col}' contains integer (pre-tokenized) data. "
              "Set tokenize_train_data or tokenize_eval_data to False if your dataset is already tokenized."
          )
        if isinstance(val, (list, tuple)) and val and isinstance(val[0], (int, np.integer)):
          raise ValueError(
              f"tokenize_data=True but column '{col}' contains integer (pre-tokenized) data. "
              "Set tokenize_train_data or tokenize_eval_data to False if your dataset is already tokenized."
          )
      else:
        if isinstance(val, (str, bytes)):
          raise ValueError(
              f"tokenize_data=False but column '{col}' contains text data. "
              "Set tokenize_train_data or tokenize_eval_data to True if your dataset needs tokenization."
          )
    return filtered


@dataclasses.dataclass
class Rekey(grain.MapTransform):
  """Rename keys according to a mapping dict"""

  def __init__(self, mapping_dict, keep_old_keys=False):
    self.mapping_dict = mapping_dict
    self.keep_old_keys = keep_old_keys

  def map(self, element):
    old_keys = set()
    for new_key, old_key in self.mapping_dict.items():
      element[new_key] = element[old_key]
      old_keys.add(old_key)
    if not self.keep_old_keys:
      for key in old_keys:
        del element[key]
    return element


class DropKeys(grain.MapTransform):
  """Remove the given keys from each element (e.g. packer-emitted junk columns)."""

  def __init__(self, keys):
    self.keys = tuple(keys)

  def map(self, element):
    for k in self.keys:
      element.pop(k, None)
    return element


@dataclasses.dataclass
class ReformatPacking(grain.MapTransform):
  """Reformat packing outputs."""

  def __init__(self, column_names):
    self.column_names = column_names

  def map(self, element):
    ret = {}
    for col in self.column_names:
      ret[f"{col}"] = element[0][col]
      ret[f"{col}_segmentation"] = element[1][col]
      ret[f"{col}_position"] = element[2][col]
    return ret


@dataclasses.dataclass
class PadOrTrimToMaxLength(grain.MapTransform):
  """Pads or trims each input to the specified length.
  And optionally add true length for the input."""

  def __init__(
      self,
      max_length: int,
      pad_id: int = 0,
      config=None,
      add_true_length: bool = False,
      max_num_images_per_example: int = -1,
  ):
    self.max_length = max_length
    self.pad_id = pad_id
    self.config = config
    self.add_true_length = add_true_length
    self.max_num_images_per_example = max_num_images_per_example

  def _pad_text(self, x: np.ndarray, max_length: int, pad_id: int) -> np.ndarray:
    pad_amount = max(max_length - x.shape[0], 0)
    pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
    return np.pad(x, pad_amount, constant_values=pad_id)[: self.max_length]

  def _pad_image_and_mask(self, preprocessed_image: mm_utils.PreprocessorOutput) -> mm_utils.PreprocessorOutput:
    """Pads the input tensors (image and mask) of a PreprocessorOutput to a maximum number of items.

    This function unifies padding logic for image tensors (standard or tiled) and
    mask tensors. It determines the tensor type based on its dimensions and applies
    the appropriate padding along the first axis.

    The maximum number of items is calculated based on model constraints or a
    user-defined limit, ensuring that sequence length limits are respected while
    reserving space for at least one text token. If the input tensor has fewer
    items than this maximum, it is padded with zeros.

    Args:
        preprocessed_image (multimodal_utils.PreprocessorOutput): The input numpy arrays to pad.
            - For masks, the expected shape is (num_masks, num_tiles).
            - For standard images, the shape is (num_images, H, W, C).
            - For tiled images, the shape is (num_images, num_tiles, H, W, C).

    Returns:
        np.ndarray: The tensor, padded with zeros up to the maximum number of
        items along the first axis.

    Raises:
        ValueError: If the input tensor's dimension is not 2, 4, or 5.
        ValueError: If the number of items in the input tensor exceeds the
        allowed maximum.

    Notes:
      - The computation of maximum images ensures that space is reserved in the sequence
        for at least one text token.
      - The dummy images used for padding are based on the image shape for initialization
        of this model (ignoring batch size).
    """
    if not isinstance(preprocessed_image, mm_utils.PreprocessorOutput):
      raise TypeError(f"Input must be multimodal_utils.PreprocessorOutput, but got {type(preprocessed_image)}")

    if preprocessed_image.pixel_values is None:
      raise ValueError("Input preprocessed_image must have pixel_values to pad images.")

    if self.config.model_name and self.config.model_name.startswith("qwen3-omni"):  # pyrefly: ignore[missing-attribute]
      return preprocessed_image

    # Determine the maximum number of images/masks allowed.
    image_offsets = mm_processor.get_image_offsets(self.config, preprocessed_image)
    single_image_offset = image_offsets // preprocessed_image.pixel_values.shape[0]

    # Reserve space for at least one text token.
    max_num_items = (self.max_length - 1) // single_image_offset
    if self.max_num_images_per_example > 0:
      max_num_items = min(self.max_num_images_per_example, max_num_items)

    image_tensor = preprocessed_image.pixel_values
    mask_tensor = preprocessed_image.pixel_mask

    def _pad(tensor: np.ndarray) -> np.ndarray:
      # Validate tensor dimensions.
      if tensor.ndim in (4, 5):  # Standard or Tiled Image
        tensor_type = "images"
      elif tensor.ndim == 2:  # Mask
        tensor_type = "masks"
      else:
        raise ValueError(
            "Input tensor must be 2D (mask), 4D (image), or 5D (tiled image), " f"but got {tensor.ndim} dimensions."
        )

      # Assert that the input tensor does not exceed the maximum size.
      if tensor.shape[0] > max_num_items:
        raise ValueError(f"Number of {tensor_type} ({tensor.shape[0]}) exceeds the maximum allowed ({max_num_items}).")

      # Apply padding if the tensor is smaller than the maximum size.
      if tensor.shape[0] < max_num_items:
        pad_size = max_num_items - tensor.shape[0]
        pad_shape_suffix = tensor.shape[1:]
        pad_shape = (pad_size,) + pad_shape_suffix
        pad_tensor = np.zeros(pad_shape, dtype=tensor.dtype)

        if tensor.size > 0:
          tensor = np.concatenate([tensor, pad_tensor], axis=0)
        else:
          # If the input tensor is empty, the result is just the padding.
          tensor = pad_tensor

      return tensor

    preprocessed_image.pixel_values = _pad(image_tensor)

    if mask_tensor is not None:
      preprocessed_image.pixel_mask = _pad(mask_tensor)

    return preprocessed_image

  def map(
      self, element: dict[str, np.ndarray | mm_utils.PreprocessorOutput]
  ) -> dict[str, np.ndarray | mm_utils.PreprocessorOutput]:
    """map to each element"""
    data_columns = list(element.keys())
    for data_column in data_columns:
      if data_column != "images":
        if isinstance(element[data_column], mm_utils.PreprocessorOutput):
          raise TypeError("Only 'images' column can be of type PreprocessorOutput.")

        element[f"{data_column}_segmentation"] = (
            element[data_column] != self.pad_id  # pyrefly: ignore[unsupported-operation]
        )  # pyrefly: ignore[unsupported-operation]
        # pyrefly: ignore[missing-attribute]
        element[f"{data_column}_segmentation"] = element[
            f"{data_column}_segmentation"
        ].astype(  # pyrefly: ignore[missing-attribute]
            np.int32
        )
        element[f"{data_column}_position"] = np.arange(
            element[data_column].shape[0], dtype=np.int32  # pyrefly: ignore[missing-attribute]
        )  # pyrefly: ignore[missing-attribute]
        if self.add_true_length:
          element[f"{data_column}_true_length"] = np.array(
              [element[data_column].shape[0]], dtype=np.int32  # pyrefly: ignore[missing-attribute]
          )  # pyrefly: ignore[missing-attribute]

    for key, _ in element.items():
      if key == "images":
        if self.config.model_name is None:  # pyrefly: ignore[missing-attribute]
          raise ValueError("model_name must be provided when padding images")

        element["images"] = self._pad_image_and_mask(element["images"])  # pyrefly: ignore[bad-argument-type]

      elif "true_length" not in key:
        element[key] = self._pad_text(element[key], self.max_length, self.pad_id)  # pyrefly: ignore[bad-argument-type]
    return element


@dataclasses.dataclass
class ExtractImagesAndMasks(grain.MapTransform):
  """Extracts images and masks from a PreprocessorOutput object.

  This transform is used in multi-modal data pipelines to extract the image
  tensors and their corresponding masks from a PreprocessorOutput object.
  The extracted images and masks are then added to the data element under
  the keys 'images' and 'image_masks', respectively.

  If the 'images' key is not present in the input element, the transform
  returns the element unchanged.
  """

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Applies the extraction transformation to the 'images' field if present."""
    preprocessed_image = element.get("images")
    if preprocessed_image is None:
      return element

    if not isinstance(preprocessed_image, mm_utils.PreprocessorOutput):
      raise TypeError(f"'images' must be of type PreprocessorOutput, but got {type(preprocessed_image)}")

    output = element.copy()
    output["images"] = preprocessed_image.pixel_values  # pyrefly: ignore[unsupported-operation]
    if preprocessed_image.pixel_mask is not None:
      output["image_masks"] = preprocessed_image.pixel_mask
    # Qwen MRoPE needs per-image (t, h, w) grids to build 3D positions.
    pixel_grid_thw = getattr(preprocessed_image, "pixel_grid_thw", None)
    if pixel_grid_thw is not None:
      output["image_grid_thw"] = pixel_grid_thw

    return output


@dataclasses.dataclass
class FoldImagesIntoBatch(grain.MapTransform):
  """Folds the 'image' dimension into the batch dimension.

  This transform is used in multi-modal data pipelines where each data example
  might have multiple associated images. For model processing, it's often
  efficient to treat each image as a separate item in a larger batch.

  This operation reshapes the 'images' tensor from a shape like
  (B, N, T, H, W, C) to (B * N, T, H, W, C), where B is the batch size, N is
  the number of images per example, and T is the number of image tiles.

  The transformation is triggered only if the input 'images' tensor has more
  dimensions than the expected batched image tensor.
  """

  model_name: str | None = None

  def __post_init__(self):
    """Initializes the target shape after the dataclass is created."""
    self.target_shape = mm_processor.get_dummy_image_shape_for_init(self.model_name)

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Applies the folding transformation to the 'images' field if present."""
    images = element.get("images")
    if images is None:
      return element

    # If ndim is greater than the expected ndim for a batched image tensor,
    # it implies an extra dimension (e.g., number of images per example)
    # that needs to be folded into the batch dimension.
    if images.ndim > len(self.target_shape):
      # Compute the new shape by merging the batch and image count dimensions.
      trailing_dims = self.target_shape[1:]

      # Reshape merges the leading dimensions (B, N) into one (-1) and
      # appends the correct trailing dimensions.
      element["images"] = images.reshape(-1, *trailing_dims)

    return element


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
  return padded[tuple(slices)]


def shift_left(x, pad_id, axis=1):
  """Shift to the left and pad."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, 1)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(1, None)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(pad_id))
  return padded[tuple(slices)]


def shift_and_refine(x, ignored_ids, axis=1):
  """Shift inputs, set segmentation to 0 when target element is in ignored_ids if provided"""
  x["targets"] = shift_left(x["targets"], ignored_ids[0], axis=axis)
  x["targets_segmentation"] = shift_left(x["targets_segmentation"], 0, axis=axis)
  if "dataset_id" in x:
    x["dataset_id"] = shift_left(x["dataset_id"], 0, axis=axis)
  for ignore_id in ignored_ids:
    x["targets_segmentation"] = np.where(x["targets"] != ignore_id, x["targets_segmentation"], 0)

  return x


@dataclasses.dataclass
class ShiftData(grain.MapTransform):
  """Shift inputs and refine annotations."""

  def __init__(self, ignored_ids, axis=1):
    self.ignored_ids = ignored_ids
    self.axis = axis

  def map(self, element):
    return shift_and_refine(element, ignored_ids=self.ignored_ids, axis=self.axis)


@dataclasses.dataclass
class ComputeQwen3OmniPositions(grain.MapTransform):
  """Computes 3D position IDs for Qwen3-Omni multimodal sequences.

  This transform replaces the standard 1D sequential positions with 3D
  positions (temporal, height, width) for multimodal models like Qwen3-Omni.

  For text-only sequences, all 3 dimensions receive the same sequential values.
  For multimodal sequences with vision/audio, vision tokens get true 3D positions
  and text tokens continue sequentially from max(vision_pos) + 1.

  The actual position computation is delegated to multimodal_utils.get_rope_index(),
  which can be tested and modified independently.
  """

  def __init__(
      self,
      data_column: str = "inputs",
      spatial_merge_size: int = 2,
      position_id_per_seconds: int = 25,
      use_audio_in_video: bool = False,
      config=None,
      keep_aux_fields: bool = False,
  ):
    """Initialize the Qwen3-Omni position computation transform.

    Args:
      data_column: Name of the data column to compute positions for (default: "inputs").
      spatial_merge_size: Number of patches merged spatially (e.g., 2 for 2x2→1).
      position_id_per_seconds: Temporal granularity (tokens per second, typically 25).
      use_audio_in_video: If True, audio tokens are interleaved with video tokens.
      config: Optional model config for model-specific special token IDs.
      keep_aux_fields: If True, keep auxiliary (aux) fields — mrope deltas / grid
        metadata — on the element for inference. Training should leave this False
        so the batch matches get_shaped_batch.
    """
    self.data_column = data_column
    self.spatial_merge_size = spatial_merge_size
    self.position_id_per_seconds = position_id_per_seconds
    self.use_audio_in_video = use_audio_in_video
    self.config = config
    self.keep_aux_fields = keep_aux_fields

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute 3D position IDs for the batch element.

    Args:
      element: Dictionary containing:
        - {data_column}: Token IDs with shape (batch, seq_len)
        - {data_column}_segmentation: Attention mask (1=real, 0=padding)
        - image_grid_thw: Optional (num_images, 3) or (batch, num_images, 3) array
        - video_grid_thw: Optional (num_videos, 3) array
        - audio_lengths: Optional (num_audios,) array
        - second_per_grids: Optional (num_videos,) array

    Returns:
      element with {data_column}_position updated to shape (batch, seq_len, 3)
      for 3D positions (always 3D, even for text-only sequences).
    """

    # Extract inputs and metadata
    input_ids = element[self.data_column]
    attention_mask = element.get(f"{self.data_column}_segmentation")

    # Extract multimodal metadata (if present)
    image_grid_thw = element.get("image_grid_thw")
    video_grid_thw = element.get("video_grid_thw")
    audio_lengths = element.get("audio_lengths")
    second_per_grids = element.get("second_per_grids")

    # grain.Batch stacks per-example (N, 3) grids to (B, N, 3). get_rope_index
    # resets image_idx per sequence against a shared (N, 3) table, which is
    # correct when training force-resizes all images to the same grid.
    if image_grid_thw is not None and image_grid_thw.ndim == 3:
      image_grid_thw = image_grid_thw[0]
    if video_grid_thw is not None and video_grid_thw.ndim == 3:
      video_grid_thw = video_grid_thw[0]

    # Call the standalone get_rope_index function from multimodal_utils
    from maxtext.multimodal import processor_qwen3_omni  # pylint: disable=import-outside-toplevel

    # TODO(jfacevedo/hengtaoguo): Now get_rope_index is Qwen3-Omni specific. We should generalize it for other models
    position_ids, mrope_position_deltas = processor_qwen3_omni.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        use_audio_in_video=self.use_audio_in_video,
        audio_lengths=audio_lengths,
        second_per_grids=second_per_grids,
        spatial_merge_size=self.spatial_merge_size,
        position_id_per_seconds=self.position_id_per_seconds,
        config=self.config,
    )

    # Update element with 3D positions
    # Shape: (batch, seq_len, 3) for multimodal, or (batch, seq_len) for text-only
    element[f"{self.data_column}_position"] = position_ids.astype(np.int32)
    if self.keep_aux_fields:
      element[f"{self.data_column}_mrope_deltas"] = mrope_position_deltas
    else:
      # Drop metadata that is not part of the training shaped batch.
      element.pop(f"{self.data_column}_mrope_deltas", None)
      element.pop("image_grid_thw", None)
      element.pop("video_grid_thw", None)
      element.pop("audio_lengths", None)
      element.pop("second_per_grids", None)

    return element
