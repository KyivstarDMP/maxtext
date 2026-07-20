# Copyright 2023–2026 Google LLC
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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

from typing import Any, Sequence
import datetime
import functools
import os

from absl import app

import numpy as np
import optax

import pathwaysutils  # pylint: disable=unused-import

import tensorflow as tf

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from flax import linen as nn, nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.configs import pyconfig
from maxtext.utils.globals import EPS
from maxtext.utils import elastic_utils
# Placeholder: internal

# pylint: disable=too-many-positional-arguments
from maxtext.layers.multi_token_prediction import calculate_mtp_acceptance_rate, calculate_mtp_loss
from maxtext.common import checkpointing, profiler
from maxtext.common.goodput import (
    GoodputEvent,
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
    record_goodput,
)
from maxtext.common.gcloud_stub import vertex_tensorboard_modules
from maxtext.common import metric_logger
from maxtext.common.metric_logger import record_activation_metrics
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import qk_clip_utils
from maxtext.utils import sharding
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import train_utils
from maxtext.utils.gradient_accumulation import gradient_accumulation_loss_and_grad
from maxtext.utils.vocabulary_tiling import vocab_tiling_linen_loss, vocab_tiling_nnx_loss

VertexTensorboardManager, _vertex_tb_is_stub = vertex_tensorboard_modules()


def get_first_step(model, state):
  if isinstance(model, nn.Module):
    return int(state.step)
  return int(state.optimizer.step.get_value())


# Constant tag folded into the per-step rng to derive DITTO's coin/sentence stream
# without consuming the dropout rng stream.
_DITTO_RNG_TAG = 0xD1770


def _unlikelihood_loss_full(logits, data, config):
  """Token-level unlikelihood loss on full (non-tiled) ``[B, S, V]`` logits.

  Flattens the batch-sequence axis and defers to ``max_utils.unlikelihood_loss_from_logits``,
  the same kernel used per tile in the vocab-tiling path, so the two paths agree.
  Candidate scope B: ``targets`` / ``targets_segmentation`` are the candidate context
  (the model's own completion tokens).
  """
  batch_size, seq_len = logits.shape[0], logits.shape[1]
  positions = jnp.arange(batch_size * seq_len, dtype=jnp.int32)
  return max_utils.unlikelihood_loss_from_logits(
      logits.reshape(batch_size * seq_len, config.vocab_size),
      positions,
      data["targets"],
      data["targets_segmentation"],
      data["targets"].reshape(-1),
      (data["targets_segmentation"] != 0).reshape(-1),
      seq_len=seq_len,
      window=config.unlikelihood_window,
      eps=config.unlikelihood_eps,
  )


def _ditto_loss_full(logits, data, config):
  """DITTO decay loss on full (non-tiled) ``[B, S, V]`` logits.

  Flattens the batch-sequence axis and defers to ``max_utils.ditto_loss_from_logits``,
  the same kernel used per tile in the vocab-tiling path, so the two paths agree. The
  baseline ``gold_probs`` is the detached per-token gold probability of the *same* logits.
  Reads the precomputed period maps ``ditto_baseline_pos`` / ``ditto_pen_mask`` from
  ``data`` (built by :func:`build_pseudo_repetition`).
  """
  batch_size, seq_len = logits.shape[0], logits.shape[1]
  flat_logits = logits.reshape(batch_size * seq_len, config.vocab_size)
  gold = data["targets"].reshape(-1)
  gold_probs = jax.lax.stop_gradient(max_utils.gold_prob_from_logits(flat_logits, gold)).reshape(
      batch_size, seq_len
  )
  positions = jnp.arange(batch_size * seq_len, dtype=jnp.int32)
  return max_utils.ditto_loss_from_logits(
      flat_logits,
      positions,
      gold,
      gold_probs,
      data["ditto_baseline_pos"],
      data["ditto_pen_mask"],
      seq_len=seq_len,
      gamma=config.ditto_gamma,
      eps=config.ditto_eps,
      loss_type=config.ditto_loss_type,
  )


def build_pseudo_repetition(inputs, targets, targets_segmentation, delim_ids, rng, max_reps=0):
  """Build a synthetic pseudo-repetition batch for a DITTO step (Xu et al., 2022).

  Reproduces the paper's ``re_orgnize_sentence`` for MaxText's completion-only SFT layout
  (``targets[i] = inputs[i+1]`` after ``shift_and_refine``, ``targets_segmentation != 0``
  on the assistant completion). Per row it makes the assistant **start repeating one of
  its own sentences partway through the answer**, keeping the prompt and earlier answer as
  context. See ``docs/008``.

  Args:
    inputs: ``[B, S]`` int decoder input tokens.
    targets: ``[B, S]`` int next-token targets (``= inputs`` shifted left).
    targets_segmentation: ``[B, S]`` int; non-zero marks the assistant completion.
    delim_ids: a (static) tuple of token ids that mark sentence boundaries.
    rng: a PRNGKey used to pick the repeated sentence per row.
    max_reps: cap on the number of repetitions of the chosen sentence (``<= 0`` = fill the
      whole completion run). Capping bounds the geometric decay depth ``gamma^n`` so it cannot
      drive deep-repetition probabilities toward 0 — uncapped + a long run is what previously
      taught the model to stop immediately (empty output). Recommended ~5.

  Returns:
    ``(new_inputs, new_targets, baseline_pos, pen_mask)``. ``baseline_pos[b, i] = i -
    period`` (clamped) and ``pen_mask[b, i]`` marks the 2nd-or-later repetition (the
    positions the decay loss penalizes). Rows with < 3 completion sentences are left
    unchanged with an all-zero ``pen_mask`` (a jit-friendly per-row relaxation of the
    original's whole-batch skip).
  """
  b_dim, s_dim = inputs.shape
  idx = jnp.arange(s_dim, dtype=jnp.int32)[None, :]  # [1, S]
  if not delim_ids:
    # No sentence delimiters configured -> DITTO is a no-op (empty pen_mask).
    zeros = jnp.zeros((b_dim, s_dim), dtype=jnp.int32)
    return inputs, targets, jnp.broadcast_to(idx, (b_dim, s_dim)).astype(jnp.int32), zeros

  comp_lab = targets_segmentation != 0  # [B, S] label-space completion
  # inputs[j] is a completion token  <=>  targets_segmentation[j-1] != 0  (shift_right).
  cmask_in = jnp.concatenate([jnp.zeros((b_dim, 1), dtype=bool), comp_lab[:, :-1]], axis=1)

  delim_arr = jnp.asarray(delim_ids, dtype=inputs.dtype)
  is_delim = jnp.isin(inputs, delim_arr) & cmask_in  # [B, S]
  cum = jnp.cumsum(is_delim.astype(jnp.int32), axis=1)  # [B, S]
  num_delims = cum[:, -1]  # [B]
  valid = num_delims >= 3  # need >= 3 delimiters to define a prefix + a repeated sentence

  # Pick a 0-indexed delimiter rank r in [1, num_delims - 2] per row.
  u = jax.random.uniform(rng, (b_dim,))
  r = (1 + jnp.floor(u * jnp.maximum(num_delims - 2, 1)).astype(jnp.int32)).astype(jnp.int32)

  def pos_of_rank(k):  # position of the k-th (0-indexed) delimiter = first j with cum[j] >= k+1
    return jnp.argmax(cum >= (k + 1)[:, None], axis=1).astype(jnp.int32)

  s_start = pos_of_rank(r)  # e_r
  e_next = pos_of_rank(r + 1)  # e_{r+1}
  period = jnp.maximum(e_next - s_start, 1)  # [B]

  # run_end: first completion-input position after s_start that ends the run, else S.
  after = (~cmask_in) & (idx > s_start[:, None])  # [B, S]
  run_end = jnp.where(jnp.any(after, axis=1), jnp.argmax(after, axis=1).astype(jnp.int32), s_dim)  # [B]

  # Cap the repetition depth: fill at most `max_reps` copies of the sentence, then let the
  # original completion resume. This bounds the geometric decay `gamma^n` (a long uncapped run
  # drives deep-repetition probabilities to ~0 -> the model learns to stop immediately).
  fill_end = run_end if max_reps <= 0 else jnp.minimum(run_end, s_start + period * max_reps)  # [B]

  # Overwrite inputs on [s_start, fill_end) with the period-`period` repeat of the unit.
  in_region = (idx >= s_start[:, None]) & (idx < fill_end[:, None]) & valid[:, None]  # [B, S]
  src = s_start[:, None] + ((idx - s_start[:, None]) % period[:, None])
  src = jnp.clip(src, 0, s_dim - 1)
  repeated = jnp.take_along_axis(inputs, src, axis=1)
  new_inputs = jnp.where(in_region, repeated, inputs)

  # targets[i] = new_inputs[i+1] (shift_left); only changed where i+1 is in the region.
  shifted = jnp.concatenate([new_inputs[:, 1:], inputs[:, -1:]], axis=1)
  tgt_changed = jnp.concatenate([in_region[:, 1:], jnp.zeros((b_dim, 1), dtype=bool)], axis=1)
  new_targets = jnp.where(tgt_changed, shifted, targets)

  # Penalize label positions in the 2nd-or-later repetition: targets[i] is repeated for
  # i >= s_start - 1 (i.e. i+1 >= s_start); the baseline at i-period exists in the 1st
  # repetition for i >= s_start + period - 1; and targets[i] stays in-region for i < fill_end - 1.
  pen_start = s_start + period - 1
  pen_mask = (
      (idx >= pen_start[:, None]) & (idx < (fill_end - 1)[:, None]) & valid[:, None] & comp_lab
  ).astype(jnp.int32)
  baseline_pos = jnp.clip(idx - period[:, None], 0, s_dim - 1).astype(jnp.int32)
  baseline_pos = jnp.broadcast_to(baseline_pos, (b_dim, s_dim))
  return new_inputs, new_targets, baseline_pos, pen_mask


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------


def _num_datasets_plus1(config):
  """Static [num_datasets + 1]: slot 0 = pad/unknown, 1..num_datasets = mixture components."""
  return len([n for n in config.per_dataset_names.split(",") if n]) + 1


def _per_dataset_from_logits(logits, masked_xent, data, config):
  """Non-tiled per-dataset (xent_sum, correct_count) vectors of shape [num_datasets+1].

  `masked_xent` is the per-token cross-entropy already multiplied by the completion mask, so a
  segment-sum by `dataset_id` gives each component's summed loss. `correct` is next-token accuracy
  over the same mask. Both emit an SPMD all-reduce over the DP-sharded batch inside pjit.
  """
  num_seg = _num_datasets_plus1(config)
  ids = data["dataset_id"].reshape(-1)
  mask = data["targets_segmentation"] != 0
  xent_sum_by_ds = jax.ops.segment_sum(masked_xent.reshape(-1), ids, num_segments=num_seg)
  correct = (jnp.argmax(logits, axis=-1) == data["targets"]) & mask
  correct_by_ds = jax.ops.segment_sum(correct.reshape(-1).astype(jnp.int32), ids, num_segments=num_seg)
  return xent_sum_by_ds, correct_by_ds


def _assemble_per_dataset_aux(config, data, xent_sum_by_ds, correct_by_ds, use_ditto, ditto_step):
  """Per-dataset aux dict (xent_sum / correct / token_count, each [num_datasets+1]) or None.

  Only for the train path (batches carry `dataset_id`); eval Option B runs per-dataset passes and
  has no `dataset_id`, so this returns None there. DITTO steps train on synthetic data, so their
  per-dataset train metrics are zeroed out.
  """
  if not (config.per_dataset_metrics and "dataset_id" in data):
    return None
  num_seg = _num_datasets_plus1(config)
  ids = data["dataset_id"].reshape(-1)
  token_count_by_ds = jax.ops.segment_sum(
      (data["targets_segmentation"] != 0).reshape(-1).astype(jnp.int32), ids, num_segments=num_seg
  )
  if xent_sum_by_ds is None:  # tiled loss path not yet wired for per-dataset (Phase 2) — emit zeros
    xent_sum_by_ds = jnp.zeros(num_seg, jnp.float32)
    correct_by_ds = jnp.zeros(num_seg, jnp.int32)
  if use_ditto:
    xent_sum_by_ds = jnp.where(ditto_step, jnp.zeros(num_seg, jnp.float32), xent_sum_by_ds)
    correct_by_ds = jnp.where(ditto_step, jnp.zeros(num_seg, jnp.int32), correct_by_ds)
    token_count_by_ds = jnp.where(ditto_step, jnp.zeros(num_seg, jnp.int32), token_count_by_ds)
  return {
      "xent_sum_by_ds": xent_sum_by_ds,
      "correct_by_ds": correct_by_ds,
      "token_count_by_ds": token_count_by_ds,
  }


def loss_fn(model, config, data, dropout_rng, params, sparsity_state=None, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A nn.Module (Linen) or nnx.Module (NNX).
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout (Linen); unused for NNX.
    params: Model params (Linen); unused for NNX (params are part of the model).
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, xent_sum, and total_weights
  """
  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]
  else:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_eval_on, :]
  # Anti-repetition regularizers. Unlikelihood adds `alpha * L_UL` to the NLL on the real
  # batch. DITTO (paper-faithful) instead alternates: with probability
  # `ditto_sequence_level_train_rate` the step trains on a synthetic pseudo-repetition
  # batch with a pure DITTO decay loss; otherwise a normal MLE step.
  use_unlikelihood = is_train and config.unlikelihood_alpha > 0
  use_ditto = is_train and config.ditto_alpha > 0
  ul_sum = 0.0
  ditto_sum = 0.0
  ditto_step = jnp.array(False)
  # Per-dataset (per mixture component) accumulators; filled in the non-tiled loss branch (train only).
  xent_sum_by_ds = None
  correct_by_ds = None
  if use_ditto:
    # Derive an independent per-step rng (fold_in does not consume the dropout stream).
    base_rng = dropout_rng if dropout_rng is not None else jax.random.PRNGKey(0)
    ditto_rng = jax.random.fold_in(base_rng, _DITTO_RNG_TAG)
    coin_rng, sentence_rng = jax.random.split(ditto_rng)
    ditto_step = jax.random.uniform(coin_rng, ()) < config.ditto_sequence_level_train_rate
    pr_inputs, pr_targets, ditto_baseline_pos, ditto_pen_mask = build_pseudo_repetition(
        data["inputs"],
        data["targets"],
        data["targets_segmentation"],
        tuple(config.ditto_sentence_delim_ids),
        sentence_rng,
        max_reps=config.ditto_max_reps,
    )
    # On a DITTO step swap in the synthetic sequence; otherwise leave the batch untouched
    # and zero the penalty mask so the DITTO term vanishes (the loss select below also
    # discards it, but this avoids any wasted/garbage penalty on MLE steps).
    data["inputs"] = jnp.where(ditto_step, pr_inputs, data["inputs"])
    data["targets"] = jnp.where(ditto_step, pr_targets, data["targets"])
    data["ditto_baseline_pos"] = ditto_baseline_pos
    data["ditto_pen_mask"] = jnp.where(ditto_step, ditto_pen_mask, jnp.zeros_like(ditto_pen_mask))
  mutable_collections = ["intermediates"]
  if config.mtp_num_layers > 0 and is_train:
    # The single model.apply call now triggers the entire chain if MTP is enabled:
    # Decoder runs -> returns hidden_state -> MTPBlock uses it -> MTPBlock sows losses -> we reap them here.
    mutable_collections.append("mtp_losses")

  # During evaluation, if the acceptance rate test is enabled, we must
  # make its specific collection mutable so the MTPBlock can sow into it.
  if config.mtp_eval_target_module > 0 and not is_train:
    mutable_collections.append("mtp_acceptance")
  sparsity_enabled = is_train and config.weight_sparsity_n and config.weight_sparsity_m
  if sparsity_enabled:
    mutable_collections.append("batch_stats")
  if isinstance(model, nn.Module):
    # inputs, targets, segments, positions = apply_args
    if dropout_rng is not None:
      rng1, aqt_rng = jax.random.split(dropout_rng)
    else:
      rng1, aqt_rng = None, None

    # Flax Linen model
    if sparsity_enabled:
      model_vars = {"params": params}
      if sparsity_state:
        model_vars["batch_stats"] = sparsity_state
    else:
      model_vars = params
    logits, intermediate_outputs = model.apply(
        model_vars,
        data["inputs"],
        data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        encoder_images=data["images"] if config.use_multimodal else None,
        encoder_image_masks=data["image_masks"] if config.use_multimodal and "image_masks" in data else None,
        enable_dropout=config.enable_dropout if is_train else False,
        rngs={"dropout": rng1, "params": aqt_rng},
        mutable=mutable_collections,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )

    if (config.use_indexer and not config.indexer_sparse_training) and is_train:
      # In Dense Warm-up stage, we skip main model loss calculation for efficiency.
      # The main model parameters are frozen and only the indexer is trained via KL divergence.
      xent_sum = 0.0
      total_z_loss = 0.0
    elif config.num_vocab_tiling > 1:
      hidden_state_key = ("intermediates", "decoder", "hidden_states")
      hidden_states = maxtext_utils.get_nested_value(intermediate_outputs, hidden_state_key)[0]
      if config.per_dataset_metrics and "dataset_id" in data:
        xent_sum, total_z_loss, ul_sum, ditto_sum, xent_sum_by_ds, correct_by_ds = vocab_tiling_linen_loss(
            hidden_states, data, config, model, params, is_train
        )
      else:
        xent_sum, total_z_loss, ul_sum, ditto_sum = vocab_tiling_linen_loss(
            hidden_states, data, config, model, params, is_train
        )
    else:
      one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
      xent, z_loss = max_utils.cross_entropy_with_logits(logits, one_hot_targets, z_loss=config.z_loss_multiplier)

      xent = sharding.maybe_shard_with_logical(
          xent,
          ("activation_embed_and_logits_batch", "activation_length"),
          model.mesh,
          config.shard_mode,
          debug_sharding=config.debug_sharding,
      )
      z_loss = sharding.maybe_shard_with_logical(
          z_loss,
          ("activation_embed_and_logits_batch", "activation_length"),
          model.mesh,
          config.shard_mode,
          debug_sharding=config.debug_sharding,
      )

      # Mask out paddings at the end of each example.
      xent = xent * (data["targets_segmentation"] != 0)
      z_loss = z_loss * (data["targets_segmentation"] != 0)

      xent_sum = jnp.sum(xent)
      total_z_loss = jnp.sum(z_loss)
      if config.per_dataset_metrics and "dataset_id" in data:
        xent_sum_by_ds, correct_by_ds = _per_dataset_from_logits(logits, xent, data, config)
      if use_unlikelihood:
        ul_sum = _unlikelihood_loss_full(logits, data, config)
      if use_ditto:
        ditto_sum = _ditto_loss_full(logits, data, config)
  else:
    # Flax NNX model: forward pass, then pop Intermediates sown during it.
    logits = model(
        decoder_input_tokens=data["inputs"],
        decoder_positions=data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        encoder_images=data["images"] if config.use_multimodal else None,
        encoder_image_masks=data["image_masks"] if config.use_multimodal and "image_masks" in data else None,
        enable_dropout=config.enable_dropout if is_train else False,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )
    intermediates = nnx.pop(model, nnx.Intermediate)
    intermediate_outputs = intermediates.to_pure_dict()

    if config.num_vocab_tiling > 1:
      hidden_state_key = ("decoder", "hidden_states")
      hidden_states = maxtext_utils.get_nested_value(intermediate_outputs, hidden_state_key)[0]
      xent_sum, total_z_loss, ul_sum, ditto_sum = vocab_tiling_nnx_loss(
          model, hidden_states, data, config, is_train
      )
    elif (config.use_indexer and not config.indexer_sparse_training) and is_train:
      # In Dense Warm-up stage, we skip main model loss calculation for efficiency.
      # The main model parameters are frozen and only the indexer is trained via KL divergence.
      xent_sum = 0.0
      total_z_loss = 0.0
    else:
      one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
      xent, z_loss = max_utils.cross_entropy_with_logits(logits, one_hot_targets, z_loss=config.z_loss_multiplier)

      xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
      z_loss = nn.with_logical_constraint(z_loss, ("activation_embed_and_logits_batch", "activation_length"))

      # Mask out paddings at the end of each example.
      xent = xent * (data["targets_segmentation"] != 0)
      z_loss = z_loss * (data["targets_segmentation"] != 0)

      xent_sum = jnp.sum(xent)
      total_z_loss = jnp.sum(z_loss)
      if config.per_dataset_metrics and "dataset_id" in data:
        xent_sum_by_ds, correct_by_ds = _per_dataset_from_logits(logits, xent, data, config)
      if use_unlikelihood:
        ul_sum = _unlikelihood_loss_full(logits, data, config)
      if use_ditto:
        ditto_sum = _ditto_loss_full(logits, data, config)

  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  per_dataset_aux = _assemble_per_dataset_aux(config, data, xent_sum_by_ds, correct_by_ds, use_ditto, ditto_step)
  # MLE-step loss: NLL plus the optional alpha-weighted unlikelihood term. On a DITTO step
  # we replace it with the pure DITTO decay loss (paper-faithful alternation): the data is
  # already the synthetic pseudo-repetition batch, so the NLL there is meaningless and the
  # `jnp.where` discards it (and its gradient). Each term is a distinct differentiable
  # output of the vocab-tiling custom_vjp, so the chain rule scales the gradient correctly.
  combined_sum = xent_sum + config.unlikelihood_alpha * ul_sum
  if use_ditto:
    combined_sum = jnp.where(ditto_step, config.ditto_alpha * ditto_sum, combined_sum)
  # If gradient accumulation is enabled, we don't need to divide xent_sum
  # by total_weights and then multiply the computed gradient by total_weights,
  # since it's equivalent to computing the gradient from xent_sum.
  # This simplification reduces the number of operations and makes it easier
  # for XLA to move all-reduce out of the gradient accumulation loop when use
  # Zero1+GA to reduce communication overhead.
  # EPS was used to avoid division by zero, but it's not needed when gradient
  # accumulation is enabled since there's no division.
  if config.gradient_accumulation_steps > 1 and not config.use_tunix_gradient_accumulation:
    loss = combined_sum
  else:
    # When using Tunix gradient accumulation, we revert to standard normalization.
    # Unlike the manual accumulation path above, Tunix (via optax.MultiSteps) expects
    # a normalized loss for each step. It handles the accumulation state
    # updates and scaling internally.
    loss = combined_sum / (total_weights + EPS)

  # We keep z-loss and the (reported) unlikelihood / DITTO losses normalized by total_weights.
  total_z_loss = total_z_loss / (total_weights + EPS)
  ul_loss = ul_sum / (total_weights + EPS)
  ditto_loss = ditto_sum / (total_weights + EPS)

  # Calculate and Add MTP Loss
  mtp_loss = 0.0
  if config.mtp_num_layers > 0 and is_train:
    mtp_loss = calculate_mtp_loss(intermediate_outputs, config)
    loss += mtp_loss

  # get indexer loss
  indexer_loss = 0.0
  if config.use_indexer and config.indexer_loss_scaling_factor > 0.0:
    indexer_losses = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "self_attention", "indexer_loss")
    if indexer_losses:
      indexer_loss = jnp.mean(jnp.concatenate(indexer_losses))
      loss += indexer_loss
    else:
      max_logging.debug("No indexer loss found.")

  # get MoE load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    moe_lb_losses = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_lb_loss")
    if moe_lb_losses:
      moe_lb_loss = jnp.mean(jnp.concatenate(moe_lb_losses))
      loss += moe_lb_loss
    else:
      max_logging.debug("\nNo MoE load balance loss found. Defaulting to 0.0.")

  # get MoE routed bias term updates
  moe_bias_updates = None
  if config.routed_bias and config.routed_bias_update_rate > 0.0:
    nested_key = ("intermediates", "decoder", "moe_layers", "moe_bias_updates")
    moe_bias_updates = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, None)

  # Add the model's primary output to the intermediates dict so it can be used
  # by the acceptance rate calculation in eval_step.
  intermediate_outputs["logits"] = logits

  aux = {
      "intermediate_outputs": intermediate_outputs,
      "xent_sum": xent_sum,
      "z_loss": total_z_loss,
      "ul_loss": ul_loss,
      "ditto_loss": ditto_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "indexer_loss": indexer_loss,
      "moe_bias_updates": moe_bias_updates,
      "mtp_loss": mtp_loss,
      "batch_stats": (intermediate_outputs.get("batch_stats", None) if hasattr(intermediate_outputs, "get") else None),
  }
  if per_dataset_aux is not None:
    aux["per_dataset"] = per_dataset_aux
  return loss, aux


def train_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng=None):
  """Training step for both Linen and NNX models.

  Args:
    model: A nn.Module (Linen) or nnx.GraphDef of the TrainStateNNX (NNX).
    config: Hyperparameters.
    state_mesh_shardings: PyTree of PartitionSpecs for the train state.
    params_shardings: PyTree of PartitionSpecs for model parameters, used for gradient accumulation.
    state: Linen TrainState or NNX pure State.
    data: Training data batch.
    dropout_rng: A key to use to generate rng for dropout (Linen); unused for NNX.

  Returns:
    new_state: Updated Linen TrainState or NNX pure State.
    metrics: Dictionary of model metrics such as loss, training rate, etc.
  """
  # --- Per-path initialization ---
  if isinstance(model, nn.Module):
    params = state.params
    loss_model, loss_params, loss_rng = model, params, dropout_rng
  else:
    state = nnx.merge(model, state)  # reconstruct TrainStateNNX
    loss_model, loss_params, loss_rng = state.model, None, None
    # The NNX path is not handed a per-step rng through the jit signature (in_shardings has
    # no rng slot), so DITTO's coin/sentence selection would be frozen across steps. Derive
    # a step-varying key from the optimizer step (the same accessor get_first_step uses) so
    # the DITTO alternation works on NNX as it does on Linen. Only when DITTO is enabled.
    if config.ditto_alpha > 0:
      loss_rng = jax.random.fold_in(jax.random.PRNGKey(0), state.optimizer.step.get_value().astype(jnp.uint32))

  # --- Gradient computation ---
  if config.gradient_accumulation_steps > 1:
    loss, aux, raw_grads = gradient_accumulation_loss_and_grad(
        loss_fn,
        config,
        loss_model,
        loss_params,
        params_shardings,
        data,
        loss_rng,
    )
  else:
    if isinstance(model, nn.Module):
      if config.shard_optimizer_over_data:
        params = jax.tree.map(
            functools.partial(sharding.maybe_shard_with_name, shard_mode=config.shard_mode),
            params,
            params_shardings,
        )
      sparsity_enabled = config.weight_sparsity_n and config.weight_sparsity_m
      pure_params = params["params"] if sparsity_enabled else params
      batch_stats = params.get("batch_stats", {})

      grad_func = jax.value_and_grad(loss_fn, argnums=4, has_aux=True)
      (loss, aux), raw_grads = grad_func(
          model,
          config,
          data,
          dropout_rng,
          pure_params,
          sparsity_state=batch_stats,
          is_train=True,
      )
    else:
      model_graphdef, curr_params, rest = nnx.split(state.model, nnx.Param, ...)
      if config.parameter_memory_host_offload:
        # Params are kept on host (pinned_host) in in_shardings. Move only Param
        # variables to device before the forward/backward pass so that all dot_general
        # operands share the same memory space (XLA on GPU requires this).
        # Using params_shardings (Param-only) avoids Shardy rank mismatches that
        # occur when applying PartitionSpec() (rank-0 in SDY) to rank-1 RNG key tensors.
        device_param_shardings = jax.tree_util.tree_map_with_path(
            maxtext_utils_nnx.move_memory_to_device,
            params_shardings,
            is_leaf=lambda x: isinstance(x, NamedSharding),
        )
        curr_params = jax.device_put(curr_params, device_param_shardings)
        nnx.update(state.model, curr_params)  # ensure state.model has device params for optimizer update
      if config.shard_optimizer_over_data:
        curr_params = jax.tree.map(
            functools.partial(sharding.maybe_shard_with_name, shard_mode=config.shard_mode),
            curr_params,
            params_shardings,
        )
        nnx.update(state.model, curr_params)

      def diff_wrapper(param, rest, config, data):
        local_model = nnx.merge(model_graphdef, param, rest, copy=True)
        # Pass the step-derived rng (loss_rng) so DITTO's per-step coin varies on NNX too.
        loss, aux = loss_fn(local_model, config, data, loss_rng, None, is_train=True)
        _, _, new_rest = nnx.split(local_model, nnx.Param, ...)
        return loss, (aux, new_rest)

      grad_func = jax.value_and_grad(diff_wrapper, argnums=0, has_aux=True)
      (loss, (aux, new_rest)), raw_grads = grad_func(curr_params, rest, config, data)
      nnx.update(state.model, new_rest)

  raw_grads = jax.tree_util.tree_map(
      lambda x: x.astype(config.grad_dtype) if x.dtype == jnp.float32 else x,
      raw_grads,
  )
  if config.parameter_memory_host_offload:
    raw_grads = jax.device_put(
        raw_grads,
        max_utils.with_memory_kind(params_shardings, "device"),
    )

  # Extract aux fields into locals
  intermediate_outputs = aux["intermediate_outputs"]
  xent_sum = aux["xent_sum"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  indexer_loss = aux.get("indexer_loss", 0.0)
  z_loss = aux.get("z_loss", 0.0)
  ul_loss = aux.get("ul_loss", 0.0)
  ditto_loss = aux.get("ditto_loss", 0.0)
  moe_bias_updates = aux.get("moe_bias_updates")
  mtp_loss = aux.get("mtp_loss", 0.0)
  per_dataset = aux.get("per_dataset")
  new_opt_state = None

  if isinstance(model, nn.Module):
    if config.gradient_clipping_threshold > 0:
      grads = maxtext_utils.apply_gradient_clipping(raw_grads, state, config.gradient_clipping_threshold)
    else:
      grads = raw_grads
    if config.optimizer_memory_host_offload:
      state = state.replace(
          opt_state=jax.device_put(
              state.opt_state,
              jax.tree_util.tree_map(
                  lambda x: x.with_memory_kind(kind="device"),
                  state_mesh_shardings.opt_state,
              ),
          )
      )
    # Move all parameters to device before optimizer update
    if config.parameter_memory_host_offload:
      max_logging.log("\nMoving all parameters to device before optimizer update")

      def move(path, value):
        max_logging.log(f"train.py: Moving f{path} to device")
        return value.with_memory_kind(kind="device")

      state = state.replace(
          params=jax.device_put(
              state.params,
              jax.tree_util.tree_map_with_path(move, state_mesh_shardings.params),
          )
      )
    # Re-wrap grads to match state.params structure if it's a dict of collections
    # (when weight_sparsity is enabled, params has both 'params' and 'batch_stats' keys).
    sparsity_enabled = config.weight_sparsity_n and config.weight_sparsity_m
    if sparsity_enabled:
      full_grads = {"params": grads}
      if "batch_stats" in state.params:
        batch_stats_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params.get("batch_stats", {}))
        full_grads["batch_stats"] = batch_stats_grads
      full_grads = max_utils.unbox_logicallypartioned(full_grads)
    else:
      full_grads = grads

    if getattr(config, "skip_step_on_spikes", False):
      grad_norm = max_utils.l2norm_pytree(grads)
      # TrainState.apply_gradients doesn't pass **kwargs to tx.update, so we unpack it manually.
      updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params, loss=loss, grad_norm=grad_norm)
      new_params = optax.apply_updates(state.params, updates)

      new_state = state.replace(
          step=state.step + 1,
          params=new_params,
          opt_state=new_opt_state,
      )
    else:
      new_state = state.apply_gradients(grads=full_grads)

    # Apply updates for Auxiliary-Loss-Free load balancing for DeepSeek family
    if config.routed_bias and config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
      target_path = ("params", "decoder", "moe_layers", "DeepSeekMoeBlock_0", "MoeBlock_0", "gate", "bias")
      # Updates the shape to be aligned with state.
      moe_bias_updates = jnp.array(moe_bias_updates[0]).transpose()
      new_state = maxtext_utils.update_state_param(new_state, target_path, moe_bias_updates)
  else:
    if config.gradient_clipping_threshold > 0:
      grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, config.gradient_clipping_threshold)
    else:
      grads = raw_grads
    if config.optimizer_memory_host_offload:
      # state.optimizer is an NNX Optimizer module; state_mesh_shardings.optimizer
      # is an NNX State. Use nnx.state() to get a compatible State for device_put.
      device_opt_shardings = jax.tree_util.tree_map_with_path(
          maxtext_utils_nnx.move_memory_to_device,
          state_mesh_shardings.optimizer,
          is_leaf=lambda x: isinstance(x, NamedSharding),
      )
      opt_state = nnx.state(state.optimizer)
      new_opt_state = jax.device_put(opt_state, device_opt_shardings)
      nnx.update(state.optimizer, new_opt_state)
    state.apply_gradients(grads)
    new_state = state

    # Apply updates for Auxiliary-Loss-Free load balancing for DeepSeek family
    if config.routed_bias and config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
      target_bias = new_state.model.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias
      target_bias.value = target_bias.value + jnp.array(moe_bias_updates[0]).transpose()

  lm_loss = xent_sum / (total_weights + EPS)
  scalar_metrics = {
      "learning/loss": loss,
      "learning/lm_loss": lm_loss,
      "learning/perplexity": jnp.exp(lm_loss),
      "learning/z_loss": z_loss,
      "learning/ul_loss": ul_loss,
      "learning/ditto_loss": ditto_loss,
      "learning/moe_lb_loss": moe_lb_loss,
      "learning/indexer_loss": indexer_loss,
      "learning/mtp_loss": mtp_loss,
      "learning/total_weights": total_weights,
  }
  if config.use_qk_clip:
    if isinstance(model, nn.Module):
      new_state = qk_clip_utils.apply_qk_clip(new_state, intermediate_outputs, config)
    else:
      new_state = qk_clip_utils.apply_qk_clip_nnx(new_state, intermediate_outputs, config)

    global_max_logit = qk_clip_utils.calculate_max_logit_metric(intermediate_outputs)
    if global_max_logit is not None:
      scalar_metrics["learning/max_logits"] = global_max_logit

  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
    if isinstance(model, nn.Module):
      scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(new_state.params)
    else:
      model_params = nnx.state(new_state.model, nnx.Param)
      scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(model_params)

  # Surface skip-step rejections as a TB metric. Linen path only — the NNX
  # branch doesn't apply skip-step, so new_opt_state stays None.
  if config.skip_step_on_spikes:
    is_skipped = new_opt_state.get("is_skipped") if isinstance(new_opt_state, dict) else None
    if is_skipped is not None:
      scalar_metrics["optim/step_skipped"] = is_skipped.astype(jnp.float32)
  metrics = {
      "scalar": scalar_metrics,
      "scalars": {},
  }
  if per_dataset is not None:
    metrics["per_dataset"] = per_dataset
  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

  if isinstance(model, nn.Module):
    return new_state, metrics
  # Drop Intermediates (e.g. sowed max_logits for QK-Clip) before returning;
  # they're absent from state_mesh_shardings and would cause a leaf-count mismatch.
  return nnx.state(new_state, nnx.Not(nnx.Intermediate)), metrics


def eval_step(model, config, state, data, dropout_rng=None):
  """eval_step no backprop and new state compared with train_step."""
  if isinstance(model, nn.Module):
    sparsity_enabled = config.weight_sparsity_n and config.weight_sparsity_m
    pure_params = state.params["params"] if sparsity_enabled else state.params
    batch_stats = state.params.get("batch_stats", {})

    eval_loss_fn = functools.partial(loss_fn, model, config, data, dropout_rng, is_train=False)
    loss, aux = eval_loss_fn(pure_params, sparsity_state=batch_stats)
  else:
    state = nnx.merge(model, state)  # reconstruct TrainStateNNX
    loss, aux = loss_fn(state.model, config, data, None, None, is_train=False)

  mtp_acceptance_rate = 0.0
  if config.mtp_eval_target_module > 0:
    mtp_acceptance_rate = calculate_mtp_acceptance_rate(aux["intermediate_outputs"], config)

  xent_sum = aux["xent_sum"]
  z_loss = aux.get("z_loss", 0.0)
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  indexer_loss = aux.get("indexer_loss", 0.0)
  mtp_loss = aux.get("mtp_loss", 0.0)
  eval_total_loss = xent_sum
  metrics = {
      "scalar": {
          "evaluation/loss": loss,
          "evaluation/z_loss": z_loss,
          "evaluation/total_loss": eval_total_loss,
          "evaluation/total_weights": total_weights,
          "evaluation/moe_lb_loss": moe_lb_loss,
          "evaluation/indexer_loss": indexer_loss,
          "evaluation/mtp_loss": mtp_loss,
          "evaluation/mtp_acceptance_rate_percent": mtp_acceptance_rate,
      },
  }

  return metrics


def train_loop(config, recorder, state=None):
  """Main Training loop."""
  (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      data_loader,
      rampup_manager,
      eval_data_iterator,
      state,
  ) = train_utils.setup_train_loop(config, recorder)

  start_step = get_first_step(model, state)  # this is the start_step for training
  train_utils.validate_completed_steps(start_step, config.steps)

  if isinstance(model, nn.Module):
    jit_model = model
  else:
    jit_model, state = nnx.split(state)

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)

  p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
      config,
      jit_model,
      mesh,
      state,
      state_mesh_shardings,
      train_step,
      eval_step,
      eval_data_iterator,
      params_shardings,
  )

  with jax.set_mesh(mesh), mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    shaped_batch = maxtext_utils.get_shaped_batch(config)
    if config.shard_optimizer_over_data and isinstance(model, nn.Module):
      state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
    elif config.shard_optimizer_over_data:
      # NNX: reshard state so params match the data-sharded in_shardings (Zero-1 layout)
      state = jax.device_put(state, state_mesh_shardings)
    if isinstance(model, nn.Module):
      lower_args = (state, shaped_batch, init_rng)
    else:
      lower_args = (state, shaped_batch)
    maxtext_utils.maybe_dump_jaxpr(config, p_train_step, lower_args)
    if config.compiled_trainstep_file == "":  # compile only when there is no pre-compiled file loaded
      compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)
      compiled = p_train_step.lower(*lower_args).compile(compiler_options=compiler_options)
      compiled_stats = compiled.memory_analysis()
      max_utils.print_compiled_memory_stats(compiled_stats)
  prof = profiler.Profiler(config, offset_step=start_step)
  metric_logger_instance = metric_logger.MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

  # Write train config params, num model params, and XLA flags to tensorboard
  if isinstance(model, nn.Module):
    setup_params = state.params
  else:
    _, setup_params, _ = nnx.split(state.model, nnx.Param, ...)
  metric_logger_instance.write_setup_info_to_tensorboard(setup_params)

  elastic_utils.record_elastic_reinit_end()

  _job_completed_gracefully = False
  try:
    last_step_completion = datetime.datetime.now()
    for step in np.arange(start_step, config.steps):
      prof.maybe_activate_profiler(step, state)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
        if isinstance(model, nn.Module):
          # pylint: disable=not-callable
          step_rng_args = (jax.jit(jax.random.fold_in)(init_rng, step),)
        else:
          step_rng_args = ()
        with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            if config.shard_optimizer_over_data and isinstance(model, nn.Module):
              state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
            state, metrics = p_train_step(state, example_batch, *step_rng_args)

        step_time_delta = datetime.datetime.now() - last_step_completion

        checkpointing.maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator, step)

        if config.dump_hlo and step == (config.dump_step if config.dump_step >= 0 else start_step):
          jax.block_until_ready(state)  # Ensure compilation has finished.
          gcs_utils.upload_dump(
              config.dump_hlo_local_dir,
              config.dump_hlo_gcs_dir,
              module_name=config.dump_hlo_module_name,
              delete_local_after=config.dump_hlo_delete_local_after,
              all_host_upload=config.dump_hlo_upload_all,
          )

        eval_step_count = None
        if config.eval_interval > 0 and step > start_step and (step + 1) % config.eval_interval == 0:
          assert eval_data_iterator
          # Explicitly reset the eval iterator and counters before starting the eval loop
          eval_data_iterator.reset()
          metric_logger_instance.reset_eval_metrics()
          max_logging.log(f"Starting eval after train step {step}")

          eval_step_count = 0
          last_eval_step_completion = datetime.datetime.now()
          # pylint: disable=not-callable
          for eval_batch in eval_data_iterator:
            # Shard input eval data
            eval_batch = jax.device_put(eval_batch, sharding.get_input_data_sharding(config, mesh))
            if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
              break
            with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
              eval_metrics = p_eval_step(state, eval_batch, *step_rng_args)
            eval_step_time_delta = datetime.datetime.now() - last_eval_step_completion
            last_eval_step_completion = datetime.datetime.now()
            metric_logger_instance.buffer_and_write_metrics(
                eval_metrics, eval_step_count, step_time_delta=eval_step_time_delta, is_training=False
            )
            eval_step_count += 1

        prof.maybe_deactivate_profiler(step, state)

        if step == start_step:
          max_utils.print_mem_stats("After params initialized")

        last_step_completion = datetime.datetime.now()
        metric_logger_instance.buffer_and_write_metrics(metrics, step, step_time_delta)

    if config.save_checkpoint_on_completion:
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator)
    if checkpoint_manager is not None:
      # in case the last checkpoint_period checkpoint is still in progress
      checkpoint_manager.wait_until_finished()
    _job_completed_gracefully = True
  except exceptions.StopTraining as e:
    prof.deactivate()
    max_logging.log(f"Training stopped: {str(e)}")
    _job_completed_gracefully = True
  finally:
    if _job_completed_gracefully:
      record_goodput(recorder, RECORD_JOB_END_TIME)
    metric_logger_instance.flush_metrics_and_cleanup()

  return state


def initialize(argv: Sequence[str]) -> tuple[pyconfig.HyperParameters, Any]:
  """Initialization of hyperparameters and utilities"""
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )
  # TODO: mazumdera@ : ensure missing mandatory fields in base.yml are filled in in argv,
  # or fill in here
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  train_utils.validate_train_config(config)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  jax.config.update("jax_remove_size_one_mesh_axis_from_type", config.remove_size_one_mesh_axis_from_type)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  # Create the Goodput recorder
  recorder = create_goodput_recorder(config)

  return config, recorder


def run(config, recorder):
  """Run the job given hyperparameters and utilities."""
  with (max_utils.maybe_get_transformer_engine_context(config),):
    train_loop(config, recorder)


def get_train_func(config, recorder, argv):
  """Returns the train function, wrapping in elastic_retry if elastic training is enabled."""
  if config.elastic_enabled:
    max_logging.log("Elastic utils: Elastic training enabled.")

    def on_elastic_event():
      elastic_utils.record_elastic_event_start(recorder, config)

    def on_slices_ready():
      elastic_utils.record_elastic_wait_end_and_reinit_start(recorder)

    def elastic_train_wrapper(argv: Sequence[str]) -> None:
      """Wrapper for elastic training initializes variables and runs the train loop."""
      elastic_config, elastic_recorder = initialize(argv)
      run(
          elastic_config,
          elastic_recorder,
      )

    train_func = elastic_utils.elastic_retry(
        config,
        callback_fn=on_elastic_event,
        pre_callback_fn=on_slices_ready,
    )(functools.partial(elastic_train_wrapper, argv=argv))
  else:
    # Use the already initialized variables
    def train_func():
      run(config, recorder)

  return train_func


def main(argv: Sequence[str]) -> None:
  config, recorder = initialize(argv)
  record_goodput(recorder, RECORD_JOB_START_TIME)
  train_func = get_train_func(config, recorder, argv)
  with maybe_monitor_goodput(config):
    train_func()


if __name__ == "__main__":
  app.run(main)
