# Copyright 2025-2026 Google LLC
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

"""Functions for vocabulary tiling (VT)"""

import functools

from flax import linen as nn
from flax import nnx

import jax
import jax.numpy as jnp
from maxtext.utils.sharding import (
    maybe_shard_with_name,
    all_gather_over_fsdp,
    create_sharding,
)
from maxtext.common.common_types import ShardMode
from maxtext.utils import max_utils


def vocab_tiling_linen_loss(
    hidden_states,
    data,
    config,
    model,
    params,
    is_train,
):
  """Calculates cross-entropy loss using vocab tiling for Linen models.

  This function implements a memory-efficient approach for calculating loss when the
  vocabulary is too large to fit in memory. It works by breaking the computation
  into chunks (tiles) and processing them sequentially using `jax.lax.scan`.
  A custom VJP rule is defined to handle the backward pass efficiently.

  Args:
    hidden_states: The final hidden states from the decoder.
    data: A dictionary containing the input data, including 'targets' and 'targets_segmentation'.
    config: The model and training configuration.
    model: The Linen model instance.
    params: The model parameters.
    is_train: A boolean indicating if the model is in training mode.
  Returns:
    A tuple of (total_loss, total_z_loss, total_ul_loss, total_ditto_loss) computed via
    vocab tiling. ``total_loss`` is the summed cross-entropy (NLL, including z-loss);
    ``total_ul_loss`` is the summed token-level unlikelihood loss and ``total_ditto_loss``
    the summed DITTO repetition-penalization loss (each 0 when its ``*_alpha == 0`` or not
    training), weighted by their respective ``alpha`` at the call site.
  """
  labels = data["targets"]
  segmentation = data["targets_segmentation"]
  use_unlikelihood = is_train and config.unlikelihood_alpha > 0
  use_ditto = is_train and config.ditto_alpha > 0
  # Unlikelihood (scope B): the negative candidates are the model's own prior output —
  # the full `labels`/`segmentation` serve as the candidate context, since
  # targets_segmentation is non-zero only on completion tokens.
  # DITTO operates on a synthetic pseudo-repetition `data` (built in the trainer) with a
  # known period: `ditto_baseline_pos[b,i] = i - period` and `ditto_pen_mask[b,i]` (the
  # 2nd-or-later repetition) arrive precomputed in `data`.
  ditto_baseline_pos = data["ditto_baseline_pos"] if use_ditto else None
  ditto_pen_mask = data["ditto_pen_mask"] if use_ditto else None
  deterministic = not config.enable_dropout if is_train else True

  param_spec = nn.get_partition_spec(params)
  hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length", "activation_embed"),
  )
  label_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length"),
  )
  reshaped_hidden_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  reshaped_data_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence"),
  )
  chunked_hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  chunked_data_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence",),
  )
  chunked_logits_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_vocab"),
  )

  _maybe_shard_with_name = functools.partial(
      maybe_shard_with_name,
      shard_mode=config.shard_mode,
      debug_sharding=config.debug_sharding,
      extra_stack_level=1,
  )

  def _reshape(inputs, out_shape, out_sharding):
    reshape_out_sharding = out_sharding if config.shard_mode == ShardMode.EXPLICIT else None
    inputs = jax.lax.reshape(inputs, out_shape, out_sharding=reshape_out_sharding)
    return _maybe_shard_with_name(inputs, out_sharding)

  hidden_states = _maybe_shard_with_name(hidden_states, hidden_spec)
  labels = _maybe_shard_with_name(labels, label_spec)
  segmentation = _maybe_shard_with_name(segmentation, label_spec)
  if use_ditto:
    ditto_baseline_pos = _maybe_shard_with_name(ditto_baseline_pos, label_spec)
    ditto_pen_mask = _maybe_shard_with_name(ditto_pen_mask, label_spec)
  # TODO (chengnuojin) all gather only embedding table instead of all params after NNX module is enabled
  gathered_params = all_gather_over_fsdp(params, param_spec, model.mesh, config.logical_axis_rules, config.shard_mode)

  def _gold_probs_linen():
    """Detached per-token gold probabilities ``[B, S]`` via a forward-only tiled scan.

    The DITTO baseline (the previous-occurrence probability the penalty decays toward)
    may live in an earlier tile, so it is precomputed once here over the whole batch and
    captured in the custom_vjp closure like ``labels``. Wrapped in ``stop_gradient`` —
    DITTO detaches the baseline — so it contributes no gradient. Costs one extra forward
    over the logits when DITTO is enabled.
    """
    bsz, slen, edim = hidden_states.shape
    tile = (bsz * slen) // config.num_vocab_tiling
    rh = _reshape(hidden_states, (config.num_vocab_tiling, tile, edim), reshaped_hidden_spec)
    rl = _reshape(labels, (config.num_vocab_tiling, tile), reshaped_data_spec)

    def _gp_body(_, chunk):
      h, gp_label = chunk
      h = _maybe_shard_with_name(h, chunked_hidden_spec)
      gp_label = _maybe_shard_with_name(gp_label, chunked_data_spec)
      logits = model.apply(
          {"params": gathered_params["params"]},
          h,
          deterministic=deterministic,
          method="logits_from_hidden_states_for_vocab_tiling",
      )
      logits = _maybe_shard_with_name(logits, chunked_logits_spec)
      return None, max_utils.gold_prob_from_logits(logits, gp_label)

    _, gp_tiles = jax.lax.scan(_gp_body, None, (rh, rl))
    return jax.lax.stop_gradient(_reshape(gp_tiles, (bsz, slen), label_spec))

  gold_probs = _gold_probs_linen() if use_ditto else None

  per_dataset = config.per_dataset_metrics and "dataset_id" in data

  def _per_dataset_linen():
    """Forward-only tiled per-dataset (xent_sum, correct_count) [num_datasets+1] vectors (no grad).

    Mirrors :func:`_gold_probs_linen`: one extra forward over the tiled logits, wrapped in
    stop_gradient (these are metrics, not part of the training objective). Each chunk holds full
    per-token logits, so next-token accuracy is a per-chunk argmax; both quantities are
    segment-summed by ``dataset_id`` and accumulated across chunks.
    """
    num_seg = len([n for n in config.per_dataset_names.split(",") if n]) + 1
    bsz, slen, edim = hidden_states.shape
    tile = (bsz * slen) // config.num_vocab_tiling
    rh = _reshape(hidden_states, (config.num_vocab_tiling, tile, edim), reshaped_hidden_spec)
    rl = _reshape(labels, (config.num_vocab_tiling, tile), reshaped_data_spec)
    rs = _reshape(segmentation, (config.num_vocab_tiling, tile), reshaped_data_spec)
    rd = _reshape(data["dataset_id"], (config.num_vocab_tiling, tile), reshaped_data_spec)

    def _pd_body(acc, chunk):
      xent_acc, correct_acc = acc
      h, lbl, seg, dsid = chunk
      h = _maybe_shard_with_name(h, chunked_hidden_spec)
      logits = model.apply(
          {"params": gathered_params["params"]},
          h,
          deterministic=deterministic,
          method="logits_from_hidden_states_for_vocab_tiling",
      )
      logits = _maybe_shard_with_name(logits, chunked_logits_spec)
      chunk_xent, _ = max_utils.cross_entropy_with_logits(logits, jax.nn.one_hot(lbl, config.vocab_size), z_loss=0.0)
      m = seg != 0
      xent_acc = xent_acc + jax.ops.segment_sum(chunk_xent * m, dsid, num_segments=num_seg)
      correct = (jnp.argmax(logits, axis=-1) == lbl) & m
      correct_acc = correct_acc + jax.ops.segment_sum(correct.astype(jnp.int32), dsid, num_segments=num_seg)
      return (xent_acc, correct_acc), None

    (xent_by_ds, correct_by_ds), _ = jax.lax.scan(
        _pd_body, (jnp.zeros(num_seg, jnp.float32), jnp.zeros(num_seg, jnp.int32)), (rh, rl, rs, rd)
    )
    return jax.lax.stop_gradient(xent_by_ds), jax.lax.stop_gradient(correct_by_ds)

  pd_xent_by_ds, pd_correct_by_ds = _per_dataset_linen() if per_dataset else (None, None)

  # Customized forward and backward maps for the embedding tiling
  @jax.custom_vjp
  def chunked_cross_entropy_loss(gathered_params, hidden_states, labels, segmentation):
    """
    Calculates the total cross-entropy loss using vocab tiling.
    """
    outputs, _ = _chunked_cross_entropy_loss_fwd(gathered_params, hidden_states, labels, segmentation)
    return outputs

  def _chunked_cross_entropy_loss_fwd(gathered_params, hidden_states, labels, segmentation):
    batch_size, seq_len, emb_dim = hidden_states.shape
    vocab_tile_size = (batch_size * seq_len) // config.num_vocab_tiling

    reshaped_hidden_states = _reshape(
        hidden_states, (config.num_vocab_tiling, vocab_tile_size, emb_dim), reshaped_hidden_spec
    )
    reshaped_labels = _reshape(labels, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)
    reshaped_segmentation = _reshape(segmentation, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)
    # Global flat positions (b * seq_len + i) for each token, tiled the same way,
    # so the unlikelihood kernel can recover (b, i) and look up the input context.
    flat_positions = jnp.arange(batch_size * seq_len, dtype=jnp.int32).reshape(batch_size, seq_len)
    reshaped_positions = _reshape(flat_positions, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)

    # Scan body accumulates loss from each tile given chunked hidden states and labels
    def _fwd_scan_body(accumulators, chunk_data):
      loss_accumulator, z_loss_accumulator, ul_accumulator, ditto_accumulator = accumulators
      hidden_chunk, label_chunk, segmentation_chunk, positions_chunk = chunk_data
      hidden_chunk = _maybe_shard_with_name(hidden_chunk, chunked_hidden_spec)
      label_chunk = _maybe_shard_with_name(label_chunk, chunked_data_spec)
      segmentation_chunk = _maybe_shard_with_name(segmentation_chunk, chunked_data_spec)

      # Calculate logits for the current chunk
      chunk_logits = model.apply(
          {"params": gathered_params["params"]},
          hidden_chunk,
          deterministic=deterministic,
          method="logits_from_hidden_states_for_vocab_tiling",
      )
      chunk_logits = _maybe_shard_with_name(chunk_logits, chunked_logits_spec)
      one_hot_label_chunk = jax.nn.one_hot(label_chunk, config.vocab_size)
      chunk_xent, chunk_z_loss = max_utils.cross_entropy_with_logits(
          chunk_logits, one_hot_label_chunk, z_loss=config.z_loss_multiplier
      )

      masked_xent = jnp.sum(chunk_xent * (segmentation_chunk != 0))
      masked_z_loss = jnp.sum(chunk_z_loss * (segmentation_chunk != 0))

      loss_accumulator += masked_xent
      z_loss_accumulator += masked_z_loss
      if use_unlikelihood:
        ul_accumulator += max_utils.unlikelihood_loss_from_logits(
            chunk_logits,
            positions_chunk,
            labels,
            segmentation,
            label_chunk,
            segmentation_chunk,
            seq_len=seq_len,
            window=config.unlikelihood_window,
            eps=config.unlikelihood_eps,
        )
      if use_ditto:
        ditto_accumulator += max_utils.ditto_loss_from_logits(
            chunk_logits,
            positions_chunk,
            label_chunk,
            gold_probs,
            ditto_baseline_pos,
            ditto_pen_mask,
            seq_len=seq_len,
            gamma=config.ditto_gamma,
            eps=config.ditto_eps,
            loss_type=config.ditto_loss_type,
        )
      return (loss_accumulator, z_loss_accumulator, ul_accumulator, ditto_accumulator), None

    initial_acc = (0.0, 0.0, 0.0, 0.0)
    (total_loss, total_z_loss, total_ul_loss, total_ditto_loss), _ = jax.lax.scan(
        _fwd_scan_body,
        initial_acc,
        (reshaped_hidden_states, reshaped_labels, reshaped_segmentation, reshaped_positions),
    )
    residuals = (
        gathered_params,
        reshaped_hidden_states,
        reshaped_labels,
        reshaped_segmentation,
        reshaped_positions,
        batch_size,
        seq_len,
        emb_dim,
    )

    return (total_loss, total_z_loss, total_ul_loss, total_ditto_loss), residuals

  def _chunked_cross_entropy_loss_bwd(residuals, cotangents):
    # Unpack the cotangents tuple. We ignore the z_loss cotangent since the gradients
    # of the z_loss term are already factored into the cross-entropy cotangent. The
    # cross-entropy, unlikelihood and DITTO cotangents are folded into the per-chunk loss
    # so the chain rule produces grad(c_xent * NLL + c_ul * UL + c_ditto * DITTO) directly.
    loss_cotangent, _, ul_cotangent, ditto_cotangent = cotangents

    (
        gathered_params,
        reshaped_hidden_states,
        reshaped_labels,
        reshaped_segmentation,
        reshaped_positions,
        batch_size,
        seq_len,
        emb_dim,
    ) = residuals

    def _single_chunk_loss_fn(
        input_params, input_hidden_chunk, input_label_chunk, input_segmentation_chunk, input_positions_chunk
    ):
      chunk_logits = model.apply(
          {"params": input_params["params"]},
          input_hidden_chunk,
          deterministic=deterministic,
          method="logits_from_hidden_states_for_vocab_tiling",
      )
      chunk_logits = _maybe_shard_with_name(chunk_logits, chunked_logits_spec)
      one_hot_label_chunk = jax.nn.one_hot(input_label_chunk, config.vocab_size)
      xent, _ = max_utils.cross_entropy_with_logits(chunk_logits, one_hot_label_chunk, z_loss=config.z_loss_multiplier)
      chunk_loss = loss_cotangent * jnp.sum(xent * (input_segmentation_chunk != 0))
      if use_unlikelihood:
        ul_sum = max_utils.unlikelihood_loss_from_logits(
            chunk_logits,
            input_positions_chunk,
            labels,
            segmentation,
            input_label_chunk,
            input_segmentation_chunk,
            seq_len=seq_len,
            window=config.unlikelihood_window,
            eps=config.unlikelihood_eps,
        )
        chunk_loss = chunk_loss + ul_cotangent * ul_sum
      if use_ditto:
        ditto_sum = max_utils.ditto_loss_from_logits(
            chunk_logits,
            input_positions_chunk,
            input_label_chunk,
            gold_probs,
            ditto_baseline_pos,
            ditto_pen_mask,
            seq_len=seq_len,
            gamma=config.ditto_gamma,
            eps=config.ditto_eps,
            loss_type=config.ditto_loss_type,
        )
        chunk_loss = chunk_loss + ditto_cotangent * ditto_sum
      return chunk_loss

    def _bwd_scan_body(grad_params_acc, chunk_data):
      hidden_chunk, label_chunk, segmentation_chunk, positions_chunk = chunk_data

      # Apply sharding constraints to the chunk data
      hidden_chunk = _maybe_shard_with_name(hidden_chunk, chunked_hidden_spec)
      label_chunk = _maybe_shard_with_name(label_chunk, chunked_data_spec)
      segmentation_chunk = _maybe_shard_with_name(segmentation_chunk, chunked_data_spec)

      # Create a loss function closure that captures the current chunk's labels and segmentation.
      # This gives `jax.vjp` a function with the required signature: `loss(params, hidden_states)`.
      # pylint: disable=unnecessary-lambda-assignment
      loss_fn_for_vjp = lambda p, h: _single_chunk_loss_fn(p, h, label_chunk, segmentation_chunk, positions_chunk)

      # Get the vector-Jacobian product function wrt both params and hidden states
      _, vjp_fn = jax.vjp(loss_fn_for_vjp, gathered_params, hidden_chunk)

      # 1.0 since the output cotangents are already folded into _single_chunk_loss_fn.
      (grad_params_update, grad_hidden_chunk) = vjp_fn(1.0)
      grad_hidden_chunk = _maybe_shard_with_name(grad_hidden_chunk, chunked_hidden_spec)

      grad_params_acc = jax.tree_util.tree_map(
          lambda acc, update: acc + update,
          grad_params_acc,
          grad_params_update,
      )
      return grad_params_acc, grad_hidden_chunk

    initial_grad_params_acc = jax.tree_util.tree_map(jnp.zeros_like, gathered_params)

    # The scan now returns the total gradients for the params in the final carry
    grad_params, grad_reshaped_hidden_states = jax.lax.scan(
        _bwd_scan_body,
        initial_grad_params_acc,
        (reshaped_hidden_states, reshaped_labels, reshaped_segmentation, reshaped_positions),
    )
    grad_reshaped_hidden_states = _maybe_shard_with_name(grad_reshaped_hidden_states, reshaped_hidden_spec)
    # Cast cotangents back to each primal's dtype; custom_vjp requires dtype match.
    grad_params = jax.tree_util.tree_map(lambda x, y: y.astype(x.dtype), gathered_params, grad_params)
    # Give back sharding constraint
    grad_reshaped_hidden_states = _reshape(grad_reshaped_hidden_states, (batch_size, seq_len, emb_dim), hidden_spec)
    return (
        grad_params,  # grad for params
        grad_reshaped_hidden_states.astype(reshaped_hidden_states.dtype),
        None,  # grad for reshaped_labels
        None,  # grad for reshaped_segmentation
    )

  chunked_cross_entropy_loss.defvjp(_chunked_cross_entropy_loss_fwd, _chunked_cross_entropy_loss_bwd)

  total_loss, total_z_loss, total_ul_loss, total_ditto_loss = chunked_cross_entropy_loss(
      gathered_params,
      hidden_states,
      labels,
      segmentation,
  )

  if per_dataset:
    return total_loss, total_z_loss, total_ul_loss, total_ditto_loss, pd_xent_by_ds, pd_correct_by_ds
  return total_loss, total_z_loss, total_ul_loss, total_ditto_loss


def vocab_tiling_nnx_loss(model, hidden_states, data, config, is_train):
  """Computes cross-entropy loss with vocab tiling for NNX models.

  NNX equivalent of ``vocab_tiling_linen_loss``. Scans the vocab dimension
  and calls ``model.logits_from_hidden_states_for_vocab_tiling`` per chunk. The NNX model
  carries its own parameters, so no explicit gather is needed.

  Uses default autograd; a custom_vjp for backward memory savings can be
  added later if needed.

  Args:
    model: NNX model exposing ``logits_from_hidden_states_for_vocab_tiling``.
    hidden_states: Final hidden states from the decoder.
    data: Dict with ``targets`` and ``targets_segmentation``.
    config: Model and training config.
    is_train: Whether the model is in training mode.

  Returns:
    A tuple ``(total_loss, total_z_loss, total_ul_loss, total_ditto_loss)``.
    ``total_ul_loss`` / ``total_ditto_loss`` are the summed token-level unlikelihood and
    DITTO losses (0 when disabled), weighted by their respective ``alpha`` at the call site.
  """
  labels = data["targets"]
  segmentation = data["targets_segmentation"]
  use_unlikelihood = is_train and config.unlikelihood_alpha > 0
  use_ditto = is_train and config.ditto_alpha > 0
  # Unlikelihood (scope B): `labels`/`segmentation` are the candidate context. DITTO uses
  # the precomputed `ditto_baseline_pos` / `ditto_pen_mask` maps from `data` (see
  # vocab_tiling_linen_loss).
  ditto_baseline_pos = data["ditto_baseline_pos"] if use_ditto else None
  ditto_pen_mask = data["ditto_pen_mask"] if use_ditto else None
  deterministic = not config.enable_dropout if is_train else True
  model_mode = "train"

  hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length", "activation_embed"),
  )
  label_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length"),
  )
  reshaped_hidden_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  reshaped_data_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence"),
  )
  chunked_hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  chunked_data_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence",),
  )
  chunked_logits_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_vocab"),
  )

  _maybe_shard_with_name = functools.partial(
      maybe_shard_with_name,
      shard_mode=config.shard_mode,
      debug_sharding=config.debug_sharding,
      extra_stack_level=1,
  )

  def _reshape(inputs, out_shape, out_sharding):
    reshape_out_sharding = out_sharding if config.shard_mode == ShardMode.EXPLICIT else None
    inputs = jax.lax.reshape(inputs, out_shape, out_sharding=reshape_out_sharding)
    return _maybe_shard_with_name(inputs, out_sharding)

  hidden_states = _maybe_shard_with_name(hidden_states, hidden_spec)
  labels = _maybe_shard_with_name(labels, label_spec)
  segmentation = _maybe_shard_with_name(segmentation, label_spec)
  if use_ditto:
    ditto_baseline_pos = _maybe_shard_with_name(ditto_baseline_pos, label_spec)
    ditto_pen_mask = _maybe_shard_with_name(ditto_pen_mask, label_spec)

  batch_size, seq_len, emb_dim = hidden_states.shape
  vocab_tile_size = (batch_size * seq_len) // config.num_vocab_tiling

  reshaped_hidden_states = _reshape(
      hidden_states, (config.num_vocab_tiling, vocab_tile_size, emb_dim), reshaped_hidden_spec
  )
  reshaped_labels = _reshape(labels, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)
  reshaped_segmentation = _reshape(segmentation, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)
  # Global flat positions (b * seq_len + i) for each token, tiled the same way,
  # so the unlikelihood kernel can recover (b, i) and look up the input context.
  flat_positions = jnp.arange(batch_size * seq_len, dtype=jnp.int32).reshape(batch_size, seq_len)
  reshaped_positions = _reshape(flat_positions, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)

  # Rebuild the model per chunk inside the scan: the output head pulls an rng stream, and
  # mutating the outer model's rng inside scan's sub-trace raises TraceContextError.
  # nnx.merge(..., copy=True) makes fresh Variables local to each iteration.
  graphdef, model_state = nnx.split(model)

  # DITTO baseline: detached per-token gold probabilities [B, S] over the whole batch
  # (the previous-occurrence probability may live in an earlier tile). Forward-only scan,
  # stop_gradient -> no gradient contribution; one extra forward when DITTO is enabled.
  gold_probs = None
  if use_ditto:

    def _gp_body(_, chunk):
      h, gp_label = chunk
      h = _maybe_shard_with_name(h, chunked_hidden_spec)
      gp_label = _maybe_shard_with_name(gp_label, chunked_data_spec)
      gp_model = nnx.merge(graphdef, model_state, copy=True)
      logits = gp_model.logits_from_hidden_states_for_vocab_tiling(h, deterministic, model_mode)
      logits = _maybe_shard_with_name(logits, chunked_logits_spec)
      return None, max_utils.gold_prob_from_logits(logits, gp_label)

    _, gp_tiles = jax.lax.scan(_gp_body, None, (reshaped_hidden_states, reshaped_labels))
    gold_probs = jax.lax.stop_gradient(_reshape(gp_tiles, (batch_size, seq_len), label_spec))

  def _scan_body(accumulators, chunk_data):
    loss_accumulator, z_loss_accumulator, ul_accumulator, ditto_accumulator = accumulators
    hidden_chunk, label_chunk, segmentation_chunk, positions_chunk = chunk_data
    hidden_chunk = _maybe_shard_with_name(hidden_chunk, chunked_hidden_spec)
    label_chunk = _maybe_shard_with_name(label_chunk, chunked_data_spec)
    segmentation_chunk = _maybe_shard_with_name(segmentation_chunk, chunked_data_spec)

    chunk_model = nnx.merge(graphdef, model_state, copy=True)
    chunk_logits = chunk_model.logits_from_hidden_states_for_vocab_tiling(hidden_chunk, deterministic, model_mode)
    chunk_logits = _maybe_shard_with_name(chunk_logits, chunked_logits_spec)
    one_hot_label_chunk = jax.nn.one_hot(label_chunk, config.vocab_size)
    chunk_xent, chunk_z_loss = max_utils.cross_entropy_with_logits(
        chunk_logits, one_hot_label_chunk, z_loss=config.z_loss_multiplier
    )

    masked_xent = jnp.sum(chunk_xent * (segmentation_chunk != 0))
    masked_z_loss = jnp.sum(chunk_z_loss * (segmentation_chunk != 0))

    ul_update = ul_accumulator
    if use_unlikelihood:
      ul_update = ul_accumulator + max_utils.unlikelihood_loss_from_logits(
          chunk_logits,
          positions_chunk,
          labels,
          segmentation,
          label_chunk,
          segmentation_chunk,
          seq_len=seq_len,
          window=config.unlikelihood_window,
          eps=config.unlikelihood_eps,
      ).astype(ul_accumulator.dtype)
    ditto_update = ditto_accumulator
    if use_ditto:
      ditto_update = ditto_accumulator + max_utils.ditto_loss_from_logits(
          chunk_logits,
          positions_chunk,
          label_chunk,
          gold_probs,
          ditto_baseline_pos,
          ditto_pen_mask,
          seq_len=seq_len,
          gamma=config.ditto_gamma,
          eps=config.ditto_eps,
          loss_type=config.ditto_loss_type,
      ).astype(ditto_accumulator.dtype)
    return (
        loss_accumulator + masked_xent,
        z_loss_accumulator + masked_z_loss,
        ul_update,
        ditto_update,
    ), None

  initial_acc = (
      jnp.zeros((), dtype=hidden_states.dtype),
      jnp.zeros((), dtype=hidden_states.dtype),
      jnp.zeros((), dtype=hidden_states.dtype),
      jnp.zeros((), dtype=hidden_states.dtype),
  )
  (total_loss, total_z_loss, total_ul_loss, total_ditto_loss), _ = jax.lax.scan(
      _scan_body, initial_acc, (reshaped_hidden_states, reshaped_labels, reshaped_segmentation, reshaped_positions)
  )
  return total_loss, total_z_loss, total_ul_loss, total_ditto_loss
