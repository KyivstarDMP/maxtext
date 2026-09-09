# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare batch-sharded NNX tiled metrics with a separate single-device reference.

Both workers use CPU devices, including on a TPU host. This covers fsdp_transpose
batch sharding, not tensor-parallel vocabulary sharding.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


@pytest.mark.cpu_only
def test_nnx_tiled_metrics_on_eight_cpu_devices(tmp_path):
  """Force separate 1-device and 8-device workers before either initializes JAX."""
  source_root = Path(__file__).resolve().parents[2]
  for mode, device_count in (("reference", 1), ("tiled", 8)):
    env = os.environ.copy()
    flags = re.sub(r"--xla_force_host_platform_device_count(?:=|\s+)\d+", "", env.get("XLA_FLAGS", ""))
    env.update(
        JAX_PLATFORMS="cpu",
        JAX_PLATFORM_NAME="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={device_count}",
        PYTHONPATH=os.pathsep.join((str(source_root / "src"), str(source_root), env.get("PYTHONPATH", ""))),
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode, str(tmp_path)],
        cwd=source_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, f"{mode} worker failed:\n{completed.stdout}\n{completed.stderr}"
  report = json.loads((tmp_path / "tiled.json").read_text())
  assert report["devices"] == 8
  assert len(report["cases"]) == 4
  assert all(case["status"] == "PASS" for case in report["cases"])


def _probe(mode, output_dir):
  """Run real NNX heads without mocks; reference reductions use NumPy selection."""
  # These imports must happen after the parent configures the subprocess devices.
  # pylint: disable=import-outside-toplevel
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import nnx
  from flax.linen import partitioning
  from maxtext.common.common_types import MODEL_MODE_TRAIN
  from maxtext.configs import pyconfig
  from maxtext.utils import maxtext_utils, maxtext_utils_nnx, model_creation_utils, vocabulary_tiling
  from maxtext.utils.sharding import create_sharding

  assert jax.device_count() == (1 if mode == "reference" else 8)
  records = []
  for tied in (False, True):
    for gather_once in (False, True):
      cfg = pyconfig.initialize(
          [None, str(Path(__file__).resolve().parents[2] / "src/maxtext/configs/base.yml")],
          run_name="cpu_spmd_metrics",
          enable_checkpointing=False,
          enable_dropout=False,
          max_target_length=32,
          per_device_batch_size=8 / jax.device_count(),
          base_emb_dim=16,
          base_num_query_heads=4,
          base_num_kv_heads=4,
          head_dim=4,
          base_mlp_dim=32,
          base_num_decoder_layers=0,
          vocab_size=32,
          dtype="float32",
          matmul_precision="high",
          num_vocab_tiling=16,
          per_dataset_metrics=True,
          per_dataset_names="a,b,absent",
          z_loss_multiplier=0.1,
          logits_via_embedding=tied,
          pure_nnx=True,
          enable_nnx=True,
          pure_nnx_decoder=True,
          ici_fsdp_transpose_parallelism=-1,
          ici_fsdp_parallelism=1,
          shardy=True,
          shard_mode="auto",
          vocab_tiling_ag_once=gather_once,
      )
      mesh = maxtext_utils.get_mesh_from_config(cfg)
      assert mesh.shape["fsdp_transpose"] == jax.device_count()
      assert mesh.shape["fsdp"] == 1
      rngs = maxtext_utils_nnx.create_nnx_rngs(cfg, rng_key=jax.random.key(17))
      with partitioning.axis_rules(cfg.logical_axis_rules):
        model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)
      graphdef, params, rest = nnx.split(model, nnx.Param, ...)
      hidden = np.random.default_rng(123).normal(size=(8, 32, 16)).astype(np.float32)
      ids = np.where((np.arange(256).reshape(8, 32) * 7) % 11 < 3, 2, 1).astype(np.int32)
      mask = np.ones((8, 32), dtype=np.int32)
      for row in range(8):
        mask[row, 32 - (row + 1) :] = 0
      ids[mask == 0] = 0
      labels = (np.arange(256).reshape(8, 32) * 3 % 32).astype(np.int32)
      data = {"targets": jnp.asarray(labels), "targets_segmentation": jnp.asarray(mask), "dataset_id": jnp.asarray(ids)}

      def logits(p, h, graphdef=graphdef, rest=rest):
        return nnx.merge(graphdef, p, rest, copy=True).logits_from_hidden_states_for_vocab_tiling(h, True, "train")

      def reference(p, h, logits=logits, targets=data["targets"], cfg=cfg, mask=mask):
        out = logits(p, h).astype(jnp.float32)
        log_z = jax.scipy.special.logsumexp(out, axis=-1)
        gold = jnp.take_along_axis(out, targets[..., None], axis=-1)[..., 0]
        total = jnp.sum((log_z - gold + cfg.z_loss_multiplier * log_z**2) * mask)
        return total / mask.sum()

      def tiled(p, h, d, graphdef=graphdef, rest=rest, cfg=cfg, mask=mask):
        out = vocabulary_tiling.vocab_tiling_nnx_loss(nnx.merge(graphdef, p, rest, copy=True), h, d, cfg, True)
        tokens = jax.ops.segment_sum(
            (d["targets_segmentation"] != 0).reshape(-1).astype(jnp.int32),
            d["dataset_id"].reshape(-1),
            num_segments=4,
        )
        return out[0] / mask.sum(), (*out, tokens)

      key = f"tied{int(tied)}_gather{int(gather_once)}"
      target = output_dir / f"reference_{key}.npz"
      with partitioning.axis_rules(cfg.logical_axis_rules):
        if mode == "reference":
          value, (gp, gh) = jax.jit(jax.value_and_grad(reference, argnums=(0, 1)))(params, jnp.asarray(hidden))
          out = np.asarray(jax.jit(logits)(params, jnp.asarray(hidden)), dtype=np.float64)
          shift = out.max(axis=-1)
          log_z = shift + np.log(np.exp(out - shift[..., None]).sum(axis=-1))
          xent = log_z - np.take_along_axis(out, labels[..., None], axis=-1)[..., 0] + cfg.z_loss_multiplier * log_z**2
          losses = np.array([xent[(ids == d) & (mask != 0)].sum() for d in range(4)])
          correct = np.array([((out.argmax(axis=-1) == labels) & (ids == d) & (mask != 0)).sum() for d in range(4)])
          tokens = np.array([((ids == d) & (mask != 0)).sum() for d in range(4)])
          payload = {
              "value": value,
              "total": losses.sum(),
              "z_loss": (cfg.z_loss_multiplier * log_z**2 * mask).sum(),
              "loss_by_ds": losses,
              "correct": correct,
              "tokens": tokens,
              "grad_hidden": gh,
          }
          payload.update({f"param_{i}": p for i, p in enumerate(jax.tree_util.tree_leaves(params))})
          payload.update({f"grad_{i}": p for i, p in enumerate(jax.tree_util.tree_leaves(gp))})
          np.savez(target, **payload)
        else:
          with np.load(target) as ref:
            for i, param in enumerate(jax.tree_util.tree_leaves(params)):
              np.testing.assert_array_equal(param, ref[f"param_{i}"])
            h = jax.device_put(
                hidden,
                create_sharding(mesh, ("activation_embed_and_logits_batch", "activation_length", "activation_embed")),
            )
            data_spec = create_sharding(mesh, ("activation_embed_and_logits_batch", "activation_length"))
            d = jax.tree_util.tree_map(lambda a, spec=data_spec: jax.device_put(a, spec), data)
            (value, (total, z_loss, losses, correct, tokens)), (gp, gh) = jax.jit(
                jax.value_and_grad(tiled, argnums=(0, 1), has_aux=True)
            )(params, h, d)
            for name, actual in {"value": value, "total": total, "z_loss": z_loss, "loss_by_ds": losses}.items():
              np.testing.assert_allclose(actual, ref[name], rtol=3e-5, atol=3e-5)
            np.testing.assert_array_equal(correct, ref["correct"])
            np.testing.assert_array_equal(tokens, ref["tokens"])
            np.testing.assert_allclose(gh, ref["grad_hidden"], rtol=4e-5, atol=3e-6)
            for i, (path, actual) in enumerate(jax.tree_util.tree_leaves_with_path(gp)):
              expected = ref[f"grad_{i}"]
              if tied and "token_embedder" in jax.tree_util.keystr(path):
                # The unchanged shared output head casts the table to bf16,
                # rounding each tile's table gradient before summation.
                np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)
                assert np.linalg.norm(np.asarray(actual - expected)) <= 0.01 * np.linalg.norm(expected)
              else:
                np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=3e-6)
          assert int(tokens.sum()) == int(mask.sum())
          assert int(tokens[0]) == int(tokens[3]) == int(correct[3]) == 0
          assert float(losses[1]) > 0 and float(losses[2]) > 0
      records.append({"case": key, "status": "PASS", "mesh": dict(mesh.shape)})
  (output_dir / f"{mode}.json").write_text(json.dumps({"devices": jax.device_count(), "cases": records}))


if __name__ == "__main__":
  _probe(sys.argv[1], Path(sys.argv[2]))
