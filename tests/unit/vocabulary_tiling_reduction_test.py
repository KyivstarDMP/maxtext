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

"""Check tiled dataset sums at production token geometry against a NumPy oracle.

The TPU test requires an already initialized distributed runtime when used on
multiple hosts. CPU success does not substitute for this backend regression.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from maxtext.utils.vocabulary_tiling import _sum_by_dataset


def _check_tiled_sums(dtype, pattern):
  """Compile only the tested reduction; keep the independent oracle outside JAX."""
  mesh = Mesh(np.asarray(jax.devices()), ("fsdp_transpose",))
  assert 65536 % mesh.size == 0
  batch_spec = NamedSharding(mesh, P("fsdp_transpose", None))
  tile_spec = NamedSharding(mesh, P(None, "fsdp_transpose"))
  chunk_spec = NamedSharding(mesh, P("fsdp_transpose"))
  values = np.ones((128, 8192), dtype=dtype)
  ids = np.ones(values.shape, dtype=np.int32)
  if pattern == "tile_markers":
    values.fill(0)
    values.reshape(-1)[[0, 512, 65536, 1048575]] = [1, 2, 4, 8]
    ids.reshape(-1)[[0, 512, 65536, 1048575]] = [0, 1, 2, 1]
  elif pattern == "masks_and_invalid_ids":
    positions = np.arange(values.size).reshape(values.shape)
    values = (positions % 5).astype(dtype)
    if dtype == np.float32:
      values /= 16  # Exactly representable sums; a rounding tolerance cannot hide lost tiles.
    values[positions % 7 == 0] = 0
    ids = np.where(positions % 11 == 0, -1, positions % 3).astype(np.int32)
    ids[positions % 13 == 0] = 3  # No slot 2: also check a truly absent dataset.
    ids[ids == 2] = 1

  def tiled(v, d):
    v = jax.lax.with_sharding_constraint(v.reshape(16, 65536), tile_spec)
    d = jax.lax.with_sharding_constraint(d.reshape(16, 65536), tile_spec)

    def body(carry, chunk):
      total, grouped = carry
      chunk_values = jax.lax.with_sharding_constraint(chunk[0], chunk_spec)
      chunk_ids = jax.lax.with_sharding_constraint(chunk[1], chunk_spec)
      return (total + jnp.sum(chunk_values), grouped + _sum_by_dataset(chunk_values, chunk_ids, 3)), None

    return jax.lax.scan(body, (jnp.zeros((), dtype=v.dtype), jnp.zeros(3, dtype=v.dtype)), (v, d))[0]

  # A callback gives each process precisely the global indices its shards own.
  v = jax.make_array_from_callback(values.shape, batch_spec, lambda index: values[index])
  d = jax.make_array_from_callback(ids.shape, batch_spec, lambda index: ids[index])
  lowered = jax.jit(tiled).lower(v, d)
  # This guard intentionally covers only the metric reducer: the full training
  # graph may legitimately contain unrelated scatters. This pattern triggered
  # the TPU loop-carry miscompilation even when CPU numerical checks passed.
  assert "stablehlo.scatter" not in lowered.as_text()
  total, grouped = lowered.compile()(v, d)
  expected = np.asarray([values[ids == slot].sum(dtype=np.float64) for slot in range(3)])
  np.testing.assert_array_equal(np.asarray(grouped), expected)
  assert float(total) == values.sum(dtype=np.float64)


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
@pytest.mark.parametrize("pattern", ["all_ones", "tile_markers", "masks_and_invalid_ids"])
def test_tiled_dataset_sums_cpu(dtype, pattern):
  _check_tiled_sums(dtype, pattern)


@pytest.mark.tpu_only
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
@pytest.mark.parametrize("pattern", ["all_ones", "tile_markers", "masks_and_invalid_ids"])
def test_tiled_dataset_sums_tpu(dtype, pattern):
  if jax.device_count() < 2:
    pytest.skip("The carry-reset regression requires at least two TPU partitions")
  _check_tiled_sums(dtype, pattern)
