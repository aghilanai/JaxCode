"""INT8 Quantized Linear task."""

TASK = {
    "title": "INT8 Quantized Linear",
    "difficulty": "Hard",
    "function_name": "Int8Linear",
    "hint": "Per-channel scale = abs(weight).max(axis=1) / 127. Quantize: round(weight/scale).clip(-128, 127).astype(int8). Store weight_int8 and scale as nnx.Variable (not nnx.Param). Forward: dequantize and matmul.",
    "tests": [
        {
            "name": "Weight is int8",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

w = jax.random.normal(jax.random.PRNGKey(0), (32, 16))
q = {fn}(w)
assert isinstance(q, nnx.Module), 'Must inherit from nnx.Module'
assert q.weight_int8.value.dtype == jnp.int8, f'dtype: {q.weight_int8.value.dtype}'
""",
        },
        {
            "name": "Values in [-128, 127]",
            "code": """
import jax
import jax.numpy as jnp

q = {fn}(jax.random.normal(jax.random.PRNGKey(0), (64, 32)) * 10)
w_int8 = q.weight_int8.value
assert w_int8.min() >= -128 and w_int8.max() <= 127
""",
        },
        {
            "name": "Dequantized close to original",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
w = jax.random.normal(key, (16, 8))
q = {fn}(w)
w_recon = q.weight_int8.value.astype(jnp.float32) * q.scale.value
assert jnp.max(jnp.abs(w - w_recon)) < 0.1, 'Quantization error too large'
""",
        },
        {
            "name": "Forward output shape",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
q = {fn}(jax.random.normal(k1, (8, 4)), jax.random.normal(k2, (8,)))
out = q(jax.random.normal(jax.random.PRNGKey(1), (2, 4)))
assert out.shape == (2, 8), f'Shape: {out.shape}'
""",
        },
        {
            "name": "Weight is buffer not parameter",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp

q = {fn}(jax.random.normal(jax.random.PRNGKey(0), (4, 4)))
assert isinstance(q.weight_int8, nnx.Variable), 'weight_int8 should be an nnx.Variable'
assert isinstance(q.scale, nnx.Variable), 'scale should be an nnx.Variable'
assert not isinstance(q.weight_int8, nnx.Param), 'weight_int8 should not be an nnx.Param'
assert not isinstance(q.scale, nnx.Param), 'scale should not be an nnx.Param'
""",
        },
    ],
}
