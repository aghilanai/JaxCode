"""Grouped Query Attention task."""

TASK = {
    "title": "Grouped Query Attention",
    "difficulty": "Hard",
    "function_name": "GroupQueryAttention",
    "hint": (
        "Like MHA but fewer KV heads. W_k/W_v project to num_kv_heads * d_k dims. "
        "Repeat/broadcast KV heads to match Q heads (e.g. jnp.repeat along head axis). "
        "Call as gqa(x)."
    ),
    "tests": [
        {
            "name": "Output shape",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
gqa = {fn}(d_model=32, num_heads=8, num_kv_heads=2, rngs=nnx.Rngs(0))
out = gqa(jax.random.normal(key, (2, 6, 32)))
assert out.shape == (2, 6, 32), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "nnx.Linear with correct shapes",
            "code": """
from flax import nnx
gqa = {fn}(d_model=32, num_heads=8, num_kv_heads=2, rngs=nnx.Rngs(0))
d_k = 32 // 8
assert isinstance(gqa.W_q, nnx.Linear) and gqa.W_q.kernel.value.shape == (32, 32)
assert isinstance(gqa.W_k, nnx.Linear) and gqa.W_k.kernel.value.shape == (32, 2 * d_k)
assert isinstance(gqa.W_v, nnx.Linear) and gqa.W_v.kernel.value.shape == (32, 2 * d_k)
assert isinstance(gqa.W_o, nnx.Linear), 'W_o should be nnx.Linear'
""",
        },
        {
            "name": "Degenerates to MHA when kv_heads == heads",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(42)
gqa = {fn}(d_model=16, num_heads=4, num_kv_heads=4, rngs=nnx.Rngs(42))
out = gqa(jax.random.normal(key, (1, 4, 16)))
assert out.shape == (1, 4, 16)
assert gqa.W_k.kernel.value.shape == (16, 16), 'Full KV when kv_heads == heads'
""",
        },
        {
            "name": "KV heads are shared correctly",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
D, H, KV = 16, 4, 2
d_k = D // H
gqa = {fn}(d_model=D, num_heads=H, num_kv_heads=KV, rngs=nnx.Rngs(0))
x = jax.random.normal(key, (1, 4, D))
k = gqa.W_k(x).reshape(1, 4, KV, d_k).transpose(0, 2, 1, 3)
k_exp = jnp.repeat(k, H // KV, axis=1)
assert jnp.array_equal(k_exp[:, 0], k_exp[:, 1]), 'Heads 0,1 should share same K'
assert not jnp.array_equal(k_exp[:, 0], k_exp[:, 2]), 'Different groups need different K'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
gqa = {fn}(d_model=16, num_heads=4, num_kv_heads=2, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 16))

def loss(x):
    return gqa(x).sum()

g = jax.grad(loss)(x)
assert g is not None, 'x gradient is None'

def loss_fn(model):
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 16))
    return model(x).sum()

_, grads = nnx.value_and_grad(loss_fn)(gqa)
assert grads.W_q.kernel.value is not None and grads.W_k.kernel.value is not None, (
    'Missing weight gradients'
)
""",
        },
    ],
}
