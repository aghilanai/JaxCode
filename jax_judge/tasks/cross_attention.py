"""Multi-Head Cross-Attention task."""

TASK = {
    "title": "Multi-Head Cross-Attention",
    "difficulty": "Medium",
    "function_name": "MultiHeadCrossAttention",
    "hint": (
        "q_BLD from decoder, kv_BMD from encoder. Project, reshape to q_BHLK / k_BHMK, "
        "scaled dot-product attention (no causal mask), concat to _BLD, output projection. "
        "Call as attn(q_BLD, kv_BMD)."
    ),
    "tests": [
        {
            "name": "Output shape",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
attn = {fn}(d_model=64, num_heads=4, rngs=nnx.Rngs(0))
assert isinstance(attn, nnx.Module), 'Must inherit from nnx.Module'
out = attn(
    jax.random.normal(key, (2, 6, 64)),
    jax.random.normal(jax.random.fold_in(key, 1), (2, 10, 64)),
)
assert out.shape == (2, 6, 64), f'Output shape: {out.shape}'
""",
        },
        {
            "name": "Q and KV different lengths",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
attn = {fn}(d_model=32, num_heads=2, rngs=nnx.Rngs(0))
out = attn(
    jax.random.normal(key, (1, 3, 32)),
    jax.random.normal(jax.random.fold_in(key, 1), (1, 20, 32)),
)
assert out.shape == (1, 3, 32), f'Shape: {out.shape}'
""",
        },
        {
            "name": "No causal mask — all KV affects all Q",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
attn = {fn}(d_model=32, num_heads=2, rngs=nnx.Rngs(0))
x_q = jax.random.normal(key, (1, 4, 32))
x_kv = jax.random.normal(jax.random.fold_in(key, 1), (1, 6, 32))
out1 = attn(x_q, x_kv)
x_kv2 = x_kv.copy()
x_kv2 = x_kv2.at[:, -1].set(jax.random.normal(jax.random.fold_in(key, 2), (1, 32)))
out2 = attn(x_q, x_kv2)
assert not jnp.allclose(out1[:, 0], out2[:, 0], atol=1e-5), (
    'Changing last KV should affect all Q positions'
)
""",
        },
        {
            "name": "Gradient flow",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
attn = {fn}(d_model=32, num_heads=2, rngs=nnx.Rngs(0))
x_q = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 32))
x_kv = jax.random.normal(jax.random.PRNGKey(1), (1, 6, 32))

def loss_q(x_q):
    return attn(x_q, x_kv).sum()

def loss_kv(x_kv):
    return attn(x_q, x_kv).sum()

gq = jax.grad(loss_q)(x_q)
gkv = jax.grad(loss_kv)(x_kv)
assert gq is not None and gkv is not None, 'Missing input gradients'
""",
        },
    ],
}
