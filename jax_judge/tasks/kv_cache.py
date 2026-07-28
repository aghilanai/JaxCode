"""KV Cache Attention task."""

TASK = {
    "title": "KV Cache Attention",
    "difficulty": "Hard",
    "function_name": "KVCacheAttention",
    "hint": "Project Q/K/V with nnx.Linear, reshape to (B, num_heads, S, d_k). If cache exists, concat new K/V with cached along axis=2. Apply causal mask during prefill. Return (output, (K_all, V_all)). Cache tensors: (B, num_heads, S_total, d_k).",
    "tests": [
        {
            "name": "Output shape (no cache)",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

attn = {fn}(d_model=64, num_heads=4, rngs=nnx.Rngs(0))
assert isinstance(attn, nnx.Module), 'Must inherit from nnx.Module'
x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))
out, cache = attn(x)
assert out.shape == (2, 8, 64), f'Output shape: {out.shape}'
""",
        },
        {
            "name": "Cache structure",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

attn = {fn}(d_model=64, num_heads=4, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))
out, cache = attn(x)
assert isinstance(cache, tuple) and len(cache) == 2, 'Cache must be a (K, V) tuple'
k_cache, v_cache = cache
assert k_cache.shape == (2, 4, 8, 16), f'K cache shape: {k_cache.shape}, expected (2, 4, 8, 16)'
assert v_cache.shape == (2, 4, 8, 16), f'V cache shape: {v_cache.shape}, expected (2, 4, 8, 16)'
""",
        },
        {
            "name": "Decode step appends to cache",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

attn = {fn}(d_model=32, num_heads=2, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 32))
_, cache = attn(x)
new_token = jax.random.normal(jax.random.PRNGKey(1), (1, 1, 32))
out, new_cache = attn(new_token, cache=cache)
assert out.shape == (1, 1, 32), f'Decode output shape: {out.shape}'
k_cache, v_cache = new_cache
assert k_cache.shape[2] == 5, f'Cache should grow: K has {k_cache.shape[2]} positions, expected 5'
assert v_cache.shape[2] == 5, f'Cache should grow: V has {v_cache.shape[2]} positions, expected 5'
""",
        },
        {
            "name": "Incremental decode matches full forward",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

attn = {fn}(d_model=32, num_heads=2, rngs=nnx.Rngs(42))
x = jax.random.normal(jax.random.PRNGKey(42), (1, 6, 32))
full_out, _ = attn(x)
out1, cache = attn(x[:, :4])
out2, cache = attn(x[:, 4:5], cache=cache)
out3, cache = attn(x[:, 5:6], cache=cache)
inc_out = jnp.concatenate([out1, out2, out3], axis=1)
assert jnp.allclose(full_out, inc_out, atol=1e-5), 'Incremental decode must match full forward'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

attn = {fn}(d_model=32, num_heads=2, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 32))

def input_loss(x):
    out, _ = attn(x)
    return out.sum()

g_x = jax.grad(input_loss)(x)
assert g_x is not None, 'input grad is None'

def param_loss(model):
    out, _ = model(x)
    return out.sum()

grads = nnx.value_and_grad(param_loss)(attn)
grad_params = nnx.state(grads, nnx.Param)
n_grad = len(jax.tree_util.tree_leaves(grad_params))
n_total = len(jax.tree_util.tree_leaves(nnx.state(attn, nnx.Param)))
assert n_grad == n_total, f'Only {n_grad}/{n_total} params got gradients'
""",
        },
    ],
}
