"""Flash Attention (Tiled) task."""

TASK = {
    "title": "Flash Attention (Tiled)",
    "difficulty": "Hard",
    "function_name": "flash_attention",
    "hint": "Process Q in blocks. For each Q-block, iterate over K/V blocks. Use online softmax: track running max and sum, rescale accumulator when max changes. output = acc / row_sum.",
    "tests": [
        {
            "name": "Matches standard attention",
            "code": """
import jax
import jax.numpy as jnp
import math

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
B, S, D = 2, 16, 8
Q = jax.random.normal(k1, (B, S, D))
K = jax.random.normal(k2, (B, S, D))
V = jax.random.normal(k3, (B, S, D))
out = {fn}(Q, K, V, block_size=4)
scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(D)
ref = jnp.matmul(jax.nn.softmax(scores, axis=-1), V)
assert jnp.allclose(out, ref, atol=1e-4), f'Max diff: {jnp.max(jnp.abs(out - ref)):.6f}'
""",
        },
        {
            "name": "Non-aligned block size",
            "code": """
import jax
import jax.numpy as jnp
import math

key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)
B, S, D = 1, 7, 4
Q = jax.random.normal(k1, (B, S, D))
K = jax.random.normal(k2, (B, S, D))
V = jax.random.normal(k3, (B, S, D))
out = {fn}(Q, K, V, block_size=3)
scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(D)
ref = jnp.matmul(jax.nn.softmax(scores, axis=-1), V)
assert jnp.allclose(out, ref, atol=1e-4), 'Mismatch with non-aligned block size'
""",
        },
        {
            "name": "Block size invariant",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
Q = jax.random.normal(k1, (1, 12, 8))
K = jax.random.normal(k2, (1, 12, 8))
V = jax.random.normal(k3, (1, 12, 8))
out4 = {fn}(Q, K, V, block_size=4)
out6 = {fn}(Q, K, V, block_size=6)
assert jnp.allclose(out4, out6, atol=1e-4), 'Different block sizes should give same result'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
Q = jax.random.normal(k1, (1, 8, 4))
K = jax.random.normal(k2, (1, 8, 4))
V = jax.random.normal(k3, (1, 8, 4))
gQ = jax.grad(lambda q: {fn}(q, K, V, block_size=4).sum())(Q)
assert gQ is not None, 'Q gradient is None'
""",
        },
    ],
}
