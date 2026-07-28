"""Linear Self-Attention task."""

TASK = {
    "title": "Linear Self-Attention",
    "difficulty": "Hard",
    "function_name": "linear_attention",
    "hint": "Feature map: phi(x) = elu(x) + 1. Compute phi(Q) @ (phi(K)^T @ V) instead of softmax(Q @ K^T) @ V. Normalize by phi(Q) @ sum(phi(K)).",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
out = {fn}(
    jax.random.normal(k1, (2, 8, 16)),
    jax.random.normal(k2, (2, 8, 16)),
    jax.random.normal(k3, (2, 8, 32)),
)
assert out.shape == (2, 8, 32), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "No NaN or Inf",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
out = {fn}(
    jax.random.normal(k1, (2, 16, 8)),
    jax.random.normal(k2, (2, 16, 8)),
    jax.random.normal(k3, (2, 16, 8)),
)
assert not jnp.isnan(out).any(), 'NaN in output'
assert not jnp.isinf(out).any(), 'Inf in output'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
Q = jax.random.normal(k1, (1, 4, 8))
K = jax.random.normal(k2, (1, 4, 8))
V = jax.random.normal(k3, (1, 4, 8))
gQ, gK, gV = jax.grad(
    lambda q, k, v: {fn}(q, k, v).sum(), argnums=(0, 1, 2)
)(Q, K, V)
assert gQ is not None and gK is not None and gV is not None, 'Missing gradients'
""",
        },
        {
            "name": "Runs fast on long sequences (linear complexity)",
            "code": """
import jax
import jax.numpy as jnp
import time

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
Q = jax.random.normal(k1, (1, 2048, 64))
K = jax.random.normal(k2, (1, 2048, 64))
V = jax.random.normal(k3, (1, 2048, 64))
t0 = time.perf_counter()
for _ in range(10):
    {fn}(Q, K, V)
elapsed = time.perf_counter() - t0
assert elapsed < 5.0, f'Too slow: {elapsed:.2f}s — should be O(S*D^2) not O(S^2*D)'
""",
        },
    ],
}
