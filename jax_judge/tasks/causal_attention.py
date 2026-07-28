"""Causal Self-Attention task."""

TASK = {
    "title": "Causal Self-Attention",
    "difficulty": "Hard",
    "function_name": "causal_attention",
    "hint": "Same as softmax attention but mask future positions with -inf before softmax. `jnp.triu(..., k=1)` gives the upper triangle.",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
out = {fn}(
    jax.random.normal(k1, (2, 6, 16)),
    jax.random.normal(k2, (2, 6, 16)),
    jax.random.normal(k3, (2, 6, 16)),
)
assert out.shape == (2, 6, 16), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Future tokens don't affect past",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)
B, S, D = 1, 8, 16
Q = jax.random.normal(k1, (B, S, D))
K = jax.random.normal(k2, (B, S, D))
V = jax.random.normal(k3, (B, S, D))
out1 = {fn}(Q, K, V)
K2 = K.at[:, 4:].set(jax.random.normal(k4, (B, 4, D)))
V2 = V.at[:, 4:].set(jax.random.normal(k5, (B, 4, D)))
out2 = {fn}(Q, K2, V2)
assert jnp.allclose(out1[:, :4], out2[:, :4], atol=1e-5), 'Changing future K/V affected past outputs'
""",
        },
        {
            "name": "First position only sees itself",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
Q = jax.random.normal(k1, (1, 4, 8))
K = jax.random.normal(k2, (1, 4, 8))
V = jax.random.normal(k3, (1, 4, 8))
out = {fn}(Q, K, V)
assert jnp.allclose(out[:, 0], V[:, 0], atol=1e-5), 'Position 0 should output V[0]'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
Q = jax.random.normal(k1, (2, 4, 8))
K = jax.random.normal(k2, (2, 4, 8))
V = jax.random.normal(k3, (2, 4, 8))
gQ, gK, gV = jax.grad(
    lambda q, k, v: {fn}(q, k, v).sum(), argnums=(0, 1, 2)
)(Q, K, V)
assert gQ is not None and gK is not None and gV is not None, 'Missing gradients'
""",
        },
    ],
}
