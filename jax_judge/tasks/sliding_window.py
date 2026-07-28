"""Sliding Window Attention task."""

TASK = {
    "title": "Sliding Window Attention",
    "difficulty": "Hard",
    "function_name": "sliding_window_attention",
    "hint": "Like softmax attention but position i only attends to positions j where |i-j| <= window_size. Mask the rest with -inf.",
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
    jax.random.normal(k3, (2, 8, 16)),
    window_size=2,
)
assert out.shape == (2, 8, 16), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "window_size=0 — only sees itself",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
Q = jax.random.normal(k1, (1, 4, 8))
K = jax.random.normal(k2, (1, 4, 8))
V = jax.random.normal(k3, (1, 4, 8))
out = {fn}(Q, K, V, window_size=0)
assert jnp.allclose(out, V, atol=1e-5), 'window=0: each position should output V[i]'
""",
        },
        {
            "name": "Large window equals full attention",
            "code": """
import jax
import jax.numpy as jnp
import math

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
B, S, D = 2, 6, 8
Q = jax.random.normal(k1, (B, S, D))
K = jax.random.normal(k2, (B, S, D))
V = jax.random.normal(k3, (B, S, D))
out_win = {fn}(Q, K, V, window_size=S)
d_k = K.shape[-1]
scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(d_k)
ref = jnp.matmul(jax.nn.softmax(scores, axis=-1), V)
assert jnp.allclose(out_win, ref, atol=1e-5), 'Large window should equal full attention'
""",
        },
        {
            "name": "Distant tokens don't affect output",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)
B, S, D = 1, 10, 8
Q = jax.random.normal(k1, (B, S, D))
K = jax.random.normal(k2, (B, S, D))
V = jax.random.normal(k3, (B, S, D))
out1 = {fn}(Q, K, V, window_size=1)
K2 = K.at[:, 5:].set(jax.random.normal(k4, (B, 5, D)))
V2 = V.at[:, 5:].set(jax.random.normal(k5, (B, 5, D)))
out2 = {fn}(Q, K2, V2, window_size=1)
assert jnp.allclose(out1[:, 0], out2[:, 0], atol=1e-5), 'Distant tokens should not affect output'
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
gQ = jax.grad(lambda q: {fn}(q, K, V, window_size=1).sum())(Q)
assert gQ is not None, 'Q gradient is None'
""",
        },
    ],
}
