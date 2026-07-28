"""Softmax Attention task."""

TASK = {
    "title": "Softmax Attention",
    "difficulty": "Hard",
    "function_name": "scaled_dot_product_attention",
    "hint": (
        "Shape suffixes: B=batch, L=query len, M=KV len, K=head dim. "
        "scores_BLM = (q_BLK @ k_BMK^T) / sqrt(K), then softmax(scores_BLM) @ v_BMK."
    ),
    "tests": [
        {
            "name": "Output shape",
            "code": """
import jax
import jax.numpy as jnp
import math

key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)
B, S, D = 2, 4, 8
Q = jax.random.normal(k1, (B, S, D))
K = jax.random.normal(k2, (B, S, D))
V = jax.random.normal(k3, (B, S, D))
out = {fn}(Q, K, V)
assert out.shape == (B, S, D), f'Shape mismatch: {out.shape} vs {(B, S, D)}'
""",
        },
        {
            "name": "Numerical correctness",
            "code": """
import jax
import jax.numpy as jnp
import math

key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)
B, S, D = 2, 4, 8
Q = jax.random.normal(k1, (B, S, D))
K = jax.random.normal(k2, (B, S, D))
V = jax.random.normal(k3, (B, S, D))
out = {fn}(Q, K, V)
scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(D)
weights = jax.nn.softmax(scores, axis=-1)
ref = jnp.matmul(weights, V)
assert jnp.allclose(out, ref, atol=1e-5), 'Value mismatch vs reference'
""",
        },
        {
            "name": "Gradient check",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)
Q = jax.random.normal(k1, (2, 4, 8))
K = jax.random.normal(k2, (2, 4, 8))
V = jax.random.normal(k3, (2, 4, 8))
gQ, gK, gV = jax.grad(
    lambda q, k, v: {fn}(q, k, v).sum(), argnums=(0, 1, 2)
)(Q, K, V)
assert gQ is not None, 'Q gradient is None'
assert gK is not None, 'K gradient is None'
assert gV is not None, 'V gradient is None'
""",
        },
        {
            "name": "Cross-attention (seq_q != seq_k)",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
Q = jax.random.normal(k1, (1, 3, 16))
K = jax.random.normal(k2, (1, 5, 16))
V = jax.random.normal(k3, (1, 5, 32))
out = {fn}(Q, K, V)
assert out.shape == (1, 3, 32), f'Cross-attention shape: {out.shape}'
""",
        },
    ],
}
