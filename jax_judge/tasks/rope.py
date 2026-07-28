"""Rotary Position Embedding (RoPE) task."""

TASK = {
    "title": "Rotary Position Embedding (RoPE)",
    "difficulty": "Hard",
    "function_name": "apply_rope",
    "hint": "Split into pairs $(x_{\\text{even}}, x_{\\text{odd}})$. Compute $\\theta = \\text{pos} \\cdot 1/(10000^{2i/d})$. Rotate: $[x_e\\cos\\theta - x_o\\sin\\theta, x_e\\sin\\theta + x_o\\cos\\theta]$. Stack and flatten.",
    "tests": [
        {
            "name": "Output shapes",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
q = jax.random.normal(k1, (2, 8, 64))
k = jax.random.normal(k2, (2, 8, 64))
q_rot, k_rot = {fn}(q, k)
assert q_rot.shape == q.shape, f'Q shape: {q_rot.shape}'
assert k_rot.shape == k.shape, f'K shape: {k_rot.shape}'
""",
        },
        {
            "name": "Preserves norm",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
q = jax.random.normal(k1, (1, 16, 32))
k = jax.random.normal(k2, (1, 16, 32))
q_rot, k_rot = {fn}(q, k)
assert jnp.allclose(jnp.linalg.norm(q, axis=-1), jnp.linalg.norm(q_rot, axis=-1), atol=1e-4), 'RoPE should preserve norms'
""",
        },
        {
            "name": "Relative position property",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
q = jax.random.normal(k1, (1, 8, 16))
k = jax.random.normal(k2, (1, 8, 16))
q_rot, k_rot = {fn}(q, k)
q2 = jnp.concatenate([jnp.zeros((1, 3, 16)), q], axis=1)
k2 = jnp.concatenate([jnp.zeros((1, 3, 16)), k], axis=1)
q2_rot, k2_rot = {fn}(q2, k2)
dot1 = jnp.sum(q_rot[:, 0] * k_rot[:, 0], axis=-1)
dot2 = jnp.sum(q2_rot[:, 3] * k2_rot[:, 3], axis=-1)
assert jnp.allclose(dot1, dot2, atol=1e-4), 'Dot product should depend on relative position only'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
q = jax.random.normal(k1, (1, 4, 8))
k = jax.random.normal(k2, (1, 4, 8))

def loss(q, k):
    qr, kr = {fn}(q, k)
    return qr.sum() + kr.sum()

gq = jax.grad(loss, argnums=0)(q, k)
gk = jax.grad(loss, argnums=1)(q, k)
assert gq is not None and gk is not None, 'Missing gradients'
""",
        },
    ],
}
