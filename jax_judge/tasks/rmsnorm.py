"""RMSNorm implementation task."""

TASK = {
    "title": "Implement RMSNorm",
    "difficulty": "Medium",
    "function_name": "rms_norm",
    "hint": "$\\text{RMS}(x) = \\sqrt{\\text{mean}(x^2) + \\epsilon}$. $\\text{RMSNorm}(x) = \\frac{x}{\\text{RMS}(x)} \\cdot \\text{weight}$. Simpler than LayerNorm — no mean subtraction.",
    "tests": [
        {
            "name": "Basic behavior",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (2, 8))
weight = jnp.ones(8)
out = {fn}(x, weight)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
ref = x / rms * weight
assert jnp.allclose(out, ref, atol=1e-5), 'Value mismatch'
""",
        },
        {
            "name": "With learned weight",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
x = jax.random.normal(k1, (4, 16))
weight = jax.random.normal(k2, (16,))
out = {fn}(x, weight)
rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
ref = x / rms * weight
assert jnp.allclose(out, ref, atol=1e-5), 'Value mismatch with non-trivial weight'
""",
        },
        {
            "name": "3-D input",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (2, 4, 32))
weight = jnp.ones(32)
out = {fn}(x, weight)
assert out.shape == x.shape, f'Shape mismatch on 3-D: {out.shape}'
rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
ref = x / rms * weight
assert jnp.allclose(out, ref, atol=1e-5), 'Value mismatch on 3-D'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (2, 8))
weight = jnp.ones(8)

def loss(x, weight):
    return {fn}(x, weight).sum()

gx = jax.grad(loss, argnums=0)(x, weight)
gw = jax.grad(loss, argnums=1)(x, weight)
assert gx is not None, 'gradient w.r.t. x is None'
assert gw is not None, 'gradient w.r.t. weight is None'
""",
        },
    ],
}
