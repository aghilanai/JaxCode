"""LayerNorm implementation task."""

TASK = {
    "title": "Implement LayerNorm",
    "difficulty": "Medium",
    "function_name": "my_layer_norm",
    "hint": "Normalize over the last dim: $(x - \\mu) / \\sqrt{\\sigma^2 + \\epsilon}$, then scale by $\\gamma$ and shift by $\\beta$.",
    "tests": [
        {
            "name": "Shape and basic behavior",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (2, 3, 8))
gamma = jnp.ones(8)
beta = jnp.zeros(8)
out = {fn}(x, gamma, beta)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
mean = jnp.mean(x, axis=-1, keepdims=True)
var = jnp.var(x, axis=-1, keepdims=True)
ref = (x - mean) / jnp.sqrt(var + 1e-5) * gamma + beta
assert jnp.allclose(out, ref, atol=1e-4), 'Value mismatch vs reference layer norm'
""",
        },
        {
            "name": "With learned parameters",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
x = jax.random.normal(k1, (4, 16))
gamma = jax.random.normal(k2, (16,))
beta = jax.random.normal(k3, (16,))
out = {fn}(x, gamma, beta)
mean = jnp.mean(x, axis=-1, keepdims=True)
var = jnp.var(x, axis=-1, keepdims=True)
ref = (x - mean) / jnp.sqrt(var + 1e-5) * gamma + beta
assert jnp.allclose(out, ref, atol=1e-4), 'Value mismatch with non-trivial gamma/beta'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (2, 8))
gamma = jnp.ones(8)
beta = jnp.zeros(8)

def loss(x, gamma, beta):
    return {fn}(x, gamma, beta).sum()

gx = jax.grad(loss, argnums=0)(x, gamma, beta)
gg = jax.grad(loss, argnums=1)(x, gamma, beta)
gb = jax.grad(loss, argnums=2)(x, gamma, beta)
assert gx is not None, 'gradient w.r.t. x is None'
assert gg is not None, 'gradient w.r.t. gamma is None'
""",
        },
    ],
}
