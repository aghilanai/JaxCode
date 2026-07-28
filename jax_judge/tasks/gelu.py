"""GELU Activation task."""

TASK = {
    "title": "GELU Activation",
    "difficulty": "Easy",
    "function_name": "my_gelu",
    "hint": "Exact: $x \\cdot 0.5 \\cdot (1 + \\text{erf}(x / \\sqrt{2}))$. Or approximate: $0.5x(1+\\tanh(\\sqrt{2/\\pi}(x+0.044715x^3)))$.",
    "tests": [
        {
            "name": "Matches jax.nn.gelu",
            "code": """
import jax
import jax.numpy as jnp
import jax.nn as nn
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (4, 8))
out = {fn}(x)
ref = nn.gelu(x)
assert jnp.allclose(out, ref, atol=1e-4), 'Does not match jax.nn.gelu'
""",
        },
        {
            "name": "gelu(0) = 0",
            "code": """
import jax.numpy as jnp
out = {fn}(jnp.array([0.0]))
assert jnp.allclose(out, jnp.array([0.0]), atol=1e-7), f'gelu(0) = {out[0]}'
""",
        },
        {
            "name": "Shape preservation",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (2, 3, 4))
assert {fn}(x).shape == x.shape, 'Shape mismatch'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (4, 8))
g = jax.grad(lambda t: {fn}(t).sum())(x)
assert g is not None and g.shape == x.shape, 'Gradient issue'
""",
        },
    ],
}
