"""Softmax implementation task."""

TASK = {
    "title": "Implement Softmax",
    "difficulty": "Easy",
    "function_name": "my_softmax",
    "hint": "softmax(x)_i = exp(x_i) / sum(exp(x_j)). Subtract max(x) first for numerical stability.",
    "tests": [
        {
            "name": "Basic 1-D",
            "code": """
import jax.numpy as jnp
import jax.nn as nn
x = jnp.array([1.0, 2.0, 3.0])
out = {fn}(x, dim=-1)
expected = nn.softmax(x, axis=-1)
assert jnp.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'
""",
        },
        {
            "name": "2-D along dim=-1",
            "code": """
import jax
import jax.numpy as jnp
import jax.nn as nn
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (4, 8))
out = {fn}(x, dim=-1)
expected = nn.softmax(x, axis=-1)
assert out.shape == expected.shape, f'Shape mismatch'
assert jnp.allclose(out, expected, atol=1e-5), 'Values differ'
assert jnp.allclose(out.sum(axis=-1), jnp.ones(4), atol=1e-5), 'Rows must sum to 1'
""",
        },
        {
            "name": "Numerical stability",
            "code": """
import jax.numpy as jnp
import jax.nn as nn
x = jnp.array([1000., 1001., 1002.])
out = {fn}(x, dim=-1)
assert not jnp.isnan(out).any(), 'NaN in output — not numerically stable'
assert not jnp.isinf(out).any(), 'Inf in output — not numerically stable'
expected = nn.softmax(x, axis=-1)
assert jnp.allclose(out, expected, atol=1e-5), 'Values differ on large input'
""",
        },
    ],
}
