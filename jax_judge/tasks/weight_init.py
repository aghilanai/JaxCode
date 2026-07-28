"""Kaiming Initialization task."""

TASK = {
    "title": "Kaiming Initialization",
    "difficulty": "Easy",
    "function_name": "kaiming_init",
    "hint": "For fan_in mode: std = sqrt(2 / fan_in) where fan_in = shape[-1] if len(shape) >= 2 else shape[0]. Fill with jax.random.normal(key, shape) * std. Return the array.",
    "tests": [
        {
            "name": "Mean approximately 0",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
w = {fn}((256, 512), key)
assert abs(float(jnp.mean(w))) < 0.02, f'Mean too far from 0: {float(jnp.mean(w)):.4f}'
""",
        },
        {
            "name": "Std matches sqrt(2/fan_in)",
            "code": """
import jax
import jax.numpy as jnp
import math
key = jax.random.PRNGKey(0)
fan_in = 1024
w = {fn}((256, fan_in), key)
expected = math.sqrt(2.0 / fan_in)
assert abs(float(jnp.std(w)) - expected) < 0.005, f'Std {float(jnp.std(w)):.4f} vs expected {expected:.4f}'
""",
        },
        {
            "name": "Returns array with correct shape",
            "code": """
import jax
key = jax.random.PRNGKey(0)
out = {fn}((64, 32), key)
assert out.shape == (64, 32), 'Shape should be (64, 32)'
""",
        },
        {
            "name": "Smaller fan_in gives larger std",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
w1 = {fn}((64, 16), k1)
w2 = {fn}((64, 256), k2)
assert float(jnp.std(w1)) > float(jnp.std(w2)), 'Smaller fan_in should give larger std'
""",
        },
    ],
}
