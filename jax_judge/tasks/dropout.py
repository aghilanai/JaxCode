"""Implement Dropout task."""

TASK = {
    "title": "Implement Dropout",
    "difficulty": "Easy",
    "function_name": "MyDropout",
    "hint": (
        "Subclass nnx.Module. __call__(x, deterministic=False): when deterministic=True "
        "return x unchanged; otherwise randomly zero elements with probability p and scale "
        "survivors by 1/(1-p). Use rngs for randomness — no train()/eval() modes."
    ),
    "tests": [
        {
            "name": "Deterministic mode is identity",
            "code": """
from flax import nnx
import jax.numpy as jnp
d = {fn}(p=0.5, rngs=nnx.Rngs(42))
assert isinstance(d, nnx.Module), 'Must inherit from nnx.Module'
x = jnp.ones((4, 8))
out = d(x, deterministic=True)
assert jnp.allclose(out, x), 'deterministic=True should return input unchanged'
""",
        },
        {
            "name": "Training: zeros and scaling",
            "code": """
from flax import nnx
import jax.numpy as jnp
d = {fn}(p=0.5, rngs=nnx.Rngs(42))
x = jnp.ones(1000)
out = d(x, deterministic=False)
assert (out == 0).any(), 'No zeros found during dropout'
non_zero = out[out != 0]
assert jnp.allclose(non_zero, jnp.full_like(non_zero, 2.0), atol=1e-5), (
    'Non-zeros should be scaled by 1/(1-p)=2.0'
)
""",
        },
        {
            "name": "Drop rate is approximately p",
            "code": """
from flax import nnx
import jax.numpy as jnp
d = {fn}(p=0.3, rngs=nnx.Rngs(0))
out = d(jnp.ones(10000), deterministic=False)
frac = (out == 0).mean()
assert 0.25 < frac < 0.35, f'Expected ~30% zeros, got {frac*100:.1f}%'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
d = {fn}(p=0.5, rngs=nnx.Rngs(42))
x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
# Grad through deterministic path (identity) — avoids Rng mutation under jax.grad
g = jax.grad(lambda t: d(t, deterministic=True).sum())(x)
assert g is not None, 'x gradient is None'
assert jnp.allclose(g, jnp.ones_like(x)), 'Identity dropout should have unit grad'
""",
        },
    ],
}
