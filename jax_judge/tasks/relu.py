"""ReLU implementation task."""

TASK = {
    "title": "Implement ReLU",
    "difficulty": "Easy",
    "function_name": "relu",
    "hint": "ReLU(x) = max(0, x). Think about element-wise comparison with zero.",
    "tests": [
        {
            "name": "Basic values",
            "code": """
import jax.numpy as jnp
x = jnp.array([-2., -1., 0., 1., 2.])
out = {fn}(x)
expected = jnp.array([0., 0., 0., 1., 2.])
assert out.shape == expected.shape, f'Shape mismatch: {out.shape} vs {expected.shape}'
assert jnp.allclose(out, expected), f'Wrong Answer: {out} vs {expected}'
""",
        },
        {
            "name": "2-D tensor",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (4, 8))
out = {fn}(x)
assert out.shape == x.shape, f'Shape mismatch on 2-D input'
assert (out >= 0).all(), 'ReLU output must be non-negative'
assert jnp.allclose(out, jnp.maximum(x, 0)), 'Value mismatch on random input'
""",
        },
        {
            "name": "Gradient check",
            "code": """
import jax
import jax.numpy as jnp
x = jnp.array([-1., 0., 1., 2.])
g = jax.grad(lambda t: {fn}(t).sum())(x)
assert g[0] == 0., f'grad at x=-1 should be 0, got {g[0]}'
assert g[2] == 1., f'grad at x=1 should be 1, got {g[2]}'
assert g[3] == 1., f'grad at x=2 should be 1, got {g[3]}'
assert g[1] in (0., 1.), f'grad at x=0 should be 0 or 1, got {g[1]}'
""",
        },
        {
            "name": "Performance",
            "code": """
import jax
import jax.numpy as jnp
import time
key = jax.random.PRNGKey(0)
big = jax.random.normal(key, (1024, 1024))
t0 = time.perf_counter()
for _ in range(100):
    {fn}(big)
elapsed = time.perf_counter() - t0
assert elapsed < 5.0, f'Too slow: {elapsed:.2f}s for 100 iterations'
""",
        },
    ],
}
