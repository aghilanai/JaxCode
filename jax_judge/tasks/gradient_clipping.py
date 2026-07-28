"""Gradient Norm Clipping task."""

TASK = {
    "title": "Gradient Norm Clipping",
    "difficulty": "Easy",
    "function_name": "clip_grad_norm",
    "hint": "parameters is a list of gradient arrays. Total norm = sqrt(sum(||g||^2)). If total > max_norm, scale all gradients by max_norm/total (update list elements in place). Return original total norm.",
    "tests": [
        {
            "name": "Clips to max_norm",
            "code": """
import jax.numpy as jnp
grads = [jnp.ones(10) * 10.0, jnp.ones(10) * 10.0]
{fn}(grads, max_norm=1.0)
new_norm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads)))
assert new_norm <= 1.0 + 1e-5, f'Clipped norm {new_norm:.4f} > 1.0'
""",
        },
        {
            "name": "Returns original norm",
            "code": """
import jax.numpy as jnp
g = jnp.ones(10) * 3.0
grads = [g]
expected = float(jnp.linalg.norm(g))
returned = {fn}(grads, max_norm=100.0)
assert abs(returned - expected) < 1e-4, f'Returned {returned:.4f}, expected {expected:.4f}'
""",
        },
        {
            "name": "No change when norm < max_norm",
            "code": """
import jax.numpy as jnp
g = jnp.ones(4) * 0.001
grads = [g]
grad_before = grads[0]
{fn}(grads, max_norm=100.0)
assert jnp.array_equal(grads[0], grad_before), 'Should not change when norm < max_norm'
""",
        },
        {
            "name": "Preserves direction",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
g = jax.random.normal(key, (100,)) * 10.0
grads = [g]
dir_before = grads[0] / jnp.linalg.norm(grads[0])
{fn}(grads, max_norm=1.0)
dir_after = grads[0] / jnp.linalg.norm(grads[0])
assert jnp.allclose(dir_before, dir_after, atol=1e-5), 'Should preserve direction'
""",
        },
    ],
}
