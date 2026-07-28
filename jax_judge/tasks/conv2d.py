"""2D Convolution task."""

TASK = {
    "title": "2D Convolution",
    "difficulty": "Medium",
    "function_name": "my_conv2d",
    "hint": "Extract patches with nested loops or jnp.pad for zero-padding. For each output position, sum(patch * kernel). Support stride and padding.",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
x = jax.random.normal(k1, (1, 3, 8, 8))
w = jax.random.normal(k2, (16, 3, 3, 3))
out = {fn}(x, w)
assert out.shape == (1, 16, 6, 6), f'Shape: {out.shape}'
""",
        },
        {
            "name": "Matches jax.lax.conv",
            "code": """
import jax
import jax.numpy as jnp
import jax.lax as lax
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
x = jax.random.normal(k1, (2, 3, 8, 8))
w = jax.random.normal(k2, (4, 3, 3, 3))
b = jax.random.normal(k3, (4,))
out = {fn}(x, w, b)
ref = lax.conv_general_dilated(
    x, w, (1, 1), padding='VALID', dimension_numbers=('NCHW', 'OIHW', 'NCHW')
) + b.reshape(1, 4, 1, 1)
assert jnp.allclose(out, ref, atol=1e-4), f'Max diff: {float(jnp.max(jnp.abs(out - ref))):.6f}'
""",
        },
        {
            "name": "With padding",
            "code": """
import jax
import jax.numpy as jnp
import jax.lax as lax
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
x = jax.random.normal(k1, (1, 1, 5, 5))
w = jax.random.normal(k2, (1, 1, 3, 3))
out = {fn}(x, w, padding=1)
ref = lax.conv_general_dilated(
    x, w, (1, 1), padding=((1, 1), (1, 1)), dimension_numbers=('NCHW', 'OIHW', 'NCHW')
)
assert out.shape == ref.shape and jnp.allclose(out, ref, atol=1e-4), 'Padding mismatch'
""",
        },
        {
            "name": "With stride",
            "code": """
import jax
import jax.numpy as jnp
import jax.lax as lax
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
x = jax.random.normal(k1, (1, 1, 8, 8))
w = jax.random.normal(k2, (1, 1, 3, 3))
out = {fn}(x, w, stride=2)
ref = lax.conv_general_dilated(
    x, w, (2, 2), padding='VALID', dimension_numbers=('NCHW', 'OIHW', 'NCHW')
)
assert out.shape == ref.shape and jnp.allclose(out, ref, atol=1e-4), 'Stride mismatch'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
x = jax.random.normal(k1, (1, 1, 4, 4))
w = jax.random.normal(k2, (2, 1, 3, 3))

def loss(x, w):
    return {fn}(x, w).sum()

gx = jax.grad(loss, argnums=0)(x, w)
gw = jax.grad(loss, argnums=1)(x, w)
assert gx is not None and gw is not None, 'Missing gradients'
""",
        },
    ],
}
