"""Simple Linear Layer task."""

TASK = {
    "title": "Simple Linear Layer",
    "difficulty": "Medium",
    "function_name": "SimpleLinear",
    "hint": (
        "Subclass flax.nnx.Module. Store weight (out, in) and bias as nnx.Param. "
        "__call__(x) = x @ W.T + b. Kaiming init: randn * (1/sqrt(in_features)). "
        "Pass rngs=nnx.Rngs(0) for random initialization."
    ),
    "tests": [
        {
            "name": "Weight & bias shape",
            "code": """
from flax import nnx
layer = {fn}(8, 4, rngs=nnx.Rngs(0))
assert isinstance(layer, nnx.Module), 'Must inherit from nnx.Module'
assert layer.weight.value.shape == (4, 8), f'Weight shape: {layer.weight.value.shape}'
assert layer.bias.value.shape == (4,), f'Bias shape: {layer.bias.value.shape}'
""",
        },
        {
            "name": "Forward pass",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
layer = {fn}(8, 4, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(1), (2, 8))
y = layer(x)
assert y.shape == (2, 4), f'Output shape: {y.shape}'
expected = x @ layer.weight.value.T + layer.bias.value
assert jnp.allclose(y, expected, atol=1e-5), 'Forward != x @ W^T + b'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
layer = {fn}(8, 4, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(1), (2, 8))

def loss(x):
    return layer(x).sum()

g = jax.grad(loss)(x)
assert g is not None, 'Input gradient is None'

def loss_fn(model):
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 8))
    return model(x).sum()

_, grads = nnx.value_and_grad(loss_fn)(layer)
assert grads.weight.value is not None, 'weight grad is None'
assert grads.bias.value is not None, 'bias grad is None'
""",
        },
    ],
}
