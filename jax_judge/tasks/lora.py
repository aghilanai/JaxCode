"""LoRA (Low-Rank Adaptation) task."""

TASK = {
    "title": "LoRA (Low-Rank Adaptation)",
    "difficulty": "Medium",
    "function_name": "LoRALinear",
    "hint": (
        "Freeze base nnx.Linear. Add lora_A (rank, in) and lora_B (out, rank) as nnx.Param; "
        "initialize B to zeros. output = linear(x) + (x @ A.T @ B.T) * (alpha/rank)."
    ),
    "tests": [
        {
            "name": "Base weights frozen",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
layer = {fn}(in_features=16, out_features=8, rank=4, rngs=nnx.Rngs(0))
assert isinstance(layer, nnx.Module)
# Base linear should not receive gradients (frozen / stop_gradient)
def loss_fn(model):
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16))
    return model(x).sum()

_, grads = nnx.value_and_grad(loss_fn)(layer)
base_grad = grads.linear.kernel.value
assert base_grad is None or jnp.allclose(base_grad, 0), 'Base weight must be frozen'
""",
        },
        {
            "name": "LoRA parameter shapes",
            "code": """
from flax import nnx
layer = {fn}(in_features=16, out_features=8, rank=4, rngs=nnx.Rngs(0))
assert layer.lora_A.value.shape == (4, 16), f'lora_A: {layer.lora_A.value.shape}'
assert layer.lora_B.value.shape == (8, 4), f'lora_B: {layer.lora_B.value.shape}'
""",
        },
        {
            "name": "B=0 means output equals base",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
layer = {fn}(in_features=8, out_features=4, rank=2, rngs=nnx.Rngs(0))
x = jax.random.normal(key, (2, 8))
assert jnp.allclose(layer(x), layer.linear(x), atol=1e-5), (
    'With B=0, should equal base linear'
)
""",
        },
        {
            "name": "Only LoRA params get gradients",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
layer = {fn}(in_features=8, out_features=4, rank=2, rngs=nnx.Rngs(0))
# B starts at 0 so A would get zero grad; set B nonzero first
layer.lora_B.value = jax.random.normal(jax.random.PRNGKey(1), layer.lora_B.value.shape)

def loss_fn(model):
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 8))
    return model(x).sum()

_, grads = nnx.value_and_grad(loss_fn)(layer)
assert grads.lora_A.value is not None, 'lora_A grad is None'
assert grads.lora_B.value is not None, 'lora_B grad is None'
assert jnp.abs(grads.lora_A.value).sum() > 0, 'lora_A should have non-zero grad'
assert jnp.abs(grads.lora_B.value).sum() > 0, 'lora_B should have non-zero grad'
""",
        },
        {
            "name": "Forward computation",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
layer = {fn}(in_features=8, out_features=4, rank=2, alpha=2.0, rngs=nnx.Rngs(0))
layer.lora_B.value = jax.random.normal(jax.random.fold_in(key, 1), layer.lora_B.value.shape)
x = jax.random.normal(key, (3, 8))
ref = layer.linear(x) + (x @ layer.lora_A.value.T @ layer.lora_B.value.T) * (2.0 / 2)
assert jnp.allclose(layer(x), ref, atol=1e-5), 'Forward mismatch'
""",
        },
    ],
}
