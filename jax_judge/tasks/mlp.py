"""SwiGLU MLP task."""

TASK = {
    "title": "SwiGLU MLP",
    "difficulty": "Medium",
    "function_name": "SwiGLUMLP",
    "hint": (
        "Three nnx.Linear layers: gate_proj(d, d_ff), up_proj(d, d_ff), down_proj(d_ff, d). "
        "__call__(x) = down_proj(silu(gate_proj(x)) * up_proj(x)). SiLU(x) = x * sigmoid(x)."
    ),
    "tests": [
        {
            "name": "Parameter shapes",
            "code": """
from flax import nnx
mlp = {fn}(d_model=64, d_ff=128, rngs=nnx.Rngs(0))
assert isinstance(mlp, nnx.Module), 'Must inherit from nnx.Module'
assert hasattr(mlp, 'gate_proj'), 'Need self.gate_proj'
assert hasattr(mlp, 'up_proj'), 'Need self.up_proj'
assert hasattr(mlp, 'down_proj'), 'Need self.down_proj'
assert mlp.gate_proj.kernel.value.shape == (64, 128), (
    f'gate_proj shape: {mlp.gate_proj.kernel.value.shape}'
)
assert mlp.up_proj.kernel.value.shape == (64, 128), (
    f'up_proj shape: {mlp.up_proj.kernel.value.shape}'
)
assert mlp.down_proj.kernel.value.shape == (128, 64), (
    f'down_proj shape: {mlp.down_proj.kernel.value.shape}'
)
""",
        },
        {
            "name": "Forward output shape",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
mlp = {fn}(d_model=32, d_ff=64, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 32))
out = mlp(x)
assert out.shape == (2, 8, 32), f'Output shape: {out.shape}'
""",
        },
        {
            "name": "Numerical correctness",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
mlp = {fn}(d_model=16, d_ff=32, rngs=nnx.Rngs(0))
x = jax.random.normal(key, (1, 4, 16))
out = mlp(x)
gate = mlp.gate_proj(x)
up = mlp.up_proj(x)
ref = mlp.down_proj(jax.nn.silu(gate) * up)
assert jnp.allclose(out, ref, atol=1e-5), 'Output != down(silu(gate(x)) * up(x))'
""",
        },
        {
            "name": "2-D input",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
mlp = {fn}(d_model=32, d_ff=64, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(0), (4, 32))
out = mlp(x)
assert out.shape == (4, 32), f'2-D output shape: {out.shape}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
mlp = {fn}(d_model=32, d_ff=64, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 32))

def loss(x):
    return mlp(x).sum()

g = jax.grad(loss)(x)
assert g is not None, 'x gradient is None'

def loss_fn(model):
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 32))
    return model(x).sum()

_, grads = nnx.value_and_grad(loss_fn)(mlp)
param_leaves = jax.tree_util.tree_leaves(nnx.state(mlp))
grad_leaves = jax.tree_util.tree_leaves(grads)
assert len(grad_leaves) == len(param_leaves), (
    f'grad/param count mismatch: {len(grad_leaves)} vs {len(param_leaves)}'
)
assert all(g is not None for g in grad_leaves), 'Some params missing gradients'
""",
        },
    ],
}
