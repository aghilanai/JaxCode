"""GPT-2 Transformer Block task."""

TASK = {
    "title": "GPT-2 Transformer Block",
    "difficulty": "Hard",
    "function_name": "GPT2Block",
    "hint": (
        "Pre-norm: x = x + attn(ln1(x)), x = x + mlp(ln2(x)). MLP: Linear(d, 4d) -> GELU -> "
        "Linear(4d, d). Attention must be causal. Subclass nnx.Module; use nnx.LayerNorm."
    ),
    "tests": [
        {
            "name": "Output shape",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
block = {fn}(d_model=64, num_heads=4, rngs=nnx.Rngs(0))
assert isinstance(block, nnx.Module), 'GPT2Block should inherit from nnx.Module'
out = block(jax.random.normal(key, (2, 8, 64)))
assert out.shape == (2, 8, 64), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Has LayerNorm (pre-norm architecture)",
            "code": """
from flax import nnx
block = {fn}(d_model=32, num_heads=4, rngs=nnx.Rngs(0))
assert hasattr(block, 'ln1') and isinstance(block.ln1, nnx.LayerNorm), (
    'Need self.ln1 = nnx.LayerNorm'
)
assert hasattr(block, 'ln2') and isinstance(block.ln2, nnx.LayerNorm), (
    'Need self.ln2 = nnx.LayerNorm'
)
""",
        },
        {
            "name": "MLP has 4x expansion with GELU",
            "code": """
from flax import nnx
block = {fn}(d_model=32, num_heads=4, rngs=nnx.Rngs(0))
assert hasattr(block, 'mlp'), 'Need self.mlp'
linears = [m for _, m in nnx.iter_modules(block.mlp) if isinstance(m, nnx.Linear)]
assert len(linears) >= 2, f'MLP needs >= 2 Linear layers, got {len(linears)}'
assert linears[0].kernel.value.shape == (32, 128), (
    f'MLP first layer: {linears[0].kernel.value.shape}, expected (32, 128)'
)
assert linears[-1].kernel.value.shape == (128, 32), (
    f'MLP last layer: {linears[-1].kernel.value.shape}, expected (128, 32)'
)
""",
        },
        {
            "name": "Causal masking — future doesn't affect past",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
block = {fn}(d_model=32, num_heads=4, rngs=nnx.Rngs(0))
x = jax.random.normal(key, (1, 8, 32))
out1 = block(x)
x2 = x.copy()
x2 = x2.at[:, 4:].set(jax.random.normal(jax.random.fold_in(key, 1), (1, 4, 32)))
out2 = block(x2)
assert jnp.allclose(out1[:, :4], out2[:, :4], atol=1e-5), (
    'Future tokens affected past — not causal'
)
""",
        },
        {
            "name": "Gradient flow to all parameters",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
block = {fn}(d_model=32, num_heads=4, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 32))

def loss(x):
    return block(x).sum()

g = jax.grad(loss)(x)
assert g is not None, 'x gradient is None'

def loss_fn(model):
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 32))
    return model(x).sum()

_, grads = nnx.value_and_grad(loss_fn)(block)
param_leaves = jax.tree_util.tree_leaves(nnx.state(block))
grad_leaves = jax.tree_util.tree_leaves(grads)
assert len(grad_leaves) == len(param_leaves), (
    f'grad/param count mismatch: {len(grad_leaves)} vs {len(param_leaves)}'
)
assert all(g is not None for g in grad_leaves), 'Some params missing gradients'
""",
        },
    ],
}
