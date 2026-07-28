"""Mixture of Experts (MoE) task."""

TASK = {
    "title": "Mixture of Experts (MoE)",
    "difficulty": "Hard",
    "function_name": "MixtureOfExperts",
    "hint": (
        "Router: nnx.Linear(d, num_experts) -> top-k -> softmax. Each expert: "
        "Linear -> ReLU -> Linear. Weighted sum of top-k expert outputs per token."
    ),
    "tests": [
        {
            "name": "Output shape",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
moe = {fn}(d_model=32, d_ff=64, num_experts=4, top_k=2, rngs=nnx.Rngs(0))
assert isinstance(moe, nnx.Module)
out = moe(jax.random.normal(key, (2, 8, 32)))
assert out.shape == (2, 8, 32), f'Shape: {out.shape}'
""",
        },
        {
            "name": "Has router and experts",
            "code": """
from flax import nnx
moe = {fn}(d_model=32, d_ff=64, num_experts=4, top_k=2, rngs=nnx.Rngs(0))
assert hasattr(moe, 'router'), 'Need self.router'
assert hasattr(moe, 'experts'), 'Need self.experts'
assert len(moe.experts) == 4, f'Expected 4 experts, got {len(moe.experts)}'
""",
        },
        {
            "name": "Router logits shape",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
moe = {fn}(d_model=16, d_ff=32, num_experts=8, top_k=2, rngs=nnx.Rngs(0))
logits = moe.router(jax.random.normal(jax.random.PRNGKey(0), (4, 16)))
assert logits.shape == (4, 8), f'Router output: {logits.shape}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
moe = {fn}(d_model=16, d_ff=32, num_experts=4, top_k=2, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 16))

def loss(x):
    return moe(x).sum()

g = jax.grad(loss)(x)
assert g is not None, 'x gradient is None'
""",
        },
    ],
}
