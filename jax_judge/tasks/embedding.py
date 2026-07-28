"""Embedding Layer task."""

TASK = {
    "title": "Embedding Layer",
    "difficulty": "Easy",
    "function_name": "MyEmbedding",
    "hint": (
        "Subclass nnx.Module. Store weight matrix (num_embeddings, embedding_dim) as nnx.Param. "
        "__call__(indices) returns weight[indices]. Pass rngs for initialization."
    ),
    "tests": [
        {
            "name": "Weight shape",
            "code": """
from flax import nnx
emb = {fn}(100, 32, rngs=nnx.Rngs(0))
assert isinstance(emb, nnx.Module), 'Must inherit from nnx.Module'
assert hasattr(emb, 'weight'), 'Need self.weight'
assert emb.weight.value.shape == (100, 32), f'Weight shape: {emb.weight.value.shape}'
""",
        },
        {
            "name": "Lookup correctness",
            "code": """
from flax import nnx
import jax.numpy as jnp
emb = {fn}(10, 4, rngs=nnx.Rngs(0))
idx = jnp.array([0, 3, 7])
out = emb(idx)
assert out.shape == (3, 4), f'Output shape: {out.shape}'
assert jnp.array_equal(out[0], emb.weight.value[0]), 'Mismatch at index 0'
assert jnp.array_equal(out[1], emb.weight.value[3]), 'Mismatch at index 3'
""",
        },
        {
            "name": "Batch of indices",
            "code": """
from flax import nnx
import jax.numpy as jnp
emb = {fn}(20, 8, rngs=nnx.Rngs(0))
idx = jnp.array([[1, 2], [3, 4]])
out = emb(idx)
assert out.shape == (2, 2, 8), f'Batch output shape: {out.shape}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
from flax import nnx
import jax.numpy as jnp
emb = {fn}(10, 4, rngs=nnx.Rngs(0))
idx = jnp.array([2, 5])

def loss_fn(model):
    return model(idx).sum()

_, grads = nnx.value_and_grad(loss_fn)(emb)
wgrad = grads.weight.value
assert wgrad is not None, 'weight grad is None'
assert jnp.abs(wgrad[2]).sum() > 0, 'Grad at used index should be non-zero'
assert jnp.abs(wgrad[0]).sum() == 0, 'Grad at unused index should be zero'
""",
        },
    ],
}
