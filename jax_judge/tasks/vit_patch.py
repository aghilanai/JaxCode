"""ViT Patch Embedding task."""

TASK = {
    "title": "ViT Patch Embedding",
    "difficulty": "Medium",
    "function_name": "PatchEmbedding",
    "hint": "Reshape image into patches: (B, C, H, W) -> (B, num_patches, C*P*P). Then project with nnx.Linear(C*P*P, embed_dim). num_patches = (img_size/patch_size)^2.",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(img_size=32, patch_size=8, in_channels=3, embed_dim=64, rngs=nnx.Rngs(0))
assert isinstance(pe, nnx.Module), 'Must inherit from nnx.Module'
out = pe(jax.random.normal(jax.random.PRNGKey(0), (2, 3, 32, 32)))
assert out.shape == (2, 16, 64), f'Shape: {out.shape}, expected (2, 16, 64)'
""",
        },
        {
            "name": "num_patches attribute",
            "code": """
from flax import nnx

pe = {fn}(img_size=224, patch_size=16, in_channels=3, embed_dim=768, rngs=nnx.Rngs(0))
assert pe.num_patches == 196, f'num_patches: {pe.num_patches}'
""",
        },
        {
            "name": "Different image sizes",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(img_size=64, patch_size=16, in_channels=1, embed_dim=32, rngs=nnx.Rngs(0))
out = pe(jax.random.normal(jax.random.PRNGKey(0), (1, 1, 64, 64)))
assert out.shape == (1, 16, 32), f'Shape: {out.shape}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(img_size=32, patch_size=8, in_channels=3, embed_dim=64, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(0), (1, 3, 32, 32))

def loss(x):
    return pe(x).sum()

g = jax.grad(loss)(x)
assert g is not None, 'input grad is None'
""",
        },
    ],
}
