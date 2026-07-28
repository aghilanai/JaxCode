"""Multi-Head Attention task."""

TASK = {
    "title": "Multi-Head Attention",
    "difficulty": "Hard",
    "function_name": "MultiHeadAttention",
    "hint": (
        "Use nnx.Linear for W_q/W_k/W_v/W_o. K = d_model // num_heads. "
        "Shape suffixes: B=batch, L=query len, M=KV len, D=d_model, H=heads, K=d_k. "
        "Reshape to q_BHLK / k_BHMK, SDPA, concat to _BLD, then W_o. Call as mha(q_BLD, k_BMD, v_BMD)."
    ),
    "tests": [
        {
            "name": "Output shape",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
B, L, D, H = 2, 6, 32, 4
mha = {fn}(d_model=D, num_heads=H, rngs=nnx.Rngs(0))
x_BLD = jax.random.normal(key, (B, L, D))
out_BLD = mha(x_BLD, x_BLD, x_BLD)
assert out_BLD.shape == (B, L, D), f'Shape mismatch: {out_BLD.shape} vs {(B, L, D)}'
""",
        },
        {
            "name": "Uses nnx.Linear with correct shapes",
            "code": """
from flax import nnx
mha = {fn}(d_model=32, num_heads=4, rngs=nnx.Rngs(0))
for name in ['W_q', 'W_k', 'W_v', 'W_o']:
    layer = getattr(mha, name)
    assert isinstance(layer, nnx.Linear), f'{name} should be nnx.Linear, got {type(layer)}'
    assert layer.kernel.value.shape == (32, 32), f'{name}.kernel shape: {layer.kernel.value.shape}'
""",
        },
        {
            "name": "Numerical correctness vs reference",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
import math
key = jax.random.PRNGKey(0)
D, H = 16, 2
K = D // H
mha = {fn}(d_model=D, num_heads=H, rngs=nnx.Rngs(0))
q_BLD = jax.random.normal(key, (1, 4, D))
k_BMD = jax.random.normal(jax.random.fold_in(key, 1), (1, 4, D))
v_BMD = jax.random.normal(jax.random.fold_in(key, 2), (1, 4, D))
out_BLD = mha(q_BLD, k_BMD, v_BMD)
q_BHLK = mha.W_q(q_BLD).reshape(1, 4, H, K).transpose(0, 2, 1, 3)
k_BHMK = mha.W_k(k_BMD).reshape(1, 4, H, K).transpose(0, 2, 1, 3)
v_BHMK = mha.W_v(v_BMD).reshape(1, 4, H, K).transpose(0, 2, 1, 3)
scores_BHLM = jnp.matmul(q_BHLK, jnp.swapaxes(k_BHMK, -2, -1)) / math.sqrt(K)
weights_BHLM = jax.nn.softmax(scores_BHLM, axis=-1)
attn_BHLK = jnp.matmul(weights_BHLM, v_BHMK)
ref_BLD = mha.W_o(attn_BHLK.transpose(0, 2, 1, 3).reshape(1, 4, D))
assert jnp.allclose(out_BLD, ref_BLD, atol=1e-5), 'Output does not match reference'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
mha = {fn}(d_model=16, num_heads=2, rngs=nnx.Rngs(0))
x_BLD = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 16))

def loss(x_BLD):
    return mha(x_BLD, x_BLD, x_BLD).sum()

g_BLD = jax.grad(loss)(x_BLD)
assert g_BLD is not None, 'x gradient is None'

def loss_fn(model):
    x_BLD = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 16))
    return model(x_BLD, x_BLD, x_BLD).sum()

_, grads = nnx.value_and_grad(loss_fn)(mha)
assert grads.W_q.kernel.value is not None, 'W_q.kernel grad is None'
assert grads.W_o.kernel.value is not None, 'W_o.kernel grad is None'
""",
        },
        {
            "name": "Cross-attention (seq_q != seq_k)",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
mha = {fn}(d_model=32, num_heads=4, rngs=nnx.Rngs(0))
q_BLD = jax.random.normal(key, (1, 3, 32))
k_BMD = jax.random.normal(jax.random.fold_in(key, 1), (1, 7, 32))
v_BMD = jax.random.normal(jax.random.fold_in(key, 2), (1, 7, 32))
out_BLD = mha(q_BLD, k_BMD, v_BMD)
assert out_BLD.shape == (1, 3, 32), f'Cross-attention shape: {out_BLD.shape}'
""",
        },
        {
            "name": "Different heads give different outputs",
            "code": """
from flax import nnx
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(42)
D, H = 16, 4
K = D // H
mha = {fn}(d_model=D, num_heads=H, rngs=nnx.Rngs(42))
x_BLD = jax.random.normal(key, (1, 4, D))
q_BHLK = mha.W_q(x_BLD).reshape(1, 4, H, K).transpose(0, 2, 1, 3)
assert not jnp.allclose(q_BHLK[:, 0], q_BHLK[:, 1], atol=1e-3), 'Heads produce identical queries'
""",
        },
    ],
}
