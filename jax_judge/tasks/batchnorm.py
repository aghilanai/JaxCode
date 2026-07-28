"""BatchNorm implementation task."""

TASK = {
    "title": "Implement BatchNorm",
    "difficulty": "Medium",
    "function_name": "my_batch_norm",
    "hint": (
        "Implement train/eval BatchNorm: in training, use batch stats over axis=0 "
        "and update running_mean/running_var with momentum (use list wrappers for "
        "mutable running stats); in inference, normalize using the running statistics only."
    ),
    "tests": [
        {
            "name": "Training mode — zero mean per feature",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (8, 4))
gamma = jnp.ones(4)
beta = jnp.zeros(4)
running_mean = [jnp.zeros(4)]
running_var = [jnp.ones(4)]
out = {fn}(x, gamma, beta, running_mean, running_var, training=True)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
col_means = jnp.mean(out, axis=0)
assert jnp.allclose(col_means, jnp.zeros(4), atol=1e-5), f'Column means not zero: {col_means}'
""",
        },
        {
            "name": "Training mode — numerical correctness and running stats update",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (16, 8))
gamma = jax.random.normal(jax.random.fold_in(key, 1), (8,))
beta = jax.random.normal(jax.random.fold_in(key, 2), (8,))
running_mean = [jnp.zeros(8)]
running_var = [jnp.ones(8)]
momentum = 0.1
out = {fn}(x, gamma, beta, running_mean, running_var, momentum=momentum, training=True)

# Reference using batch stats
mean = jnp.mean(x, axis=0)
var = jnp.var(x, axis=0, ddof=0)
ref = gamma * (x - mean) / jnp.sqrt(var + 1e-5) + beta
assert jnp.allclose(out, ref, atol=1e-4), 'Value mismatch'

# Running stats should have moved toward batch stats
expected_mean = (1 - momentum) * jnp.zeros_like(mean) + momentum * mean
expected_var = (1 - momentum) * jnp.ones_like(var) + momentum * var
assert jnp.allclose(running_mean[0], expected_mean, atol=1e-6), 'running_mean not updated correctly'
assert jnp.allclose(running_var[0], expected_var, atol=1e-6), 'running_var not updated correctly'
""",
        },
        {
            "name": "Inference mode — uses running statistics",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (4, 8))
gamma = jax.random.normal(jax.random.fold_in(key, 1), (8,))
beta = jax.random.normal(jax.random.fold_in(key, 2), (8,))

# Pretend these came from previous training
running_mean = [jax.random.normal(jax.random.fold_in(key, 3), (8,))]
running_var = [jax.random.uniform(jax.random.fold_in(key, 4), (8,)) + 0.5]

rm_copy = jnp.array(running_mean[0])
rv_copy = jnp.array(running_var[0])
out = {fn}(x, gamma, beta, running_mean, running_var, training=False)
ref = gamma * (x - rm_copy) / jnp.sqrt(rv_copy + 1e-5) + beta
assert jnp.allclose(out, ref, atol=1e-4), 'Inference should use running stats'
""",
        },
        {
            "name": "Gradient flow w.r.t inputs and affine params",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
x = jax.random.normal(k1, (4, 8))
gamma = jnp.ones(8)
beta = jnp.zeros(8)
running_mean = [jnp.zeros(8)]
running_var = [jnp.ones(8)]

def loss_fn(x, gamma, beta):
    return {fn}(x, gamma, beta, running_mean, running_var, training=True).sum()

gx, gg, gb = jax.grad(loss_fn, argnums=(0, 1, 2))(x, gamma, beta)
assert gx is not None, 'x gradient is None'
assert gg is not None, 'gamma gradient is None'
assert gb is not None, 'beta gradient is None'
""",
        },
    ],
}
