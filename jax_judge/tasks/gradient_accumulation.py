"""Gradient Accumulation task."""

TASK = {
    "title": "Gradient Accumulation",
    "difficulty": "Easy",
    "function_name": "accumulated_step",
    "hint": (
        "Zero grads once. For each micro-batch: forward, loss/n_batches, accumulate grads "
        "via nnx.value_and_grad. Then optimizer.step(). Loss scaling ensures accumulated "
        "grads match a single large batch."
    ),
    "tests": [
        {
            "name": "Matches full batch update",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx


class SGD:
    def __init__(self, model, lr=0.1):
        self.model = model
        self.lr = lr
        self.grads = None

    def zero_grad(self):
        self.grads = None

    def step(self):
        params = nnx.state(self.model, nnx.Param)
        new_params = jax.tree.map(lambda p, g: p - self.lr * g, params, self.grads)
        nnx.update(self.model, new_params)


def mse_loss(pred, y):
    return jnp.mean((pred - y) ** 2)


key = jax.random.PRNGKey(0)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)
model = nnx.Linear(4, 2, use_bias=False, rngs=nnx.Rngs(k1))
model_ref = nnx.Linear(4, 2, use_bias=False, rngs=nnx.Rngs(k1))
nnx.update(model_ref, nnx.state(model))
opt = SGD(model, lr=0.1)
opt_ref = SGD(model_ref, lr=0.1)
x1, y1 = jax.random.normal(k2, (2, 4)), jax.random.normal(k3, (2, 2))
x2, y2 = jax.random.normal(k4, (2, 4)), jax.random.normal(k5, (2, 2))
{fn}(model, opt, mse_loss, [(x1, y1), (x2, y2)])
opt_ref.zero_grad()

def full_batch_loss(m):
    pred = m(jnp.concatenate([x1, x2], axis=0))
    return mse_loss(pred, jnp.concatenate([y1, y2], axis=0))

_, opt_ref.grads = nnx.value_and_grad(full_batch_loss)(model_ref)
opt_ref.step()
params_match = jax.tree.map(
    lambda a, b: jnp.allclose(a, b, atol=1e-5),
    nnx.state(model, nnx.Param),
    nnx.state(model_ref, nnx.Param),
)
assert jax.tree.reduce(lambda x, y: x and y, params_match), 'Must match full batch'
""",
        },
        {
            "name": "Returns loss value",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx


class SGD:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr
        self.grads = None

    def zero_grad(self):
        self.grads = None

    def step(self):
        params = nnx.state(self.model, nnx.Param)
        new_params = jax.tree.map(lambda p, g: p - self.lr * g, params, self.grads)
        nnx.update(self.model, new_params)


model = nnx.Linear(4, 2, rngs=nnx.Rngs(0))
opt = SGD(model, lr=0.01)
loss = {fn}(
    model,
    opt,
    lambda pred, y: jnp.mean((pred - y) ** 2),
    [(jax.random.normal(jax.random.PRNGKey(0), (2, 4)), jax.random.normal(jax.random.PRNGKey(1), (2, 2)))],
)
assert isinstance(loss, float), f'Should return float, got {type(loss)}'
assert loss > 0, 'Loss should be positive'
""",
        },
        {
            "name": "Parameters actually update",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx


class SGD:
    def __init__(self, model, lr=0.1):
        self.model = model
        self.lr = lr
        self.grads = None

    def zero_grad(self):
        self.grads = None

    def step(self):
        params = nnx.state(self.model, nnx.Param)
        new_params = jax.tree.map(lambda p, g: p - self.lr * g, params, self.grads)
        nnx.update(self.model, new_params)


model = nnx.Linear(4, 2, rngs=nnx.Rngs(0))
opt = SGD(model, lr=0.1)
params_before = jax.tree.map(lambda p: jnp.array(p), nnx.state(model, nnx.Param))
{fn}(
    model,
    opt,
    lambda pred, y: jnp.mean((pred - y) ** 2),
    [(jax.random.normal(jax.random.PRNGKey(0), (2, 4)), jax.random.normal(jax.random.PRNGKey(1), (2, 2)))],
)
params_after = jax.tree.map(lambda p: jnp.array(p), nnx.state(model, nnx.Param))
changed = jax.tree.map(lambda a, b: not jnp.array_equal(a, b), params_before, params_after)
assert jax.tree.reduce(lambda x, y: x or y, changed), 'Should change'
""",
        },
    ],
}
