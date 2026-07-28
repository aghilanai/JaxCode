"""Adam Optimizer task."""

TASK = {
    "title": "Adam Optimizer",
    "difficulty": "Medium",
    "function_name": "MyAdam",
    "hint": (
        "Track m (1st moment) and v (2nd moment). m = beta1*m + (1-beta1)*grad, "
        "v = beta2*v + (1-beta2)*grad^2. Bias correct: m_hat = m/(1-beta1^t), "
        "v_hat = v/(1-beta2^t). Update: p -= lr * m_hat / (sqrt(v_hat) + eps). "
        "Implement step() and zero_grad(); compare trajectory to optax.adam."
    ),
    "tests": [
        {
            "name": "Parameters change after step",
            "code": """
import jax
import jax.numpy as jnp

class Param:
    def __init__(self, value):
        self.value = jnp.asarray(value)
        self.grad = None

key = jax.random.PRNGKey(0)
p = Param(jax.random.normal(key, (4, 3)))
opt = {fn}([p], lr=0.01)
p.grad = 2 * p.value
before = p.value.copy()
opt.step()
assert not jnp.allclose(p.value, before), 'Should change after step'
""",
        },
        {
            "name": "Matches optax.adam",
            "code": """
import jax
import jax.numpy as jnp
import optax

class Param:
    def __init__(self, value):
        self.value = jnp.asarray(value)
        self.grad = None

key = jax.random.PRNGKey(0)
init = jax.random.normal(key, (8, 4))
p1 = Param(init.copy())
w2 = init.copy()
opt1 = {fn}([p1], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
tx = optax.adam(0.001, b1=0.9, b2=0.999, eps=1e-8)
opt_state = tx.init(w2)
for _ in range(5):
    p1.grad = 2 * p1.value
    opt1.step()
    opt1.zero_grad()
    g2 = 2 * w2
    updates, opt_state = tx.update(g2, opt_state, w2)
    w2 = optax.apply_updates(w2, updates)
assert jnp.allclose(p1.value, w2, atol=1e-5), (
    f'Max diff: {jnp.max(jnp.abs(p1.value - w2)):.6f}'
)
""",
        },
        {
            "name": "zero_grad works",
            "code": """
import jax
import jax.numpy as jnp

class Param:
    def __init__(self, value):
        self.value = jnp.asarray(value)
        self.grad = None

key = jax.random.PRNGKey(0)
p = Param(jax.random.normal(key, (4,)))
opt = {fn}([p], lr=0.01)
p.grad = 2 * p.value
assert jnp.abs(p.grad).sum() > 0
opt.zero_grad()
assert p.grad is None or jnp.abs(p.grad).sum() == 0, (
    'zero_grad should clear all gradients'
)
""",
        },
    ],
}
