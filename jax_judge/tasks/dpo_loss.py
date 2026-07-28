"""DPO (Direct Preference Optimization) Loss task."""

TASK = {
    "title": "DPO (Direct Preference Optimization) Loss",
    "difficulty": "Hard",
    "function_name": "dpo_loss",
    "hint": "L = -log(sigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)))). Mean over batch.",
    "tests": [
        {
            "name": "Easy pair: small loss",
            "code": """
import jax.numpy as jnp

chosen = jnp.array([0.0, 0.0])
rejected = jnp.array([-10.0, -10.0])
ref_c = jnp.array([-1.0, -1.0])
ref_r = jnp.array([-1.0, -1.0])
loss = {fn}(chosen, rejected, ref_c, ref_r, beta=0.1)
assert loss.ndim == 0, 'Must be scalar'
assert float(loss) < 0.5, f'Easy pair loss too high: {float(loss):.4f}'
""",
        },
        {
            "name": "Hard pair: large loss",
            "code": """
import jax.numpy as jnp

loss = {fn}(
    jnp.array([-10.0]),
    jnp.array([0.0]),
    jnp.array([-1.0]),
    jnp.array([-1.0]),
    beta=0.1,
)
assert float(loss) > 0.5, f'Hard pair loss too low: {float(loss):.4f}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3, k4 = jax.random.split(key, 4)
c = jax.random.normal(k1, (4,))
r = jax.random.normal(k2, (4,))
gc, gr = jax.grad(
    lambda ch, rej: {fn}(ch, rej, jax.random.normal(k3, (4,)), jax.random.normal(k4, (4,))),
    argnums=(0, 1),
)(c, r)
assert gc is not None and gr is not None, 'Missing gradients'
""",
        },
        {
            "name": "Mathematical correctness",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3, k4 = jax.random.split(key, 4)
c = jax.random.normal(k1, (3,))
r = jax.random.normal(k2, (3,))
rc = jax.random.normal(k3, (3,))
rr = jax.random.normal(k4, (3,))
beta = 0.5
loss = {fn}(c, r, rc, rr, beta=beta)
ref = -jnp.log(jax.nn.sigmoid(beta * ((c - rc) - (r - rr)))).mean()
assert jnp.allclose(loss, ref, atol=1e-5), f'{float(loss):.6f} vs {float(ref):.6f}'
""",
        },
        {
            "name": "Beta scaling",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3, k4 = jax.random.split(key, 4)
c = jax.random.normal(k1, (4,))
r = jax.random.normal(k2, (4,))
rc = jax.random.normal(k3, (4,))
rr = jax.random.normal(k4, (4,))
l1 = {fn}(c, r, rc, rr, beta=0.1)
l2 = {fn}(c, r, rc, rr, beta=1.0)
assert not jnp.allclose(l1, l2), 'Different beta should give different loss'
""",
        },
    ],
}
