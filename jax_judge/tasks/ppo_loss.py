"""PPO (Proximal Policy Optimization) clipped loss task."""

TASK = {
    "title": "PPO (Proximal Policy Optimization) Clipped Loss",
    "difficulty": "Hard",
    "function_name": "ppo_loss",
    "hint": (
        "Compute ratio r = exp(new_logps - old_logps_detached). "
        "Form unclipped = r * adv_detached and clipped = clip(r, 1-clip, 1+clip) * adv_detached. "
        "Return the negative mean of min(unclipped, clipped). "
        "Use `jax.lax.stop_gradient` on old_logps and advantages so gradients flow only through new_logps."
    ),
    "tests": [
        {
            "name": "Basic shape & type",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
new_logps = jax.random.normal(k1, (16,))
old_logps = jax.random.normal(k2, (16,))
advantages = jax.random.normal(k3, (16,))
loss = {fn}(new_logps, old_logps, advantages)
assert isinstance(loss, jax.Array) and loss.ndim == 0, 'Loss must be scalar Array'
""",
        },
        {
            "name": "Numeric check vs fixed value",
            "code": """
import jax.numpy as jnp

new_logps = jnp.array([0.0, -0.2, -0.4, -0.6])
old_logps = jnp.array([0.0, -0.1, -0.5, -0.5])
advantages = jnp.array([1.0, -1.0, 0.5, -0.5])
loss = {fn}(new_logps, old_logps, advantages, clip_ratio=0.2)
expected = jnp.array(-0.0488)
assert jnp.allclose(loss, expected, atol=1e-4, rtol=0), 'Loss should match the expected numeric value on the fixed example'
""",
        },
        {
            "name": "Gradient flows to new_logps only",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
new_logps = jax.random.normal(k1, (8,))
old_logps = jax.random.normal(k2, (8,))
advantages = jax.random.normal(k3, (8,))
g_new = jax.grad(lambda n: {fn}(n, old_logps, advantages))(new_logps)
g_old = jax.grad(lambda o: {fn}(new_logps, o, advantages))(old_logps)
g_adv = jax.grad(lambda a: {fn}(new_logps, old_logps, a))(advantages)
assert g_new is not None, 'Gradients should flow through new_logps'
assert jnp.allclose(g_old, 0.0), 'Gradients should not flow through old_logps (treat as constant baseline)'
assert jnp.allclose(g_adv, 0.0), 'Gradients should not flow through advantages (treat as constant advantages)'
""",
        },
    ],
}
