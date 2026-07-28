"""GRPO (Group Relative Policy Optimization) Loss task."""

TASK = {
    "title": "GRPO (Group Relative Policy Optimization) Loss",
    "difficulty": "Hard",
    "function_name": "grpo_loss",
    "hint": (
        "Per group, normalize rewards: A_i = (r_i - mean_g) / (std_g + eps). "
        "Detach A_i from graph with `jax.lax.stop_gradient`, then return -mean(A_i * logps)."
    ),
    "tests": [
        {
            "name": "Basic shape & type",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
logps = jax.random.normal(k1, (6,))
rewards = jax.random.normal(k2, (6,))
group_ids = jnp.array([0, 0, 0, 1, 1, 1])
loss = {fn}(logps, rewards, group_ids)
assert isinstance(loss, jax.Array) and loss.ndim == 0, 'Loss must be scalar Array'
""",
        },
        {
            "name": "Numeric check vs reference",
            "code": """
import jax
import jax.numpy as jnp

def _reference_grpo_loss(logps, rewards, group_ids, eps=1e-5):
    logps = jnp.asarray(logps).reshape(-1)
    rewards = jnp.asarray(rewards).reshape(-1)
    group_ids = jnp.asarray(group_ids).reshape(-1)
    advantages = jnp.zeros_like(rewards)
    for gid in [int(g) for g in jnp.unique(group_ids)]:
        mask = group_ids == gid
        r_g = rewards[mask]
        mean_g = jnp.mean(r_g)
        std_g = jnp.std(r_g)
        advantages = advantages.at[mask].set((r_g - mean_g) / (std_g + eps))
    advantages_detached = jax.lax.stop_gradient(advantages)
    return -(advantages_detached * logps).mean()

logps = jnp.array([0.0, -0.5, -1.0, -1.5])
rewards = jnp.array([1.0, 0.8, 0.2, 0.0])
group_ids = jnp.array([0, 0, 1, 1])
loss_student = {fn}(logps, rewards, group_ids)
loss_ref = _reference_grpo_loss(logps, rewards, group_ids)
assert jnp.allclose(loss_student, loss_ref, atol=1e-5, rtol=1e-5), 'Loss should match reference implementation numerically on a fixed example'
""",
        },
        {
            "name": "Gradient flows to logps only",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
logps = jax.random.normal(k1, (4,))
rewards = jax.random.normal(k2, (4,))
group_ids = jnp.array([0, 0, 1, 1])
g_logps = jax.grad(lambda l: {fn}(l, rewards, group_ids))(logps)
g_rewards = jax.grad(lambda r: {fn}(logps, r, group_ids))(rewards)
assert g_logps is not None, 'Gradients should flow through logps'
assert jnp.allclose(g_rewards, 0.0), 'Gradients should not flow through rewards'
""",
        },
        {
            "name": "Group-wise normalization",
            "code": """
import jax
import jax.numpy as jnp

logps = jnp.zeros(4)
rewards = jnp.array([0.0, 1.0, 10.0, 11.0])
group_ids = jnp.array([0, 0, 1, 1])
grad = jax.grad(lambda l: {fn}(l, rewards, group_ids))(logps)
# Since each group has rewards [0,1] and [10,11], the normalized advantages
# should be identical across groups, leading to identical gradients per position.
assert jnp.allclose(grad[:2], grad[2:]), 'Groups should be treated independently but symmetrically'
""",
        },
    ],
}
