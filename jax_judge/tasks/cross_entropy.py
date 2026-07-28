"""Cross-Entropy Loss task."""

TASK = {
    "title": "Cross-Entropy Loss",
    "difficulty": "Easy",
    "function_name": "cross_entropy_loss",
    "hint": "log_probs = logits - logsumexp(logits, dim=-1, keepdim=True). Loss = -log_probs[arange(B), targets].mean(). Subtract max for stability (logsumexp handles this).",
    "tests": [
        {
            "name": "Matches reference cross-entropy",
            "code": """
import jax
import jax.numpy as jnp
import jax.nn as nn
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
logits = jax.random.normal(k1, (4, 10))
targets = jax.random.randint(k2, (4,), 0, 10)
out = {fn}(logits, targets)
log_probs = nn.log_softmax(logits, axis=-1)
ref = -jnp.mean(log_probs[jnp.arange(4), targets])
assert jnp.allclose(out, ref, atol=1e-5), f'Mismatch: {float(out):.4f} vs {float(ref):.4f}'
""",
        },
        {
            "name": "Numerical stability",
            "code": """
import jax.numpy as jnp
logits = jnp.array([[1000., 0., 0.], [0., 1000., 0.]])
targets = jnp.array([0, 1])
out = {fn}(logits, targets)
assert not jnp.isnan(out), 'NaN with large logits'
assert not jnp.isinf(out), 'Inf with large logits'
assert float(out) < 0.01, 'Should be ~0 for confident correct predictions'
""",
        },
        {
            "name": "Scalar output",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
out = {fn}(jax.random.normal(k1, (8, 5)), jax.random.randint(k2, (8,), 0, 5))
assert out.ndim == 0, 'Loss must be a scalar'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
logits = jax.random.normal(k1, (8, 5))
targets = jax.random.randint(k2, (8,), 0, 5)
g = jax.grad(lambda l, t: {fn}(l, t))(logits, targets)
assert g is not None, 'gradient w.r.t. logits is None'
""",
        },
    ],
}
