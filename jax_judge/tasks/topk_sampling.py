"""Top-k / Top-p Sampling task."""

TASK = {
    "title": "Top-k / Top-p Sampling",
    "difficulty": "Medium",
    "function_name": "sample_top_k_top_p",
    "hint": "Apply temperature first. For top-k: set logits below the k-th largest to -inf. For top-p: sort, compute cumsum of probs, mask where cumsum > p. Sample with `jax.random.categorical(key, logits)`.",
    "tests": [
        {
            "name": "top_k=1 always returns argmax",
            "code": """
import jax
import jax.numpy as jnp

logits = jnp.array([1.0, 5.0, 2.0, 0.5])
key = jax.random.PRNGKey(0)
for _ in range(10):
    key, subkey = jax.random.split(key)
    assert {fn}(logits, top_k=1, key=subkey) == 1, 'top_k=1 should return argmax'
""",
        },
        {
            "name": "Low temperature concentrates",
            "code": """
import jax
import jax.numpy as jnp

logits = jnp.array([1.0, 3.0, 2.0])
counts = [0, 0, 0]
key = jax.random.PRNGKey(42)
for _ in range(100):
    key, subkey = jax.random.split(key)
    counts[{fn}(logits, temperature=0.01, key=subkey)] += 1
assert counts[1] > 90, f'Low temp should pick argmax, got {counts}'
""",
        },
        {
            "name": "All tokens reachable (no filtering)",
            "code": """
import jax
import jax.numpy as jnp

logits = jnp.zeros(5)
seen = set()
for i in range(200):
    key = jax.random.PRNGKey(i)
    seen.add(int({fn}(logits, key=key)))
assert len(seen) == 5, f'Only saw {seen}'
""",
        },
        {
            "name": "Returns valid index",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
V = 100
logits = jax.random.normal(k1, (V,))
for _ in range(20):
    key, subkey = jax.random.split(key)
    t = {fn}(logits, top_k=10, top_p=0.9, key=subkey)
    assert 0 <= t < V, f'Token {t} out of range'
""",
        },
    ],
}
