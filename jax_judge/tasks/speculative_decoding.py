"""Speculative Decoding task."""

TASK = {
    "title": "Speculative Decoding",
    "difficulty": "Hard",
    "function_name": "speculative_decode",
    "hint": "For each draft token i: accept with prob min(1, p_target[i,token]/p_draft[i,token]). If rejected, sample from max(0, p_target - p_draft) normalized. Return list of accepted tokens (may include one resampled).",
    "tests": [
        {
            "name": "Perfect draft: all accepted",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
probs = jax.nn.softmax(jax.random.normal(key, (4, 10)), axis=-1)
tokens = jnp.array([2, 5, 1, 8])
accepted = {fn}(probs, probs, tokens)
assert len(accepted) == 4, f'Perfect draft should accept all, got {len(accepted)}'
for i in range(4):
    assert accepted[i] == int(tokens[i]), f'Token {i} mismatch'
""",
        },
        {
            "name": "Output length bounded",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
K = 5
target = jax.nn.softmax(jax.random.normal(k1, (K, 8)), axis=-1)
draft = jax.nn.softmax(jax.random.normal(k2, (K, 8)), axis=-1)
tokens = jax.random.randint(k3, (K,), 0, 8)
accepted = {fn}(target, draft, tokens)
assert 1 <= len(accepted) <= K, f'Length {len(accepted)} not in [1, {K}]'
""",
        },
        {
            "name": "All tokens valid",
            "code": """
import jax
import jax.numpy as jnp

V = 8
for seed in range(20):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    target = jax.nn.softmax(jax.random.normal(k1, (3, V)), axis=-1)
    draft = jax.nn.softmax(jax.random.normal(k2, (3, V)), axis=-1)
    tokens = jax.random.randint(k3, (3,), 0, V)
    for t in {fn}(target, draft, tokens):
        assert 0 <= t < V, f'Token {t} out of range'
""",
        },
    ],
}
