"""Beam Search Decoding task."""

TASK = {
    "title": "Beam Search Decoding",
    "difficulty": "Medium",
    "function_name": "beam_search",
    "hint": "Maintain beam_width hypotheses. Each step: expand each hypothesis with all tokens, keep top beam_width by total score. Stop when all beams end with eos or max_len reached.",
    "tests": [
        {
            "name": "Returns list starting with start_token",
            "code": """
import jax.numpy as jnp

def dummy(tokens):
    return jnp.zeros(10)

seq = {fn}(dummy, start_token=0, max_len=5, beam_width=3, eos_token=9)
assert isinstance(seq, list), 'Must return a list'
assert seq[0] == 0, f'First token: {seq[0]}'
""",
        },
        {
            "name": "Greedy path (beam=1)",
            "code": """
import jax.numpy as jnp

def greedy_fn(tokens):
    lp = jnp.full((5,), -10.0)
    lp = lp.at[min(len(tokens), 4)].set(0.0)
    return lp

seq = {fn}(greedy_fn, start_token=0, max_len=5, beam_width=1, eos_token=4)
assert seq == [0, 1, 2, 3, 4], f'Greedy: {seq}'
""",
        },
        {
            "name": "Beam finds better path than greedy",
            "code": """
import jax.numpy as jnp

def tricky(tokens):
    lp = jnp.full((6,), -100.0)
    if len(tokens) == 1:
        lp = lp.at[1].set(-1.0).at[2].set(-0.5)
    elif tokens[-1] == 1:
        lp = lp.at[5].set(0.0)
    elif tokens[-1] == 2:
        lp = lp.at[5].set(-10.0)
    else:
        lp = lp.at[5].set(0.0)
    return lp

seq = {fn}(tricky, start_token=0, max_len=5, beam_width=2, eos_token=5)
assert seq == [0, 1, 5], f'Beam should find [0,1,5], got {seq}'
""",
        },
        {
            "name": "Stops at eos",
            "code": """
import jax.numpy as jnp

def eos_fn(tokens):
    lp = jnp.zeros(4)
    lp = lp.at[3].set(10.0)
    return lp

seq = {fn}(eos_fn, start_token=0, max_len=100, beam_width=2, eos_token=3)
assert seq[-1] == 3 and len(seq) == 2, f'Should be [0,3], got {seq}'
""",
        },
    ],
}
