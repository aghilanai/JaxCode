"""Cosine LR Scheduler with Warmup task."""

TASK = {
    "title": "Cosine LR Scheduler with Warmup",
    "difficulty": "Medium",
    "function_name": "cosine_lr_schedule",
    "hint": "Warmup: linear ramp from 0 to max_lr over warmup_steps. Then cosine decay: min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress)).",
    "tests": [
        {
            "name": "Start of warmup",
            "code": """
lr = {fn}(step=0, total_steps=100, warmup_steps=10, max_lr=0.001, min_lr=0.0)
assert abs(lr) < 1e-8, f'lr at step 0: {lr}'
""",
        },
        {
            "name": "End of warmup",
            "code": """
lr = {fn}(step=10, total_steps=100, warmup_steps=10, max_lr=0.001)
assert abs(lr - 0.001) < 1e-8, f'lr at warmup end: {lr}'
""",
        },
        {
            "name": "End of schedule",
            "code": """
lr = {fn}(step=100, total_steps=100, warmup_steps=10, max_lr=0.001, min_lr=0.0001)
assert abs(lr - 0.0001) < 1e-6, f'lr at end: {lr}'
""",
        },
        {
            "name": "Warmup is monotonically increasing",
            "code": """
lrs = [{fn}(step=i, total_steps=100, warmup_steps=10, max_lr=0.001) for i in range(11)]
for i in range(len(lrs) - 1):
    assert lrs[i] <= lrs[i+1] + 1e-10, f'Not increasing at step {i}'
""",
        },
        {
            "name": "Cosine shape",
            "code": """
import math
lr = {fn}(step=55, total_steps=100, warmup_steps=10, max_lr=0.001, min_lr=0.0)
progress = (55 - 10) / (100 - 10)
expected = 0.5 * 0.001 * (1 + math.cos(math.pi * progress))
assert abs(lr - expected) < 1e-8, f'{lr} vs {expected}'
""",
        },
    ],
}
