"""JaxCode — JAX/Flax NNX practice engine. Used in Jupyter Notebooks.

Example:
    from jax_judge import status, check

    # View progress for all tasks
    status()

    # After implementing the function, run the judge
    check("relu")
"""

from jax_judge._version import __version__
from jax_judge.engine import check, hint
from jax_judge.progress import status, reset_progress

__all__ = ["__version__", "check", "hint", "status", "reset_progress"]
