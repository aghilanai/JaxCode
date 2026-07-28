"""Linear Regression Three Ways task."""

TASK = {
    "title": "Linear Regression",
    "difficulty": "Medium",
    "function_name": "LinearRegression",
    "hint": "Closed-form: augment $X$ with ones column, solve $w = (X^T X)^{-1} X^T y$ via `jnp.linalg.lstsq`. Gradient descent: $\\nabla w = \\frac{2}{N} X^T (\\hat{y} - y)$, update $w \\leftarrow w - \\text{lr} \\cdot \\nabla w$ (no autograd). `nn_linear`: create `nnx.Linear(D, 1)`, use MSE + `nnx.Optimizer` with optax SGD loop.",
    "tests": [
        {
            "name": "Closed-form returns correct shapes",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(42)
X = jax.random.normal(key, (50, 3))
y = X @ jnp.array([2.0, -1.0, 0.5]) + 3.0 + jax.random.normal(jax.random.PRNGKey(1), (50,)) * 0.01
model = {fn}()
w, b = model.closed_form(X, y)
assert w.shape == (3,), f'w shape: {w.shape}, expected (3,)'
assert b.shape == (), f'b shape: {b.shape}, expected scalar'
""",
        },
        {
            "name": "Closed-form finds correct weights",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(42)
true_w = jnp.array([2.0, -1.0, 0.5])
true_b = 3.0
X = jax.random.normal(key, (100, 3))
y = X @ true_w + true_b
model = {fn}()
w, b = model.closed_form(X, y)
assert jnp.allclose(w, true_w, atol=1e-4), f'w: {w} vs true: {true_w}'
assert jnp.allclose(b, jnp.array(true_b), atol=1e-4), f'b: {float(b):.4f} vs true: {true_b}'
""",
        },
        {
            "name": "Gradient descent converges",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(42)
true_w = jnp.array([2.0, -1.0, 0.5])
true_b = 3.0
X = jax.random.normal(key, (100, 3))
y = X @ true_w + true_b
model = {fn}()
w, b = model.gradient_descent(X, y, lr=0.05, steps=2000)
assert jnp.allclose(w, true_w, atol=0.1), f'GD w: {w} vs true: {true_w}'
assert abs(float(b) - true_b) < 0.1, f'GD b: {float(b):.4f} vs true: {true_b}'
""",
        },
        {
            "name": "nn.Linear approach works",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(42)
true_w = jnp.array([2.0, -1.0, 0.5])
true_b = 3.0
X = jax.random.normal(key, (100, 3))
y = X @ true_w + true_b
model = {fn}()
w, b = model.nn_linear(X, y, lr=0.05, steps=2000)
assert jnp.allclose(w, true_w, atol=0.1), f'nn w: {w} vs true: {true_w}'
assert abs(float(b) - true_b) < 0.1, f'nn b: {float(b):.4f} vs true: {true_b}'
""",
        },
        {
            "name": "All three methods agree",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (200, 2))
true_w = jnp.array([1.5, -2.0])
y = X @ true_w + 1.0 + jax.random.normal(jax.random.PRNGKey(1), (200,)) * 0.1
model = {fn}()
w_cf, b_cf = model.closed_form(X, y)
w_gd, b_gd = model.gradient_descent(X, y, lr=0.05, steps=3000)
w_nn, b_nn = model.nn_linear(X, y, lr=0.05, steps=3000)
assert jnp.allclose(w_cf, w_gd, atol=0.15), f'CF vs GD: max diff {float(jnp.max(jnp.abs(w_cf - w_gd))):.4f}'
assert jnp.allclose(w_cf, w_nn, atol=0.15), f'CF vs NN: max diff {float(jnp.max(jnp.abs(w_cf - w_nn))):.4f}'
assert abs(float(b_cf) - float(b_gd)) < 0.15, f'Bias CF vs GD: {float(b_cf):.4f} vs {float(b_gd):.4f}'
assert abs(float(b_cf) - float(b_nn)) < 0.15, f'Bias CF vs NN: {float(b_cf):.4f} vs {float(b_nn):.4f}'
""",
        },
        {
            "name": "Closed-form uses no autograd",
            "code": """
import jax
import jax.numpy as jnp
import numpy as np

X = jax.random.normal(jax.random.PRNGKey(0), (30, 2))
y = X @ jnp.array([1.0, 2.0]) + 0.5
model = {fn}()
w, b = model.closed_form(X, y)
assert isinstance(np.asarray(w), np.ndarray), 'Closed-form w should be a concrete array'
assert not isinstance(w, jax.core.Tracer), 'Closed-form should not use autograd tracing'
""",
        },
    ],
}
