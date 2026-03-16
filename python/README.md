# ucminf — Python Bindings

Python bindings for the **ucminf** C++ unconstrained nonlinear optimization
library.  The bindings are implemented with [pybind11](https://pybind11.readthedocs.io/).

## Installation

```bash
# from the python/ directory (or the repository root)
pip install python/
```

Requirements: a C++17-capable compiler, Python ≥ 3.8, and pybind11
(`pip install pybind11`).

## Quick start

```python
import ucminf

# Rosenbrock Banana function — minimum at (1, 1)
fn = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
gr = lambda x: [
    -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
     200*(x[1] - x[0]**2)
]

result = ucminf.minimize(x0=[2.0, 0.5], fn=fn, gr=gr)

print(result.x)           # [1.0, 1.0]
print(result.f)           # ~0.0
print(result.status)      # Status.SmallGradient
print(result.message)     # "Stopped by small gradient (grtol)."
```

## API

### `ucminf.minimize(x0, fn, gr=None, control=None, gradstep_rel=1e-6, gradstep_abs=1e-8)`

Minimize `fn(x)` starting from `x0`.

| Parameter      | Type                         | Description                                                           |
|----------------|------------------------------|-----------------------------------------------------------------------|
| `x0`           | `list[float]`                | Initial guess                                                         |
| `fn`           | `callable(list) -> float`    | Objective function                                                    |
| `gr`           | `callable(list) -> list`     | Gradient function (optional; central finite differences if omitted)   |
| `control`      | `ucminf.Control`             | Algorithmic parameters (optional)                                     |
| `gradstep_rel` | `float`                      | Relative finite-difference step (default `1e-6`)                      |
| `gradstep_abs` | `float`                      | Absolute finite-difference step (default `1e-8`)                      |

Returns an `ucminf.Result` object.

### `ucminf.Control`

| Attribute        | Default | Description                                               |
|------------------|---------|-----------------------------------------------------------|
| `grtol`          | `1e-6`  | Stop when max\|grad\| ≤ grtol                            |
| `xtol`           | `1e-12` | Stop when step length ≤ xtol                              |
| `stepmax`        | `1.0`   | Initial trust-region radius                               |
| `maxeval`        | `500`   | Maximum function + gradient evaluations                   |
| `inv_hessian_lt` | `[]`    | Initial inverse Hessian (packed lower triangle, optional) |

### `ucminf.Result`

| Attribute        | Description                                              |
|------------------|----------------------------------------------------------|
| `x`              | Parameter values at the minimum                          |
| `f`              | Objective value at x                                     |
| `n_eval`         | Total function/gradient evaluations                      |
| `max_gradient`   | max\|grad(x)\| at the solution                           |
| `last_step`      | Length of the last accepted step                         |
| `status`         | `ucminf.Status` enum value                               |
| `message`        | Human-readable convergence message                       |
| `inv_hessian_lt` | Packed lower triangle of the final inverse-Hessian       |

### `ucminf.Status`

| Value                       | Meaning                                  |
|-----------------------------|------------------------------------------|
| `SmallGradient`             | Converged: max\|grad\| ≤ grtol          |
| `SmallStep`                 | Converged: step ≤ xtol                   |
| `EvaluationLimitReached`    | Stopped: maxeval reached                 |
| `ZeroStepFromLineSearch`    | Stopped: line search returned alpha = 0  |

## Example without an analytic gradient

```python
import ucminf

fn = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

# gr=None → central finite differences are used automatically
result = ucminf.minimize(x0=[2.0, 0.5], fn=fn)
print(result.x, result.status)
```

## Example with custom control

```python
import ucminf

ctrl = ucminf.Control()
ctrl.grtol   = 1e-9   # tighter gradient tolerance
ctrl.maxeval = 1000

result = ucminf.minimize(x0=[2.0, 0.5], fn=fn, gr=gr, control=ctrl)
```
