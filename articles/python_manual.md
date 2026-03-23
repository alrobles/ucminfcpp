# Using ucminfcpp from Python

## Overview

`ucminfcpp` ships a **pybind11**-based Python extension module located
in the `python/` directory of the source tree. This allows Python
developers to use the same high-performance C++ optimizer core without
going through R at all.

## Prerequisites

- Python ≥ 3.8
- A C++17-capable compiler (GCC ≥ 7, Clang ≥ 5, MSVC 2019+)
- CMake ≥ 3.15
- pybind11 (`pip install pybind11`)

## Building the Python Module

Clone the repository and build the Python extension:

``` bash
git clone https://github.com/alrobles/ucminfcpp.git
cd ucminfcpp

cmake -S . -B build \
      -DBUILD_PYTHON=ON \
      -DBUILD_JULIA=OFF \
      -DBUILD_TESTS=OFF

cmake --build build
```

After a successful build the shared library (e.g. `ucminfcpp.so` on
Linux, `ucminfcpp.pyd` on Windows) is placed in the `build/` directory.
Add it to your `PYTHONPATH` or copy it to your project:

``` bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/build"
```

## Basic Usage

Import the module and call `minimize()` with an initial point and a
callable that computes the function value **and** gradient
simultaneously:

``` python
import ucminfcpp

def banana(x, g):
    """Rosenbrock's Banana Function with in-place gradient."""
    f       = 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2
    g[0]    = -400.0 * x[0] * (x[1] - x[0]**2) - 2.0 * (1.0 - x[0])
    g[1]    =  200.0 * (x[1] - x[0]**2)
    return f

result = ucminfcpp.minimize([-1.2, 1.0], banana)
print("par  =", result.par)        # [1.0, 1.0]
print("f    =", result.value)      # ~0.0
print("conv =", result.convergence)
```

## Step-by-Step Example: Fitting a Quadratic

``` python
import ucminfcpp

def quadratic(x, g):
    # f(x) = (x0 - 3)^2 + (x1 + 5)^2
    f    = (x[0] - 3.0)**2 + (x[1] + 5.0)**2
    g[0] = 2.0 * (x[0] - 3.0)
    g[1] = 2.0 * (x[1] + 5.0)
    return f

x0     = [0.0, 0.0]
result = ucminfcpp.minimize(x0, quadratic)

print(f"Minimum at x = {result.par}")   # [3.0, -5.0]
print(f"f(x*)       = {result.value}")  # 0.0
```

## Using Finite Differences for the Gradient

If providing an analytical gradient is inconvenient, you can approximate
it numerically using `scipy.optimize.approx_fprime` and wrap the result:

``` python
import ucminfcpp
from scipy.optimize import approx_fprime
import numpy as np

def rosenbrock(x):
    return 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2

eps = np.sqrt(np.finfo(float).eps)

def rosenbrock_with_grad(x, g):
    grad = approx_fprime(x, rosenbrock, eps)
    for i, gi in enumerate(grad):
        g[i] = gi
    return rosenbrock(x)

result = ucminfcpp.minimize([-1.2, 1.0], rosenbrock_with_grad)
print(result.par)   # close to [1.0, 1.0]
```

## Comparison with scipy.optimize.minimize

`ucminfcpp` is a strong alternative to `scipy.optimize.minimize` with
the `"BFGS"` method for smooth, unconstrained problems:

| Aspect             | scipy BFGS              | ucminfcpp               |
|--------------------|-------------------------|-------------------------|
| Algorithm          | BFGS                    | BFGS + soft line search |
| Gradient input     | Separate `jac` argument | Combined `f+g` function |
| Installation       | via pip                 | Build from source       |
| Fortran-equivalent | No                      | Yes (matches ucminf)    |

## Controlling Convergence

Pass keyword arguments to adjust optimizer settings:

``` python
result = ucminfcpp.minimize(
    [-1.2, 1.0],
    banana,
    maxeval = 1000,
    eps     = 1e-10
)
```
