# ucminfcpp

Unconstrained nonlinear optimization via **modern C++17** and Rcpp — a
complete reimplementation of the original Fortran-based
[ucminf](https://CRAN.R-project.org/package=ucminf) R package with
multi-language support.

## Overview

**ucminfcpp** provides the `ucminf()` function for general-purpose
unconstrained nonlinear optimization. The algorithm is a quasi-Newton method
with BFGS updating of the inverse Hessian and a soft line search with
adaptive trust-region radius monitoring. It is designed as a drop-in
replacement for the original `ucminf::ucminf()` function.

The original algorithm was written in Fortran by Hans Bruun Nielsen (IMM,
DTU, 2000) and described in:

> H.B. Nielsen, "UCMINF — An Algorithm for Unconstrained, Nonlinear
> Optimization", Report IMM-REP-2000-19, DTU, December 2000.

### Key features

| Feature | Description |
|---------|-------------|
| **Modern C++17 core** | No raw pointers, no manual memory management. `std::vector`, `std::function`, RAII throughout. |
| **R interface** | Drop-in replacement for `ucminf::ucminf()` via Rcpp. |
| **Python bindings** | `pybind11`-based module in `python/`. |
| **Julia bindings** | `CxxWrap`-based module in `julia/`. |
| **C++ unit tests** | Catch2 test suite covering the pure C++ API. |
| **CMake build** | Standalone C++ build for embedding in other projects. |

## Installation

### R package

Install from GitHub with `remotes`:

```r
# install.packages("remotes")
remotes::install_github("alrobles/ucminfcpp")
```

### Python module

```bash
pip install pybind11
pip install python/    # from the repository root
```

### Standalone C++ / tests

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

## Usage

### R

```r
library(ucminfcpp)

# Rosenbrock Banana function
fn <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
gr <- function(x) c(
  -400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]),
   200 * (x[2] - x[1]^2)
)

result <- ucminf(par = c(2, 0.5), fn = fn, gr = gr)

result$par          # optimum: c(1, 1)
result$value        # minimum: ~0
result$convergence  # 1 = converged (small gradient)
```

### Python

```python
import ucminf

fn = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
gr = lambda x: [-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]),
                 200*(x[1]-x[0]**2)]

result = ucminf.minimize(x0=[2.0, 0.5], fn=fn, gr=gr)
print(result.x)      # [1.0, 1.0]
print(result.f)      # ~0.0
print(result.status) # Status.SmallGradient
```

See [python/README.md](python/README.md) for the full Python API.

### Pure C++

```cpp
#include "ucminf_core.hpp"

auto rosenbrock = [](const std::vector<double>& x,
                     std::vector<double>& g, double& f) {
    f    = (1-x[0])*(1-x[0]) + 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
    g[0] = -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]*x[0]);
    g[1] =  200*(x[1]-x[0]*x[0]);
};

ucminf::Control ctrl;
ctrl.grtol = 1e-8;

ucminf::Result result = ucminf::minimize({2.0, 0.5}, rosenbrock, ctrl);
// result.x            → {1.0, 1.0}
// result.f            → ~0
// result.status       → ucminf::Status::SmallGradient
// result.n_eval       → number of f+g evaluations
// result.inv_hessian_lt → packed lower triangle of final inv-Hessian
```

### Julia

See [julia/README.md](julia/README.md) for build instructions and usage.

## C++ API reference

### `ucminf::minimize()`

```cpp
ucminf::Result minimize(
    std::vector<double> x0,
    ucminf::ObjFun      fdf,
    const ucminf::Control& control = {}
);
```

Minimizes `f(x)` starting from `x0`. Throws `std::invalid_argument` if
control parameters are invalid.

### `ucminf::ObjFun`

```cpp
using ObjFun = std::function<void(
    const std::vector<double>& x,   // current point (input)
    std::vector<double>&       g,   // gradient (output)
    double&                    f    // function value (output)
)>;
```

### `ucminf::Control`

| Field | Default | Description |
|-------|---------|-------------|
| `grtol` | `1e-6` | Stop when max\|grad\| ≤ grtol |
| `xtol` | `1e-12` | Stop when step ≤ xtol |
| `stepmax` | `1.0` | Initial trust-region radius |
| `maxeval` | `500` | Maximum function+gradient evaluations |
| `inv_hessian_lt` | `{}` | Optional initial inverse Hessian (packed lower triangle) |

### `ucminf::Result`

| Field | Description |
|-------|-------------|
| `x` | Parameter values at the minimum |
| `f` | Objective value at x |
| `n_eval` | Total evaluations used |
| `max_gradient` | max\|grad(x)\| at solution |
| `last_step` | Length of the last accepted step |
| `status` | `ucminf::Status` convergence code |
| `inv_hessian_lt` | Final inverse Hessian (packed lower triangle) |

### `ucminf::Status`

| Value | Code | Meaning |
|-------|------|---------|
| `SmallGradient` | 1 | Converged: max\|grad\| ≤ grtol |
| `SmallStep` | 2 | Converged: step ≤ xtol |
| `EvaluationLimitReached` | 3 | Stopped: maxeval reached |
| `ZeroStepFromLineSearch` | 4 | Stopped: line search returned alpha=0 |

## R interface (`ucminf()`)

```r
ucminf(par, fn, gr = NULL, ..., control = list(), hessian = 0)
```

| Argument  | Description |
|-----------|-------------|
| `par`     | Numeric vector of initial parameter values. |
| `fn`      | Objective function to minimize (must return a scalar). |
| `gr`      | Optional gradient function. If `NULL`, finite differences are used. |
| `...`     | Additional arguments passed to `fn` and `gr`. |
| `control` | Named list of algorithmic parameters (see below). |
| `hessian` | `0` = no Hessian output, `2` = inverse Hessian, `3` = both. |

### R control parameters

| Name | Default | Description |
|------|---------|-------------|
| `trace` | `0` | Print level. |
| `grtol` | `1e-6` | Gradient tolerance. |
| `xtol` | `1e-12` | Step tolerance. |
| `stepmax` | `1` | Initial trust-region radius. |
| `maxeval` | `500` | Maximum function evaluations. |
| `grad` | `"forward"` | Finite-difference type: `"forward"` or `"central"`. |
| `gradstep` | `c(1e-6, 1e-8)` | Step size for finite differences. |
| `invhessian.lt` | (identity) | Initial inverse Hessian (packed lower-triangle vector). |

### R return value

A list of class `"ucminf"` with:

| Name | Description |
|------|-------------|
| `par` | Parameter values at the minimum. |
| `value` | Objective function value at the minimum. |
| `convergence` | Integer code (1=small gradient, 2=small step, 3=eval limit, 4=zero step). |
| `message` | Human-readable convergence message. |
| `info` | Named vector: `maxgradient`, `laststep`, `stepmax`, `neval`. |
| `invhessian.lt` | Lower triangle of the final inverse Hessian. |
| `invhessian` | Full inverse Hessian matrix (when `hessian >= 2`). |
| `hessian` | Full Hessian matrix (when `hessian == 3`). |

## Algorithm details

The UCMINF algorithm combines three techniques:

1. **Quasi-Newton direction.** At each iteration the search direction is
   `h = -D * g`, where `D` is the current inverse-Hessian approximation and
   `g` is the gradient.

2. **BFGS inverse-Hessian update.** After each accepted step the
   inverse-Hessian approximation `D` is updated with the standard BFGS
   rank-2 formula, ensuring that `D` remains symmetric and positive definite.

3. **Soft line search with trust region.** A line search satisfying the
   strong Wolfe conditions is performed along the search direction. The
   step length is bounded by an adaptive trust-region radius that shrinks
   when steps are rejected and grows when the optimizer is making good
   progress.

The inverse Hessian is stored in packed lower-triangular form, so memory
usage scales as `n(n+1)/2` rather than `n²`.

## Migration history

| Phase | Source | Target | Description |
|-------|--------|--------|-------------|
| 1 | Fortran 77 | C99 (`ucminf_core.c`) | Line-by-line translation; Fortran BLAS replaced with inline helpers |
| 2 | C99 | Rcpp (`ucminf_rcpp.cpp`) | C callback wrapped for R; Rcpp manages R↔C bridge |
| 3 (this PR) | C99 + Rcpp | C++17 + Rcpp + pybind11 + CxxWrap | Full modernization; raw pointers → `std::vector`; C function pointer → `std::function`; multi-language support |

### Why migrate to C++17?

| Concern | C99 + Rcpp | C++17 + Rcpp |
|---------|-----------|--------------|
| Memory management | Manual (`malloc`/`free` pattern via workspace arrays) | RAII (`std::vector` lifetime) |
| Callback interface | C function pointer + `void *userdata` | `std::function<…>` — type-safe, captures lambdas |
| Error handling | Return integer codes | `std::invalid_argument` exceptions |
| Multi-language API | R only | R, Python, Julia, standalone C++ |
| Unit testing | R `testthat` only | R `testthat` + C++ Catch2 |

## License

GPL (>= 2)
