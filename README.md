# ucminfcpp

Unconstrained nonlinear optimization via C and Rcpp — a modern
reimplementation of the original Fortran-based
[ucminf](https://CRAN.R-project.org/package=ucminf) R package.

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

## Installation

Install from GitHub with `remotes`:

```r
# install.packages("remotes")
remotes::install_github("alrobles/ucminfcpp")
```

## Usage

### Basic example — Rosenbrock function

```r
library(ucminfcpp)

# Objective function
fn <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

# Analytic gradient (optional)
gr <- function(x) c(
  -400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]),
   200 * (x[2] - x[1]^2)
)

result <- ucminf(par = c(2, 0.5), fn = fn, gr = gr)

result$par          # optimum: c(1, 1)
result$value        # minimum: ~0
result$convergence  # 1 = converged (small gradient)
```

### Using finite-difference gradients

If no gradient function is supplied, the gradient is approximated
automatically via finite differences:

```r
# Forward differences (default)
result <- ucminf(par = c(2, 0.5), fn = fn)

# Central differences (more accurate, roughly twice as many evaluations)
result <- ucminf(par = c(2, 0.5), fn = fn,
                 control = list(grad = "central"))
```

### Retrieving the Hessian

```r
result <- ucminf(par = c(2, 0.5), fn = fn, gr = gr, hessian = 3)

result$invhessian   # approximate inverse Hessian at solution
result$hessian      # approximate Hessian at solution
```

## Function signature

```r
ucminf(par, fn, gr = NULL, ..., control = list(), hessian = 0)
```

| Argument  | Description |
|-----------|-------------|
| `par`     | Numeric vector of initial parameter values. |
| `fn`      | Objective function to minimize (must return a scalar). |
| `gr`      | Optional gradient function (must return a vector the same length as `par`). If `NULL`, finite differences are used. |
| `...`     | Additional arguments passed to `fn` and `gr`. |
| `control` | Named list of algorithmic parameters (see below). |
| `hessian` | `0` = no Hessian output, `2` = return inverse Hessian, `3` = return both inverse Hessian and Hessian. |

### Control parameters

| Name              | Default   | Description |
|-------------------|-----------|-------------|
| `trace`           | `0`       | Print level. `0` = silent, positive values increase verbosity. |
| `grtol`           | `1e-6`    | Convergence tolerance on the infinity norm of the gradient. |
| `xtol`            | `1e-12`   | Convergence tolerance on step size. |
| `stepmax`         | `1`       | Initial trust-region radius. |
| `maxeval`         | `500`     | Maximum number of function evaluations. |
| `grad`            | `"forward"` | Finite-difference type when `gr = NULL`: `"forward"` or `"central"`. |
| `gradstep`        | `c(1e-6, 1e-8)` | Step size parameters for finite differences. The step for element `i` is `abs(x[i]) * gradstep[1] + gradstep[2]`. |
| `invhessian.lt`   | (identity) | Initial inverse Hessian as a packed lower-triangle vector. |

### Return value

A list of class `"ucminf"` with components:

| Name              | Description |
|-------------------|-------------|
| `par`             | Parameter values at the minimum. |
| `value`           | Objective function value at the minimum. |
| `convergence`     | Integer code: `1` = small gradient, `2` = small step, `3` = evaluation limit, `4` = zero step, negative = error. |
| `message`         | Human-readable convergence message. |
| `info`            | Named vector with `maxgradient`, `laststep`, `stepmax`, and `neval`. |
| `invhessian.lt`   | Lower triangle of the final inverse Hessian (packed). |
| `invhessian`      | Full inverse Hessian matrix (when `hessian >= 2`). |
| `hessian`         | Full Hessian matrix (when `hessian == 3`). |

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

## Migration from Fortran

The original `ucminf` R package relied on a Fortran subroutine for the core
optimization loop, called from R through C glue code. **ucminfcpp**
eliminates the Fortran dependency entirely through a two-phase migration:

### Phase 1 — Fortran to C (`src/ucminf_core.c`)

The Fortran subroutine was translated line-by-line into portable C99. The
translation preserves the algorithmic structure of the original code so that
the two implementations can be compared side by side. Key changes from the
Fortran source:

| Aspect | Fortran original | C translation |
|--------|-----------------|---------------|
| Array indexing | 1-based | 0-based |
| BLAS routines | External library (DDOT, DNRM2, DCOPY, DSCAL, DAXPY, IDAMAX, DSPMV, DSPR2, DSPR) | Inline helper functions (`my_ddot`, `my_dnrm2`, etc.) |
| Function/gradient evaluation | `EXTERNAL FDF` subroutine | C callback via `ucminf_fdf_t` function pointer with `void *userdata` |
| Matrix storage | Packed lower triangle (column-major) | Same layout, unchanged |
| Workspace | Passed as a single `DOUBLE PRECISION W(IW)` array | Same layout, unchanged |

The resulting `ucminf_optimize()` C function is a self-contained, pure-C
implementation with no external dependencies beyond the C standard library.

### Phase 2 — C to R via Rcpp (`src/ucminf_rcpp.cpp`)

An Rcpp wrapper bridges the C core to R:

- A **callback trampoline** (`r_fdf_callback`) converts between the C
  function-pointer interface and R function calls. It handles evaluation of
  the objective function and either calls the user-supplied gradient or
  computes a finite-difference approximation (forward or central).
- The exported `ucminf_cpp()` function translates R control parameters into
  the C interface, allocates workspace, invokes `ucminf_optimize()`, and
  packs the results back into an R list.

### Phase 3 — R interface (`R/ucminf.R`)

The user-facing `ucminf()` function in R provides the same API as the
original `ucminf` package. It validates inputs, builds the control list,
calls the Rcpp layer, and reconstructs the full inverse-Hessian matrix from
the packed storage returned by the C core.

### Why migrate?

| Concern | Fortran + C + R | Pure C + Rcpp |
|---------|----------------|---------------|
| Compiler toolchain | Requires a Fortran compiler (gfortran) | C and C++ compilers only |
| CRAN portability | Fortran compiler availability varies across platforms | Broader platform support |
| Maintainability | Three-language stack | Two-language stack (C + C++/R) |
| External BLAS | Depends on system BLAS | Self-contained BLAS-like helpers |
| Build complexity | Makevars must link Fortran runtime | Standard Rcpp build |

## License

GPL (>= 2)
