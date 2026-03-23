# General-Purpose Unconstrained Non-Linear Optimization (C/Rcpp)

An implementation of the UCMINF algorithm translated from Fortran to C
and wrapped with Rcpp. The algorithm uses a quasi-Newton method with
BFGS updating of the inverse Hessian and a soft line search with
trust-region radius monitoring.

## Usage

``` r
ucminf(
  par,
  fn = NULL,
  gr = NULL,
  ...,
  fdfun = NULL,
  control = list(),
  hessian = 0
)
```

## Arguments

- par:

  Initial estimate of the minimum (numeric vector).

- fn:

  Objective function to be minimized. Must return a scalar. Ignored when
  `fdfun` is supplied.

- gr:

  Gradient function. If `NULL` a finite-difference approximation is
  used. Ignored when `fdfun` is supplied.

- ...:

  Optional arguments passed to `fn` and `gr`.

- fdfun:

  Optional combined value+gradient function `fdfun(x)` returning
  `list(f = scalar, g = numeric_vector)`. When supplied, `fn` and `gr`
  are ignored.

- control:

  A list of control parameters (see Details), or a
  [`ucminf_control`](https://alrobles.github.io/ucminfcpp/reference/ucminf_control.md)
  object.

- hessian:

  Integer controlling Hessian output:

  0

  :   No Hessian (default).

  2

  :   Returns the final inverse-Hessian approximation from BFGS.

  3

  :   Returns both the inverse Hessian (2) and its inverse (Hessian).

## Value

A list of class `"ucminf"` with elements:

- par:

  Computed minimizer.

- value:

  Objective value at the minimizer.

- convergence:

  Termination code:

  1

  :   Small gradient (`grtol`).

  2

  :   Small step (`xtol`).

  3

  :   Evaluation limit (`maxeval`).

  4

  :   Zero step from line search.

  -2

  :   n \<= 0.

  -4

  :   stepmax \<= 0.

  -5

  :   grtol or xtol \<= 0.

  -6

  :   maxeval \<= 0.

  -7

  :   Given inverse Hessian not positive definite.

- message:

  Human-readable termination message.

- invhessian.lt:

  Lower triangle of the final inverse-Hessian approximation (packed).

- invhessian:

  Full inverse-Hessian matrix (when `hessian >= 2`).

- hessian:

  Hessian matrix, inverse of `invhessian` (when `hessian == 3`).

- info:

  Named vector:

  maxgradient

  :   \\\\g(x)\\\_\infty\\ at solution.

  laststep

  :   Length of the last step.

  stepmax

  :   Final trust-region radius.

  neval

  :   Number of function/gradient evaluations.

## Details

This package is a two-phase migration of the original ucminf R package:

- **Phase 1**: The Fortran core is translated to C
  (`src/ucminf_core.c`).

- **Phase 2**: The C core is wrapped with an Rcpp interface
  (`src/ucminf_rcpp.cpp`).

The interface is designed to be interchangeable with the original
[`ucminf::ucminf`](https://rdrr.io/pkg/ucminf/man/ucminf.html) and with
[`optim`](https://rdrr.io/r/stats/optim.html).

The `control` argument accepts:

- `trace`:

  If positive, print convergence info after optimization. Default `0`.

- `grtol`:

  Stop when \\\\g(x)\\\_\infty \le\\ `grtol`. Default `1e-6`.

- `xtol`:

  Stop when \\\\x - x\_{\rm prev}\\^2 \le\\ `xtol`\*(`xtol` +
  \\\\x\\^2\\). Default `1e-12`.

- `stepmax`:

  Initial trust-region radius. Default `1`.

- `maxeval`:

  Maximum function evaluations. Default `500`.

- `grad`:

  Finite-difference type when `gr = NULL`: `"forward"` (default) or
  `"central"`.

- `gradstep`:

  Length-2 vector; step is \\\|x_i\| \cdot \code{gradstep\[1\]} +
  \code{gradstep\[2\]}\\. Default `c(1e-6, 1e-8)`.

- `invhessian.lt`:

  A vector containing the lower triangle of the initial inverse Hessian
  (packed column-major). Default: identity.

## References

Nielsen, H. B. (2000). *UCMINF – An Algorithm for Unconstrained,
Nonlinear Optimization*. Report IMM-REP-2000-19, DTU.

## Examples

``` r
## Rosenbrock Banana function
fR <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
gR <- function(x) c(-400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1]),
                    200 * (x[2] - x[1] * x[1]))

## Find minimum with analytic gradient
ucminf(par = c(2, 0.5), fn = fR, gr = gR)
#> ucminf result
#>   Converged: Stopped by small gradient (grtol). 
#>   Minimum value: 8.073e-20 
#>   Minimizer:
#> [1] 1 1
#>   Evaluations: 21 
#>   Max |gradient|: 1.021e-08 

## Find minimum with finite-difference gradient
ucminf(par = c(2, 0.5), fn = fR)
#> ucminf result
#>   Converged: Stopped by zero step from line search. 
#>   Minimum value: 9.007e-08 
#>   Minimizer:
#> [1] 0.9997 0.9994
#>   Evaluations: 24 
#>   Max |gradient|: 8.065e-05 
```
