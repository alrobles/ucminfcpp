# Build a validated control list for ucminf()

Returns a named list of control parameters that can be passed directly
to
[`ucminf()`](https://alrobles.github.io/ucminfcpp/reference/ucminf.md),
[`ucminf_xptr()`](https://alrobles.github.io/ucminfcpp/reference/ucminf_xptr.md),
etc.

## Usage

``` r
ucminf_control(
  trace = 0,
  grtol = 1e-06,
  xtol = 1e-12,
  stepmax = 1,
  maxeval = 500L,
  grad = c("forward", "central"),
  gradstep = c(1e-06, 1e-08),
  invhessian.lt = NULL
)
```

## Arguments

- trace:

  Integer. If \> 0, print convergence info. Default 0.

- grtol:

  Gradient tolerance. Default 1e-6.

- xtol:

  Step tolerance. Default 1e-12.

- stepmax:

  Initial trust-region radius. Default 1.

- maxeval:

  Maximum function evaluations. Default 500.

- grad:

  Finite-difference type: "forward" or "central". Default "forward".

- gradstep:

  Length-2 step vector. Default c(1e-6, 1e-8).

- invhessian.lt:

  Packed lower-triangle of initial inverse Hessian. Default NULL.

## Value

A list of class `"ucminf_control"`.
