# Unconstrained Nonlinear Optimization (C++/Rcpp implementation)

Internal function called by
[`ucminf`](https://alrobles.github.io/ucminfcpp/reference/ucminf.md).
Use that instead.

## Usage

``` r
ucminf_cpp(par, fn, gr, has_gr, control)
```

## Arguments

- par:

  Numeric vector of starting values.

- fn:

  R function returning the objective value.

- gr:

  R function returning the gradient, or NULL.

- has_gr:

  Logical: TRUE if gr is provided.

- control:

  Named list of control parameters: `grtol`, `xtol`, `stepmax`,
  `maxeval`, `grad_type`, `gradstep`, `invhessian_lt`.

## Value

A list with elements `par`, `value`, `convergence`, `neval`,
`maxgradient`, `laststep`, `stepmax`, and `invhessian_lt`.
