# Optimize using a compiled C++ objective (XPtr interface)

Calls the UCMINF optimizer with a compiled C++ objective function that
has been wrapped in an `Rcpp::XPtr<ucminf::ObjFun>`. This path bypasses
the R interpreter on every function evaluation, giving maximum
performance for non-trivial objectives.

## Usage

``` r
ucminf_xptr(par, xptr, control = list(), hessian = 0)
```

## Arguments

- par:

  Numeric starting vector.

- xptr:

  An `externalptr` created by wrapping a `ucminf::ObjFun*` in
  `Rcpp::XPtr<ucminf::ObjFun>`.

- control:

  A named list of control parameters (see
  [`ucminf`](https://alrobles.github.io/ucminfcpp/reference/ucminf.md)),
  or a
  [`ucminf_control`](https://alrobles.github.io/ucminfcpp/reference/ucminf_control.md)
  object.

- hessian:

  Integer: 0 = none, 2 = inv-Hessian, 3 = both.

## Value

A list of class `"ucminf"` (same structure as
[`ucminf`](https://alrobles.github.io/ucminfcpp/reference/ucminf.md)).

## See also

[`ucminf`](https://alrobles.github.io/ucminfcpp/reference/ucminf.md),
[`ucminf_control`](https://alrobles.github.io/ucminfcpp/reference/ucminf_control.md)
