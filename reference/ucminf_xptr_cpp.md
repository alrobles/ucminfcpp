# Optimize using a compiled C++ objective function passed as an external pointer

Optimize using a compiled C++ objective function passed as an external
pointer

## Usage

``` r
ucminf_xptr_cpp(par, xptr, control)
```

## Arguments

- par:

  Numeric starting vector.

- xptr:

  An `externalptr` wrapping a heap-allocated `ucminf::ObjFun*` created
  via `Rcpp::XPtr<ucminf::ObjFun>`.

- control:

  Named list of control parameters (same as `ucminf_cpp`).

## Value

Same list structure as `ucminf_cpp`.
