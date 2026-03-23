# Optimize using a combined value+gradient R function

Internal function called by
[`ucminf`](https://alrobles.github.io/ucminfcpp/reference/ucminf.md)
when `fdfun` is supplied. Use
[`ucminf()`](https://alrobles.github.io/ucminfcpp/reference/ucminf.md)
instead.

## Usage

``` r
ucminf_fdf_cpp(par, fdfun, control)
```

## Arguments

- par:

  Numeric starting vector.

- fdfun:

  R function `fdfun(x)` returning
  `list(f = scalar, g = numeric_vector_of_length_n)`.

- control:

  Named list of control parameters (same as `ucminf_cpp`).

## Value

Same list structure as `ucminf_cpp`.
