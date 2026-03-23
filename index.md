# `ucminfcpp`

R package that provides unconstrained nonlinear optimization through a
modern **C++17** reimplementation of the classic Fortran-based
[ucminf](https://CRAN.R-project.org/package=ucminf) algorithm. It is a
**drop-in replacement** for
[`ucminf::ucminf()`](https://rdrr.io/pkg/ucminf/man/ucminf.html).

The reimplementation of the C++ library offers multi-language support:
[**Python**](https://alrobles.github.io/ucminfcpp/articles/python_manual.html)
and
[**Julia**](https://alrobles.github.io/ucminfcpp/articles/julia_manual.html)
bindings are available. \## Features

- **Drop-in replacement** for the original
  [`ucminf::ucminf()`](https://rdrr.io/pkg/ucminf/man/ucminf.html)
  function.
- **Modern C++17** implementation, enabling easier extension and
  integration.
- **Multi-language support** – use the same algorithm in R, Python, and
  Julia.
- **Efficient and robust** – retains the numerical properties of the
  original Fortran code.
- **Easy installation** – no Fortran compiler required.

## Installation

Install the latest development version from GitHub:

``` r
# Install directly from GitHub
devtools::install_github("alrobles/ucminfcpp")
```

## Basic Example

The example below optimises **Rosenbrock’s Banana Function** and
confirms that `ucminfcpp` and `ucminf` produce identical results.

``` r
# Rosenbrock's Banana Function
banana <- function(x) {
  100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
}

# Initial point
starting_point <- c(-1.2, 1)

# Optimization with ucminfcpp
result_cpp <- ucminfcpp::ucminf(starting_point, banana)
cat("ucminfcpp result:\n")
print(result_cpp$par)

# Optimization with ucminf
result_fortran <- ucminf::ucminf(starting_point, banana)
cat("ucminf result:\n")
print(result_fortran$par)

# Check similarity
identical_result <- all.equal(result_cpp$par, result_fortran$par)
cat("Are the results the same? ", identical_result, "\n")
```

## Overview

`ucminfcpp` provides the
[`ucminf()`](https://alrobles.github.io/ucminfcpp/reference/ucminf.md)
function for general-purpose unconstrained nonlinear optimization. The
algorithm is a quasi-Newton method with BFGS updating of the inverse
Hessian and a soft line search with adaptive trust-region radius
monitoring.

It is designed as a drop-in replacement for the original Fortran-based
[ucminf](https://CRAN.R-project.org/package=ucminf)
[`ucminf::ucminf()`](https://rdrr.io/pkg/ucminf/man/ucminf.html)
function. The original algorithm was written in Fortran by Hans Bruun
Nielsen.

## Complete Credit

This reimplementation is based on the original repository
[ucminf](https://github.com/hdakpo/ucminf).

The implementation and techniques are derived from the original
Fortran-based algorithm described by: Nielsen, H. B. (2000) UCMINF - An
Algorithm For Unconstrained, Nonlinear Optimization, Report
IMM-REP-2000-19, Department of Mathematical Modelling, Technical
University of Denmark.

- You can consult this document
  [`here`](http://www.imm.dtu.dk/documents/ftp/tr00/tr19_00.pdf),

- The original Fortran source code is archived
  [`here`](https://web.archive.org/web/20050418082240/http://www.imm.dtu.dk/~hbn/Software/ucminf.f).

Dr. Nielsen passed away in 2015; the code was later modified for
integration with R packages. The structure of `ucminf` in R draws from
the [FortranCallsR](https://github.com/cran/FortranCallsR) package by
Diethelm Wuertz. Dr. Wuertz passed away in 2016

## Further Reading

- [Detailed walkthroughs and performance
  benchmarks:](https://alrobles.github.io/ucminfcpp/articles/ucminf_features.html)

- [Portability from Fortran to
  C++:](https://alrobles.github.io/ucminfcpp/articles/portability_from_fortran.html)

- [Use with C++
  :](https://alrobles.github.io/ucminfcpp/articles/cpp_use.html)

- [Use with `Python`
  :](https://alrobles.github.io/ucminfcpp/articles/python_manual.html)

- [Use with
  `Julia`:](https://alrobles.github.io/ucminfcpp/articles/julia_manual.html)
