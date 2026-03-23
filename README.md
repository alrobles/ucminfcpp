
<!-- badges: start -->

<!-- badges: end -->

# ucminfcpp

`ucminfcpp` is an R package that provides unconstrained nonlinear
optimization through a modern **C++17** reimplementation of the classic
Fortran-based [ucminf](https://CRAN.R-project.org/package=ucminf)
algorithm. It is a **drop-in replacement** for `ucminf::ucminf()` and
was primarily developed to power the
[`@alrobles/nicher`](https://github.com/alrobles/nicher) and
[`@alrobles/xsdmMle`](https://github.com/alrobles/xsdmMle) packages.

## Installation

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

## Further Reading

For detailed walkthroughs, performance benchmarks, and language-specific
guides (C++, Python, Julia), visit the [`ucminfcpp` documentation
site](https://alrobles.github.io/ucminfcpp/).
