# Introduction to ucminfcpp

## Overview

`ucminfcpp` is an R package that provides unconstrained nonlinear
optimization through a modern **C++17** reimplementation of the classic
Fortran-based [ucminf](https://CRAN.R-project.org/package=ucminf)
algorithm. It is a **drop-in replacement** for
[`ucminf::ucminf()`](https://rdrr.io/pkg/ucminf/man/ucminf.html).

The package was primarily developed to power the
[`nicher`](https://github.com/alrobles/nicher) and
[`xsdmMle`](https://github.com/alrobles/xsdmMle) packages, where fast
and reliable unconstrained optimization is central to fitting ecological
niche models and maximum-likelihood estimators.

## Optimization Showcases

### 1. Rosenbrock’s Banana Function

The Banana Function is a classic benchmark for optimization algorithms.
Its global minimum is at `(1, 1)`.

``` r
banana <- function(x) {
  100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
}

result <- ucminfcpp::ucminf(c(-1.2, 1), banana)
print(result$par)   # should be c(1, 1)
#> [1] 0.9996971 0.9993937
```

### 2. Quadratic Function

A simple quadratic with a known minimum at `(3, -5)`.

``` r
quadratic <- function(x) {
  (x[1] - 3)^2 + (x[2] + 5)^2
}

result_q <- ucminfcpp::ucminf(c(0, 0), quadratic)
print(result_q$par)   # should be c(3, -5)
#> [1]  3.000000 -5.000002
```

### 3. User-Defined Function

Any smooth, differentiable objective can be passed to
[`ucminfcpp::ucminf()`](https://alrobles.github.io/ucminfcpp/reference/ucminf.md).

``` r
custom_fn <- function(x) {
  sum(x^2 - log(1 + x^2))
}

result_c <- ucminfcpp::ucminf(c(1, 1, 1), custom_fn)
print(result_c$par)   # should be close to c(0, 0, 0)
#> [1] 0.00602089 0.00602089 0.00602089
```

## Verifying Consistency with ucminf

`ucminfcpp` is designed to be numerically equivalent to `ucminf`. The
following code confirms the roots match for the Banana Function.

``` r
starting_point <- c(-1.2, 1)

result_cpp     <- ucminfcpp::ucminf(starting_point, banana)
result_fortran <- ucminf::ucminf(starting_point, banana)

cat("ucminfcpp result:\n");  print(result_cpp$par)
#> ucminfcpp result:
#> [1] 0.9996971 0.9993937
cat("ucminf result:\n");     print(result_fortran$par)
#> ucminf result:
#> [1] 0.9996971 0.9993937

identical_result <- all.equal(result_cpp$par, result_fortran$par)
cat("Are the results the same?", identical_result, "\n")
#> Are the results the same? TRUE
```

## Performance Benchmarks

Even when calling pure R objective functions (which cross the
interpreter boundary on every evaluation), `ucminfcpp` typically
outperforms the Fortran-based `ucminf` due to reduced C/Fortran call
overhead in its C++ core.

``` r
library(microbenchmark)

benchmark_results <- microbenchmark(
  ucminfcpp = ucminfcpp::ucminf(c(-1.2, 1), banana),
  ucminf    = ucminf::ucminf(c(-1.2, 1), banana),
  times = 100L
)
print(benchmark_results)
#> Unit: microseconds
#>       expr     min       lq      mean   median       uq     max neval
#>  ucminfcpp  88.165  91.2055  96.49987  93.9005  97.2670 164.287   100
#>     ucminf 123.722 126.1805 131.60768 129.3415 131.9015 201.597   100
```

The benchmark shows the median execution time for each implementation
across 100 runs. `ucminfcpp` achieves comparable or better performance
while maintaining full numerical equivalence.

For even greater speed, implement the objective function in C++ and call
it via `ucminfcpp::minimize_direct<F>()` to eliminate all interpreter
overhead.
