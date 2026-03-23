# Key Features of ucminfcpp

## Unconstrained Optimization

`ucminfcpp` solves **unconstrained** nonlinear optimization problems of
the form:

    minimise  f(x),   x ∈ ℝⁿ

The underlying algorithm is a quasi-Newton method with BFGS updating of
the inverse Hessian and a soft line search with an adaptive trust-region
radius. No constraints on `x` are imposed; if you need bounds, consider
wrapping the problem with a barrier or penalty function.

### Example — Himmelblau’s Function

Himmelblau’s function has four local minima. Starting from different
initial points illustrates how the optimizer converges to the nearest
minimum.

``` r
himmelblau <- function(x) {
  (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
}

# Four different starting points → four different minima
starts <- list(c(3, 2), c(-2.805, 3.131), c(-3.779, -3.283), c(3.584, -1.848))
for (s in starts) {
  res <- ucminfcpp::ucminf(s, himmelblau)
  cat(sprintf("start = (%.2f, %.2f)  ->  par = (%.4f, %.4f),  f = %.2e\n",
              s[1], s[2], res$par[1], res$par[2], res$value))
}
#> start = (3.00, 2.00)  ->  par = (3.0000, 2.0000),  f = 0.00e+00
#> start = (-2.81, 3.13)  ->  par = (-2.8051, 3.1313),  f = 5.97e-11
#> start = (-3.78, -3.28)  ->  par = (-3.7793, -3.2832),  f = 1.26e-10
#> start = (3.58, -1.85)  ->  par = (3.5844, -1.8481),  f = 7.80e-12
```

## Flexibility in Multiple Dimensions

`ucminfcpp` handles objective functions of arbitrary dimension `n`. The
only requirement is that the function accepts a numeric vector of length
`n` and returns a scalar.

### Example — Sphere Function in 10 Dimensions

``` r
sphere <- function(x) sum(x^2)

x0  <- rep(2, 10)
res <- ucminfcpp::ucminf(x0, sphere)
cat("Minimum:", res$value, "\n")
#> Minimum: 3.131017e-16
cat("Par (first 5):", round(res$par[1:5], 8), "\n")
#> Par (first 5): -1e-08 -1e-08 -1e-08 -1e-08 -1e-08
```

### Example — Rastrigin Function in 5 Dimensions

The Rastrigin function is highly multimodal, but starting near the
origin still converges to the global minimum.

``` r
rastrigin <- function(x) {
  n <- length(x)
  10 * n + sum(x^2 - 10 * cos(2 * pi * x))
}

res_r <- ucminfcpp::ucminf(rep(0.1, 5), rastrigin)
cat("Minimum:", res_r$value, "\n")
#> Minimum: 1.421085e-14
cat("Par:", round(res_r$par, 6), "\n")
#> Par: 0 0 0 0 0
```

## Handling Edge Cases

### Supplying an Analytical Gradient

Providing an exact gradient via the `gr` argument speeds up convergence
and improves accuracy, because finite-difference approximation is
skipped.

``` r
banana      <- function(x) 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
banana_grad <- function(x) {
  c(-400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]),
     200 * (x[2] - x[1]^2))
}

res_g <- ucminfcpp::ucminf(c(-1.2, 1), banana, gr = banana_grad)
cat("par:", res_g$par, "\n")
#> par: 1 1
cat("convergence:", res_g$convergence, "\n")
#> convergence: 1
```

### Already-at-Minimum Starting Point

Starting exactly at the minimum should return immediately without
degrading the result.

``` r
quadratic <- function(x) (x[1] - 3)^2 + (x[2] + 5)^2
res_min   <- ucminfcpp::ucminf(c(3, -5), quadratic)
cat("f(3,-5) =", res_min$value, "\n")   # should be ~0
#> f(3,-5) = 0
```

### Flat Regions

Functions with a broad, nearly flat minimum are handled gracefully; the
optimizer stops when the gradient norm falls below the tolerance.

``` r
flat_fn <- function(x) {
  exp(-0.01 * sum(x^2))   # very flat; minimum spread over a large area
}

# Negate to turn maximum into minimum
res_flat <- ucminfcpp::ucminf(c(5, 5), function(x) -flat_fn(x))
cat("convergence code:", res_flat$convergence, "\n")
#> convergence code: 1
```

## Controlling the Optimizer

All tuning parameters of the original `ucminf` are supported:

| Parameter         | Default                     | Description                            |
|-------------------|-----------------------------|----------------------------------------|
| `control$maxeval` | 500                         | Maximum number of function evaluations |
| `control$trace`   | 0                           | Verbosity level (0 = silent)           |
| `control$eps`     | `sqrt(.Machine$double.eps)` | Gradient convergence tolerance         |
| `control$stepmax` | 1                           | Maximum step length                    |

``` r
res_ctrl <- ucminfcpp::ucminf(
  c(-1.2, 1), banana,
  control = list(maxeval = 1000, trace = 0)
)
cat("evaluations:", res_ctrl$info["neval"], "\n")
#> evaluations: 42
```
