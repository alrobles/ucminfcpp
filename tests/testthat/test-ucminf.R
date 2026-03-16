## Tests for the ucminfcpp package
## These tests verify the Fortran-to-C translation and Rcpp interface by
## optimizing the Rosenbrock Banana function (known solution: x = (1, 1)).

library(testthat)
library(ucminfcpp)

## Known objective: Rosenbrock Banana function
## Minimum at (1, 1) with F(1,1) = 0
fR <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
gR <- function(x) c(
    -400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1]),
     200 * (x[2] - x[1] * x[1])
)

## ---- Basic optimization with analytic gradient ----

test_that("Rosenbrock with analytic gradient converges to (1,1)", {
    res <- ucminf(par = c(2, 0.5), fn = fR, gr = gR)
    expect_true(res$convergence > 0,
                info = paste("Expected positive convergence code, got", res$convergence))
    expect_equal(res$par, c(1, 1), tolerance = 1e-5,
                 info = "Minimizer should be (1, 1)")
    expect_equal(res$value, 0, tolerance = 1e-10,
                 info = "Minimum value should be 0")
})

test_that("Rosenbrock with analytic gradient: gradient near zero at solution", {
    res <- ucminf(par = c(2, 0.5), fn = fR, gr = gR)
    g_sol <- gR(res$par)
    expect_lt(max(abs(g_sol)), 1e-5,
              label = "gradient norm at solution")
})

## ---- Basic optimization with finite-difference gradient ----

test_that("Rosenbrock with forward finite-difference gradient converges", {
    res <- ucminf(par = c(2, 0.5), fn = fR,
                  control = list(grad = "forward"))
    expect_true(res$convergence > 0)
    expect_equal(res$par, c(1, 1), tolerance = 1e-3)   # FD is less accurate
    expect_equal(res$value, 0, tolerance = 1e-5)
})

test_that("Rosenbrock with central finite-difference gradient converges", {
    res <- ucminf(par = c(2, 0.5), fn = fR,
                  control = list(grad = "central"))
    expect_true(res$convergence > 0)
    expect_equal(res$par, c(1, 1), tolerance = 1e-5)
    expect_equal(res$value, 0, tolerance = 1e-9)
})

## ---- Starting from the minimum ----

test_that("Starting exactly at the minimum returns convergence=1 immediately", {
    res <- ucminf(par = c(1, 1), fn = fR, gr = gR)
    expect_equal(res$convergence, 1L)   # small gradient
    expect_equal(res$par, c(1, 1), tolerance = 1e-10)
})

## ---- Return value structure ----

test_that("ucminf returns a list with expected elements", {
    res <- ucminf(par = c(2, 0.5), fn = fR, gr = gR)
    expect_s3_class(res, "ucminf")
    expect_true(all(c("par", "value", "convergence", "message",
                      "invhessian.lt", "info") %in% names(res)))
})

test_that("info vector contains neval, maxgradient, laststep, stepmax", {
    res <- ucminf(par = c(2, 0.5), fn = fR, gr = gR)
    expect_named(res$info, c("maxgradient", "laststep", "stepmax", "neval"))
    expect_gt(res$info["neval"], 0)
})

## ---- Hessian output ----

test_that("hessian=2 returns inverse Hessian matrix", {
    res <- ucminf(par = c(2, 0.5), fn = fR, gr = gR, hessian = 2)
    expect_true(!is.null(res$invhessian))
    expect_equal(dim(res$invhessian), c(2, 2))
    ## Inverse Hessian at (1,1) for Rosenbrock should be positive definite
    ev <- eigen(res$invhessian, symmetric = TRUE)$values
    expect_true(all(ev > 0), info = "inverse Hessian should be positive definite")
})

## ---- Named parameters ----

test_that("parameter names are preserved in output", {
    res <- ucminf(par = c(x1 = 2, x2 = 0.5), fn = fR, gr = gR)
    expect_named(res$par, c("x1", "x2"))
})

## ---- Control parameters ----

test_that("maxeval limit is respected", {
    res <- ucminf(par = c(2, 0.5), fn = fR, gr = gR,
                  control = list(maxeval = 5))
    expect_lte(res$info["neval"], 10)  # allow some slack for line search
})

test_that("grtol control is respected", {
    res <- ucminf(par = c(2, 0.5), fn = fR, gr = gR,
                  control = list(grtol = 1e-3))
    expect_lte(res$info["maxgradient"], 1e-3 + 1e-10)
})

## ---- Higher-dimensional test ----

test_that("3D Rosenbrock converges", {
    ## Extended Rosenbrock (3D):  sum_{i=1}^{2} [(1-x_i)^2 + 100*(x_{i+1}-x_i^2)^2]
    ## Minimum at (1,1,1) with F=0
    fR3 <- function(x)
        (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2 +
        (1 - x[2])^2 + 100 * (x[3] - x[2]^2)^2
    gR3 <- function(x) c(
        -400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]),
         200 * (x[2] - x[1]^2) - 400 * x[2] * (x[3] - x[2]^2) - 2 * (1 - x[2]),
         200 * (x[3] - x[2]^2)
    )
    res <- ucminf(par = c(2, 1, 0.5), fn = fR3, gr = gR3,
                  control = list(maxeval = 2000))
    expect_true(res$convergence %in% c(1L, 2L))
    expect_equal(res$par, c(1, 1, 1), tolerance = 1e-4)
})

## ---- Simple quadratic (exact in one step) ----

test_that("Simple quadratic F(x)=x^2 converges to 0", {
    fQ <- function(x) x^2
    gQ <- function(x) 2 * x
    res <- ucminf(par = 3, fn = fQ, gr = gQ)
    expect_true(res$convergence > 0)
    expect_equal(as.numeric(res$par), 0, tolerance = 1e-6)
    expect_equal(res$value, 0, tolerance = 1e-12)
})

## ---- Error handling ----

test_that("unknown control parameters raise an error", {
    expect_error(
        ucminf(par = c(1, 1), fn = fR, gr = gR, control = list(unknown_param = 1)),
        regexp = "Unknown control"
    )
})
