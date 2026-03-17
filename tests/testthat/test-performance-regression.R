library(testthat)
library(ucminfcpp)

testthat::skip_if_not_installed("ucminf")
library(ucminf)

# ---- helper functions ----
fn_rosen <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
gr_rosen <- function(x) c(-400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]), 200*(x[2]-x[1]^2))

fn_wood <- function(x) {
  100*(x[2]-x[1]^2)^2 + (1-x[1])^2 + 90*(x[4]-x[3]^2)^2 + (1-x[3])^2 +
    10.1*((x[2]-1)^2 + (x[4]-1)^2) + 19.8*(x[2]-1)*(x[4]-1)
}
gr_wood <- function(x) {
  c(-400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]),
     200*(x[2]-x[1]^2) + 20.2*(x[2]-1) + 19.8*(x[4]-1),
    -360*x[3]*(x[4]-x[3]^2) - 2*(1-x[3]),
     180*(x[4]-x[3]^2) + 20.2*(x[4]-1) + 19.8*(x[2]-1))
}

fn_powell <- function(x) {
  (x[1]+10*x[2])^2 + 5*(x[3]-x[4])^2 + (x[2]-2*x[3])^4 + 10*(x[1]-x[4])^4
}

fn_beale <- function(x) {
  (1.5 - x[1]*(1-x[2]))^2 + (2.25 - x[1]*(1-x[2]^2))^2 + (2.625 - x[1]*(1-x[2]^3))^2
}

tol <- 1e-5

test_that("Rosenbrock 2D: ucminfcpp matches ucminf numerically", {
  r1 <- ucminf::ucminf(c(2, 0.5), fn_rosen, gr_rosen)
  r2 <- ucminfcpp::ucminf(c(2, 0.5), fn_rosen, gr_rosen)
  expect_equal(r1$par,   r2$par,   tolerance = tol)
  expect_equal(r1$value, r2$value, tolerance = tol)
  expect_equal(r1$convergence, r2$convergence)
})

test_that("Wood 4D: ucminfcpp matches ucminf numerically", {
  x0 <- c(-3, -1, -3, -1)
  r1 <- ucminf::ucminf(x0, fn_wood, gr_wood)
  r2 <- ucminfcpp::ucminf(x0, fn_wood, gr_wood)
  expect_equal(r1$par,   r2$par,   tolerance = tol)
  expect_equal(r1$value, r2$value, tolerance = tol)
})

test_that("Powell 4D: ucminfcpp matches ucminf numerically", {
  x0 <- c(3, -1, 0, 1)
  r1 <- ucminf::ucminf(x0, fn_powell)
  r2 <- ucminfcpp::ucminf(x0, fn_powell)
  expect_equal(r1$value, r2$value, tolerance = 1e-4)
})

test_that("Beale 2D: ucminfcpp matches ucminf numerically", {
  x0 <- c(1, 1)
  r1 <- ucminf::ucminf(x0, fn_beale)
  r2 <- ucminfcpp::ucminf(x0, fn_beale)
  expect_equal(r1$par,   r2$par,   tolerance = tol)
  expect_equal(r1$value, r2$value, tolerance = tol)
})

test_that("Wood 4D: ucminfcpp median not more than 2x slower than ucminf", {
  # Performance threshold is intentionally generous (2x) to be stable across
  # different hardware and CI environments. The goal is to catch severe
  # regressions, not to enforce a tight ratio.
  skip_if_not_installed("microbenchmark")
  library(microbenchmark)
  x0 <- c(-3, -1, -3, -1)
  b <- microbenchmark(
    ucminf    = ucminf::ucminf(x0, fn_wood, gr_wood),
    ucminfcpp = ucminfcpp::ucminf(x0, fn_wood, gr_wood),
    times = 100
  )
  s <- summary(b)
  ratio <- s$median[s$expr == "ucminfcpp"] / s$median[s$expr == "ucminf"]
  expect_lte(ratio, 2.0)
})
