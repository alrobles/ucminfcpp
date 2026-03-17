library(testthat)
library(ucminfcpp)

test_that("ucminf_control() returns a list with class ucminf_control", {
  ctrl <- ucminf_control()
  expect_s3_class(ctrl, "ucminf_control")
  expect_type(ctrl, "list")
})

test_that("ucminf_control() has correct default values and types", {
  ctrl <- ucminf_control()
  expect_equal(ctrl$grtol,    1e-6)
  expect_equal(ctrl$xtol,     1e-12)
  expect_equal(ctrl$stepmax,  1.0)
  expect_equal(ctrl$maxeval,  500L)
  expect_equal(ctrl$grad,     "forward")
  expect_equal(ctrl$gradstep, c(1e-6, 1e-8))
  expect_null(ctrl$invhessian.lt)
})

test_that("ucminf_control() accepts and overrides individual values", {
  ctrl <- ucminf_control(grtol = 1e-8, maxeval = 200L, grad = "central")
  expect_equal(ctrl$grtol,   1e-8)
  expect_equal(ctrl$maxeval, 200L)
  expect_equal(ctrl$grad,    "central")
})

test_that("ucminf() accepts a ucminf_control() object unchanged", {
  fn <- function(x) (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
  gr <- function(x) c(-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]), 200*(x[2]-x[1]^2))
  ctrl <- ucminf_control(grtol = 1e-8)
  expect_no_error(ucminfcpp::ucminf(c(2, 0.5), fn, gr, control = ctrl))
})

test_that("ucminf_control() rejects unknown parameter names via ucminf()", {
  fn <- function(x) sum(x^2)
  expect_error(ucminfcpp::ucminf(c(1,1), fn, control = list(badparam = 1)))
})

test_that("ucminf_control() rejects non-positive maxeval", {
  expect_error(ucminf_control(maxeval = 0))
  expect_error(ucminf_control(maxeval = -1))
})

test_that("ucminf_control() rejects non-positive tolerances", {
  expect_error(ucminf_control(grtol = 0))
  expect_error(ucminf_control(xtol  = -1e-12))
})

test_that("ucminf_control() rejects wrong grad value", {
  expect_error(ucminf_control(grad = "complex"))
})
