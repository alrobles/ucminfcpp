library(testthat)
library(ucminfcpp)

# Helper: compile the XPtr demo inline (avoids dependency on inst/examples path)
make_xptr <- function() {
  skip_if_not_installed("Rcpp")
  # Inline version of the rosenbrock XPtr demo
  Rcpp::cppFunction(
    depends  = "ucminfcpp",
    includes = '#include "ucminf_core.hpp"',
    code     = '
SEXP make_rosen() {
  auto* fn = new ucminf::ObjFun(
    [](const std::vector<double>& x, std::vector<double>& g, double& f) {
      double a = 1.0 - x[0], b = x[1] - x[0]*x[0];
      f    =  a*a + 100.0*b*b;
      g[0] = -2.0*a - 400.0*x[0]*b;
      g[1] =  200.0*b;
    });
  return Rcpp::XPtr<ucminf::ObjFun>(fn, true);
}
')
  make_rosen()
}

test_that("ucminf_xptr returns an externalptr", {
  xp <- make_xptr()
  expect_true(is(xp, "externalptr"))
})

test_that("XPtr Rosenbrock gives correct minimum", {
  xp  <- make_xptr()
  res <- ucminfcpp::ucminf_xptr(c(2, 0.5), xp)
  expect_equal(res$par,   c(1, 1), tolerance = 1e-5)
  expect_equal(res$value, 0,       tolerance = 1e-10)
})

test_that("XPtr result matches R-function result", {
  skip_if_not_installed("ucminf")
  xp  <- make_xptr()
  r1  <- ucminf::ucminf(c(2, 0.5),
           function(x) (1-x[1])^2+100*(x[2]-x[1]^2)^2,
           function(x) c(-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]), 200*(x[2]-x[1]^2)))
  r2  <- ucminfcpp::ucminf_xptr(c(2, 0.5), xp)
  expect_equal(r1$par,   r2$par,   tolerance = 1e-6)
  expect_equal(r1$value, r2$value, tolerance = 1e-10)
})

test_that("ucminf_xptr result has ucminf class", {
  xp  <- make_xptr()
  res <- ucminfcpp::ucminf_xptr(c(2, 0.5), xp)
  expect_s3_class(res, "ucminf")
})

test_that("ucminf_xptr returns hessian when requested", {
  xp  <- make_xptr()
  res <- ucminfcpp::ucminf_xptr(c(2, 0.5), xp, hessian = 2)
  expect_true(!is.null(res$invhessian))
  expect_equal(dim(res$invhessian), c(2, 2))
})
