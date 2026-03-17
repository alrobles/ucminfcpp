library(testthat)
library(ucminfcpp)

fn_rosen <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
gr_rosen <- function(x) c(-400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]), 200*(x[2]-x[1]^2))

fdf_rosen <- function(x) {
  a <- 1 - x[1]; b <- x[2] - x[1]^2
  list(f = a^2 + 100*b^2,
       g = c(-2*a - 400*x[1]*b, 200*b))
}

fn_wood <- function(x) {
  100*(x[2]-x[1]^2)^2+(1-x[1])^2+90*(x[4]-x[3]^2)^2+(1-x[3])^2+
    10.1*((x[2]-1)^2+(x[4]-1)^2)+19.8*(x[2]-1)*(x[4]-1)
}
gr_wood <- function(x) {
  c(-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]),
     200*(x[2]-x[1]^2)+20.2*(x[2]-1)+19.8*(x[4]-1),
    -360*x[3]*(x[4]-x[3]^2)-2*(1-x[3]),
     180*(x[4]-x[3]^2)+20.2*(x[4]-1)+19.8*(x[2]-1))
}
fdf_wood <- function(x) list(f = fn_wood(x), g = gr_wood(x))

test_that("fdfun interface matches fn+gr interface on Rosenbrock", {
  r1 <- ucminfcpp::ucminf(c(2, 0.5), fn_rosen, gr_rosen)
  r2 <- ucminfcpp::ucminf(c(2, 0.5), fdfun = fdf_rosen)
  expect_equal(r1$par,   r2$par,   tolerance = 1e-8)
  expect_equal(r1$value, r2$value, tolerance = 1e-8)
})

test_that("fdfun interface matches fn+gr interface on Wood 4D", {
  x0 <- c(-3, -1, -3, -1)
  r1 <- ucminfcpp::ucminf(x0, fn_wood, gr_wood)
  r2 <- ucminfcpp::ucminf(x0, fdfun = fdf_wood)
  expect_equal(r1$par,   r2$par,   tolerance = 1e-8)
  expect_equal(r1$value, r2$value, tolerance = 1e-8)
})

test_that("fdfun interface errors when fdfun returns wrong-length g", {
  bad_fdf <- function(x) list(f = sum(x^2), g = c(1.0))  # wrong length for n=2
  expect_error(ucminfcpp::ucminf(c(1, 2), fdfun = bad_fdf))
})

test_that("fdfun interface errors when fdfun returns non-list", {
  bad_fdf <- function(x) sum(x^2)
  expect_error(ucminfcpp::ucminf(c(1, 2), fdfun = bad_fdf))
})
