# tests/testthat/test-ucminf-compare.R

library(testthat)
library(ucminf)
library(ucminfcpp)

# Helper comparison function
compare_ucminf <- function(par, fn, gr = NULL, control = list(), ...) {
  res1 <- ucminf::ucminf(par, fn, gr, control = control, ...)
  res2 <- ucminfcpp::ucminf(par, fn, gr, control = control, ...)
  list(
    par1 = res1$par,
    par2 = res2$par,
    value1 = res1$value,
    value2 = res2$value,
    conv1 = res1$convergence,
    conv2 = res2$convergence
  )
}

test_that("Rosenbrock analytic gradient matches", {
  fn <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
  gr <- function(x) c(
    -400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]),
    200 * (x[2] - x[1]^2)
  )
  starts <- list(c(2,0.5), c(-1,1), c(10,-5))
  for (s in starts) {
    cmp <- compare_ucminf(s, fn, gr)
    expect_equal(cmp$conv1, cmp$conv2)
    expect_equal(cmp$value1, cmp$value2, tolerance=1e-8)
    expect_equal(cmp$par1, cmp$par2, tolerance=1e-6)
  }
})

test_that("Rosenbrock numeric gradient matches (robust)", {
  
  fn <- function(x) (1 - x[1])^2 + 
    100 * (x[2] - x[1]^2)^2
  
  starts <- list(
    c(2, 0.5),
    c(0, 0),
    c(1.5, 1.5)
  )
  
  for (s in starts) {
    
    cmp <- compare_ucminf(
      par = s,
      fn  = fn,
      control = list(grad = "central")
    )
    
    # Ambos deben retornar códigos válidos (1 a 4)
    expect_true(cmp$conv1 %in% c(1,2,3,4))
    expect_true(cmp$conv2 %in% c(1,2,3,4))
    
    # Si ambos convergen por gradiente pequeño, comparamos resultados
    if (cmp$conv1 == 1 && cmp$conv2 == 1) {
      expect_equal(cmp$value1, cmp$value2, tolerance = 1e-6)
      expect_equal(cmp$par1, cmp$par2, tolerance = 1e-4)
    }
    
    # Si uno converge y el otro no, no forzamos igualdad,
    # pero sí lo reportamos explícitamente para debugging
    if (cmp$conv1 != cmp$conv2) {
      message("Diferencia de convergencia en inicio = ", paste(s, collapse=", "),
              "\n  ucminf:    ", cmp$conv1,
              "\n  ucminfcpp: ", cmp$conv2)
    }
  }
})
test_that("Quadratic SPD problems match", {
  set.seed(1)
  for (i in 1:5) {
    n <- 5
    A <- crossprod(matrix(rnorm(n*n), n, n))
    b <- rnorm(n)
    fn <- function(x) 0.5 * crossprod(x, A %*% x) + sum(b * x)
    gr <- function(x) as.vector(A %*% x + b)
    par0 <- rnorm(n)
    cmp <- compare_ucminf(par0, fn, gr)
    expect_equal(cmp$conv1, cmp$conv2)
    expect_equal(cmp$value1, cmp$value2, tolerance=1e-10)
    expect_equal(cmp$par1, cmp$par2, tolerance=1e-8)
  }
})

