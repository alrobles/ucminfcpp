# Reproducible benchmark: ucminf vs ucminfcpp (R functions + fdfun + XPtr)
# Run from R with:
#   source(system.file("benchmarks/bench_r_vs_cpp.R", package = "ucminfcpp"))

library(ucminf)
library(ucminfcpp)
library(microbenchmark)

# ---- Test functions ----

# Rosenbrock 2D
fn_rosen <- function(x) (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2
gr_rosen <- function(x) c(-400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]), 200*(x[2]-x[1]^2))
fdf_rosen <- function(x) {
  a <- 1-x[1]; b <- x[2]-x[1]^2
  list(f=a^2+100*b^2, g=c(-2*a-400*x[1]*b, 200*b))
}

# Wood 4D
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
fdf_wood <- function(x) list(f=fn_wood(x), g=gr_wood(x))

# Powell 4D
fn_powell <- function(x) {
  (x[1]+10*x[2])^2+5*(x[3]-x[4])^2+(x[2]-2*x[3])^4+10*(x[1]-x[4])^4
}

# ---- Benchmark: R functions (ucminf vs ucminfcpp vs fdfun) ----
cat("=== Rosenbrock 2D (analytic gradient) ===\n")
print(microbenchmark(
  ucminf        = ucminf::ucminf(c(2, 0.5), fn_rosen, gr_rosen),
  ucminfcpp     = ucminfcpp::ucminf(c(2, 0.5), fn_rosen, gr_rosen),
  ucminfcpp_fdf = ucminfcpp::ucminf(c(2, 0.5), fdfun = fdf_rosen),
  times = 1000
))

cat("\n=== Wood 4D (analytic gradient) ===\n")
x0_wood <- c(-3, -1, -3, -1)
print(microbenchmark(
  ucminf        = ucminf::ucminf(x0_wood, fn_wood, gr_wood),
  ucminfcpp     = ucminfcpp::ucminf(x0_wood, fn_wood, gr_wood),
  ucminfcpp_fdf = ucminfcpp::ucminf(x0_wood, fdfun = fdf_wood),
  times = 1000
))

cat("\n=== Powell 4D (numeric gradient) ===\n")
x0_pw <- c(3, -1, 0, 1)
print(microbenchmark(
  ucminf    = ucminf::ucminf(x0_pw, fn_powell),
  ucminfcpp = ucminfcpp::ucminf(x0_pw, fn_powell),
  times = 500
))

cat("\n=== XPtr Rosenbrock (compiled C++ lambda, no R callback overhead) ===\n")
cat("# To run the XPtr benchmark, compile the example first:\n")
cat("#   Rcpp::sourceCpp(system.file('examples/rosenbrock_xptr.cpp',\n")
cat("#                               package='ucminfcpp'))\n")
cat("#   xp <- make_rosenbrock_xptr()\n")
cat("#   microbenchmark(\n")
cat("#     ucminf_ref  = ucminf::ucminf(c(2,0.5), fn_rosen, gr_rosen),\n")
cat("#     ucminfcpp_r = ucminfcpp::ucminf(c(2,0.5), fn_rosen, gr_rosen),\n")
cat("#     ucminfcpp_x = ucminfcpp::ucminf_xptr(c(2,0.5), xp),\n")
cat("#     times=1000)\n")
