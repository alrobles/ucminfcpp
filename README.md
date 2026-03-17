# ucminfcpp

Unconstrained nonlinear optimization via **modern C++17** and Rcpp — a
complete reimplementation of the original Fortran-based
[ucminf](https://CRAN.R-project.org/package=ucminf) R package with
multi-language support.

## Overview

**ucminfcpp** provides the `ucminf()` function for general-purpose
unconstrained nonlinear optimization. The algorithm is a quasi-Newton method
with BFGS updating of the inverse Hessian and a soft line search with
adaptive trust-region radius monitoring. It is designed as a drop-in
replacement for the original `ucminf::ucminf()` function.

The original algorithm was written in Fortran by Hans Bruun Nielsen (IMM,
DTU, 2000) and described in:

> H.B. Nielsen, "UCMINF — An Algorithm for Unconstrained, Nonlinear
> Optimization", Report IMM-REP-2000-19, DTU, December 2000.

**Complete Credit:** This reimplementation is based on the original repository [hdakpo/ucminf](https://github.com/hdakpo/ucminf). The implementation and techniques are derived from the original Fortran-based algorithm described by:

- Nielsen, H. B. (2000) ‘UCMINF - An Algorithm For Unconstrained, Nonlinear Optimization’, Report IMM-REP-2000-19, Department of Mathematical Modelling, Technical University of Denmark. http://www.imm.dtu.dk/documents/ftp/tr00/tr19_00.pdf

The original Fortran source code is located at
http://www2.imm.dtu.dk/projects/hbn_software/ucminf.f (no longer available but archived at https://web.archive.org/web/20050418082240/http://www.imm.dtu.dk/~hbn/Software/ucminf.f). Dr. Nielsen passed away in 2015, and the code was slightly modified to integrate with R-based packages. The structure of the implementation in R draws from the ‘FortranCallsR’ package by Diethelm Wuertz.

### Key features

| Feature | Description |
|---------|-------------|
| **Dual-layer C++ API** | `minimize()` (high-level, `std::function`) and `minimize_direct<F>()` (zero-overhead template). |
| **R interface** | Drop-in replacement for `ucminf::ucminf()` via Rcpp. |
| **Python bindings** | `pybind11`-based module in `python/`. |
| **Julia bindings** | `CxxWrap`-based module in `julia/`. |
| **C++ unit tests** | Catch2 test suite covering both API entry points. |
| **CMake build** | Standalone C++ build for embedding in other projects. |

## Installation

### R package

Install from GitHub with `remotes`:

```r
# install.packages("remotes")
remotes::install_github("alrobles/ucminfcpp")
```
...

## Algorithm details

The UCMINF algorithm combines three techniques...

## C++ API

Two entry points share the same optimised kernel.

### High-level API — `ucminf::minimize()`

Accepts an `ObjFun` (`std::function`) callback. Convenient for any context
where the callable type is not known at compile time (R/Python/Julia bridges).

```cpp
#include "ucminf_core.hpp"

ucminf::Result res = ucminf::minimize(
    {2.0, 0.5},                                          // x0
    [](const std::vector<double>& x,
       std::vector<double>& g, double& f) {              // ObjFun callback
        double a = 1.0 - x[0], b = x[1] - x[0]*x[0];
        f    = a*a + 100.0*b*b;
        g[0] = -2.0*a - 400.0*x[0]*b;
        g[1] =  200.0*b;
    }
);
```

### Low-level template API — `ucminf::minimize_direct<F>()`

Accepts any callable `F` without `std::function` type erasure. When `F` is
a plain function pointer, a captureless lambda, or a trivially inlinable
functor the compiler can inline the entire `fdf` call, eliminating
virtual-dispatch overhead and enabling constant-propagation across the hot
loop. Prefer this when calling from pure C++ or from statically-typed
language bindings.

```cpp
// Plain function pointer — fully inlinable
void rosenbrock(const std::vector<double>& x,
                std::vector<double>& g, double& f) { /* ... */ }

ucminf::Result res = ucminf::minimize_direct({2.0, 0.5}, rosenbrock);

// Lambda — also fully inlinable by the compiler
auto fn = [](const std::vector<double>& x,
             std::vector<double>& g, double& f) { /* ... */ };
ucminf::Result res2 = ucminf::minimize_direct({2.0, 0.5}, fn);
```

Both overloads accept the same `ucminf::Control` struct and return the same
`ucminf::Result`. Workspace buffers (for the line search) are allocated once
per call and reused across all iterations.

## R usage and performance notes

When `ucminfcpp::ucminf()` is called from R, every evaluation of `fn` and
`gr` crosses the R interpreter boundary (an R → C++ → R round-trip). This
overhead dominates runtime for most problems and is independent of the C++
implementation quality:

```r
library(microbenchmark)
library(ucminf)
library(ucminfcpp)

fn <- function(x) (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2
gr <- function(x) c(-400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]),
                     200*(x[2]-x[1]^2))

microbenchmark(
  ucminf    = ucminf::ucminf(c(2, 0.5), fn, gr),
  ucminfcpp = ucminfcpp::ucminf(c(2, 0.5), fn, gr),
  times = 200
)
```

**To minimize R callback overhead:**

* Pre-compute expensive invariants outside `fn`/`gr`.
* Implement `fn`/`gr` in C++ and call via `minimize_direct<F>()`.
* Use compiled R code (e.g. `Rcpp::cppFunction`) for the objective.
* For high-dimensional problems (n ≥ 20), the benefit of the optimized
  C++ kernel increases relative to per-call R overhead.

