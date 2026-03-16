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
| **Modern C++17 core** | No raw pointers, no manual memory management. `std::vector`, `std::function`, RAII throughout. |
| **R interface** | Drop-in replacement for `ucminf::ucminf()` via Rcpp. |
| **Python bindings** | `pybind11`-based module in `python/`. |
| **Julia bindings** | `CxxWrap`-based module in `julia/`. |
| **C++ unit tests** | Catch2 test suite covering the pure C++ API. |
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