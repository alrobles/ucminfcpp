# Using ucminfcpp from Julia

## Overview

`ucminfcpp` provides **Julia bindings** via `CxxWrap.jl`, located in the
`julia/` directory of the source tree. Julia users can call the same C++
optimizer core directly from Julia code, achieving performance on par
with native Julia optimization libraries.

## Prerequisites

- Julia ≥ 1.6
- A C++17-capable compiler (GCC ≥ 7, Clang ≥ 5)
- CMake ≥ 3.15
- [`CxxWrap.jl`](https://github.com/JuliaInterop/CxxWrap.jl) Julia
  package

Install `CxxWrap.jl` from the Julia REPL:

``` julia
using Pkg
Pkg.add("CxxWrap")
```

## Building the Julia Module

Clone the repository and build the Julia shared library:

``` bash
git clone https://github.com/alrobles/ucminfcpp.git
cd ucminfcpp

cmake -S . -B build \
      -DBUILD_JULIA=ON \
      -DBUILD_PYTHON=OFF \
      -DBUILD_TESTS=OFF

cmake --build build
```

The shared library (e.g. `libucminfcpp_julia.so`) is placed in the
`build/` directory.

## Loading the Module in Julia

``` julia
using CxxWrap

# Load the compiled library
@wrapmodule(() -> joinpath(@__DIR__, "build", "libucminfcpp_julia"))

@initcxx
```

## Basic Usage

Define your objective function as a Julia function that fills a gradient
vector in-place and returns the function value, then call `minimize`:

``` julia
function banana(x::Vector{Float64}, g::Vector{Float64})
    f    = 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2
    g[1] = -400.0 * x[1] * (x[2] - x[1]^2) - 2.0 * (1.0 - x[1])
    g[2] =  200.0 * (x[2] - x[1]^2)
    return f
end

x0     = [-1.2, 1.0]
result = minimize(x0, banana)

println("par  = ", result.par)         # [1.0, 1.0]
println("f    = ", result.value)       # ~0.0
println("conv = ", result.convergence)
```

## Step-by-Step Example: Quadratic Function

``` julia
function quadratic(x::Vector{Float64}, g::Vector{Float64})
    f    = (x[1] - 3.0)^2 + (x[2] + 5.0)^2
    g[1] = 2.0 * (x[1] - 3.0)
    g[2] = 2.0 * (x[2] + 5.0)
    return f
end

result = minimize([0.0, 0.0], quadratic)
println("Minimum at x = ", result.par)   # [3.0, -5.0]
```

## Numerical Gradient via FiniteDiff.jl

If deriving the gradient analytically is inconvenient, use
[FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl):

``` julia
using FiniteDiff

function rosenbrock_scalar(x::Vector{Float64})
    return 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2
end

function rosenbrock_with_grad(x::Vector{Float64}, g::Vector{Float64})
    FiniteDiff.finite_difference_gradient!(g, rosenbrock_scalar, x)
    return rosenbrock_scalar(x)
end

result = minimize([-1.2, 1.0], rosenbrock_with_grad)
println(result.par)   # close to [1.0, 1.0]
```

## Comparison with Optim.jl

`ucminfcpp` complements native Julia optimizers like
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl):

| Aspect                 | Optim.jl (BFGS)         | ucminfcpp                  |
|------------------------|-------------------------|----------------------------|
| Pure Julia             | Yes                     | No (C++ core)              |
| Auto-differentiation   | Via ForwardDiff.jl      | Not built-in               |
| Fortran-equivalent     | No                      | Yes (matches ucminf)       |
| Combined f+g interface | Optional                | Required                   |
| Use case               | General Julia workflows | Cross-language equivalence |

Use `ucminfcpp` from Julia when you need results that are **numerically
identical** to those obtained from the R or Fortran `ucminf` packages —
for example, when validating a Julia model against an R reference
implementation.

## High-Dimensional Example

``` julia
function sphere(x::Vector{Float64}, g::Vector{Float64})
    f = 0.0
    for i in eachindex(x)
        f    += x[i]^2
        g[i]  = 2.0 * x[i]
    end
    return f
end

x0     = fill(2.0, 20)
result = minimize(x0, sphere)
println("f(x*) ≈ ", result.value)   # ~0.0
```
