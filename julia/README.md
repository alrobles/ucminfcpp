# ucminf — Julia Bindings

Julia bindings for the **ucminf** C++ unconstrained nonlinear optimization
library.  The bindings use [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl)
(JlCxx).

## Prerequisites

* Julia ≥ 1.9
* [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl) ≥ 0.14
  (`] add CxxWrap` in the Julia REPL)
* A C++17-capable compiler (GCC ≥ 9 or Clang ≥ 10)
* CMake ≥ 3.16

## Building the shared library

```bash
# 1. Find the JlCxx CMake prefix from Julia
PREFIX=$(julia -e 'using CxxWrap; print(CxxWrap.prefix_path())')

# 2. Configure and build
cmake -S . -B build \
      -DBUILD_JULIA=ON \
      -DJlCxx_DIR="${PREFIX}/lib/cmake/JlCxx"
cmake --build build
```

The resulting shared library (`ucminf_julia.so` / `.dylib` / `.dll`) can be
loaded directly from Julia.

## Usage

```julia
using CxxWrap

# Load the shared library
@wrapmodule(() -> joinpath(@__DIR__, "build", "julia", "libucminf_julia"))

@initcxx

# Objective function (Rosenbrock Banana)
fn = x -> (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2

# Analytic gradient
gr = x -> [
    -400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]),
     200*(x[2]-x[1]^2)
]

ctrl = Control()
set_grtol!(ctrl, 1e-8)

result = minimize(Float64[2.0, 0.5], fn, gr, ctrl)

println("x       = ", result.x())
println("f       = ", result.f())
println("n_eval  = ", result.n_eval())
println("message = ", result.message())
```

## Without an analytic gradient

```julia
# Omit `gr`; a central finite difference is used automatically.
result = minimize(Float64[2.0, 0.5], fn, ctrl)
```

## Control parameters

| Julia setter          | Default  | Description                                           |
|-----------------------|----------|-------------------------------------------------------|
| `set_grtol!(c, v)`    | `1e-6`   | Stop when max\|grad\| ≤ v                            |
| `set_xtol!(c, v)`     | `1e-12`  | Stop when step ≤ v                                    |
| `set_stepmax!(c, v)`  | `1.0`    | Initial trust-region radius                           |
| `set_maxeval!(c, v)`  | `500`    | Maximum function + gradient evaluations               |
| `set_inv_hessian_lt!(c, v)` | `[]` | Initial inverse Hessian (packed lower triangle)   |

## Result accessors

| Julia method      | Description                                          |
|-------------------|------------------------------------------------------|
| `result.x()`      | Parameter values at the minimum                      |
| `result.f()`      | Objective value at x                                 |
| `result.n_eval()` | Total evaluations used                               |
| `result.max_gradient()` | max\|grad(x)\| at solution                    |
| `result.last_step()` | Length of the last step                           |
| `result.status()` | Status enum value                                    |
| `result.message()`| Human-readable convergence message                   |
| `result.inv_hessian_lt()` | Packed lower triangle of final inv-Hessian  |
