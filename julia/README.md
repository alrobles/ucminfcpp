# ucminf — Julia Bindings

Julia bindings for the **ucminf** C++ unconstrained nonlinear optimization
library.  The bindings use [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl)
(JlCxx).

## Directory layout

```
julia/
├── Project.toml            # Julia package manifest
├── src/
│   └── Ucminfcpp.jl        # Julia module (wraps the CxxWrap shared lib)
├── test/
│   └── runtests.jl         # Test suite
├── ucminf_julia.cpp        # C++ source for the shared library
├── CMakeLists.txt          # CMake build for the shared library
└── README.md               # This file
```

## Prerequisites

* Julia ≥ 1.9
* [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl) ≥ 0.14
  (`] add CxxWrap` in the Julia REPL)
* A C++17-capable compiler (GCC ≥ 9 or Clang ≥ 10)
* CMake ≥ 3.16

## Step 1 — Build the shared library

```bash
# From the repository root
PREFIX=$(julia -e 'using CxxWrap; print(CxxWrap.prefix_path())')

cmake -S julia -B julia/build \
      -DBUILD_JULIA=ON \
      -DJlCxx_DIR="${PREFIX}/lib/cmake/JlCxx"
cmake --build julia/build
```

The resulting shared library (`libucminf_julia.so` / `.dylib` / `.dll`) is
placed in `julia/build/`.

## Step 2 — Use the Julia package

```julia
# Activate the package (from the repository root)
import Pkg
Pkg.activate("julia")
Pkg.instantiate()

using Ucminfcpp

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

## Running the tests

```bash
julia --project=julia julia/test/runtests.jl
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
