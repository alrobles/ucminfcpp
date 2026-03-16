"""
    Ucminfcpp

Julia package providing bindings to the **ucminf** C++ unconstrained
nonlinear optimization library.  The underlying algorithm is a quasi-Newton
method with BFGS inverse-Hessian updating and a soft line search with
adaptive trust-region radius monitoring.

The shared library (`libucminf_julia`) must be built with CMake before this
module can be loaded.  See the build instructions in the top-level README or
`julia/README.md`.

# Quick start

```julia
using Ucminfcpp

fn = x -> (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2
gr = x -> [-400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]),
            200*(x[2]-x[1]^2)]

result = minimize(Float64[2.0, 0.5], fn, gr)
println("x = ", result.x())   # ≈ [1.0, 1.0]
println("f = ", result.f())   # ≈ 0.0
```
"""
module Ucminfcpp

export minimize, Control, set_grtol!, set_xtol!, set_stepmax!, set_maxeval!,
       set_inv_hessian_lt!, status_message

using CxxWrap

# ---------------------------------------------------------------------------
# Load the shared library
# ---------------------------------------------------------------------------
# The default search path is <package root>/build/julia/libucminf_julia.
# Override by calling `Ucminfcpp.set_library_path!(path)` before the first
# `using Ucminfcpp` / `import Ucminfcpp`.
# ---------------------------------------------------------------------------

const _LIB_SEARCH_PATHS = [
    # Relative to the package root (works after `cmake --build build`)
    joinpath(@__DIR__, "..", "build", "julia", "libucminf_julia"),
    joinpath(@__DIR__, "..", "build", "libucminf_julia"),
    # System paths
    "libucminf_julia",
]

function _find_library()
    for p in _LIB_SEARCH_PATHS
        # CxxWrap expects just the stem; the OS picks the right extension.
        candidate = p
        # Try with and without extension
        for suffix in ("", ".so", ".dylib", ".dll")
            full = candidate * suffix
            if isfile(full)
                return candidate  # return stem, let CxxWrap add the extension
            end
        end
    end
    error("""
    ucminf Julia library not found.  Please build it first (from the repository root):

        PREFIX=\$(julia -e 'using CxxWrap; print(CxxWrap.prefix_path())')
        cmake -S julia -B julia/build \\
              -DBUILD_JULIA=ON \\
              -DJlCxx_DIR="\${PREFIX}/lib/cmake/JlCxx"
        cmake --build julia/build

    Then re-run `using Ucminfcpp`.
    """)
end

@wrapmodule(() -> _find_library())

function __init__()
    @initcxx
end

end # module Ucminfcpp
