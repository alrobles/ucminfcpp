"""
Tests for the Ucminfcpp Julia package.

Run with:
    julia --project=julia julia/test/runtests.jl

The shared library must be built before running these tests.
See julia/README.md for build instructions.
"""

using Test

# Try to load the package; skip all tests gracefully if the library is not built.
lib_available = false
try
    using Ucminfcpp
    lib_available = true
catch e
    @warn "Ucminfcpp shared library not found — skipping tests." exception=e
end

@testset "Ucminfcpp" begin

    if !lib_available
        @warn "Library not built; all tests skipped."
        return
    end

    # ------------------------------------------------------------------
    # Rosenbrock Banana — minimum at (1, 1)
    # ------------------------------------------------------------------
    @testset "Rosenbrock with gradient" begin
        fn = x -> (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2
        gr = x -> [
            -400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]),
             200*(x[2]-x[1]^2)
        ]

        ctrl = Control()
        result = minimize(Float64[2.0, 0.5], fn, gr, ctrl)

        @test result.f() < 1e-10
        @test abs(result.x()[1] - 1.0) < 1e-5
        @test abs(result.x()[2] - 1.0) < 1e-5
        @test result.n_eval() > 0
        @test !isempty(result.message())
    end

    # ------------------------------------------------------------------
    # Rosenbrock without an analytic gradient (finite differences)
    # ------------------------------------------------------------------
    @testset "Rosenbrock finite differences" begin
        fn = x -> (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2

        ctrl = Control()
        result = minimize(Float64[2.0, 0.5], fn, ctrl)  # 3-arg form

        @test result.f() < 1e-8
        @test abs(result.x()[1] - 1.0) < 1e-4
        @test abs(result.x()[2] - 1.0) < 1e-4
    end

    # ------------------------------------------------------------------
    # 1-D quadratic f(x) = x^2 — minimum at 0
    # ------------------------------------------------------------------
    @testset "1-D quadratic" begin
        fn = x -> x[1]^2
        gr = x -> [2.0*x[1]]

        ctrl = Control()
        result = minimize(Float64[3.0], fn, gr, ctrl)

        @test abs(result.x()[1]) < 1e-6
        @test result.f() < 1e-12
    end

    # ------------------------------------------------------------------
    # Control setters
    # ------------------------------------------------------------------
    @testset "Control parameter setters" begin
        ctrl = Control()
        set_grtol!(ctrl, 1e-9)
        set_maxeval!(ctrl, 1000)
        set_stepmax!(ctrl, 2.0)
        set_xtol!(ctrl, 1e-14)

        @test ctrl.grtol()   ≈ 1e-9
        @test ctrl.maxeval() == 1000
        @test ctrl.stepmax() ≈ 2.0
        @test ctrl.xtol()    ≈ 1e-14
    end

    # ------------------------------------------------------------------
    # status_message
    # ------------------------------------------------------------------
    @testset "status_message" begin
        fn = x -> x[1]^2
        gr = x -> [2.0*x[1]]
        result = minimize(Float64[1.0], fn, gr, Control())

        msg = result.message()
        @test isa(msg, AbstractString) && !isempty(msg)
    end

end # @testset
