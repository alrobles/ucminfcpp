/*
 * test_ucminf.cpp
 *
 * Unit tests for the ucminf C++ core using Catch2 v3.
 *
 * Tests verify:
 *  - Correct minimization of standard benchmark functions
 *  - Convergence codes and diagnostics
 *  - Control parameter validation (exception paths)
 *  - Warm-start with a custom initial inverse Hessian
 *  - 1-D and N-D problems
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ucminf_core.hpp"

#include <cmath>
#include <stdexcept>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// Helpers: benchmark objective + gradient functions
// ---------------------------------------------------------------------------

/// Rosenbrock Banana function: F(x,y) = (1-x)^2 + 100*(y-x^2)^2
/// Minimum at (1,1) with F=0.
static void rosenbrock_fdf(const std::vector<double>& x,
                            std::vector<double>& g,
                            double& f)
{
    double a = 1.0 - x[0];
    double b = x[1] - x[0] * x[0];
    f    = a * a + 100.0 * b * b;
    g[0] = -2.0 * a - 400.0 * x[0] * b;
    g[1] =  200.0 * b;
}

/// Simple quadratic: F(x) = x^2. Minimum at x=0 with F=0.
static void quadratic_fdf(const std::vector<double>& x,
                           std::vector<double>& g,
                           double& f)
{
    f    = x[0] * x[0];
    g[0] = 2.0 * x[0];
}

/// Sum of squares: F(x) = sum(x_i^2). Minimum at 0 with F=0.
static void sum_of_squares_fdf(const std::vector<double>& x,
                                std::vector<double>& g,
                                double& f)
{
    f = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        f    += x[i] * x[i];
        g[i]  = 2.0 * x[i];
    }
}

/// Extended Rosenbrock (3-D):
///   F(x) = (1-x0)^2 + 100*(x1-x0^2)^2 + (1-x1)^2 + 100*(x2-x1^2)^2
/// Minimum at (1,1,1) with F=0.
static void rosenbrock3_fdf(const std::vector<double>& x,
                             std::vector<double>& g,
                             double& f)
{
    double a0 = 1.0 - x[0];
    double b0 = x[1] - x[0] * x[0];
    double a1 = 1.0 - x[1];
    double b1 = x[2] - x[1] * x[1];
    f    = a0*a0 + 100.0*b0*b0 + a1*a1 + 100.0*b1*b1;
    g[0] = -2.0*a0 - 400.0*x[0]*b0;
    g[1] =  200.0*b0 - 2.0*a1 - 400.0*x[1]*b1;
    g[2] =  200.0*b1;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

TEST_CASE("1-D quadratic converges to 0", "[basic]")
{
    ucminf::Result res = ucminf::minimize({3.0}, quadratic_fdf);

    REQUIRE(static_cast<int>(res.status) > 0);
    CHECK_THAT(res.x[0], WithinAbs(0.0, 1e-6));
    CHECK_THAT(res.f,    WithinAbs(0.0, 1e-12));
    CHECK(res.n_eval > 0);
    CHECK(res.max_gradient >= 0.0);
}

TEST_CASE("Rosenbrock: converges to (1,1)", "[rosenbrock]")
{
    ucminf::Result res = ucminf::minimize({2.0, 0.5}, rosenbrock_fdf);

    REQUIRE(static_cast<int>(res.status) > 0);
    CHECK_THAT(res.x[0], WithinAbs(1.0, 1e-5));
    CHECK_THAT(res.x[1], WithinAbs(1.0, 1e-5));
    CHECK_THAT(res.f,    WithinAbs(0.0, 1e-10));
}

TEST_CASE("Rosenbrock: convergence code is SmallGradient or SmallStep", "[rosenbrock]")
{
    ucminf::Result res = ucminf::minimize({2.0, 0.5}, rosenbrock_fdf);
    bool good_code = (res.status == ucminf::Status::SmallGradient ||
                      res.status == ucminf::Status::SmallStep);
    CHECK(good_code);
}

TEST_CASE("Rosenbrock: gradient near zero at solution", "[rosenbrock]")
{
    ucminf::Result res = ucminf::minimize({2.0, 0.5}, rosenbrock_fdf);
    CHECK(res.max_gradient <= 1e-5);
}

TEST_CASE("Rosenbrock: starting at the minimum converges immediately", "[rosenbrock]")
{
    ucminf::Result res = ucminf::minimize({1.0, 1.0}, rosenbrock_fdf);
    CHECK(res.status == ucminf::Status::SmallGradient);
    CHECK_THAT(res.x[0], WithinAbs(1.0, 1e-10));
    CHECK_THAT(res.x[1], WithinAbs(1.0, 1e-10));
}

TEST_CASE("3-D extended Rosenbrock converges to (1,1,1)", "[rosenbrock3]")
{
    ucminf::Control ctrl;
    ctrl.maxeval = 2000;
    ucminf::Result res = ucminf::minimize({2.0, 1.0, 0.5}, rosenbrock3_fdf, ctrl);

    bool good_code = (res.status == ucminf::Status::SmallGradient ||
                      res.status == ucminf::Status::SmallStep);
    REQUIRE(good_code);
    CHECK_THAT(res.x[0], WithinAbs(1.0, 1e-4));
    CHECK_THAT(res.x[1], WithinAbs(1.0, 1e-4));
    CHECK_THAT(res.x[2], WithinAbs(1.0, 1e-4));
}

TEST_CASE("Sum of squares (5-D) converges to zero", "[sum_of_squares]")
{
    ucminf::Result res = ucminf::minimize({1.0, -2.0, 3.0, -4.0, 5.0},
                                          sum_of_squares_fdf);
    REQUIRE(static_cast<int>(res.status) > 0);
    for (int i = 0; i < 5; ++i)
        CHECK_THAT(res.x[i], WithinAbs(0.0, 1e-5));
    CHECK_THAT(res.f, WithinAbs(0.0, 1e-10));
}

TEST_CASE("maxeval limit is respected", "[control]")
{
    ucminf::Control ctrl;
    ctrl.maxeval = 5;
    ucminf::Result res = ucminf::minimize({2.0, 0.5}, rosenbrock_fdf, ctrl);
    // Should either hit the eval limit or converge quickly
    CHECK(res.n_eval <= 30); // 5 + a few line search evaluations
}

TEST_CASE("grtol control affects convergence tolerance", "[control]")
{
    ucminf::Control ctrl;
    ctrl.grtol = 1e-3; // coarser than default
    ucminf::Result res = ucminf::minimize({2.0, 0.5}, rosenbrock_fdf, ctrl);
    // Converged with coarser tolerance: gradient at solution should be <= 1e-3
    if (res.status == ucminf::Status::SmallGradient)
        CHECK(res.max_gradient <= 1e-3 + 1e-9);
}

TEST_CASE("Result has a correctly-sized inv_hessian_lt", "[result]")
{
    ucminf::Result res = ucminf::minimize({2.0, 0.5}, rosenbrock_fdf);
    int n  = static_cast<int>(res.x.size());
    int nn = n * (n + 1) / 2;
    CHECK(static_cast<int>(res.inv_hessian_lt.size()) == nn);
}

TEST_CASE("Warm start with given inverse Hessian (identity) produces same result", "[warm_start]")
{
    // Start from identity D0 – this should give the same or better convergence
    ucminf::Control ctrl;
    ctrl.inv_hessian_lt = {1.0, 0.0, 1.0}; // identity for n=2

    ucminf::Result res = ucminf::minimize({2.0, 0.5}, rosenbrock_fdf, ctrl);
    REQUIRE(static_cast<int>(res.status) > 0);
    CHECK_THAT(res.x[0], WithinAbs(1.0, 1e-4));
    CHECK_THAT(res.x[1], WithinAbs(1.0, 1e-4));
}

TEST_CASE("Invalid arguments throw std::invalid_argument", "[errors]")
{
    // Empty x
    CHECK_THROWS_AS(ucminf::minimize({}, quadratic_fdf), std::invalid_argument);

    // stepmax <= 0
    ucminf::Control c1; c1.stepmax = 0.0;
    CHECK_THROWS_AS(ucminf::minimize({1.0}, quadratic_fdf, c1), std::invalid_argument);

    // grtol <= 0
    ucminf::Control c2; c2.grtol = -1.0;
    CHECK_THROWS_AS(ucminf::minimize({1.0}, quadratic_fdf, c2), std::invalid_argument);

    // maxeval <= 0
    ucminf::Control c3; c3.maxeval = 0;
    CHECK_THROWS_AS(ucminf::minimize({1.0}, quadratic_fdf, c3), std::invalid_argument);

    // Non-positive-definite D0
    ucminf::Control c4; c4.inv_hessian_lt = {-1.0, 0.0, 1.0}; // negative diagonal
    CHECK_THROWS_AS(ucminf::minimize({1.0, 1.0}, rosenbrock_fdf, c4),
                    std::invalid_argument);

    // Wrong size D0
    ucminf::Control c5; c5.inv_hessian_lt = {1.0}; // should be length 3 for n=2
    CHECK_THROWS_AS(ucminf::minimize({1.0, 1.0}, rosenbrock_fdf, c5),
                    std::invalid_argument);
}

TEST_CASE("status_message returns a non-empty string for every Status", "[status_message]")
{
    using S = ucminf::Status;
    for (auto s : {S::SmallGradient, S::SmallStep, S::EvaluationLimitReached,
                   S::ZeroStepFromLineSearch, S::InvalidDimension,
                   S::InvalidStepmax, S::InvalidTolerances, S::InvalidMaxeval,
                   S::HessianNotPositiveDefinite})
    {
        CHECK(!ucminf::status_message(s).empty());
    }
}
