/*
 * ucminf_core.hpp
 *
 * Phase 3: Modern C++17 API for the UCMINF unconstrained nonlinear
 * optimization algorithm.
 *
 * Original Fortran algorithm by Hans Bruun Nielsen, IMM, DTU, 2000.
 * Reference: H.B. Nielsen, "UCMINF -- An Algorithm for Unconstrained,
 * Nonlinear Optimization", Report IMM-REP-2000-19, DTU, December 2000.
 *
 * Refactoring from C (ucminf_core.c/h) to C++17:
 *  - std::vector<double> replaces raw pointer workspace arrays
 *  - std::function replaces C function pointer + void* userdata
 *  - RAII: all memory managed automatically by vector lifetime
 *  - Named result struct replaces output-parameter workspace convention
 *  - C++ exceptions replace integer error codes for invalid arguments
 */

#pragma once

#include <functional>
#include <string>
#include <vector>

namespace ucminf {

// ---------------------------------------------------------------------------
// Termination status codes
// ---------------------------------------------------------------------------

/// Termination/convergence status returned by minimize().
enum class Status : int {
    SmallGradient              =  1, ///< Converged: max|grad| <= grtol
    SmallStep                  =  2, ///< Converged: step length <= xtol
    EvaluationLimitReached     =  3, ///< Stopped: maxeval reached
    ZeroStepFromLineSearch     =  4, ///< Stopped: line search returned alpha=0
    InvalidDimension           = -2, ///< Error: x must be non-empty
    InvalidStepmax             = -4, ///< Error: stepmax must be > 0
    InvalidTolerances          = -5, ///< Error: grtol and xtol must be > 0
    InvalidMaxeval             = -6, ///< Error: maxeval must be > 0
    HessianNotPositiveDefinite = -7  ///< Error: given inv_hessian_lt is not PD
};

/// Return a human-readable string for a Status value.
inline std::string status_message(Status s)
{
    switch (s) {
    case Status::SmallGradient:
        return "Stopped by small gradient (grtol).";
    case Status::SmallStep:
        return "Stopped by small step (xtol).";
    case Status::EvaluationLimitReached:
        return "Stopped by function evaluation limit (maxeval).";
    case Status::ZeroStepFromLineSearch:
        return "Stopped by zero step from line search.";
    case Status::InvalidDimension:
        return "Computation did not start: x must be non-empty.";
    case Status::InvalidStepmax:
        return "Computation did not start: stepmax must be > 0.";
    case Status::InvalidTolerances:
        return "Computation did not start: grtol and xtol must be > 0.";
    case Status::InvalidMaxeval:
        return "Computation did not start: maxeval must be > 0.";
    case Status::HessianNotPositiveDefinite:
        return "Computation did not start: given inv_hessian_lt is not positive definite.";
    default:
        return "Unknown status.";
    }
}

// ---------------------------------------------------------------------------
// Objective function callback type
// ---------------------------------------------------------------------------

/// Callback type for evaluating the objective function and its gradient.
///
/// On each call the implementation must set:
///   @p f  — the scalar objective value at @p x
///   @p g  — the gradient vector at @p x (must have the same length as @p x)
///
/// @param x  Current point (input, length n)
/// @param g  Gradient at x (output, length n)
/// @param f  Objective value at x (output)
using ObjFun = std::function<void(const std::vector<double>& x,
                                   std::vector<double>&       g,
                                   double&                    f)>;

// ---------------------------------------------------------------------------
// Control parameters
// ---------------------------------------------------------------------------

/// Algorithmic control parameters for minimize().
struct Control {
    /// Gradient tolerance: stop when max|g(x)| <= grtol. Default 1e-6.
    double grtol = 1e-6;

    /// Step tolerance: stop when ||step||^2 <= xtol*(xtol+||x||^2). Default 1e-12.
    double xtol = 1e-12;

    /// Initial trust-region radius. Default 1.0.
    double stepmax = 1.0;

    /// Maximum number of function+gradient evaluations. Default 500.
    int maxeval = 500;

    /// Optional initial inverse-Hessian approximation in packed lower-triangle
    /// (column-major) form. Length must equal n*(n+1)/2 when provided.
    /// If empty, the identity matrix is used.
    std::vector<double> inv_hessian_lt;
};

// ---------------------------------------------------------------------------
// Optimization result
// ---------------------------------------------------------------------------

/// Result returned by minimize().
struct Result {
    std::vector<double> x;            ///< Parameter values at the (approximate) minimum
    double              f         {}; ///< Objective value at x
    int                 n_eval    {}; ///< Total function/gradient evaluations used
    double              max_gradient{}; ///< max|grad(x)| at the solution
    double              last_step {}; ///< Length of the last accepted step
    Status              status    {}; ///< Termination status code

    /// Packed lower-triangle of the final inverse-Hessian approximation.
    /// Length n*(n+1)/2, stored in column-major order.
    std::vector<double> inv_hessian_lt;
};

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Minimize f(x) starting from @p x0 using the UCMINF quasi-Newton algorithm.
///
/// The algorithm uses a BFGS inverse-Hessian update and a soft line search
/// with adaptive trust-region radius monitoring.
///
/// @param x0       Initial parameter vector (length n > 0)
/// @param fdf      Callback that computes f(x) and grad f(x)
/// @param control  Algorithmic parameters (tolerances, limits, optional D0)
/// @return         Result containing the minimizer, diagnostics, and final
///                 inverse-Hessian approximation
///
/// @throws std::invalid_argument when control parameters are invalid and
///         the algorithm cannot start (mirrors the negative status codes)
Result minimize(std::vector<double> x0,
                ObjFun              fdf,
                const Control&      control = {});

} // namespace ucminf
