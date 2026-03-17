/*
 * ucminf_core.hpp
 *
 * Modern C++17 API for the UCMINF unconstrained nonlinear optimization
 * algorithm.
 *
 * Original Fortran algorithm by Hans Bruun Nielsen, IMM, DTU, 2000.
 * Reference: H.B. Nielsen, "UCMINF -- An Algorithm for Unconstrained,
 * Nonlinear Optimization", Report IMM-REP-2000-19, DTU, December 2000.
 *
 * Dual-layer API
 * --------------
 * This header exposes two entry points that share the same optimised kernel
 * (see ucminf_core_impl.hpp):
 *
 *  1. minimize(x0, fdf, control)
 *       High-level API.  fdf is an ObjFun (std::function) — convenient for
 *       R/Python/Julia bridges where the callable is not known at compile
 *       time, but incurs the usual std::function type-erasure overhead.
 *
 *  2. minimize_direct<F>(x0, fdf, control)
 *       Low-level template API.  F can be any callable (plain function
 *       pointer, lambda, functor) with signature
 *           void(const std::vector<double>&, std::vector<double>&, double&)
 *       Because F is a template parameter the compiler can inline the
 *       callable completely, eliminating virtual-dispatch overhead.  Prefer
 *       this API when calling directly from C++, Python (pybind11), or Julia
 *       (CxxWrap) where the callable type is statically known.
 *
 * Performance notes for R users
 * ------------------------------
 * When called from R via the Rcpp wrapper (ucminf_rcpp.cpp) every fdf
 * invocation crosses the R interpreter boundary (R → C++ → R round-trip).
 * This overhead dominates runtime for most problems and cannot be avoided
 * when fn/gr are R functions.  For latency-sensitive R workloads consider:
 *   - Pre-computing costly invariants outside fn/gr.
 *   - Implementing fn/gr in C++ and calling via minimize_direct<F>().
 *   - Using compiled R code (e.g. via Rcpp::cppFunction).
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
// Objective function callback type (high-level / R-compatible)
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
///
/// Used by the high-level minimize() overload.  For zero-overhead C++ use
/// prefer minimize_direct<F>() with any callable as F.
using ObjFun = std::function<void(const std::vector<double>& x,
                                   std::vector<double>&       g,
                                   double&                    f)>;

// ---------------------------------------------------------------------------
// Control parameters
// ---------------------------------------------------------------------------

/// Algorithmic control parameters for minimize() / minimize_direct().
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

/// Result returned by minimize() / minimize_direct().
struct Result {
    std::vector<double> x;              ///< Parameter values at the (approximate) minimum
    double              f           {}; ///< Objective value at x
    int                 n_eval      {}; ///< Total function/gradient evaluations used
    double              max_gradient{}; ///< max|grad(x)| at the solution
    double              last_step   {}; ///< Length of the last accepted step
    Status              status      {}; ///< Termination status code

    /// Packed lower-triangle of the final inverse-Hessian approximation.
    /// Length n*(n+1)/2, stored in column-major order.
    std::vector<double> inv_hessian_lt;
};

// ---------------------------------------------------------------------------
// High-level entry point (ObjFun / std::function overload)
// ---------------------------------------------------------------------------

/// Minimize f(x) starting from @p x0 using the UCMINF quasi-Newton algorithm.
///
/// Uses an ObjFun (std::function) callback — convenient when the callable is
/// not known at compile time (e.g. R/Python lambdas via Rcpp/pybind11).
/// For pure-C++ code where the callable type is statically known, prefer
/// minimize_direct<F>() which allows the compiler to inline fdf and avoids
/// std::function type-erasure overhead.
///
/// @param x0       Initial parameter vector (length n > 0)
/// @param fdf      Callback computing f(x) and grad f(x)
/// @param control  Algorithmic parameters (tolerances, limits, optional D0)
/// @return         Result containing the minimizer, diagnostics, and final
///                 inverse-Hessian approximation
///
/// @throws std::invalid_argument when control parameters are invalid
Result minimize(std::vector<double> x0,
                ObjFun              fdf,
                const Control&      control = {});

// ---------------------------------------------------------------------------
// Low-level template entry point (zero-overhead direct C++ interface)
// ---------------------------------------------------------------------------

// Forward-declare the implementation template (defined in ucminf_core_impl.hpp,
// which is included below).
namespace detail {
template<typename F>
Result minimize_impl(std::vector<double> x, F fdf, const Control& control);
} // namespace detail

/// Minimize f(x) starting from @p x0 using the UCMINF quasi-Newton algorithm.
///
/// Template overload: F can be any callable with signature
///   void(const std::vector<double>& x, std::vector<double>& g, double& f)
/// When F is a lambda or plain function pointer the compiler can inline the
/// entire fdf call, eliminating virtual-dispatch overhead compared to the
/// std::function-based minimize() overload.
///
/// Line-search workspace is allocated once per call and reused across
/// iterations; there are no heap allocations inside the hot loop.
///
/// @param x0       Initial parameter vector (length n > 0)
/// @param fdf      Any callable: lambda, function pointer, functor, ...
/// @param control  Algorithmic parameters (tolerances, limits, optional D0)
/// @return         Result containing the minimizer, diagnostics, and final
///                 inverse-Hessian approximation
///
/// @throws std::invalid_argument when control parameters are invalid
template<typename F>
inline Result minimize_direct(std::vector<double> x0,
                               F                   fdf,
                               const Control&      control = {})
{
    return detail::minimize_impl(std::move(x0), std::move(fdf), control);
}

} // namespace ucminf

// Include the template implementation.  Must come after the declarations above
// because minimize_impl references Control, Result, and Status.
#include "ucminf_core_impl.hpp"
