/*
 * ucminf_core.cpp
 *
 * Phase 3: Modern C++17 implementation of the UCMINF unconstrained nonlinear
 * optimization algorithm.
 *
 * Original Fortran algorithm by Hans Bruun Nielsen, IMM, DTU, 2000.
 * Reference: H.B. Nielsen, "UCMINF -- An Algorithm for Unconstrained,
 * Nonlinear Optimization", Report IMM-REP-2000-19, DTU, December 2000.
 *
 * This file replaces the C translation (ucminf_core.c). Key differences:
 *   - std::vector<double> replaces manual workspace arrays / raw pointers
 *   - std::function replaces C function pointer + void* userdata
 *   - RAII: memory is automatically managed by vector lifetime
 *   - Exceptions thrown on invalid inputs (no silent negative status codes)
 *   - Internal helpers are in an anonymous namespace (no external linkage)
 *   - Algorithmic logic is preserved verbatim from the C translation
 */

#include "ucminf_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ucminf {

// ============================================================================
// Internal implementation detail
// ============================================================================
namespace {

// ----------------------------------------------------------------------------
// BLAS-like helper functions (dense, stride-1 vectors)
// ----------------------------------------------------------------------------

/// Dot product: result = x^T y
inline double ddot(const std::vector<double>& x,
                   const std::vector<double>& y)
{
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        s += x[i] * y[i];
    return s;
}

/// Euclidean norm: result = ||x||_2
inline double dnrm2(const std::vector<double>& x)
{
    return std::sqrt(ddot(x, x));
}

/// Scale vector in place: x *= alpha
inline void dscal(double alpha, std::vector<double>& x)
{
    for (auto& v : x) v *= alpha;
}

/// AXPY: y += alpha * x
inline void daxpy(double alpha,
                  const std::vector<double>& x,
                  std::vector<double>& y)
{
    for (std::size_t i = 0; i < x.size(); ++i)
        y[i] += alpha * x[i];
}

/// Index of element with maximum absolute value (0-indexed)
inline int idamax(const std::vector<double>& x)
{
    int idx = 0;
    double maxv = std::abs(x[0]);
    for (int i = 1; i < static_cast<int>(x.size()); ++i) {
        double v = std::abs(x[i]);
        if (v > maxv) { maxv = v; idx = i; }
    }
    return idx;
}

// ----------------------------------------------------------------------------
// Packed lower-triangular symmetric matrix operations
//
// Storage convention (0-indexed, column-major):
//   A(i,j) with i >= j is at index j*(2*n - j - 1)/2 + i  in the packed array.
//   Equivalently: the diagonal of column j is at running offset kk, and
//   sub-diagonal elements of column j follow immediately.
// ----------------------------------------------------------------------------

/// DSPMV (lower): y = alpha * A * x + beta * y
/// A is n-by-n symmetric with lower triangle stored in packed form ap.
void dspmv_lower(int n, double alpha, const std::vector<double>& ap,
                 const std::vector<double>& x,
                 double beta, std::vector<double>& y)
{
    for (auto& v : y) v *= beta;

    int k = 0;
    for (int j = 0; j < n; ++j) {
        y[j] += alpha * ap[k] * x[j];
        for (int i = j + 1; i < n; ++i) {
            double aij = ap[k + (i - j)];
            y[i] += alpha * aij * x[j];   // A(i,j)*x(j)
            y[j] += alpha * aij * x[i];   // A(j,i)*x(i) by symmetry
        }
        k += n - j;
    }
}

/// DSPR2 (lower): ap += alpha * (x * y^T + y * x^T)
/// Rank-2 update of packed symmetric lower-triangle matrix.
void dspr2_lower(int n, double alpha,
                 const std::vector<double>& x,
                 const std::vector<double>& y,
                 std::vector<double>& ap)
{
    int k = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i)
            ap[k + (i - j)] += alpha * (x[i] * y[j] + y[i] * x[j]);
        k += n - j;
    }
}

// ----------------------------------------------------------------------------
// SPCHOL: Cholesky factorization of packed symmetric positive-definite matrix
// ----------------------------------------------------------------------------

/// In-place Cholesky factorization of a symmetric matrix stored in packed
/// lower-triangular form. The Cholesky factor L (so A = L L^T) overwrites
/// the input on success.
///
/// @return 0 on success; k+1 (1-indexed) if the leading minor of order k
///         is not positive definite.
int spchol(int n, std::vector<double>& a)
{
    int kk = 0; // index of the diagonal element of the current column
    for (int k = 0; k < n; ++k) {
        if (a[kk] <= 0.0)
            return k + 1;
        a[kk] = std::sqrt(a[kk]);

        if (k < n - 1) {
            int nk = n - k - 1;
            double inv_akk = 1.0 / a[kk];

            // Scale sub-diagonal elements in column k
            for (int i = 0; i < nk; ++i)
                a[kk + 1 + i] *= inv_akk;

            // Rank-1 downdate of the trailing lower-right sub-matrix
            // (corresponds to dspr_lower(nk, -1.0, &a[kk+1], &a[kn]) in C)
            int kn = kk + nk + 1;
            int k2 = 0;
            for (int j = 0; j < nk; ++j) {
                double xj = a[kk + 1 + j];
                for (int i = j; i < nk; ++i)
                    a[kn + k2 + (i - j)] -= xj * a[kk + 1 + i];
                k2 += nk - j;
            }
            kk = kn;
        }
    }
    return 0;
}

// ----------------------------------------------------------------------------
// SLINE: Soft line search with (strong) Wolfe conditions
// ----------------------------------------------------------------------------

/// Threshold for switching between quadratic interpolation and bisection.
constexpr double QUADRATIC_INTERP_THRESHOLD = 1e-15;

/// Soft line search along direction h from x with value f.
///
/// Finds alpha > 0 such that F(x + alpha*h) satisfies the Wolfe conditions.
/// On return:
///   alpha  - accepted step length (0 if the search failed)
///   fn     - F(x + alpha*h)
///   slps   - { phi'(0), phi'(alpha) }
///   g      - updated to grad F at the accepted point
///
/// @return Number of function evaluations used.
int sline(int n,
          const std::vector<double>& x, double f, std::vector<double>& g,
          const std::vector<double>& h,
          double& alpha, double& fn, double (&slps)[2],
          int maxeval,
          const ObjFun& fdf)
{
    alpha = 0.0;
    fn    = f;
    int nev = 0;

    // Initial directional derivative; bail out if not a descent direction
    slps[0] = ddot(g, h);
    slps[1] = slps[0];
    if (slps[0] >= 0.0)
        return 0;

    const double fi0   = f;
    const double sl0   = 5e-2  * slps[0]; // sufficient-decrease multiplier
    const double slthr = 0.995 * slps[0]; // curvature threshold

    // Lower-bound bracket (starts at alpha = 0)
    double lo_a = 0.0, lo_f = f, lo_dp = slps[0];
    // Trial point (starts at b = 1)
    double b_a = 1.0, b_f = 0.0, b_dp = 0.0;

    bool ok = false;
    std::vector<double> wx(n), wg(n);

    // ---- Expansion phase ----
    while (true) {
        wx = x;
        daxpy(b_a, h, wx);        // wx = x + b*h
        fdf(wx, wg, b_f);
        ++nev;
        b_dp = ddot(wg, h);
        if (b_a == 1.0) slps[1] = b_dp;

        if (b_f <= fi0 + sl0 * b_a) {
            if (b_dp <= std::abs(slthr)) {
                ok      = true;
                alpha   = b_a;
                fn      = b_f;
                slps[1] = b_dp;
                g = wg;
                if (b_a < 2.0 && b_dp < slthr && nev < maxeval) {
                    lo_a = b_a; lo_f = b_f; lo_dp = b_dp;
                    b_a  = 2.0;
                    continue; // try expanding further
                }
            }
        }
        break;
    }

    double d = b_a - lo_a;

    // ---- Refinement phase ----
    while (!ok && nev < maxeval) {
        double gamma;
        double c = b_f - lo_f - d * lo_dp;
        if (c > QUADRATIC_INTERP_THRESHOLD * static_cast<double>(n) * b_a) {
            // Minimizer of the quadratic interpolant
            gamma = lo_a - 0.5 * lo_dp * (d * d / c);
            // Safeguard: keep gamma strictly inside (lo_a, b_a)
            double d01 = 0.1 * d;
            if (gamma < lo_a + d01) gamma = lo_a + d01;
            if (gamma > b_a  - d01) gamma = b_a  - d01;
        } else {
            gamma = 0.5 * (lo_a + b_a); // bisection
        }

        wx = x;
        daxpy(gamma, h, wx);      // wx = x + gamma*h
        double g_f = 0.0;
        fdf(wx, wg, g_f);
        ++nev;
        double g_dp = ddot(wg, h);

        if (g_f < fi0 + sl0 * gamma) {
            ok      = true;
            alpha   = gamma;
            fn      = g_f;
            slps[1] = g_dp;
            g = wg;
            lo_a = gamma; lo_f = g_f; lo_dp = g_dp;
        } else {
            b_a = gamma; b_f = g_f; b_dp = g_dp;
        }

        ok = ok && (std::abs(g_dp) <= std::abs(slthr));
        d  = b_a - lo_a;
        ok = ok || (d <= 0.0);
    }

    return nev;
}

} // anonymous namespace

// ============================================================================
// Public API — ucminf::minimize()
// ============================================================================

Result minimize(std::vector<double> x, ObjFun fdf, const Control& control)
{
    int n = static_cast<int>(x.size());

    // --- Input validation (throw instead of returning negative status codes) ---
    if (n <= 0)
        throw std::invalid_argument("ucminf::minimize: x must be non-empty");
    if (control.stepmax <= 0.0)
        throw std::invalid_argument("ucminf::minimize: stepmax must be > 0");
    if (control.grtol <= 0.0 || control.xtol <= 0.0)
        throw std::invalid_argument("ucminf::minimize: grtol and xtol must be > 0");
    if (control.maxeval <= 0)
        throw std::invalid_argument("ucminf::minimize: maxeval must be > 0");

    int nn = n * (n + 1) / 2;
    bool has_invh = !control.inv_hessian_lt.empty();
    if (has_invh && static_cast<int>(control.inv_hessian_lt.size()) != nn)
        throw std::invalid_argument(
            "ucminf::minimize: inv_hessian_lt must have length n*(n+1)/2");

    // --- Workspace allocation (RAII: vectors own their memory) ---
    std::vector<double> x_prev(n), g_prev(n), g_curr(n), h(n);
    std::vector<double> D(nn, 0.0); // packed inverse Hessian approximation

    bool usedel = false; // whether to apply trust-region scaling unconditionally

    if (has_invh) {
        D = control.inv_hessian_lt;
        // Verify positive-definiteness via Cholesky on a copy
        std::vector<double> Dtmp(D);
        if (spchol(n, Dtmp) != 0)
            throw std::invalid_argument(
                "ucminf::minimize: given inv_hessian_lt is not positive definite");
        usedel = false;
    } else {
        // D = I in packed lower-triangle form
        int k = 0;
        for (int i = 0; i < n; ++i) {
            D[k] = 1.0;
            k += n - i;
        }
        usedel = true;
    }

    // --- First function/gradient evaluation ---
    double fx = 0.0;
    fdf(x, g_curr, fx);
    int neval = 1;

    double nmh = 0.0;
    double nmx = dnrm2(x);
    double nmg = std::abs(g_curr[idamax(g_curr)]);
    double dx  = control.stepmax;

    Status status = Status::EvaluationLimitReached; // updated below

    // Check for convergence at the starting point
    if (nmg <= control.grtol) {
        status = Status::SmallGradient;
        Result r;
        r.x              = std::move(x);
        r.f              = fx;
        r.n_eval         = neval;
        r.max_gradient   = nmg;
        r.last_step      = nmh;
        r.status         = status;
        r.inv_hessian_lt = std::move(D);
        return r;
    }

    // --- Main optimization loop ---
    while (true) {

        // Save current iterate and gradient
        x_prev = x;
        g_prev = g_curr;

        // Quasi-Newton direction: h = -D * g
        dspmv_lower(n, -1.0, D, g_curr, 0.0, h);

        // Stopping condition: step would be too small
        nmh = dnrm2(h);
        if (nmh <= control.xtol * (control.xtol + nmx)) {
            status = Status::SmallStep;
            break;
        }

        // Trust-region scaling
        bool redu = false;
        if (nmh > dx || usedel) {
            redu = true;
            dscal(dx / nmh, h);
            nmh    = dx;
            usedel = false;
        }

        // Soft line search (max 5 evaluations per iteration)
        double a = 0.0, fxn = 0.0;
        double sl[2] = {0.0, 0.0};
        int meval = sline(n, x, fx, g_curr, h, a, fxn, sl, 5, fdf);
        neval += meval;

        if (a == 0.0) {
            status = Status::ZeroStepFromLineSearch;
            nmh = 0.0;
            break;
        }

        // Accept step: update x, f, and gradient norm
        nmg = std::abs(g_curr[idamax(g_curr)]);
        fx  = fxn;
        daxpy(a, h, x);       // x = x + a*h
        nmx = dnrm2(x);

        // Compute step vector s = new_x - old_x.
        // x_prev currently holds old_x; after daxpy below it holds -(s).
        daxpy(-1.0, x, x_prev); // x_prev = old_x - new_x  (= -s)
        nmh = dnrm2(x_prev);    // ||s||

        // Trust-region radius update
        if (a < 1.0) {
            dx = 0.35 * dx;                          // shrink
        } else if (redu && sl[1] < 0.7 * sl[0]) {
            dx = 3.0 * dx;                           // expand
        }

        // BFGS inverse-Hessian update
        //
        // Let s = new_x - old_x  (step)
        //     y = g_new - g_old  (gradient change)
        //     rho = 1 / (y^T s)
        //
        // At this point:
        //   x_prev = -s   (we negate below to get s)
        //   g_prev = old gradient
        //   g_curr = new gradient
        //
        // Compute y = g_old - g_new, stored in g_prev:
        //   g_prev -= g_curr  =>  g_prev = -y
        //
        // BFGS formula (following the original Fortran structure):
        //   yh = y^T s = -dot(g_prev, x_prev) after negating x_prev
        //   u  = D * y  (= D * (-g_prev)) stored in h
        //   D += 1/yh * (A*s*s^T/yh - u*s^T - s*u^T)   (rank-2 update)

        dscal(-1.0, x_prev);              // x_prev = s
        daxpy(-1.0, g_curr, g_prev);      // g_prev = old_g - new_g = -y

        double yh = -ddot(g_prev, x_prev); // y^T s  (= 1/rho)

        if (yh > 1e-8 * nmh * dnrm2(g_prev)) {
            dspmv_lower(n, -1.0, D, g_prev, 0.0, h); // h = D*y
            double yv = -ddot(g_prev, h);             // y^T D y
            a = (1.0 + yv / yh) / yh;
            dscal(-1.0 / yh, h);                      // h = -D*y/yh
            daxpy(0.5 * a, x_prev, h);                // h = 0.5*A*s - rho*(D*y)
            dspr2_lower(n, 1.0, x_prev, h, D);        // D += s*h^T + h*s^T
        }

        // --- Stopping criteria ---
        double thrx = control.xtol * (control.xtol + nmx);
        if (dx < thrx) dx = thrx;

        if (neval >= control.maxeval) { status = Status::EvaluationLimitReached; break; }
        if (nmh   <= thrx)            { status = Status::SmallStep;              break; }
        if (nmg   <= control.grtol)   { status = Status::SmallGradient;          break; }
    }

    Result r;
    r.x              = std::move(x);
    r.f              = fx;
    r.n_eval         = neval;
    r.max_gradient   = nmg;
    r.last_step      = nmh;
    r.status         = status;
    r.inv_hessian_lt = std::move(D);
    return r;
}

} // namespace ucminf
