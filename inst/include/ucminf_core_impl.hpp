/*
 * ucminf_core_impl.hpp
 *
 * Template implementation of the UCMINF optimization algorithm kernel.
 *
 * This header is included at the bottom of ucminf_core.hpp and must NOT be
 * included directly by user code.  It exists to allow both minimize() (the
 * ObjFun / std::function overload in ucminf_core.cpp) and minimize_direct<F>()
 * (the zero-overhead template API declared in ucminf_core.hpp) to share the
 * same inlinable kernel without duplicating code.
 *
 * Performance design:
 *  - All BLAS-like helpers operate on raw pointers so that the compiler can
 *    auto-vectorise the inner loops (no std::vector indirection or alias
 *    uncertainty inside the hot path).
 *  - sline_impl<F> accepts pre-allocated workspace (wx, wg) passed in by the
 *    caller (minimize_impl) so that no heap allocation occurs inside the line
 *    search.  This eliminates O(n) malloc+free pairs per line-search call.
 *  - F is a template parameter: when F is a plain function pointer or a
 *    captureless lambda the compiler can inline the fdf call completely,
 *    removing virtual-dispatch overhead that std::function would introduce.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ucminf {
namespace detail {

// ============================================================================
// Raw-pointer BLAS-like kernels
// ============================================================================
//
// Using raw pointers (rather than std::vector references) lets the compiler
// reason about aliasing and emit SIMD / auto-vectorised code.  The helpers
// are marked inline so the compiler can merge them into the callers' loops.

/// Dot product: result = x[0..n-1]^T y[0..n-1]
inline double ddot(int n, const double* __restrict__ x,
                          const double* __restrict__ y) noexcept
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

/// Euclidean norm: result = ||x||_2
inline double dnrm2(int n, const double* x) noexcept
{
    return std::sqrt(ddot(n, x, x));
}

/// Scale in place: x[i] *= alpha
inline void dscal(int n, double alpha, double* x) noexcept
{
    for (int i = 0; i < n; ++i) x[i] *= alpha;
}

/// AXPY: y[i] += alpha * x[i]
inline void daxpy(int n, double alpha,
                  const double* __restrict__ x,
                  double*       __restrict__ y) noexcept
{
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

/// Index of element with maximum absolute value (0-indexed)
inline int idamax(int n, const double* x) noexcept
{
    int    idx  = 0;
    double maxv = std::abs(x[0]);
    for (int i = 1; i < n; ++i) {
        double v = std::abs(x[i]);
        if (v > maxv) { maxv = v; idx = i; }
    }
    return idx;
}

/// DSPMV (lower): y = alpha * A * x + beta * y
/// A is n×n symmetric with lower triangle in packed column-major form.
inline void dspmv_lower(int n, double alpha,
                         const double* __restrict__ ap,
                         const double* __restrict__ x,
                         double beta,
                         double* __restrict__ y) noexcept
{
    for (int i = 0; i < n; ++i) y[i] *= beta;

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

/// DSPR2 (lower): ap += alpha * (x*y^T + y*x^T)
/// Rank-2 update of packed symmetric lower-triangle matrix.
inline void dspr2_lower(int n, double alpha,
                          const double* __restrict__ x,
                          const double* __restrict__ y,
                          double*       __restrict__ ap) noexcept
{
    int k = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i)
            ap[k + (i - j)] += alpha * (x[i] * y[j] + y[i] * x[j]);
        k += n - j;
    }
}

// ============================================================================
// Cholesky factorisation of packed symmetric positive-definite matrix
// ============================================================================

/// In-place Cholesky factorisation (A = L L^T) stored in packed lower form.
/// @return 0 on success; k+1 (1-indexed) if the k-th leading minor is not PD.
inline int spchol(int n, double* a) noexcept
{
    int kk = 0;
    for (int k = 0; k < n; ++k) {
        if (a[kk] <= 0.0) return k + 1;
        a[kk] = std::sqrt(a[kk]);

        if (k < n - 1) {
            int    nk      = n - k - 1;
            double inv_akk = 1.0 / a[kk];

            for (int i = 0; i < nk; ++i) a[kk + 1 + i] *= inv_akk;

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

// ============================================================================
// Soft line search — template on the callable type F
// ============================================================================
//
// Takes pre-allocated workspace vectors wx and wg so that no heap allocation
// is performed inside this function.  The caller (minimize_impl) allocates
// wx and wg once for the lifetime of the outer optimisation loop.
//
// x, g, h are passed as std::vector references so that the fdf callback can
// be invoked with the same types it was declared with (ObjFun-compatible).
// All BLAS operations use the raw .data() pointers internally.

constexpr double QUADRATIC_INTERP_THRESHOLD = 1e-15;

/// Soft line search along direction h from x with current gradient g and
/// value f.  Finds alpha > 0 satisfying the Wolfe conditions.
///
/// On return:
///   alpha  — accepted step length (0 if search failed)
///   fn     — F(x + alpha*h)
///   slps   — { phi'(0), phi'(alpha) }
///   g      — updated gradient at the accepted point
///
/// @param wx  Pre-allocated scratch vector of length n (reused, no alloc).
/// @param wg  Pre-allocated scratch vector of length n (reused, no alloc).
/// @return Number of function evaluations used.
template<typename F>
int sline_impl(int n,
               const std::vector<double>& x,
               double  f,
               std::vector<double>& g,
               const std::vector<double>& h,
               double& alpha,
               double& fn,
               double (&slps)[2],
               int    maxeval,
               std::vector<double>& wx,
               std::vector<double>& wg,
               F& fdf)
{
    const double* xp = x.data();
    double*       gp = g.data();
    const double* hp = h.data();
    double*      wxp = wx.data();
    double*      wgp = wg.data();

    alpha = 0.0;
    fn    = f;
    int nev = 0;

    slps[0] = ddot(n, gp, hp);
    slps[1] = slps[0];
    if (slps[0] >= 0.0) return 0;

    const double fi0   = f;
    const double sl0   = 5e-2  * slps[0];
    const double slthr = 0.995 * slps[0];

    double lo_a = 0.0, lo_f = f, lo_dp = slps[0];
    double b_a  = 1.0, b_f  = 0.0, b_dp = 0.0;

    bool ok = false;

    // ---- Expansion phase ----
    while (true) {
        for (int i = 0; i < n; ++i) wxp[i] = xp[i];
        daxpy(n, b_a, hp, wxp);           // wx = x + b*h
        fdf(wx, wg, b_f);                 // fdf receives vectors (wx, wg)
        ++nev;
        b_dp = ddot(n, wgp, hp);
        if (b_a == 1.0) slps[1] = b_dp;

        if (b_f <= fi0 + sl0 * b_a) {
            if (b_dp <= std::abs(slthr)) {
                ok      = true;
                alpha   = b_a;
                fn      = b_f;
                slps[1] = b_dp;
                for (int i = 0; i < n; ++i) gp[i] = wgp[i];
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
            // Minimiser of the quadratic interpolant
            gamma      = lo_a - 0.5 * lo_dp * (d * d / c);
            double d01 = 0.1 * d;
            if (gamma < lo_a + d01) gamma = lo_a + d01;
            if (gamma > b_a  - d01) gamma = b_a  - d01;
        } else {
            gamma = 0.5 * (lo_a + b_a); // bisection
        }

        for (int i = 0; i < n; ++i) wxp[i] = xp[i];
        daxpy(n, gamma, hp, wxp);          // wx = x + gamma*h
        double g_f = 0.0;
        fdf(wx, wg, g_f);                  // fdf receives vectors (wx, wg)
        ++nev;
        double g_dp = ddot(n, wgp, hp);

        if (g_f < fi0 + sl0 * gamma) {
            ok      = true;
            alpha   = gamma;
            fn      = g_f;
            slps[1] = g_dp;
            for (int i = 0; i < n; ++i) gp[i] = wgp[i];
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

// ============================================================================
// Main optimisation kernel — template on the callable type F
// ============================================================================
//
// F must be callable as:
//   F(const std::vector<double>& x, std::vector<double>& g, double& f)
// (Same contract as ucminf::ObjFun but without the std::function wrapper.)
//
// All BLAS inner loops operate on raw pointers (.data()) so the compiler can
// auto-vectorise them.  Workspace vectors wx and wg are allocated once here
// and reused on every call to sline_impl, eliminating per-iteration allocations.

template<typename F>
Result minimize_impl(std::vector<double> x, F fdf, const Control& control)
{
    int n = static_cast<int>(x.size());

    // --- Input validation ---
    if (n <= 0)
        throw std::invalid_argument("ucminf::minimize: x must be non-empty");
    if (control.stepmax <= 0.0)
        throw std::invalid_argument("ucminf::minimize: stepmax must be > 0");
    if (control.grtol <= 0.0 || control.xtol <= 0.0)
        throw std::invalid_argument("ucminf::minimize: grtol and xtol must be > 0");
    if (control.maxeval <= 0)
        throw std::invalid_argument("ucminf::minimize: maxeval must be > 0");

    int  nn       = n * (n + 1) / 2;
    bool has_invh = !control.inv_hessian_lt.empty();
    if (has_invh && static_cast<int>(control.inv_hessian_lt.size()) != nn)
        throw std::invalid_argument(
            "ucminf::minimize: inv_hessian_lt must have length n*(n+1)/2");

    // --- Workspace allocation — performed ONCE per minimize() call ---
    std::vector<double> x_prev(n), g_prev(n), g_curr(n), h(n);
    std::vector<double> D(nn, 0.0);
    // Line-search scratch buffers: allocated once, reused every iteration.
    std::vector<double> wx(n), wg(n);

    bool usedel = false;

    if (has_invh) {
        D = control.inv_hessian_lt;
        std::vector<double> Dtmp(D);
        if (spchol(n, Dtmp.data()) != 0)
            throw std::invalid_argument(
                "ucminf::minimize: given inv_hessian_lt is not positive definite");
        usedel = false;
    } else {
        int k = 0;
        for (int i = 0; i < n; ++i) { D[k] = 1.0; k += n - i; }
        usedel = true;
    }

    // --- First function/gradient evaluation ---
    double fx = 0.0;
    fdf(x, g_curr, fx);
    int neval = 1;

    double* xp      = x.data();
    double* x_prevp = x_prev.data();
    double* g_prevp = g_prev.data();
    double* g_currp = g_curr.data();
    double* hp      = h.data();
    double* Dp      = D.data();

    double nmh = 0.0;
    double nmx = dnrm2(n, xp);
    double nmg = std::abs(g_currp[idamax(n, g_currp)]);
    double dx  = control.stepmax;

    Status status = Status::EvaluationLimitReached;

    // Check for convergence at starting point
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

    // --- Main optimisation loop ---
    while (true) {

        // Save current iterate and gradient
        for (int i = 0; i < n; ++i) x_prevp[i] = xp[i];
        for (int i = 0; i < n; ++i) g_prevp[i] = g_currp[i];

        // Quasi-Newton direction: h = -D * g
        dspmv_lower(n, -1.0, Dp, g_currp, 0.0, hp);

        // Stopping condition: step too small
        nmh = dnrm2(n, hp);
        if (nmh <= control.xtol * (control.xtol + nmx)) {
            status = Status::SmallStep;
            break;
        }

        // Trust-region scaling
        bool redu = false;
        if (nmh > dx || usedel) {
            redu = true;
            dscal(n, dx / nmh, hp);
            nmh    = dx;
            usedel = false;
        }

        // Soft line search (max 5 evaluations per iteration).
        // wx and wg are pre-allocated scratch — no heap allocation here.
        double a = 0.0, fxn = 0.0;
        double sl[2] = {0.0, 0.0};
        int meval = sline_impl(n, x, fx, g_curr, h, a, fxn, sl, 5, wx, wg, fdf);
        neval += meval;

        if (a == 0.0) {
            status = Status::ZeroStepFromLineSearch;
            nmh    = 0.0;
            break;
        }

        // Accept step: update x, f, gradient norm
        nmg = std::abs(g_currp[idamax(n, g_currp)]);
        fx  = fxn;
        daxpy(n, a, hp, xp);               // x = x + a*h
        nmx = dnrm2(n, xp);

        // Compute step vector s = new_x - old_x.
        // x_prev currently holds old_x; after daxpy it holds old_x - new_x = -s.
        daxpy(n, -1.0, xp, x_prevp);       // x_prev = old_x - new_x = -s
        nmh = dnrm2(n, x_prevp);           // ||s||

        // Trust-region radius update
        if (a < 1.0) {
            dx = 0.35 * dx;
        } else if (redu && sl[1] < 0.7 * sl[0]) {
            dx = 3.0 * dx;
        }

        // BFGS inverse-Hessian update
        //
        //   s = new_x - old_x      (step)
        //   y = g_new - g_old      (gradient change)
        //   rho = 1 / (y^T s)
        //
        // At this point x_prev = -s (negated below) and g_prev = old_g.
        dscal(n, -1.0, x_prevp);                   // x_prev = s
        daxpy(n, -1.0, g_currp, g_prevp);           // g_prev = old_g - new_g = -y

        double yh = -ddot(n, g_prevp, x_prevp);     // y^T s = 1/rho

        if (yh > 1e-8 * nmh * dnrm2(n, g_prevp)) {
            dspmv_lower(n, -1.0, Dp, g_prevp, 0.0, hp); // h = D*y
            double yv = -ddot(n, g_prevp, hp);            // y^T D y
            a = (1.0 + yv / yh) / yh;
            dscal(n, -1.0 / yh, hp);                  // h = -D*y/yh
            daxpy(n, 0.5 * a, x_prevp, hp);            // h = 0.5*A*s - rho*(D*y)
            dspr2_lower(n, 1.0, x_prevp, hp, Dp);      // D += s*h^T + h*s^T
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

} // namespace detail
} // namespace ucminf
