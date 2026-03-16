/*
 * ucminf_core.c
 *
 * Phase 1: C translation of the UCMINF Fortran algorithm.
 *
 * Original Fortran algorithm by Hans Bruun Nielsen, IMM, DTU, 2000.
 * Reference: H.B. Nielsen, "UCMINF -- An Algorithm for Unconstrained,
 * Nonlinear Optimization", Report IMM-REP-2000-19, DTU, December 2000.
 *
 * The structure of this file mirrors the original Fortran source to preserve
 * algorithmic integrity. The key differences from Fortran are:
 *   - 0-based array indexing (Fortran used 1-based)
 *   - Inline BLAS-like helpers instead of external BLAS library
 *   - fdf callback replaces the Fortran EXTERNAL FDF subroutine
 */

#include <math.h>
#include <string.h>
#include "ucminf_core.h"

/*
 * Threshold for using quadratic interpolation vs. bisection in the line
 * search refinement phase. If the quadratic curvature coefficient is below
 * this threshold (relative to the interval length), bisection is used instead.
 */
#define QUADRATIC_INTERP_THRESHOLD 1e-15

/* =========================================================================
 * Minimal BLAS-like helpers (double precision)
 * ========================================================================= */

/* Dot product: result = x^T y */
static double my_ddot(int n, const double *x, int incx,
                      const double *y, int incy)
{
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += x[i * incx] * y[i * incy];
    return s;
}

/* Euclidean norm: result = ||x||_2 */
static double my_dnrm2(int n, const double *x, int incx)
{
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += x[i * incx] * x[i * incx];
    return sqrt(s);
}

/* Copy vector: y = x */
static void my_dcopy(int n, const double *x, int incx, double *y, int incy)
{
    for (int i = 0; i < n; i++)
        y[i * incy] = x[i * incx];
}

/* Scale vector in place: x = alpha * x */
static void my_dscal(int n, double alpha, double *x, int incx)
{
    for (int i = 0; i < n; i++)
        x[i * incx] *= alpha;
}

/* AXPY: y = alpha * x + y */
static void my_daxpy(int n, double alpha, const double *x, int incx,
                     double *y, int incy)
{
    for (int i = 0; i < n; i++)
        y[i * incy] += alpha * x[i * incx];
}

/* Index of element with maximum absolute value (0-indexed) */
static int my_idamax(int n, const double *x, int incx)
{
    int idx = 0;
    double maxval = fabs(x[0]);
    for (int i = 1; i < n; i++) {
        double v = fabs(x[i * incx]);
        if (v > maxval) {
            maxval = v;
            idx = i;
        }
    }
    return idx;
}

/*
 * Packed lower-triangular symmetric matrix storage (0-indexed, column-major):
 *   Element A(i,j) with i >= j (0-indexed) is stored at index
 *   j*(2*n - j - 1)/2 + i in the packed array.
 */

/*
 * DSPMV (lower): y = alpha * A * x + beta * y
 * A is n-by-n symmetric with lower triangle stored in packed form AP.
 */
static void dspmv_lower(int n, double alpha, const double *ap,
                        const double *x, int incx,
                        double beta, double *y, int incy)
{
    /* y = beta * y */
    for (int i = 0; i < n; i++)
        y[i * incy] *= beta;

    /* y += alpha * A * x  (using packed lower triangle) */
    int k = 0;
    for (int j = 0; j < n; j++) {
        /* diagonal: A(j,j) = ap[k] */
        y[j * incy] += alpha * ap[k] * x[j * incx];
        /* sub-diagonal elements in column j */
        for (int i = j + 1; i < n; i++) {
            double aij = ap[k + (i - j)];
            y[i * incy] += alpha * aij * x[j * incx]; /* A(i,j)*x(j) */
            y[j * incy] += alpha * aij * x[i * incx]; /* A(j,i)*x(i) by symmetry */
        }
        k += n - j;
    }
}

/*
 * DSPR2 (lower): AP = AP + alpha * (x * y^T + y * x^T)
 * Rank-2 update of packed symmetric lower-triangle matrix.
 */
static void dspr2_lower(int n, double alpha,
                        const double *x, int incx,
                        const double *y, int incy,
                        double *ap)
{
    int k = 0;
    for (int j = 0; j < n; j++) {
        double xj = x[j * incx];
        double yj = y[j * incy];
        for (int i = j; i < n; i++) {
            ap[k + (i - j)] += alpha * (x[i * incx] * yj + y[i * incy] * xj);
        }
        k += n - j;
    }
}

/*
 * DSPR (lower): AP = AP + alpha * x * x^T
 * Rank-1 update of packed symmetric lower-triangle matrix.
 */
static void dspr_lower(int n, double alpha, const double *x, int incx,
                       double *ap)
{
    int k = 0;
    for (int j = 0; j < n; j++) {
        double xj = alpha * x[j * incx];
        for (int i = j; i < n; i++) {
            ap[k + (i - j)] += xj * x[i * incx];
        }
        k += n - j;
    }
}

/* =========================================================================
 * SPCHOL: Cholesky factorization of packed symmetric positive-definite matrix
 * ========================================================================= */

/*
 * In-place Cholesky factorization of symmetric matrix stored in lower
 * triangular packed form. On success (FAIL=0) the Cholesky factor
 * overwrites AP. FAIL>0 means the leading minor of order FAIL is not
 * positive definite.
 *
 * This is a direct C translation of the SPCHOL Fortran subroutine.
 */
static int spchol(int n, double *a)
{
    int kk = 0; /* points to the diagonal element of the current column */
    for (int k = 0; k < n; k++) {
        if (a[kk] <= 0.0)
            return k + 1; /* not positive definite (1-indexed column) */
        a[kk] = sqrt(a[kk]);
        if (k < n - 1) {
            int nk = n - k - 1;
            /* scale sub-diagonal elements in column k */
            my_dscal(nk, 1.0 / a[kk], &a[kk + 1], 1);
            /* update trailing lower-right submatrix */
            int kn = kk + nk + 1;
            dspr_lower(nk, -1.0, &a[kk + 1], 1, &a[kn]);
            kk = kn;
        }
    }
    return 0;
}

/* =========================================================================
 * SLINE: Soft line search with (strong) Wolfe conditions
 * ========================================================================= */

/*
 * C translation of the SLINE Fortran subroutine.
 *
 * Finds alpha > 0 along direction h such that F(x + alpha*h) is
 * sufficiently decreased (Wolfe/strong-Wolfe conditions).
 *
 * The Fortran original uses XFD(3,3), a 3x3 matrix where:
 *   row 1 = step value, row 2 = f value, row 3 = phi' (directional deriv)
 *   col 1 = lower bound point, col 2 = trial point b, col 3 = refined gamma
 *
 * Here we use named scalar variables for clarity.
 *
 * Parameters:
 *   n      - dimension
 *   x      - current point (length n, read-only)
 *   f      - F(x) (read-only)
 *   g      - gradient at x (length n, updated to gradient at accepted point)
 *   h      - search direction (length n, read-only)
 *   w      - workspace of length >= 2*n
 *   alpha  - on output: accepted step length (0 if failure)
 *   fn     - on output: F(x + alpha*h)
 *   slps   - on output: slps[0]=phi'(0), slps[1]=phi'(alpha)
 *   nev    - on input: max function evaluations; on output: evals used
 *   fdf    - function/gradient callback
 *   udata  - user data
 */
static void sline(int n, const double *x, double f, double *g,
                  const double *h, double *w,
                  double *alpha, double *fn, double *slps, int *nev,
                  ucminf_fdf_t fdf, void *udata)
{
    double *wx = w;       /* trial x,        length n */
    double *wg = w + n;  /* trial gradient,  length n */

    *alpha = 0.0;
    *fn    = f;
    int meval = *nev;
    *nev = 0;

    /* Initial directional derivative; check descent direction */
    slps[0] = my_ddot(n, g, 1, h, 1);
    slps[1] = slps[0];
    if (slps[0] >= 0.0)
        return;

    double fi0   = f;
    double sl0   = 5e-2  * slps[0];  /* sufficient-decrease slope multiplier */
    double slthr = 0.995 * slps[0];  /* curvature threshold */

    /* Lower-bound bracket point (starts at alpha=0) */
    double lo_a  = 0.0,  lo_f  = f,    lo_dp = slps[0];
    /* Trial point (starts at b=1) */
    double b_a   = 1.0,  b_f,          b_dp;

    int ok = 0;

    /* ---- Expansion phase ---- */
    while (1) {
        my_dcopy(n, x, 1, wx, 1);
        my_daxpy(n, b_a, h, 1, wx, 1);     /* wx = x + b*h */
        fdf(n, wx, wg, &b_f, udata);
        (*nev)++;
        b_dp = my_ddot(n, wg, 1, h, 1);
        if (b_a == 1.0) slps[1] = b_dp;

        if (b_f <= fi0 + sl0 * b_a) {
            /* Sufficient decrease: update accepted point */
            if (b_dp <= fabs(slthr)) {
                ok = 1;
                *alpha  = b_a;
                *fn     = b_f;
                slps[1] = b_dp;
                my_dcopy(n, wg, 1, g, 1);
                /* Expand if we can still double the step */
                if (b_a < 2.0 && b_dp < slthr && *nev < meval) {
                    lo_a = b_a; lo_f = b_f; lo_dp = b_dp;
                    b_a  = 2.0;
                    continue;
                }
            }
        }
        break; /* end expansion */
    }

    double d = b_a - lo_a;

    /* ---- Refinement phase ---- */
    while (!ok && *nev < meval) {
        double gamma;
        double c = b_f - lo_f - d * lo_dp;
        if (c > QUADRATIC_INTERP_THRESHOLD * (double)n * b_a) {
            /* Minimizer of quadratic interpolant */
            gamma = lo_a - 0.5 * lo_dp * (d * d / c);
            /* Safeguard: keep gamma strictly inside (lo_a, b_a) */
            double d01 = 0.1 * d;
            if (gamma < lo_a + d01) gamma = lo_a + d01;
            if (gamma > b_a  - d01) gamma = b_a  - d01;
        } else {
            gamma = 0.5 * (lo_a + b_a);
        }

        my_dcopy(n, x, 1, wx, 1);
        my_daxpy(n, gamma, h, 1, wx, 1);   /* wx = x + gamma*h */
        double g_f, g_dp;
        fdf(n, wx, wg, &g_f, udata);
        (*nev)++;
        g_dp = my_ddot(n, wg, 1, h, 1);

        if (g_f < fi0 + sl0 * gamma) {
            /* Sufficient decrease: accept gamma as new lower bound */
            ok = 1;
            *alpha  = gamma;
            *fn     = g_f;
            slps[1] = g_dp;
            my_dcopy(n, wg, 1, g, 1);
            lo_a = gamma; lo_f = g_f; lo_dp = g_dp;
        } else {
            /* Reduce upper bound */
            b_a = gamma; b_f = g_f; b_dp = g_dp;
        }

        ok = ok && (fabs(g_dp) <= fabs(slthr));
        d  = b_a - lo_a;
        ok = ok || (d <= 0.0);
    }
}

/* =========================================================================
 * UCMINF_OPTIMIZE: Main optimization routine
 * ========================================================================= */

void ucminf_optimize(int n, double *x, double dx, double eps1, double eps2,
                     int *maxfun, double *w, int iw, int *icontr,
                     ucminf_fdf_t fdf, void *userdata)
{
    int optim  = (*icontr > 0);
    int dgivn  = (*icontr > 2);

    /* Input validation */
    *icontr = 0;
    int nn = n * (n + 1) / 2;

    if (n <= 0) {
        *icontr = -2;
        return;
    }
    if (optim) {
        if (dx <= 0.0)    { *icontr = -4; return; }
        if (eps1 <= 0.0 || eps2 <= 0.0) { *icontr = -5; return; }
        if (*maxfun <= 0) { *icontr = -6; return; }
    } else {
        if (dx == 0.0)    { *icontr = -4; return; }
        int need = (n * (n + 11)) / 2;
        if (need < 7) need = 7;
        int need2 = (dgivn) ? (2 * nn > nn + 5 * n ? 2 * nn : nn + 5 * n)
                             : need;
        if (iw < need || (dgivn && iw < need2)) { *icontr = -8; return; }
    }

    /*
     * Workspace layout (0-indexed):
     *   w[0      .. n-1]        : x_prev (previous iterate)
     *   w[n      .. 2n-1]       : g_prev (previous gradient)
     *   w[2n     .. 3n-1]       : g_curr (current gradient)
     *   w[3n     .. 4n-1]       : h      (quasi-Newton step direction)
     *   w[4n     .. 4n+nn-1]    : D      (packed lower triangle of inv-Hessian)
     *   w[4n+nn  ..]            : workspace for line search (needs >= 2n)
     */
    double *x_prev = w;
    double *g_prev = w + n;
    double *g_curr = w + 2 * n;
    double *h      = w + 3 * n;
    double *D      = w + 4 * n;
    double *ws     = w + 4 * n + nn;  /* line search workspace */

    int usedel; /* whether to force trust-region scaling */

    if (dgivn) {
        /* User supplied D0: check it is positive definite via Cholesky */
        /* Copy D0 to a temp area in ws, factor it there */
        my_dcopy(nn, D, 1, ws, 1);
        int fail = spchol(n, ws);
        if (fail != 0) {
            *icontr = -7;
            return;
        }
        usedel = 0;
    } else {
        /* Initialize D = I (in packed lower-triangle form) */
        memset(D, 0, nn * sizeof(double));
        int k = 0;
        for (int i = 0; i < n; i++) {
            D[k] = 1.0;
            k += n - i;
        }
        usedel = 1;
    }

    /* First evaluation of f and g */
    double fx;
    fdf(n, x, g_curr, &fx, userdata);
    int neval = 1;

    double nmh = 0.0;
    double nmx = my_dnrm2(n, x, 1);
    double nmg = fabs(g_curr[my_idamax(n, g_curr, 1)]);

    if (nmg <= eps1) {
        *icontr = 1;
        goto done;
    }

    /* ---- Main optimization loop ---- */
    while (1) {

        /* Save current x and gradient */
        my_dcopy(n, x,      1, x_prev, 1);
        my_dcopy(n, g_curr, 1, g_prev, 1);

        /* Quasi-Newton direction: h = -D * g */
        dspmv_lower(n, -1.0, D, g_curr, 1, 0.0, h, 1);

        /* Check for near-zero step (stopping condition 2) */
        nmh = my_dnrm2(n, h, 1);
        if (nmh <= eps2 * (eps2 + nmx)) {
            *icontr = 2;
            goto done;
        }

        /* Trust-region scaling */
        int redu = 0;
        if (nmh > dx || usedel) {
            redu = 1;
            my_dscal(n, dx / nmh, h, 1);
            nmh = dx;
            usedel = 0;
        }

        /* Soft line search (max 5 evaluations) */
        int meval = 5;
        double a, fxn;
        double sl[2];
        sline(n, x, fx, g_curr, h, ws, &a, &fxn, sl, &meval, fdf, userdata);

        if (a == 0.0) {
            *icontr = 4;
            nmh = 0.0;
            goto done;
        }

        /* Update neval, x, f(x), and ||g|| */
        neval += meval;
        nmg = fabs(g_curr[my_idamax(n, g_curr, 1)]);
        fx  = fxn;

        /* x_prev currently holds old x; compute step s = new_x - old_x */
        /* new_x = old_x + a*h */
        my_daxpy(n, a, h, 1, x, 1); /* x = old_x + a*h */
        nmx = my_dnrm2(n, x, 1);

        /* x_prev = old_x, x = new_x; compute s = x - x_prev stored in x_prev */
        /* Actually we need s for BFGS: s = new_x - old_x
         * g_prev holds old gradient, g_curr holds new gradient
         * y = g_curr - g_prev
         * We'll compute: x_prev = -(old_x - new_x) = new_x - old_x = s */
        my_daxpy(n, -1.0, x, 1, x_prev, 1); /* x_prev = old_x - new_x */
        nmh = my_dnrm2(n, x_prev, 1);       /* ||step|| */

        /* Update trust-region radius */
        if (a < 1.0) {
            dx = 0.35 * dx;                             /* shrink */
        } else if (redu && (sl[1] < 0.7 * sl[0])) {
            dx = 3.0 * dx;                              /* expand */
        }

        /* BFGS update of inverse Hessian D
         *
         * Let s = x_new - x_old  (step vector)
         *     y = g_new - g_old  (gradient difference)
         *     rho = 1 / (y^T s)
         *
         * In the workspace after the code above:
         *   x_prev = old_x - new_x = -s  (we negate it below to get s)
         *   g_prev = old gradient
         *   g_curr = new gradient
         *
         * Compute y = g_old - g_new, stored in g_prev:
         *   g_prev -= g_curr  =>  g_prev = old_g - new_g = -y
         *
         * Then:
         *   x_prev *= -1  => x_prev = s
         *   g_prev  = -y
         *
         * So:
         *   YH = -dot(g_prev, x_prev) = -dot(-y, s) = y^T s    (= 1/rho)
         *   u  = D * (-g_prev) = D * y  (stored in h)
         *   YV = -dot(g_prev, h) = -dot(-y, D*y) = y^T D y
         *   A  = (1 + YV/YH) / YH  =  rho * (1 + rho * y^T D y)
         *
         * BFGS formula:
         *   D+ = D + rho * (s*v^T + v*s^T)
         * where v = 0.5*(1 + rho*y^T*D*y)*s - D*y
         *         = 0.5*A/rho * s/rho^{-1} - u ... simplified:
         *
         * Following the Fortran directly:
         *   h <- -D * g_prev / YH  =  D*y / YH  =  rho * u   (scale of u)
         *   h += 0.5*A * x_prev                              (add part of s term)
         *   D += 1.0 * x_prev * h^T + h * x_prev^T          (rank-2 update)
         */
        my_dscal(n, -1.0, x_prev, 1); /* x_prev = s */
        my_daxpy(n, -1.0, g_curr, 1, g_prev, 1); /* g_prev = old_g - new_g = -y */

        double yh = -my_ddot(n, g_prev, 1, x_prev, 1); /* y^T s */

        if (yh > 1e-8 * nmh * my_dnrm2(n, g_prev, 1)) {
            /* u = D * y = D * (-g_prev) -> store in h: h = -D * (-y) = D*y */
            dspmv_lower(n, -1.0, D, g_prev, 1, 0.0, h, 1); /* h = D*y */
            double yv = -my_ddot(n, g_prev, 1, h, 1);        /* y^T D y */
            a = (1.0 + yv / yh) / yh;
            my_dscal(n, -1.0 / yh, h, 1);                   /* h = -D*y/yh = -rho*u */
            my_daxpy(n, 0.5 * a, x_prev, 1, h, 1);           /* h = 0.5*A*s - rho*u = rho*v */
            dspr2_lower(n, 1.0, x_prev, 1, h, 1, D);         /* D += s*(rho*v)^T + (rho*v)*s^T */
        }

        /* Stopping criteria */
        double thrx = eps2 * (eps2 + nmx);
        if (dx < thrx) dx = thrx;

        if (neval >= *maxfun) { *icontr = 3; break; }
        if (nmh   <= thrx)    { *icontr = 2; break; }
        if (nmg   <= eps1)    { *icontr = 1; break; }
    }

done:
    /* Store return values in the first three positions of workspace */
    *maxfun = neval;
    w[0] = fx;
    w[1] = nmg;
    w[2] = nmh;
}
