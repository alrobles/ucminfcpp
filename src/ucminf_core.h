/*
 * ucminf_core.h
 *
 * Phase 1: C translation of the UCMINF Fortran algorithm.
 *
 * Original Fortran algorithm by Hans Bruun Nielsen, IMM, DTU, 2000.
 * Reference: H.B. Nielsen, "UCMINF -- An Algorithm for Unconstrained,
 * Nonlinear Optimization", Report IMM-REP-2000-19, DTU, December 2000.
 *
 * Translation from Fortran 77 to C by migration to ucminfcpp.
 */

#ifndef UCMINF_CORE_H
#define UCMINF_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Callback type for evaluating the objective function and its gradient.
 *
 * Parameters:
 *   n        - problem dimension
 *   x        - current point (input, length n)
 *   g        - gradient at x (output, length n)
 *   f        - objective value at x (output)
 *   userdata - user-supplied pointer passed through unchanged
 */
typedef void (*ucminf_fdf_t)(int n, const double *x, double *g, double *f,
                              void *userdata);

/*
 * Main UCMINF optimization routine.
 *
 * Minimizes F(x) starting from x using a quasi-Newton method with BFGS
 * inverse-Hessian updates and a soft line search.
 *
 * Parameters:
 *   n        - problem dimension (must be > 0)
 *   x        - on input:  initial guess (length n)
 *              on output: computed minimizer
 *   dx       - initial trust-region radius (must be > 0)
 *   eps1     - gradient tolerance (stop when max|g(x)| <= eps1)
 *   eps2     - step tolerance (stop when ||step||^2 <= eps2*(eps2+||x||^2))
 *   maxfun   - on input:  maximum number of function evaluations
 *              on output: number of evaluations used
 *   w        - workspace array of length >= n*(n+11)/2 + 10
 *   iw       - length of workspace w
 *   icontr   - control flag on input:
 *                1 = optimize with D0 = identity
 *                3 = optimize with D0 given in w[4n .. 4n+n(n+1)/2-1]
 *              on output: convergence code:
 *                1 = small gradient (grtol satisfied)
 *                2 = small step (xtol satisfied)
 *                3 = function evaluation limit reached
 *                4 = zero step from line search
 *               -2 = n <= 0
 *               -4 = dx <= 0
 *               -5 = eps1 or eps2 <= 0
 *               -6 = maxfun <= 0
 *               -7 = given initial D is not positive definite
 *               -8 = workspace too small
 *   fdf      - callback to evaluate f and g
 *   userdata - passed unchanged to fdf
 */
void ucminf_optimize(int n, double *x, double dx, double eps1, double eps2,
                     int *maxfun, double *w, int iw, int *icontr,
                     ucminf_fdf_t fdf, void *userdata);

#ifdef __cplusplus
}
#endif

#endif /* UCMINF_CORE_H */
