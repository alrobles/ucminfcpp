/*
 * ucminf_rcpp.cpp
 *
 * Phase 2: Rcpp interface wrapping the C core implementation.
 *
 * This file provides the R-callable function ucminf_cpp() which connects
 * the C UCMINF optimization algorithm (ucminf_core.c) to R via Rcpp.
 * User-supplied R functions for the objective and gradient are called back
 * through a lightweight C++ trampoline.
 */

// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>

extern "C" {
#include "ucminf_core.h"
}

using namespace Rcpp;

/* =========================================================================
 * R callback trampoline
 * ========================================================================= */

/*
 * Data passed from Rcpp into the C callback.
 */
struct RCallbackData {
    Function fn;        /* R objective function */
    Function gr;        /* R gradient function (optional) */
    bool has_gr;        /* whether gr was provided */
    int grad_type;      /* 0=user gradient, 1=forward diff, 2=central diff */
    NumericVector gradstep; /* step sizes for finite-difference gradient */
};

/*
 * C callback that evaluates the R objective function and gradient.
 * Passed as `fdf` to ucminf_optimize().
 */
static void r_fdf_callback(int n, const double *x, double *g, double *f,
                           void *userdata)
{
    RCallbackData *data = static_cast<RCallbackData *>(userdata);

    /* Wrap x in an R numeric vector (copy) */
    NumericVector xvec(x, x + n);

    /* Evaluate objective function */
    *f = as<double>(data->fn(xvec));

    /* Evaluate gradient */
    if (data->has_gr) {
        NumericVector gvec = as<NumericVector>(data->gr(xvec));
        if (gvec.size() != n)
            Rcpp::stop("gradient function must return a numeric vector of length %d", n);
        for (int i = 0; i < n; i++) g[i] = gvec[i];
    } else {
        /* Finite-difference gradient */
        double gradstep_rel = data->gradstep[0];  /* relative step component */
        double gradstep_abs = data->gradstep[1];  /* absolute step component */
        int fwd = (data->grad_type == 1);
        for (int i = 0; i < n; i++) {
            double xi  = xvec[i];
            double dx  = std::abs(xi) * gradstep_rel + gradstep_abs;
            xvec[i]    = xi + dx;
            double f1  = as<double>(data->fn(xvec));
            if (fwd) {
                g[i] = (f1 - *f) / dx;
                xvec[i] = xi;
            } else {
                xvec[i] = xi - dx;
                double f2 = as<double>(data->fn(xvec));
                g[i] = (f1 - f2) / (2.0 * dx);
                xvec[i] = xi;
            }
        }
    }
}

/* =========================================================================
 * Exported Rcpp function
 * ========================================================================= */

//' Unconstrained Nonlinear Optimization (C/Rcpp implementation)
//'
//' Internal function called by \code{\link{ucminf}}. Use that instead.
//'
//' @param par Numeric vector of starting values.
//' @param fn R function returning the objective value.
//' @param gr R function returning the gradient, or NULL.
//' @param has_gr Logical: TRUE if gr is provided.
//' @param control Named list of control parameters:
//'   \code{grtol}, \code{xtol}, \code{stepmax}, \code{maxeval},
//'   \code{grad}, \code{gradstep}, \code{invhessian_lt}.
//'
//' @return A list with elements \code{par}, \code{value}, \code{convergence},
//'   \code{neval}, \code{maxgradient}, \code{laststep}, \code{stepmax},
//'   and \code{invhessian_lt}.
//'
//' @keywords internal
// [[Rcpp::export]]
List ucminf_cpp(NumericVector par, Function fn, Function gr, bool has_gr,
                List control)
{
    int n = par.size();
    if (n <= 0)
        Rcpp::stop("par must have length >= 1");

    /* Extract control parameters */
    double eps1    = as<double>(control["grtol"]);
    double eps2    = as<double>(control["xtol"]);
    double stepmax = as<double>(control["stepmax"]);
    int    maxeval = as<int>(control["maxeval"]);
    int    grad_type = as<int>(control["grad_type"]); /* 0=user, 1=fwd, 2=central */
    NumericVector gradstep = as<NumericVector>(control["gradstep"]);
    SEXP invh_lt_sexp = control["invhessian_lt"];

    /* Workspace size: mirrors original R package formula
     *   iw = n * ceil(max(n+1, (n+11)/2)) + 10
     * Here we use: 4n + nn + 2n + extra  (for x_prev, g_prev, g_curr, h, D, sline ws)
     * Minimum needed: 4n + n(n+1)/2 + 2n = n(n+9)/2 + 6n
     * Use the same formula as the original package for compatibility. */
    int nn = n * (n + 1) / 2;
    /* Conservative: at least 4n + nn + 2n + 10 */
    int iw = 4 * n + nn + 2 * n + 10;
    /* Also ensure we match the original formula */
    int iw2 = n * (int)std::ceil(std::max((double)(n + 1),
                                          (double)(n + 11) / 2.0)) + 10;
    if (iw < iw2) iw = iw2;

    NumericVector w(iw, 0.0);

    /* Control flag: 1 = optimize with D0=I; 3 = optimize with user D0 */
    int icontr = 1;
    bool has_invh = (invh_lt_sexp != R_NilValue);
    if (has_invh) {
        NumericVector invh_lt = as<NumericVector>(invh_lt_sexp);
        if (invh_lt.size() != nn)
            Rcpp::stop("invhessian_lt must have length n*(n+1)/2 = %d", nn);
        /* Place D0 in the correct position in the workspace */
        for (int i = 0; i < nn; i++)
            w[4 * n + i] = invh_lt[i];
        icontr = 3;
    }

    /* Copy par into a working buffer */
    NumericVector x = clone(par);

    /* Set up callback data */
    RCallbackData cbdata = { fn, gr, has_gr, grad_type, gradstep };

    int maxfun = maxeval;

    /* Call the C optimizer */
    ucminf_optimize(n, x.begin(), stepmax, eps1, eps2,
                    &maxfun, w.begin(), iw, &icontr,
                    r_fdf_callback, &cbdata);

    /* Retrieve diagnostics from workspace (overwritten at done:) */
    double fx   = w[0]; /* objective value at solution */
    double nmg  = w[1]; /* max |gradient| at solution  */
    double nmh  = w[2]; /* last step length             */

    /* Pack results */
    List result = List::create(
        Named("par")            = x,
        Named("value")          = fx,
        Named("convergence")    = icontr,
        Named("neval")          = maxfun,
        Named("maxgradient")    = nmg,
        Named("laststep")       = nmh,
        Named("stepmax")        = stepmax,
        Named("invhessian_lt")  = NumericVector(w.begin() + 4 * n,
                                                w.begin() + 4 * n + nn)
    );

    return result;
}
