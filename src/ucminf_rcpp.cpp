/*
 * ucminf_rcpp.cpp
 *
 * Phase 3: Rcpp interface wrapping the modern C++ core implementation.
 *
 * This file provides the R-callable function ucminf_cpp() which connects
 * the C++17 UCMINF optimization algorithm (ucminf_core.cpp) to R via Rcpp.
 * User-supplied R functions for the objective and gradient are called back
 * through a C++ lambda captured by the ObjFun std::function.
 */

// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include "ucminf_core.hpp"

using namespace Rcpp;

/* =========================================================================
 * Exported Rcpp function
 * ========================================================================= */

//' Unconstrained Nonlinear Optimization (C++/Rcpp implementation)
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
    double eps1      = as<double>(control["grtol"]);
    double eps2      = as<double>(control["xtol"]);
    double stepmax   = as<double>(control["stepmax"]);
    int    maxeval   = as<int>(control["maxeval"]);
    int    grad_type = as<int>(control["grad_type"]); /* 0=user, 1=fwd, 2=central */
    NumericVector gradstep  = as<NumericVector>(control["gradstep"]);
    SEXP invh_lt_sexp = control["invhessian_lt"];

    /* Build the ucminf::Control struct */
    ucminf::Control ctrl;
    ctrl.grtol   = eps1;
    ctrl.xtol    = eps2;
    ctrl.stepmax = stepmax;
    ctrl.maxeval = maxeval;

    int nn = n * (n + 1) / 2;
    bool has_invh = (invh_lt_sexp != R_NilValue);
    if (has_invh) {
        NumericVector invh_lt = as<NumericVector>(invh_lt_sexp);
        if (invh_lt.size() != nn)
            Rcpp::stop("invhessian_lt must have length n*(n+1)/2 = %d", nn);
        ctrl.inv_hessian_lt.assign(invh_lt.begin(), invh_lt.end());
    }

    /* Copy par into a std::vector for the C++ optimizer */
    std::vector<double> x0(par.begin(), par.end());

    /* Build ObjFun lambda that calls R functions */
    double gradstep_rel = gradstep[0];
    double gradstep_abs = gradstep[1];
    bool   use_fwd      = (grad_type == 1);
    bool   use_gr       = has_gr;

    ucminf::ObjFun fdf =
        [&fn, &gr, use_gr, use_fwd, gradstep_rel, gradstep_abs, n]
        (const std::vector<double>& xv, std::vector<double>& gv, double& f)
    {
        NumericVector xr(xv.begin(), xv.end());

        /* Evaluate objective function */
        f = as<double>(fn(xr));

        /* Evaluate gradient */
        if (use_gr) {
            NumericVector grval = as<NumericVector>(gr(xr));
            if (grval.size() != n)
                Rcpp::stop("gradient function must return a numeric vector of length %d", n);
            for (int i = 0; i < n; ++i) gv[i] = grval[i];
        } else {
            /* Finite-difference gradient */
            NumericVector xr_mut = clone(xr);
            if (use_fwd) {
                /* Forward differences */
                for (int i = 0; i < n; ++i) {
                    double xi  = xr_mut[i];
                    double dx  = std::abs(xi) * gradstep_rel + gradstep_abs;
                    xr_mut[i]  = xi + dx;
                    double f1  = as<double>(fn(xr_mut));
                    gv[i]      = (f1 - f) / dx;
                    xr_mut[i]  = xi;
                }
            } else {
                /* Central differences */
                for (int i = 0; i < n; ++i) {
                    double xi  = xr_mut[i];
                    double dx  = std::abs(xi) * gradstep_rel + gradstep_abs;
                    xr_mut[i]  = xi + dx;
                    double f1  = as<double>(fn(xr_mut));
                    xr_mut[i]  = xi - dx;
                    double f2  = as<double>(fn(xr_mut));
                    gv[i]      = (f1 - f2) / (2.0 * dx);
                    xr_mut[i]  = xi;
                }
            }
        }
    };

    /* Run the C++ optimizer */
    ucminf::Result res;
    try {
        res = ucminf::minimize(std::move(x0), fdf, ctrl);
    } catch (const std::invalid_argument& e) {
        Rcpp::stop(e.what());
    }

    int icontr = static_cast<int>(res.status);

    /* Pack results into an R list */
    NumericVector x_out(res.x.begin(), res.x.end());
    NumericVector invh_out(res.inv_hessian_lt.begin(), res.inv_hessian_lt.end());

    List result = List::create(
        Named("par")           = x_out,
        Named("value")         = res.f,
        Named("convergence")   = icontr,
        Named("neval")         = res.n_eval,
        Named("maxgradient")   = res.max_gradient,
        Named("laststep")      = res.last_step,
        Named("stepmax")       = stepmax,
        Named("invhessian_lt") = invh_out
    );

    return result;
}
