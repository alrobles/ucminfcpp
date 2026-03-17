/*
 * ucminf_rcpp.cpp
 *
 * Rcpp interface wrapping the modern C++ core implementation.
 *
 * This file provides the R-callable functions:
 *   ucminf_cpp()     — standard fn/gr R-function interface
 *   ucminf_fdf_cpp() — combined fdfun(x)->list(f,g) interface (P2-B)
 *   ucminf_xptr_cpp()— compiled C++ ObjFun via XPtr (P3-A)
 */

// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include "include/ucminf_core.hpp"

using namespace Rcpp;

/* =========================================================================
 * Helper: extract common control fields and populate ucminf::Control
 * ========================================================================= */
static ucminf::Control extract_ctrl(int n, const List& control)
{
    ucminf::Control ctrl;
    ctrl.grtol   = as<double>(control["grtol"]);
    ctrl.xtol    = as<double>(control["xtol"]);
    ctrl.stepmax = as<double>(control["stepmax"]);
    ctrl.maxeval = as<int>(control["maxeval"]);

    SEXP invh_lt_sexp = control["invhessian_lt"];
    if (invh_lt_sexp != R_NilValue) {
        int nn = n * (n + 1) / 2;
        NumericVector invh_lt = as<NumericVector>(invh_lt_sexp);
        if (invh_lt.size() != nn)
            Rcpp::stop("invhessian_lt must have length n*(n+1)/2 = %d", nn);
        ctrl.inv_hessian_lt.assign(invh_lt.begin(), invh_lt.end());
    }
    return ctrl;
}

/* =========================================================================
 * Helper: pack optimizer result into an R list
 * ========================================================================= */
static List pack_result(const ucminf::Result& res, double stepmax)
{
    NumericVector x_out(res.x.begin(), res.x.end());
    NumericVector invh_out(res.inv_hessian_lt.begin(), res.inv_hessian_lt.end());
    return List::create(
        Named("par")           = x_out,
        Named("value")         = res.f,
        Named("convergence")   = static_cast<int>(res.status),
        Named("neval")         = res.n_eval,
        Named("maxgradient")   = res.max_gradient,
        Named("laststep")      = res.last_step,
        Named("stepmax")       = stepmax,
        Named("invhessian_lt") = invh_out
    );
}

/* =========================================================================
 * ucminf_cpp — standard fn/gr R-function interface
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
//'   \code{grad_type}, \code{gradstep}, \code{invhessian_lt}.
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

    /* P1-C: Read gradstep via REAL() to avoid materialising a NumericVector.
     * gradstep is validated to have length 2 in R (ucminf_control / ucminf.R)
     * before being passed here, so the two-element access is safe. */
    SEXP gradstep_sexp = control["gradstep"];
    if (Rf_length(gradstep_sexp) < 2)
        Rcpp::stop("gradstep must have length 2");
    double gradstep_rel = REAL(gradstep_sexp)[0];
    double gradstep_abs = REAL(gradstep_sexp)[1];

    int    grad_type = as<int>(control["grad_type"]); /* 0=user, 1=fwd, 2=central */
    bool   use_fwd   = (grad_type == 1);
    bool   use_gr    = has_gr;

    double stepmax = as<double>(control["stepmax"]);
    ucminf::Control ctrl = extract_ctrl(n, control);

    /* P1-A: Pre-allocate reusable NumericVector buffers outside the lambda */
    NumericVector xr_buf(n);
    NumericVector xr_mut_buf(n);

    /* P2-A: Use auto (lambda type) so minimize_direct<F> can inline the call */
    auto fdf = [&fn, &gr, &xr_buf, &xr_mut_buf, use_gr, use_fwd,
                gradstep_rel, gradstep_abs, n]
        (const std::vector<double>& xv, std::vector<double>& gv, double& f)
    {
        /* Overwrite pre-allocated buffer in-place — no heap allocation */
        std::copy(xv.begin(), xv.end(), xr_buf.begin());

        f = as<double>(fn(xr_buf));

        if (use_gr) {
            NumericVector grval = as<NumericVector>(gr(xr_buf));
            if (grval.size() != n)
                Rcpp::stop("gradient function must return a numeric vector of length %d", n);
            for (int i = 0; i < n; ++i) gv[i] = grval[i];
        } else {
            std::copy(xv.begin(), xv.end(), xr_mut_buf.begin());
            if (use_fwd) {
                for (int i = 0; i < n; ++i) {
                    double xi = xr_mut_buf[i];
                    double dx = std::abs(xi) * gradstep_rel + gradstep_abs;
                    xr_mut_buf[i] = xi + dx;
                    double f1 = as<double>(fn(xr_mut_buf));
                    gv[i] = (f1 - f) / dx;
                    xr_mut_buf[i] = xi;
                }
            } else {
                for (int i = 0; i < n; ++i) {
                    double xi = xr_mut_buf[i];
                    double dx = std::abs(xi) * gradstep_rel + gradstep_abs;
                    xr_mut_buf[i] = xi + dx;
                    double f1 = as<double>(fn(xr_mut_buf));
                    xr_mut_buf[i] = xi - dx;
                    double f2 = as<double>(fn(xr_mut_buf));
                    gv[i] = (f1 - f2) / (2.0 * dx);
                    xr_mut_buf[i] = xi;
                }
            }
        }
    };

    std::vector<double> x0(par.begin(), par.end());

    ucminf::Result res;
    try {
        res = ucminf::minimize_direct(std::move(x0), fdf, ctrl);
    } catch (const std::invalid_argument& e) {
        Rcpp::stop(e.what());
    }

    return pack_result(res, stepmax);
}

/* =========================================================================
 * ucminf_fdf_cpp — combined fdfun(x)->list(f,g) interface (P2-B)
 * ========================================================================= */

//' Optimize using a combined value+gradient R function
//'
//' Internal function called by \code{\link{ucminf}} when \code{fdfun} is
//' supplied. Use \code{ucminf()} instead.
//'
//' @param par    Numeric starting vector.
//' @param fdfun  R function \code{fdfun(x)} returning
//'   \code{list(f = scalar, g = numeric_vector_of_length_n)}.
//' @param control Named list of control parameters (same as \code{ucminf_cpp}).
//' @return Same list structure as \code{ucminf_cpp}.
//' @keywords internal
// [[Rcpp::export]]
List ucminf_fdf_cpp(NumericVector par, Function fdfun, List control)
{
    int n = par.size();
    if (n <= 0)
        Rcpp::stop("par must have length >= 1");

    double stepmax = as<double>(control["stepmax"]);
    ucminf::Control ctrl = extract_ctrl(n, control);

    NumericVector xr_buf(n);

    auto fdf = [&fdfun, &xr_buf, n]
        (const std::vector<double>& xv, std::vector<double>& gv, double& f)
    {
        std::copy(xv.begin(), xv.end(), xr_buf.begin());
        List ret = as<List>(fdfun(xr_buf));
        f = as<double>(ret["f"]);
        NumericVector gret = as<NumericVector>(ret["g"]);
        if (gret.size() != n)
            Rcpp::stop("fdfun must return g of length %d", n);
        for (int i = 0; i < n; ++i) gv[i] = gret[i];
    };

    std::vector<double> x0(par.begin(), par.end());

    ucminf::Result res;
    try {
        res = ucminf::minimize_direct(std::move(x0), fdf, ctrl);
    } catch (const std::invalid_argument& e) {
        Rcpp::stop(e.what());
    }

    return pack_result(res, stepmax);
}

/* =========================================================================
 * ucminf_xptr_cpp — compiled C++ ObjFun via XPtr (P3-A)
 * ========================================================================= */

//' Optimize using a compiled C++ objective function passed as an external pointer
//'
//' @param par     Numeric starting vector.
//' @param xptr    An \code{externalptr} wrapping a heap-allocated
//'   \code{ucminf::ObjFun*} created via \code{Rcpp::XPtr<ucminf::ObjFun>}.
//' @param control Named list of control parameters (same as \code{ucminf_cpp}).
//' @return Same list structure as \code{ucminf_cpp}.
//' @keywords internal
// [[Rcpp::export]]
List ucminf_xptr_cpp(NumericVector par, SEXP xptr, List control)
{
    Rcpp::XPtr<ucminf::ObjFun> ptr(xptr);
    ucminf::ObjFun& fdf_ref = *ptr;

    int n = par.size();
    if (n <= 0)
        Rcpp::stop("par must have length >= 1");

    double stepmax = as<double>(control["stepmax"]);
    ucminf::Control ctrl = extract_ctrl(n, control);

    std::vector<double> x0(par.begin(), par.end());

    ucminf::Result res;
    try {
        res = ucminf::minimize_direct(std::move(x0), fdf_ref, ctrl);
    } catch (const std::invalid_argument& e) {
        Rcpp::stop(e.what());
    }

    return pack_result(res, stepmax);
}
