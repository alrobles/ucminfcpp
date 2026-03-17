#' @useDynLib ucminfcpp, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL

#' Build a validated control list for ucminf()
#'
#' Returns a named list of control parameters that can be passed directly
#' to \code{ucminf()}, \code{ucminf_xptr()}, etc.
#'
#' @param trace       Integer. If > 0, print convergence info. Default 0.
#' @param grtol       Gradient tolerance. Default 1e-6.
#' @param xtol        Step tolerance. Default 1e-12.
#' @param stepmax     Initial trust-region radius. Default 1.
#' @param maxeval     Maximum function evaluations. Default 500.
#' @param grad        Finite-difference type: "forward" or "central". Default "forward".
#' @param gradstep    Length-2 step vector. Default c(1e-6, 1e-8).
#' @param invhessian.lt Packed lower-triangle of initial inverse Hessian. Default NULL.
#' @return A list of class \code{"ucminf_control"}.
#' @export
ucminf_control <- function(trace = 0, grtol = 1e-6, xtol = 1e-12,
                            stepmax = 1, maxeval = 500L,
                            grad = c("forward", "central"),
                            gradstep = c(1e-6, 1e-8),
                            invhessian.lt = NULL) {
    grad <- match.arg(grad)
    stopifnot(length(gradstep) == 2, maxeval > 0, grtol > 0, xtol > 0, stepmax > 0)
    structure(
        list(trace         = as.integer(trace),
             grtol         = as.double(grtol),
             xtol          = as.double(xtol),
             stepmax       = as.double(stepmax),
             maxeval       = as.integer(maxeval),
             grad          = grad,
             gradstep      = as.double(gradstep),
             invhessian.lt = invhessian.lt),
        class = "ucminf_control"
    )
}

#' General-Purpose Unconstrained Non-Linear Optimization (C/Rcpp)
#'
#' An implementation of the UCMINF algorithm translated from Fortran to C and
#' wrapped with Rcpp. The algorithm uses a quasi-Newton method with BFGS
#' updating of the inverse Hessian and a soft line search with trust-region
#' radius monitoring.
#'
#' This package is a two-phase migration of the original \pkg{ucminf} R package:
#' \itemize{
#'   \item \strong{Phase 1}: The Fortran core is translated to C
#'         (\code{src/ucminf_core.c}).
#'   \item \strong{Phase 2}: The C core is wrapped with an Rcpp interface
#'         (\code{src/ucminf_rcpp.cpp}).
#' }
#'
#' The interface is designed to be interchangeable with the original
#' \code{\link[ucminf:ucminf]{ucminf::ucminf}} and with
#' \code{\link[stats]{optim}}.
#'
#' @param par Initial estimate of the minimum (numeric vector).
#' @param fn Objective function to be minimized. Must return a scalar.
#'   Ignored when \code{fdfun} is supplied.
#' @param gr Gradient function. If \code{NULL} a finite-difference
#'   approximation is used. Ignored when \code{fdfun} is supplied.
#' @param ... Optional arguments passed to \code{fn} and \code{gr}.
#' @param fdfun Optional combined value+gradient function
#'   \code{fdfun(x)} returning \code{list(f = scalar, g = numeric_vector)}.
#'   When supplied, \code{fn} and \code{gr} are ignored.
#' @param control A list of control parameters (see Details), or a
#'   \code{\link{ucminf_control}} object.
#' @param hessian Integer controlling Hessian output:
#'   \describe{
#'     \item{0}{No Hessian (default).}
#'     \item{2}{Returns the final inverse-Hessian approximation from BFGS.}
#'     \item{3}{Returns both the inverse Hessian (2) and its inverse (Hessian).}
#'   }
#'
#' @details
#' The \code{control} argument accepts:
#' \describe{
#'   \item{\code{trace}}{If positive, print convergence info after optimization.
#'     Default \code{0}.}
#'   \item{\code{grtol}}{Stop when \eqn{\|g(x)\|_\infty \le} \code{grtol}.
#'     Default \code{1e-6}.}
#'   \item{\code{xtol}}{Stop when \eqn{\|x - x_{\rm prev}\|^2 \le}
#'     \code{xtol}*(\code{xtol} + \eqn{\|x\|^2}). Default \code{1e-12}.}
#'   \item{\code{stepmax}}{Initial trust-region radius. Default \code{1}.}
#'   \item{\code{maxeval}}{Maximum function evaluations. Default \code{500}.}
#'   \item{\code{grad}}{Finite-difference type when \code{gr = NULL}:
#'     \code{"forward"} (default) or \code{"central"}.}
#'   \item{\code{gradstep}}{Length-2 vector; step is
#'     \eqn{|x_i| \cdot \code{gradstep[1]} + \code{gradstep[2]}}.
#'     Default \code{c(1e-6, 1e-8)}.}
#'   \item{\code{invhessian.lt}}{A vector containing the lower triangle of the
#'     initial inverse Hessian (packed column-major). Default: identity.}
#' }
#'
#' @return A list of class \code{"ucminf"} with elements:
#' \item{par}{Computed minimizer.}
#' \item{value}{Objective value at the minimizer.}
#' \item{convergence}{Termination code:
#'   \describe{
#'     \item{1}{Small gradient (\code{grtol}).}
#'     \item{2}{Small step (\code{xtol}).}
#'     \item{3}{Evaluation limit (\code{maxeval}).}
#'     \item{4}{Zero step from line search.}
#'     \item{-2}{n <= 0.}
#'     \item{-4}{stepmax <= 0.}
#'     \item{-5}{grtol or xtol <= 0.}
#'     \item{-6}{maxeval <= 0.}
#'     \item{-7}{Given inverse Hessian not positive definite.}
#'   }}
#' \item{message}{Human-readable termination message.}
#' \item{invhessian.lt}{Lower triangle of the final inverse-Hessian
#'   approximation (packed).}
#' \item{invhessian}{Full inverse-Hessian matrix (when \code{hessian >= 2}).}
#' \item{hessian}{Hessian matrix, inverse of \code{invhessian}
#'   (when \code{hessian == 3}).}
#' \item{info}{Named vector:
#'   \describe{
#'     \item{maxgradient}{\eqn{\|g(x)\|_\infty} at solution.}
#'     \item{laststep}{Length of the last step.}
#'     \item{stepmax}{Final trust-region radius.}
#'     \item{neval}{Number of function/gradient evaluations.}
#'   }}
#'
#' @references
#' Nielsen, H. B. (2000). \emph{UCMINF -- An Algorithm for Unconstrained,
#' Nonlinear Optimization}. Report IMM-REP-2000-19, DTU.
#'
#' @export
#' @examples
#' ## Rosenbrock Banana function
#' fR <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
#' gR <- function(x) c(-400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1]),
#'                     200 * (x[2] - x[1] * x[1]))
#'
#' ## Find minimum with analytic gradient
#' ucminf(par = c(2, 0.5), fn = fR, gr = gR)
#'
#' ## Find minimum with finite-difference gradient
#' ucminf(par = c(2, 0.5), fn = fR)
ucminf <- function(par, fn = NULL, gr = NULL, ..., fdfun = NULL,
                   control = list(), hessian = 0) {

    ## Resolve control — accept ucminf_control objects directly
    if (inherits(control, "ucminf_control")) {
        con <- control
    } else {
        ## Default control values
        con <- list(
            trace          = 0,
            grtol          = 1e-6,
            xtol           = 1e-12,
            stepmax        = 1,
            maxeval        = 500,
            grad           = "forward",
            gradstep       = c(1e-6, 1e-8),
            invhessian.lt  = NULL
        )

        ## Apply user overrides
        unknown <- setdiff(names(control), names(con))
        if (length(unknown) > 0)
            stop("Unknown control parameters: ", paste(unknown, collapse = ", "))
        con[names(control)] <- control

        ## Validate
        stopifnot(length(con$gradstep) == 2)
        stopifnot(con$grad %in% c("forward", "central"))
    }

    ## Restore names of par helper
    nm <- names(par)

    ## ---------- fdfun path (P2-B) ----------
    if (!is.null(fdfun)) {
        ctrl_fdf <- list(
            grtol         = as.double(con$grtol),
            xtol          = as.double(con$xtol),
            stepmax       = as.double(con$stepmax),
            maxeval       = as.integer(con$maxeval),
            invhessian_lt = con$invhessian.lt
        )
        res <- ucminf_fdf_cpp(
            par     = as.double(par),
            fdfun   = fdfun,
            control = ctrl_fdf
        )
        return(.ucminf_postprocess(res, par, nm, hessian, con$trace))
    }

    ## ---------- standard fn/gr path ----------
    if (is.null(fn))
        stop("fn must be supplied when fdfun is NULL")

    ## P1-B: Avoid extra R closure when ... is empty
    if (...length() == 0L) {
        fn_wrapped <- fn
        gr_wrapped <- gr
    } else {
        fn_wrapped <- function(x) fn(x, ...)
        gr_wrapped <- if (!is.null(gr)) function(x) gr(x, ...) else NULL
    }

    ## Determine gradient type code
    has_gr    <- !is.null(gr_wrapped)
    grad_type <- if (has_gr) 0L else if (con$grad == "forward") 1L else 2L

    ## Build control list for the Rcpp function
    ctrl <- list(
        grtol          = as.double(con$grtol),
        xtol           = as.double(con$xtol),
        stepmax        = as.double(con$stepmax),
        maxeval        = as.integer(con$maxeval),
        grad_type      = as.integer(grad_type),
        gradstep       = as.double(con$gradstep),
        invhessian_lt  = con$invhessian.lt
    )

    ## Call the Rcpp optimizer
    res <- ucminf_cpp(
        par     = as.double(par),
        fn      = fn_wrapped,
        gr      = if (has_gr) gr_wrapped else fn_wrapped, # placeholder when has_gr=FALSE
        has_gr  = has_gr,
        control = ctrl
    )

    .ucminf_postprocess(res, par, nm, hessian, con$trace)
}

## Internal helper: decode result and build the "ucminf" S3 object
.ucminf_postprocess <- function(res, par, nm, hessian, trace) {
    icontr <- res$convergence
    msg <- switch(as.character(icontr),
        `1`  = "Stopped by small gradient (grtol).",
        `2`  = "Stopped by small step (xtol).",
        `3`  = "Stopped by function evaluation limit (maxeval).",
        `4`  = "Stopped by zero step from line search.",
        `-2` = "Computation did not start: length(par) = 0.",
        `-4` = "Computation did not start: stepmax is too small.",
        `-5` = "Computation did not start: grtol or xtol <= 0.",
        `-6` = "Computation did not start: maxeval <= 0.",
        `-7` = "Computation did not start: given inverse Hessian not pos. definite.",
        `-8` = "Computation did not start: workspace too small.",
        "Unknown convergence code."
    )

    sol <- res$par
    if (!is.null(nm)) names(sol) <- nm

    ans <- list(
        par         = sol,
        value       = res$value,
        convergence = icontr,
        message     = msg
    )

    if (icontr > 0) {
        n   <- length(par)
        ilt <- res$invhessian_lt

        if (isTRUE(hessian) || hessian >= 2) {
            COV <- matrix(0, n, n)
            lt_idx <- lower.tri(COV, diag = TRUE)
            COV[lt_idx] <- ilt
            COV <- t(COV) + COV - diag(diag(COV))
            ans$invhessian <- COV
            if (!is.null(nm))
                rownames(ans$invhessian) <- colnames(ans$invhessian) <- nm
        }
        if (hessian == 3)
            ans$hessian <- solve(ans$invhessian)

        ans$invhessian.lt <- ilt
        ans$info <- c(
            maxgradient = res$maxgradient,
            laststep    = res$laststep,
            stepmax     = res$stepmax,
            neval       = as.double(res$neval)
        )
    }

    if (trace > 0) {
        cat(paste(ans$message, "\n"))
        if (!is.null(ans$info))
            print(ans$info)
    }

    class(ans) <- "ucminf"
    ans
}


#' Print method for ucminf objects
#'
#' @param x A \code{ucminf} object.
#' @param digits Number of significant digits to print.
#' @param ... Unused.
#' @export
print.ucminf <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
    cat("ucminf result\n")
    cat("  Converged:", x$message, "\n")
    cat("  Minimum value:", format(x$value, digits = digits), "\n")
    cat("  Minimizer:\n")
    print(x$par, digits = digits)
    if (!is.null(x$info)) {
        cat("  Evaluations:", x$info["neval"], "\n")
        cat("  Max |gradient|:", format(x$info["maxgradient"], digits = digits), "\n")
    }
    invisible(x)
}
