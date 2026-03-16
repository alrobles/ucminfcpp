#' @useDynLib ucminfcpp, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL

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
#' @param gr Gradient function. If \code{NULL} a finite-difference
#'   approximation is used.
#' @param ... Optional arguments passed to \code{fn} and \code{gr}.
#' @param control A list of control parameters (see Details).
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
ucminf <- function(par, fn, gr = NULL, ..., control = list(), hessian = 0) {

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

    ## Wrap fn and gr to pass ... arguments
    fn_wrapped <- function(x) fn(x, ...)
    gr_wrapped <- if (!is.null(gr)) function(x) gr(x, ...) else NULL

    ## Determine gradient type code
    has_gr   <- !is.null(gr_wrapped)
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
        par    = as.double(par),
        fn     = fn_wrapped,
        gr     = if (has_gr) gr_wrapped else fn_wrapped, # placeholder, has_gr=FALSE
        has_gr = has_gr,
        control = ctrl
    )

    ## Decode convergence
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

    ## Restore names of par
    nm <- names(par)
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

        ## Optional Hessian output
        if (isTRUE(hessian) || hessian == 2 || hessian == 3) {
            n2 <- n * n
            ## Reconstruct full symmetric matrix from lower triangle
            COV <- matrix(0, n, n)
            lt_idx <- lower.tri(COV, diag = TRUE)
            COV[lt_idx] <- ilt
            COV <- t(COV) + COV - diag(diag(COV))
            ans$invhessian <- COV
            if (!is.null(nm)) {
                rownames(ans$invhessian) <- colnames(ans$invhessian) <- nm
            }
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

    if (con$trace > 0) {
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
