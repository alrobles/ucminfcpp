#' Optimize using a compiled C++ objective (XPtr interface)
#'
#' Calls the UCMINF optimizer with a compiled C++ objective function
#' that has been wrapped in an \code{Rcpp::XPtr<ucminf::ObjFun>}.
#' This path bypasses the R interpreter on every function evaluation,
#' giving maximum performance for non-trivial objectives.
#'
#' @param par     Numeric starting vector.
#' @param xptr    An \code{externalptr} created by wrapping a
#'   \code{ucminf::ObjFun*} in \code{Rcpp::XPtr<ucminf::ObjFun>}.
#' @param control A named list of control parameters (see \code{\link{ucminf}}),
#'   or a \code{\link{ucminf_control}} object.
#' @param hessian Integer: 0 = none, 2 = inv-Hessian, 3 = both.
#' @return A list of class \code{"ucminf"} (same structure as \code{\link{ucminf}}).
#' @seealso \code{\link{ucminf}}, \code{\link{ucminf_control}}
#' @export
ucminf_xptr <- function(par, xptr, control = list(), hessian = 0) {
    con <- if (inherits(control, "ucminf_control")) {
        control
    } else {
        do.call(ucminf_control, control)
    }

    ctrl <- list(
        grtol         = con$grtol,
        xtol          = con$xtol,
        stepmax       = con$stepmax,
        maxeval       = as.integer(con$maxeval),
        invhessian_lt = con$invhessian.lt
    )

    res <- ucminf_xptr_cpp(
        par     = as.double(par),
        xptr    = xptr,
        control = ctrl
    )

    nm <- names(par)
    .ucminf_postprocess(res, par, nm, hessian, trace = 0L)
}
