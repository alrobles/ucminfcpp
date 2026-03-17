// inst/examples/rosenbrock_xptr.cpp
//
// Example: create an XPtr<ucminf::ObjFun> from a compiled C++ lambda.
// Compile from R with:
//   Rcpp::sourceCpp(system.file("examples/rosenbrock_xptr.cpp",
//                               package = "ucminfcpp"))
// Then:
//   xp  <- make_rosenbrock_xptr()
//   res <- ucminfcpp::ucminf_xptr(c(2, 0.5), xp)
//   print(res)

// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>

// Use the installed public header
// (when compiled via sourceCpp with the package installed the include
//  path includes inst/include automatically via LinkingTo: ucminfcpp)
#include "ucminf_core.hpp"

//' Create an external pointer to a compiled Rosenbrock ObjFun
//'
//' @return An \code{externalptr} wrapping a \code{ucminf::ObjFun*}.
//' @export
// [[Rcpp::export]]
SEXP make_rosenbrock_xptr() {
    auto* fn = new ucminf::ObjFun(
        [](const std::vector<double>& x,
           std::vector<double>& g,
           double& f)
        {
            double a = 1.0 - x[0];
            double b = x[1] - x[0] * x[0];
            f    =  a * a + 100.0 * b * b;
            g[0] = -2.0 * a - 400.0 * x[0] * b;
            g[1] =  200.0 * b;
        });
    return Rcpp::XPtr<ucminf::ObjFun>(fn, true);
}
