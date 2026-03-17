/*
 * ucminf_core.cpp
 *
 * Defines the non-template minimize() public entry point, which wraps the
 * template kernel (detail::minimize_impl<F>, in ucminf_core_impl.hpp) with
 * the ObjFun (std::function) type.
 *
 * Original Fortran algorithm by Hans Bruun Nielsen, IMM, DTU, 2000.
 * Reference: H.B. Nielsen, "UCMINF -- An Algorithm for Unconstrained,
 * Nonlinear Optimization", Report IMM-REP-2000-19, DTU, December 2000.
 */

#include "include/ucminf_core.hpp"

namespace ucminf {

Result minimize(std::vector<double> x, ObjFun fdf, const Control& control)
{
    return detail::minimize_impl(std::move(x), std::move(fdf), control);
}

} // namespace ucminf
