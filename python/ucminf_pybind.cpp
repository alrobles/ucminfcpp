/*
 * ucminf_pybind.cpp
 *
 * Python bindings for the ucminf C++ optimization library, implemented with
 * pybind11.  Exposes the ucminf::minimize() function and its support types
 * (Control, Result, Status) to Python.
 *
 * Build
 * -----
 *   pip install pybind11
 *   pip install .            # via setup.py / pyproject.toml
 *
 *   — or via CMake —
 *   cmake -DBUILD_PYTHON=ON ..
 *   cmake --build .
 *
 * Usage (Python)
 * --------------
 *   import ucminf
 *
 *   def rosenbrock(x):
 *       return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
 *
 *   def rosenbrock_grad(x):
 *       return [
 *           -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]),
 *            200*(x[1]-x[0]**2)
 *       ]
 *
 *   ctrl = ucminf.Control()
 *   ctrl.grtol = 1e-8
 *
 *   result = ucminf.minimize(
 *       x0      = [2.0, 0.5],
 *       fn      = rosenbrock,
 *       gr      = rosenbrock_grad,  # optional; omit for finite-differences
 *       control = ctrl,
 *   )
 *   print(result.x, result.f, result.status)
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../src/ucminf_core.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helper: build an ObjFun from separate Python fn(x)->float and gr(x)->[float]
// ---------------------------------------------------------------------------

/// Build an ucminf::ObjFun that calls Python callables fn and gr.
/// If gr is py::none() a two-point central finite difference is used.
static ucminf::ObjFun make_objfun(py::object fn,
                                   py::object gr,
                                   double gradstep_rel = 1e-6,
                                   double gradstep_abs = 1e-8)
{
    bool has_gr = !gr.is_none();

    return [fn, gr, has_gr, gradstep_rel, gradstep_abs]
           (const std::vector<double>& x, std::vector<double>& g, double& f)
    {
        py::list xlist;
        for (double v : x) xlist.append(v);

        f = fn(xlist).cast<double>();

        if (has_gr) {
            auto glist = gr(xlist).cast<std::vector<double>>();
            g = std::move(glist);
        } else {
            // Central finite difference
            int n = static_cast<int>(x.size());
            py::list xm = xlist;
            for (int i = 0; i < n; ++i) {
                double xi = x[i];
                double dx = std::abs(xi) * gradstep_rel + gradstep_abs;

                xm[i] = xi + dx;
                double fp = fn(xm).cast<double>();

                xm[i] = xi - dx;
                double fm = fn(xm).cast<double>();

                xm[i] = xi; // restore
                g[i] = (fp - fm) / (2.0 * dx);
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(ucminf, m)
{
    m.doc() = R"pbdoc(
        ucminf — unconstrained nonlinear optimization (UCMINF algorithm)

        A Python extension wrapping the C++17 ucminf optimization library.
        The algorithm is a quasi-Newton method with BFGS inverse-Hessian
        updating and a soft line search with adaptive trust-region radius.

        Quick start
        -----------
        >>> import ucminf
        >>> result = ucminf.minimize(
        ...     x0  = [2.0, 0.5],
        ...     fn  = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2,
        ...     gr  = lambda x: [-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]),
        ...                       200*(x[1]-x[0]**2)],
        ... )
        >>> result.x     # [1.0, 1.0]
        >>> result.f     # ~0.0
    )pbdoc";

    // -----------------------------------------------------------------------
    // Status enum
    // -----------------------------------------------------------------------
    py::enum_<ucminf::Status>(m, "Status",
        "Termination status code returned by minimize().")
        .value("SmallGradient",              ucminf::Status::SmallGradient,
               "Converged: max|grad| <= grtol.")
        .value("SmallStep",                  ucminf::Status::SmallStep,
               "Converged: step length <= xtol.")
        .value("EvaluationLimitReached",     ucminf::Status::EvaluationLimitReached,
               "Stopped: maxeval reached.")
        .value("ZeroStepFromLineSearch",     ucminf::Status::ZeroStepFromLineSearch,
               "Stopped: line search returned alpha=0.")
        .value("InvalidDimension",           ucminf::Status::InvalidDimension)
        .value("InvalidStepmax",             ucminf::Status::InvalidStepmax)
        .value("InvalidTolerances",          ucminf::Status::InvalidTolerances)
        .value("InvalidMaxeval",             ucminf::Status::InvalidMaxeval)
        .value("HessianNotPositiveDefinite", ucminf::Status::HessianNotPositiveDefinite)
        .export_values();

    m.def("status_message", &ucminf::status_message,
          py::arg("status"),
          "Return a human-readable string for a Status value.");

    // -----------------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------------
    py::class_<ucminf::Control>(m, "Control",
        R"pbdoc(
            Algorithmic control parameters for minimize().

            Attributes
            ----------
            grtol : float
                Stop when max|grad(x)| <= grtol.  Default 1e-6.
            xtol : float
                Stop when ||step||^2 <= xtol*(xtol+||x||^2).  Default 1e-12.
            stepmax : float
                Initial trust-region radius.  Default 1.0.
            maxeval : int
                Maximum function+gradient evaluations.  Default 500.
            inv_hessian_lt : list[float]
                Optional initial inverse-Hessian in packed lower-triangle form
                (column-major, length n*(n+1)/2).  Empty list → identity.
        )pbdoc")
        .def(py::init<>())
        .def_readwrite("grtol",          &ucminf::Control::grtol)
        .def_readwrite("xtol",           &ucminf::Control::xtol)
        .def_readwrite("stepmax",        &ucminf::Control::stepmax)
        .def_readwrite("maxeval",        &ucminf::Control::maxeval)
        .def_readwrite("inv_hessian_lt", &ucminf::Control::inv_hessian_lt)
        .def("__repr__", [](const ucminf::Control& c) {
            return "<ucminf.Control grtol=" + std::to_string(c.grtol) +
                   " xtol="    + std::to_string(c.xtol) +
                   " stepmax=" + std::to_string(c.stepmax) +
                   " maxeval=" + std::to_string(c.maxeval) + ">";
        });

    // -----------------------------------------------------------------------
    // Result
    // -----------------------------------------------------------------------
    py::class_<ucminf::Result>(m, "Result",
        R"pbdoc(
            Result returned by minimize().

            Attributes
            ----------
            x : list[float]
                Parameter values at the (approximate) minimum.
            f : float
                Objective value at x.
            n_eval : int
                Total function/gradient evaluations used.
            max_gradient : float
                max|grad(x)| at the solution.
            last_step : float
                Length of the last accepted step.
            status : Status
                Termination status code.
            inv_hessian_lt : list[float]
                Packed lower-triangle of the final inverse-Hessian
                approximation (length n*(n+1)/2, column-major).
            message : str
                Human-readable termination message.
        )pbdoc")
        .def_readonly("x",              &ucminf::Result::x)
        .def_readonly("f",              &ucminf::Result::f)
        .def_readonly("n_eval",         &ucminf::Result::n_eval)
        .def_readonly("max_gradient",   &ucminf::Result::max_gradient)
        .def_readonly("last_step",      &ucminf::Result::last_step)
        .def_readonly("status",         &ucminf::Result::status)
        .def_readonly("inv_hessian_lt", &ucminf::Result::inv_hessian_lt)
        .def_property_readonly("message", [](const ucminf::Result& r) {
            return ucminf::status_message(r.status);
        })
        .def("__repr__", [](const ucminf::Result& r) {
            std::string xs = "[";
            for (std::size_t i = 0; i < r.x.size(); ++i) {
                if (i) xs += ", ";
                xs += std::to_string(r.x[i]);
            }
            xs += "]";
            return "<ucminf.Result x=" + xs +
                   " f="      + std::to_string(r.f) +
                   " status=" + ucminf::status_message(r.status) + ">";
        });

    // -----------------------------------------------------------------------
    // minimize()
    // -----------------------------------------------------------------------
    m.def(
        "minimize",
        [](std::vector<double>   x0,
           py::object            fn,
           py::object            gr,
           const ucminf::Control control,
           double                gradstep_rel,
           double                gradstep_abs) -> ucminf::Result
        {
            ucminf::ObjFun fdf = make_objfun(fn, gr, gradstep_rel, gradstep_abs);
            return ucminf::minimize(std::move(x0), fdf, control);
        },
        py::arg("x0"),
        py::arg("fn"),
        py::arg("gr")           = py::none(),
        py::arg("control")      = ucminf::Control{},
        py::arg("gradstep_rel") = 1e-6,
        py::arg("gradstep_abs") = 1e-8,
        R"pbdoc(
            Minimize an objective function using the UCMINF quasi-Newton algorithm.

            Parameters
            ----------
            x0 : list[float]
                Initial guess (length n >= 1).
            fn : callable
                Objective function.  Signature: fn(x: list[float]) -> float.
            gr : callable, optional
                Gradient function.  Signature: gr(x: list[float]) -> list[float].
                If None, a central finite-difference approximation is used.
            control : Control, optional
                Algorithmic parameters.  See Control for details.
            gradstep_rel : float, optional
                Relative step size for finite-difference gradient (default 1e-6).
            gradstep_abs : float, optional
                Absolute step size for finite-difference gradient (default 1e-8).

            Returns
            -------
            Result
                Optimization result.  Check result.status for convergence.

            Raises
            ------
            ValueError
                If control parameters are invalid (stepmax <= 0, etc.).

            Examples
            --------
            >>> import ucminf
            >>> fn = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
            >>> gr = lambda x: [-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]),
            ...                  200*(x[1]-x[0]**2)]
            >>> result = ucminf.minimize([2.0, 0.5], fn, gr)
            >>> result.x   # approximately [1.0, 1.0]
        )pbdoc");
}
