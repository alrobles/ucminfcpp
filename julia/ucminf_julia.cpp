/*
 * ucminf_julia.cpp
 *
 * Julia bindings for the ucminf C++ optimization library, implemented with
 * CxxWrap / JlCxx (https://github.com/JuliaInterop/CxxWrap.jl).
 *
 * Build
 * -----
 *   cmake -DBUILD_JULIA=ON -DJULIA_EXECUTABLE=$(which julia) ..
 *   cmake --build .
 *
 * Usage (Julia)
 * -------------
 *   using ucminf_julia
 *
 *   fn(x) = (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2
 *   gr(x) = [-400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]),
 *             200*(x[2]-x[1]^2)]
 *
 *   ctrl = ucminf_julia.Control()
 *   ctrl.grtol = 1e-8
 *
 *   result = ucminf_julia.minimize([2.0, 0.5], fn, gr, ctrl)
 *   println(result.x, " -> ", result.f)
 *
 * Notes
 * -----
 *  - Requires libcxxwrap-julia (available via Pkg: CxxWrap.jl).
 *  - The CMakeLists.txt in this directory locates the JlCxx headers/libraries
 *    using `find_package(JlCxx)`.
 *  - Julia 1.9+ and CxxWrap ≥ 0.14 are recommended.
 */

#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>

#include "../src/ucminf_core.hpp"

// ---------------------------------------------------------------------------
// Julia module registration
// ---------------------------------------------------------------------------

JLCXX_MODULE define_julia_module(jlcxx::Module& m)
{
    // -----------------------------------------------------------------------
    // Status enum
    // -----------------------------------------------------------------------
    m.add_bits<ucminf::Status>("Status", jlcxx::julia_type("CppEnum"));
    m.set_const("SmallGradient",
                static_cast<int>(ucminf::Status::SmallGradient));
    m.set_const("SmallStep",
                static_cast<int>(ucminf::Status::SmallStep));
    m.set_const("EvaluationLimitReached",
                static_cast<int>(ucminf::Status::EvaluationLimitReached));
    m.set_const("ZeroStepFromLineSearch",
                static_cast<int>(ucminf::Status::ZeroStepFromLineSearch));
    m.set_const("InvalidDimension",
                static_cast<int>(ucminf::Status::InvalidDimension));
    m.set_const("InvalidStepmax",
                static_cast<int>(ucminf::Status::InvalidStepmax));
    m.set_const("InvalidTolerances",
                static_cast<int>(ucminf::Status::InvalidTolerances));
    m.set_const("InvalidMaxeval",
                static_cast<int>(ucminf::Status::InvalidMaxeval));
    m.set_const("HessianNotPositiveDefinite",
                static_cast<int>(ucminf::Status::HessianNotPositiveDefinite));

    m.method("status_message", [](ucminf::Status s) {
        return ucminf::status_message(s);
    });

    // -----------------------------------------------------------------------
    // Control struct
    // -----------------------------------------------------------------------
    m.add_type<ucminf::Control>("Control")
        .constructor<>()
        .method("grtol",   [](const ucminf::Control& c) { return c.grtol;   })
        .method("xtol",    [](const ucminf::Control& c) { return c.xtol;    })
        .method("stepmax", [](const ucminf::Control& c) { return c.stepmax; })
        .method("maxeval", [](const ucminf::Control& c) { return c.maxeval; })
        .method("inv_hessian_lt",
                [](const ucminf::Control& c) { return c.inv_hessian_lt; })
        // Setters
        .method("set_grtol!",   [](ucminf::Control& c, double v) { c.grtol   = v; })
        .method("set_xtol!",    [](ucminf::Control& c, double v) { c.xtol    = v; })
        .method("set_stepmax!", [](ucminf::Control& c, double v) { c.stepmax = v; })
        .method("set_maxeval!", [](ucminf::Control& c, int    v) { c.maxeval = v; })
        .method("set_inv_hessian_lt!",
                [](ucminf::Control& c, const std::vector<double>& v) {
                    c.inv_hessian_lt = v;
                });

    // -----------------------------------------------------------------------
    // Result struct
    // -----------------------------------------------------------------------
    m.add_type<ucminf::Result>("Result")
        .method("x",              [](const ucminf::Result& r) { return r.x;              })
        .method("f",              [](const ucminf::Result& r) { return r.f;              })
        .method("n_eval",         [](const ucminf::Result& r) { return r.n_eval;         })
        .method("max_gradient",   [](const ucminf::Result& r) { return r.max_gradient;   })
        .method("last_step",      [](const ucminf::Result& r) { return r.last_step;      })
        .method("status",         [](const ucminf::Result& r) { return r.status;         })
        .method("inv_hessian_lt", [](const ucminf::Result& r) { return r.inv_hessian_lt; })
        .method("message",        [](const ucminf::Result& r) {
            return ucminf::status_message(r.status);
        });

    // -----------------------------------------------------------------------
    // minimize() — accepts Julia functions via std::function
    // -----------------------------------------------------------------------
    // Variant 1: with explicit gradient
    m.method("minimize",
        [](std::vector<double>            x0,
           std::function<double(std::vector<double>)>              fn,
           std::function<std::vector<double>(std::vector<double>)> gr,
           const ucminf::Control&         ctrl) -> ucminf::Result
        {
            ucminf::ObjFun fdf =
                [&fn, &gr](const std::vector<double>& x,
                           std::vector<double>& g, double& f)
            {
                f = fn(x);
                g = gr(x);
            };
            return ucminf::minimize(std::move(x0), fdf, ctrl);
        });

    // Variant 2: no gradient (central finite difference internally)
    m.method("minimize",
        [](std::vector<double>                                  x0,
           std::function<double(std::vector<double>)>           fn,
           const ucminf::Control&                              ctrl) -> ucminf::Result
        {
            constexpr double gradstep_rel = 1e-6;
            constexpr double gradstep_abs = 1e-8;
            int n = static_cast<int>(x0.size());

            ucminf::ObjFun fdf =
                [&fn, n](const std::vector<double>& x,
                         std::vector<double>& g, double& f)
            {
                f = fn(x);
                std::vector<double> xm = x;
                for (int i = 0; i < n; ++i) {
                    double xi = x[i];
                    double dx = std::abs(xi) * gradstep_rel + gradstep_abs;
                    xm[i] = xi + dx;
                    double fp = fn(xm);
                    xm[i] = xi - dx;
                    double fm = fn(xm);
                    xm[i] = xi;
                    g[i] = (fp - fm) / (2.0 * dx);
                }
            };
            return ucminf::minimize(std::move(x0), fdf, ctrl);
        });
}
