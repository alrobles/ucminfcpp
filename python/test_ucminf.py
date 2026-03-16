"""
Simple tests and usage examples for the ucminf Python module.

Run with:
    python -m pytest python/test_ucminf.py -v
or:
    python python/test_ucminf.py
"""

import math
import sys


def test_import():
    import ucminf  # noqa: F401


def test_rosenbrock_with_gradient():
    import ucminf

    fn = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    gr = lambda x: [
        -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] ** 2),
    ]

    result = ucminf.minimize(x0=[2.0, 0.5], fn=fn, gr=gr)

    assert result.status in (ucminf.Status.SmallGradient, ucminf.Status.SmallStep), (
        f"Unexpected status: {result.status}"
    )
    assert abs(result.x[0] - 1.0) < 1e-5, f"x[0] = {result.x[0]}"
    assert abs(result.x[1] - 1.0) < 1e-5, f"x[1] = {result.x[1]}"
    assert result.f < 1e-10, f"f = {result.f}"
    assert result.n_eval > 0
    assert result.max_gradient >= 0.0
    assert len(result.inv_hessian_lt) == 3  # n*(n+1)/2 = 2*3/2 = 3


def test_rosenbrock_finite_differences():
    """ucminf can estimate the gradient via finite differences when gr=None."""
    import ucminf

    fn = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    result = ucminf.minimize(x0=[2.0, 0.5], fn=fn)  # gr omitted

    assert result.status in (ucminf.Status.SmallGradient, ucminf.Status.SmallStep)
    assert abs(result.x[0] - 1.0) < 1e-4
    assert abs(result.x[1] - 1.0) < 1e-4


def test_quadratic_1d():
    """1-D quadratic f(x) = x^2 minimised at x=0."""
    import ucminf

    fn = lambda x: x[0] ** 2
    gr = lambda x: [2.0 * x[0]]

    result = ucminf.minimize(x0=[3.0], fn=fn, gr=gr)

    assert result.status in (ucminf.Status.SmallGradient, ucminf.Status.SmallStep)
    assert abs(result.x[0]) < 1e-6
    assert result.f < 1e-12


def test_control_grtol():
    """Tighter grtol should yield a smaller gradient at convergence."""
    import ucminf

    fn = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    gr = lambda x: [
        -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] ** 2),
    ]

    ctrl = ucminf.Control()
    ctrl.grtol = 1e-9

    result = ucminf.minimize(x0=[2.0, 0.5], fn=fn, gr=gr, control=ctrl)

    assert result.status in (ucminf.Status.SmallGradient, ucminf.Status.SmallStep)
    assert result.max_gradient <= 1e-8


def test_status_message():
    import ucminf

    msg = ucminf.status_message(ucminf.Status.SmallGradient)
    assert isinstance(msg, str) and len(msg) > 0


def test_result_repr():
    import ucminf

    fn = lambda x: x[0] ** 2
    result = ucminf.minimize(x0=[1.0], fn=fn)
    r = repr(result)
    assert "ucminf.Result" in r


def test_control_repr():
    import ucminf

    ctrl = ucminf.Control()
    r = repr(ctrl)
    assert "ucminf.Control" in r


def test_warm_start_identity_hessian():
    """Providing the identity as D0 should still converge."""
    import ucminf

    fn = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    gr = lambda x: [
        -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] ** 2),
    ]

    ctrl = ucminf.Control()
    ctrl.inv_hessian_lt = [1.0, 0.0, 1.0]  # identity for n=2

    result = ucminf.minimize(x0=[2.0, 0.5], fn=fn, gr=gr, control=ctrl)

    assert result.status in (ucminf.Status.SmallGradient, ucminf.Status.SmallStep)
    assert abs(result.x[0] - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_import,
        test_rosenbrock_with_gradient,
        test_rosenbrock_finite_differences,
        test_quadratic_1d,
        test_control_grtol,
        test_status_message,
        test_result_repr,
        test_control_repr,
        test_warm_start_identity_hessian,
    ]

    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1

    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    sys.exit(0 if failed == 0 else 1)
