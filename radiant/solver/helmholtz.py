import numpy as np
from ..integrate import integrate
from ..util import Approximant
from ..util import RBFParams


def lhs_integrand_factory(phi, xi, xj, delta):
    def func(x):
        return (phi(x, xi, delta, m=1) * phi(x, xj, delta, m=1) +
                phi(x, xi, delta) * phi(x, xj, delta))

    return func


def rhs_integrand_factory(phi, f, xi, delta):
    def func(x):
        return f(x) * phi(x, xi, delta)

    return func


def solve(f, centres, delta, phi, *args, combine=True, **kwargs):
    a = args[0]
    b = args[1]

    A = np.zeros((centres.shape[0], centres.shape[0]))
    fs = np.zeros_like(centres)
    for i, xi in enumerate(centres):
        fs[i] = integrate.trapezoid(
            rhs_integrand_factory(phi, f, xi, delta),
            a, b, 2500
        )

        A[i, i] = integrate.trapezoid(
            lhs_integrand_factory(phi, xi, xi, delta),
            a, b, 2500
        )

        for j, xj in enumerate(centres[:i]):
            A[i, j] = integrate.trapezoid(
                lhs_integrand_factory(phi, xi, xj, delta),
                a, b, 2500
            )

            A[j, i] = A[i, j]

    weights = np.linalg.solve(A, fs)
    params = RBFParams(phi, centres, weights, delta)

    if combine:
        return Approximant(params), A
    else:
        return params, A
