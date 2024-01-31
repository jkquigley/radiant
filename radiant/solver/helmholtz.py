import numpy as np
import cupy as cp
from .approximant import Approximant
from .approximant import RBFParams


def _lhs_integrand_factory(xi, xj, delta, phi):
    def func(x):
        return (phi(x, xi, delta, m=1) * phi(x, xj, delta, m=1) +
                phi(x, xi, delta) * phi(x, xj, delta))

    return func


def _rhs_integrand_factory(f, xi, delta, phi, approx):
    def func(x):
        return (f(x) * phi(x, xi, delta) -
                approx(x, m=1) * phi(x, xi, delta, m=1) -
                approx(x) * phi(x, xi, delta))

    return func


def solve(
        f, centres, delta, phi, integrator, *args,
        combine=True, approx=None, **kwargs
):
    if approx is None:
        approx = Approximant()

    mat = np.zeros((centres.shape[0], centres.shape[0]))
    fs = np.zeros_like(centres)

    for i, xi in enumerate(centres):
        fs[i] = integrator(_rhs_integrand_factory(f, xi, delta, phi, approx))

        mat[i, i] = integrator(_lhs_integrand_factory(xi, xi, delta, phi))

        for j, xj in enumerate(centres[:i]):
            mat[i, j] = integrator(_lhs_integrand_factory(xi, xj, delta, phi))

            mat[j, i] = mat[i, j]

    weights = cp.linalg.solve(cp.array(mat), cp.array(fs)).get()
    params = RBFParams(phi, centres, delta, weights)

    if combine:
        return Approximant(params), [np.linalg.cond(mat)]
    else:
        return params, [np.linalg.cond(mat)]
