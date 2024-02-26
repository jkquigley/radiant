from .base import BaseSolver
from ..util import dot
import numpy as np


def _helmholtz_bilinear_integrand_factory(u, v):
    def func(*x):
        return dot(u.grad(*x), v.grad(*x)) - u(*x) * v(*x)

    return func


def _poisson_bilinear_integrand_factory(u, v):
    def func(*x):
        return dot(u.grad(*x), v.grad(*x))

    return func


def _l2_integrand_factory(u, v):
    def func(*x):
        return u(*x) * v(*x)

    return func


_bilinear_integrand_factory_dict = {
    "helmholtz": _helmholtz_bilinear_integrand_factory,
    "poisson": _poisson_bilinear_integrand_factory,
}


class GalerkinSolver(BaseSolver):
    def __init__(self, phi, integrator, equation):
        super().__init__(phi)

        self.integrator = integrator
        self.bilinear_integrand_factory = _bilinear_integrand_factory_dict[
            equation.lower()
        ]

    def gen_mat(self):
        mat = np.zeros((self.phi.n, self.phi.n))

        for i in range(self.phi.n):
            phii = self.phi[i]
            mat[i, i] = self.integrator(
                self.bilinear_integrand_factory(phii, phii)
            )
            for j in range(i):
                phij = self.phi[j]
                mat[i, j] = self.integrator(
                    self.bilinear_integrand_factory(phii, phij)
                )
                mat[j, i] = mat[i, j]

        return mat

    def gen_rhs(self, *funcs):
        funcs = list(funcs)
        f = funcs.pop(0)

        b = np.zeros(self.phi.n)
        for i in range(self.phi.n):
            b[i] = self.integrator(
                _l2_integrand_factory(f, self.phi[i])
            ) - np.sum([self.integrator(
                self.bilinear_integrand_factory(g, self.phi[i])
            ) for g in funcs], axis=0)

        return b
