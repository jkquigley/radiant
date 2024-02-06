from .base import BaseSolver
import numpy as np


def _lhs_integrand_factory(xi, xj, delta, phi):
    def func(x):
        return (phi(x, xi, delta, m=1) * phi(x, xj, delta, m=1) +
                phi(x, xi, delta) * phi(x, xj, delta))

    return func


def _rhs_integrand_factory(f, xi, delta, phi, guess):
    if guess is None:
        def func(x):
            return f(x) * phi(x, xi, delta)
    else:
        def func(x):
            return (
                    f(x) * phi(x, xi, delta) -
                    guess(x, m=1) * phi(x, xi, delta, m=1) -
                    guess(x) * phi(x, xi, delta)
            )

    return func


class HelmholtzBaseSolver(BaseSolver):
    def __init__(self, phi, centres, delta, integrator):
        super().__init__(phi, centres, delta)
        self.integrator = integrator

    def gen_mat(self):
        self.mat = np.zeros((self.centres.shape[0], self.centres.shape[0]))

        for i, xi in enumerate(self.centres):
            self.mat[i, i] = self.integrator(
                _lhs_integrand_factory(xi, xi, self.delta, self.phi)
            )

            for j, xj in enumerate(self.centres[:i]):
                self.mat[i, j] = self.integrator(
                    _lhs_integrand_factory(xi, xj, self.delta, self.phi)
                )

                self.mat[j, i] = self.mat[i, j]

    def gen_rhs(self, func, guess):
        self.b = np.zeros_like(self.centres)
        for i, xi in enumerate(self.centres):
            self.b[i] = self.integrator(
                _rhs_integrand_factory(func, xi, self.delta, self.phi, guess)
            )
