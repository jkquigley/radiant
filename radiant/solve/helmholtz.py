from .base import BaseSolver
import numpy as np


def _lhs_integrand_factory(phi, delta, *args):
    idx = len(args) // 2
    ci = args[:idx]
    cj = args[idx:]

    def func(*x):
        return np.einsum(
            'i...,i...->...',
            phi.grad(delta, *x, *ci),
            phi.grad(delta, *x, *cj),
        ) + phi(delta, *x, *ci) * phi(delta, *x, *cj)

    return func


def _rhs_integrand_factory(f, phi, delta, *ci, guess=None):
    if guess is None:
        def func(*x):
            return f(*x) * phi(delta, *x, *ci)
    else:
        def func(*x):
            return (
                    f(*x) * phi(delta, *x, *ci) -
                    np.sum(guess.grad(*x) * phi.grad(delta, *x, *ci), axis=0) -
                    guess(*x) * phi(delta, *x, *ci)
            )

    return func


class HelmholtzSolver(BaseSolver):
    def __init__(self, integrator, phi, delta, *xc):
        super().__init__(phi, delta, *xc)
        self.integrator = integrator

    def gen_mat(self):
        self.mat = np.zeros((self.n, self.n))
        zipped = tuple(zip(*self.xc))

        for i, ci in enumerate(zipped):
            self.mat[i, i] = self.integrator(_lhs_integrand_factory(
                    self.phi, self.delta, *ci, *ci
            ))

            for j, cj in enumerate(zipped[:i]):
                self.mat[i, j] = self.integrator(_lhs_integrand_factory(
                    self.phi, self.delta, *cj, *cj
                ))

                self.mat[j, i] = self.mat[i, j]

    def gen_rhs(self, func, guess=None):
        self.b = np.zeros(self.n)
        for i, ci in enumerate(zip(*self.xc)):
            self.b[i] = self.integrator(_rhs_integrand_factory(
                func, self.phi, self.delta, *ci, guess=guess
            ))
