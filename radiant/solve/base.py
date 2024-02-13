import cupy as cp
import numpy as np
from ..util import flatten


class Approximant:
    def __init__(self, phi, delta, *xc, w=None):
        self.phi = phi
        self.delta = delta
        self.xc = xc
        if w is None:
            self.w = np.ones(np.size(xc[0]))
        else:
            self.w = w

    def __call__(self, *x, m=None):
        return np.einsum(
            'i,i...->...',
            self.w,
            self.phi(self.delta, *x, *self.xc, m=m),
        )

    def set_w(self, w):
        self.w = w

    def grad(self, *x):
        return np.array([
            self.__call__(*x, m=i) for i in range(self.phi.d)
        ])


class BaseSolver:
    def __init__(self, phi, delta, *xc):
        if phi.d != len(xc):
            raise ValueError(
                f"Dimension mismatch between phi ({phi.d}) and "
                f"centres ({len(xc)})."
            )

        self.phi = phi
        self.xc = tuple(map(flatten, xc))
        self.delta = delta
        self.n = np.size(xc[0])
        self.mat = None
        self.b = None

    def gen_mat(self):
        pass

    def gen_rhs(self, func, guess):
        pass

    def solve(self, func, guess=None):
        if self.mat is None:
            self.gen_mat()

        self.gen_rhs(func, guess)

        w = cp.linalg.lstsq(cp.array(self.mat), cp.array(self.b))[0].get()

        return Approximant(self.phi, self.delta, *self.xc, w=w)

    def cond(self):
        if self.mat is None:
            return -1.
        else:
            return np.linalg.cond(self.mat)
