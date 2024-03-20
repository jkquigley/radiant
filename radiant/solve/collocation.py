from .base import BaseSolver
import numpy as np


class CollocationSolver(BaseSolver):
    def __init__(self, phi, L, Lidx_func, B, Bidx_func):
        super().__init__(phi)

        self.L = L
        self.Lidx = Lidx_func(phi.xc)
        self.B = B
        self.Bidx = Bidx_func(phi.xc)

    def gen_mat(self):
        mat = np.zeros((self.phi.n, self.phi.n))

        mat[self.Lidx, :] = self.L(self.phi[self.Lidx])(*self.phi.xc)
        mat[self.Bidx, :] = self.B(self.phi[self.Bidx])(*self.phi.xc)

        return mat

    def gen_rhs(self, f, g, *funcs):
        vec = np.zeros(self.phi.n)
        vec[self.Lidx] = f(*[c[self.Lidx] for c in self.phi.xc])
        vec[self.Bidx] = g(*[c[self.Bidx] for c in self.phi.xc])

        for h in funcs:
            vec[self.Lidx] -= self.L(h)(*[c[self.Lidx] for c in self.phi.xc])
            vec[self.Bidx] -= self.B(h)(*[c[self.Bidx] for c in self.phi.xc])

        return vec
