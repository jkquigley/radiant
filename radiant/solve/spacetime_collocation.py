from .base import BaseSolver
import numpy as np


class SpaceTimeCollocation(BaseSolver):
    def __init__(self, phi, L, Lidx_func, B, Bidx_func):
        super().__init__(phi)

        self.L = L
        self.Lidx = Lidx_func(phi.xc)
        self.B = B
        self.Bidx = Bidx_func(phi.xc)

    def gen_mat(self):
        mat = np.zeros((self.phi.n, self.phi.n))

        mat[self.Lidx, :] = self.phi[self.Lidx](*self.phi.xc, m=0) + self.L(self.phi[self.Lidx])(*self.phi.xc)
        mat[self.Bidx, :] = self.B(self.phi[self.Bidx])(*self.phi.xc)

        return mat

    def gen_rhs(self, f, g, u0, *funcs):
        vec = np.zeros(self.phi.n)
        vec[self.Lidx] = f(*[c[self.Lidx] for c in self.phi.xc])
        vec[self.Bidx] = g(*[c[self.Bidx] for c in self.phi.xc])
        vec[self.phi.xc[0] == 0] = u0(
            *[c[self.phi.xc[0] == 0] for c in self.phi.xc[1:]]
        )

        for func in funcs:
            Lxc = [c[self.Lidx] for c in self.phi.xc]
            Bxc = [c[self.Bidx] for c in self.phi.xc]
            vec[self.Lidx] -= func(*Lxc, m=0) + self.L(func)(*Lxc)
            vec[self.Bidx] -= self.B(func)(*Bxc)

        return vec
