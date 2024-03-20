from .collocation import CollocationSolver
from ..function import TemporalFunction
import cupy as cp
import numpy as np


class MOLCollocation(CollocationSolver):
    def __init__(self, phi, L, Lidx_func, B, Bidx_func, tf, tn):
        super().__init__(phi, L, Lidx_func, B, Bidx_func)
        self.dt = 1 / tn
        self.tn = int(tn * tf)

    def gen_mat(self):
        mat = np.zeros((self.phi.n, self.phi.n))
        mat[self.Lidx, :] = self.phi[self.Lidx](*self.phi.xc) / self.dt

        return mat, super().gen_mat()

    def solve(self, f, g, u0):
        if self.mat is None:
            self.mat = self.gen_mat()

        approx = TemporalFunction(self.phi)
        approx.append(cp.linalg.solve(
            cp.array(self.phi(*self.phi.xc)), cp.array(u0(*self.phi.xc))
        ).get())

        lhs = cp.array(self.mat[0] + self.mat[1])

        for i in range(self.tn):
            rhs = cp.array(
                np.matmul(self.mat[0], approx[-1]) + np.where(
                    self.Lidx,
                    f(self.dt * i, *self.phi.xc),
                    g(self.dt * i, *self.phi.xc)
                )
            )
            approx.append(cp.linalg.solve(
                lhs, rhs
            ).get())

        return approx
