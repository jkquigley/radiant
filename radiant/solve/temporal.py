# from .base import BaseSolver
from .collocation import CollocationSolver
import numpy as np


# class TemporalSolver(BaseSolver):
#     def __init__(self, k, delta, xc, operator, idx_func, dt, n):
#         super().__init__(1, k, delta, xc)
#         self.operator = operator
#         self.idx = idx_func(xc)
#         self.dt = dt
#         self.n = n
#
#     def gen_mat(self):
#         phii = self.phi[self.idx]
#         bdry_idx = np.logical_not(self.idx)
#         int_xc = [c[self.idx] for c in self.phi.xc]
#         bdry_xc = [c[bdry_idx] for c in self.phi.xc]
#         bdry_n = self.phi.n - phii.n
#
#         Ainv = np.linalg.inv(self.phi(*self.phi.xc))
#         B = np.zeros((phii.n, self.phi.n))
#         B[:, self.idx] = self.operator(phii)(*int_xc)
#         B[:, bdry_idx] = phii(*bdry_xc)
#
#         C = B @ Ainv
#
#         mat = np.zeros((self.phi.n, self.phi.n))
#
#         mat[np.ix_(self.idx, self.idx)] = np.eye(phii.n, phii.n) - self.dt * C[:, self.idx]
#         mat[np.ix_(self.idx, bdry_idx)] = - self.dt * C[:, bdry_idx]
#         mat[np.ix_(bdry_idx, bdry_idx)] = np.eye(bdry_n, bdry_n)
#
#         return mat
#
#     def solve(self, f, g):
#         if self.mat is None:
#             self.mat = self.gen_mat()
#
#         int_xc = [c[self.idx] for c in self.phi.xc]
#         bdry_idx = np.logical_not(self.idx)
#         bdry_xc = [c[bdry_idx] for c in self.phi.xc]
#
#         t = 0
#         u0 = np.zeros(self.phi.n)
#         u0[self.idx] = f(*int_xc)
#         u0[bdry_idx] = g(t, *bdry_xc)
#
#         un = [u0]
#         for i in range(1, self.n):
#             t += self.dt
#             rhs = un[-1]
#             rhs[bdry_idx] = g(t, *bdry_xc)
#             un.append(np.linalg.solve(self.mat, rhs))
#
#         return un


class TemporalSolver(CollocationSolver):
    def __init__(self, d, k, delta, xc, operators, idx_funcs, dt, n):
        super().__init__(d, k, delta, xc, operators, idx_funcs)
        self.dt = dt
        self.n = n

    def gen_mat(self):
        return super().gen_mat() @ np.linalg.inv(self.phi(*self.phi.xc))

    def solve(self, f, g):
        if self.mat is None:
            self.mat = self.gen_mat()

        t = 0.
        u0 = f(*self.phi.xc)
        un = [u0]

        mat = np.eye(*np.shape(self.mat)) - self.dt * self.mat

        for i in range(1, self.n):
            t += self.dt
            un.append(np.linalg.solve(mat, un[-1]))

        return un
