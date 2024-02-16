from .base import BaseSolver
import numpy as np


class InterpolationSolver(BaseSolver):
    def __init__(self, d, k, delta, xc):
        super().__init__(d, k, delta, xc)

    def gen_mat(self):
        return self.phi(*self.phi.xc)

    def gen_rhs(self, *funcs):
        funcs = list(funcs)
        f = funcs.pop(0)
        return f(*self.phi.xc) - np.sum([
            g(*self.phi.xc) for g in funcs
        ], axis=0)
