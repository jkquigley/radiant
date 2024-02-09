import cupy as cp
import numpy as np


class BaseSolver:
    def __init__(self, phi, centres, delta):
        self.phi = phi
        self.centres = centres
        self.delta = delta
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

        weights = cp.linalg.solve(cp.array(self.mat), cp.array(self.b)).get()

        def approximant(x, *, m=None):
            return np.sum(
                self.phi(x, self.centres, self.delta, weights, m=m),
                axis=0,
            )

        return approximant

    def cond(self):
        if self.mat is None:
            return -1.
        else:
            return np.linalg.cond(self.mat)
