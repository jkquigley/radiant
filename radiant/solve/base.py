import cupy as cp
import numpy as np
from ..function import CompositeFunction
from ..util import flatten


class BaseSolver:
    def __init__(self):
        self.mat = None
        self.b = None

    def gen_mat(self, phi):
        pass

    def gen_rhs(self, phi, *funcs, guess=None):
        pass

    def solve(self, phi, *funcs, guess=None):
        if self.mat is None:
            self.gen_mat(phi)

        self.gen_rhs(*funcs, guess=guess)

        return cp.linalg.solve(cp.array(self.mat), cp.array(self.b)).get()

    def cond(self):
        if self.mat is None:
            return -1.
        else:
            return np.linalg.cond(self.mat)
