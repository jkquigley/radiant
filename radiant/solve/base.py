from ..function import Wendland
from ..function import CompositeFunction
import cupy as cp
import numpy as np


class BaseSolver:
    def __init__(self, d, k, delta, xc):
        self.phi = Wendland(d, k, delta, xc)
        self.mat = None

    def gen_mat(self):
        pass

    def gen_rhs(self, *funcs):
        pass

    def solve(self, *funcs):
        if self.mat is None:
            self.mat = self.gen_mat()

        b = self.gen_rhs(*funcs)

        return CompositeFunction([(
            cp.linalg.solve(cp.array(self.mat), cp.array(b)).get(),
            self.phi,
        )])

    def cond(self):
        if self.mat is None:
            return -1.
        else:
            return np.linalg.cond(self.mat)
