from ..function import CompositeFunction
import cupy as cp
import numpy as np
import scipy as sp


class BaseSolver:
    def __init__(self, phi):
        self.phi = phi
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

    def bandwidth(self):
        if self.mat is None:
            return -1.
        else:
            return sp.linalg.bandwidth(self.mat)
