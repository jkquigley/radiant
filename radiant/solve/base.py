import cupy as cp
import numpy as np
from ..function import CompositeFunction
from ..util import flatten


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

    def gen_rhs(self, *funcs, guess=None):
        pass

    def solve(self, *funcs, guess=None):
        if self.mat is None:
            self.gen_mat()

        self.gen_rhs(*funcs, guess=None)

        w = cp.linalg.solve(cp.array(self.mat), cp.array(self.b)).get()

        return CompositeFunction(self.phi, self.delta, *self.xc, w=w)

    def cond(self):
        if self.mat is None:
            return -1.
        else:
            return np.linalg.cond(self.mat)
