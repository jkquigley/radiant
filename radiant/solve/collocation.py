from .base import BaseSolver
import numpy as np


def map_indexing(idx, xc):
    return xc if idx is None else map(lambda x: x[idx], xc)


class CollocationSolver(BaseSolver):
    def __init__(
            self, operators, xc_idxs, phi, delta, *xc
    ):
        super().__init__(phi, delta, *xc)
        self.operators = operators
        self.xc_idxs = xc_idxs

    def gen_mat(self):
        mats = []
        for op, idx in zip(self.operators, self.xc_idxs):
            indexed_xc = map_indexing(idx, self.xc)
            mats.append(op(self.phi, self.delta, *self.xc, *indexed_xc))

        self.mat = np.vstack(mats)

    def gen_rhs(self, *funcs, guess=None):
        if len(funcs) != len(self.operators):
            raise ValueError(
                f"Expected {len(self.operators)} functions but {len(funcs)} "
                f"was provided."
            )

        vecs = []
        for f, idx in zip(funcs, self.xc_idxs):
            indexed_xc = map_indexing(idx, self.xc)
            vecs.append(f(*indexed_xc))

        self.b = np.hstack(vecs)
