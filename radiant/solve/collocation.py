from .base import BaseSolver
import numpy as np


class CollocationSolver(BaseSolver):
    def __init__(self, d, k, delta, xc, operators, idx_funcs):
        super().__init__(d, k, delta, xc)

        if len(operators) != len(idx_funcs):
            raise ValueError(
                f"Expected {len(operators)} index filters but got "
                f"{len(idx_funcs)} instead."
            )

        self.operators = operators
        self.idxs = [idx_func(xc) for idx_func in idx_funcs]

    def gen_mat(self):
        mats = []
        for op, idx in zip(self.operators, self.idxs):
            mats.append(op(self.phi[idx], *self.phi.xc))

        return np.vstack(mats)

    def gen_rhs(self, *funcs):
        if len(funcs) < len(self.operators):
            raise ValueError(
                f"Expected {len(self.operators)} or more functions but got "
                f"{len(funcs)} instead."
            )

        fs = funcs[:len(self.operators)]
        gs = funcs[len(self.operators):]

        vecs = []
        for f, idx in zip(fs, self.idxs):
            xc = [c[idx] for c in self.phi.xc]
            vecs.append(f(*xc) - np.sum([
                g(*xc)
                for g in gs[len(self.operators):]
            ], axis=0))

        return np.hstack(vecs)
