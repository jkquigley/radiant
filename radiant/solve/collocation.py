from .base import BaseSolver
import numpy as np


class CollocationSolver(BaseSolver):
    def __init__(self, phi, operators, idx_funcs):
        super().__init__(phi)

        if len(operators) != len(idx_funcs):
            raise ValueError(
                f"Expected {len(operators)} index filters but got "
                f"{len(idx_funcs)} instead."
            )

        self.operators = operators
        self.idxs = [
            idx_func(phi.xc)
            if idx_func is not None else None
            for idx_func in idx_funcs
        ]

    def gen_mat(self):
        mat = np.zeros((self.phi.n, self.phi.n))
        for op, i in zip(self.operators, self.idxs):
            if i is not None:
                mat[i, :] = op(self.phi[i])(*self.phi.xc)

        return mat

    def gen_rhs(self, *funcs):
        if len(funcs) < len(self.operators):
            raise ValueError(
                f"Expected {len(self.operators)} or more functions but got "
                f"{len(funcs)} instead."
            )

        fs = funcs[:len(self.operators)]
        gs = funcs[len(self.operators):]

        vec = np.zeros(self.phi.n)
        for f, op, i in zip(fs, self.operators, self.idxs):
            xc = [c[i] for c in self.phi.xc]
            vec[i] = f(*xc) - np.sum([op(g)(*xc) for g in gs], axis=0)

        return vec
