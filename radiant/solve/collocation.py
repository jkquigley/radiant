from .base import BaseSolver
import numpy as np


class CollocationSolver(BaseSolver):
    def __init__(self, operators, filters, phi, delta, *xc):
        if len(operators) != len(filters):
            raise ValueError(
                f"Expected {len(operators)} filters but {len(filters)} were "
                f"provided."
            )

        super().__init__(phi, delta, *xc)

        self.operators = operators
        self.filters = filters

    def gen_mat(self):
        mats = []
        for op, fil in zip(self.operators, self.filters):
            if fil is None:
                filtered_xc = self.xc
            else:
                filtered_xc = tuple(map(
                    lambda arr: fil(arr, self.xc), self.xc
                ))

            mats.append(op(self.phi, self.delta, *self.xc, *filtered_xc))

        self.mat = np.vstack(mats)

    def gen_rhs(self, *funcs, guess=None):
        if len(funcs) != len(self.operators):
            raise ValueError(
                f"Expected {len(self.operators)} functions but {len(funcs)} "
                f"were provided."
            )

        vecs = []
        for f, fil in zip(funcs, self.filters):
            if fil is None:
                filtered_xc = self.xc
            else:
                filtered_xc = tuple(map(
                    lambda arr: fil(arr, self.xc), self.xc
                ))

            if guess is None:
                vecs.append(f(*filtered_xc))
            else:
                vecs.append(f(*filtered_xc) - guess(*filtered_xc))

        self.b = np.hstack(vecs)
