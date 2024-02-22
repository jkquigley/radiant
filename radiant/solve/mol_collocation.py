from .collocation import CollocationSolver
import numpy as np


class MOLCollocation(CollocationSolver):
    def __init__(self, d, k, delta, xc, operators, idx_funcs, dt, n, **kwargs):
        super().__init__(d, k, delta, xc, operators, idx_funcs, **kwargs)
        self.dt = dt
        self.n = n

    def gen_mat(self):
        return super().gen_mat() @ np.linalg.inv(self.phi(*self.phi.xc))

    def solve(self, *funcs):
        funcs = list(funcs)
        f = funcs.pop(0)

        if self.mat is None:
            self.mat = self.gen_mat()

        t = 0.
        un = [f(*self.phi.xc)]

        mat = np.eye(*np.shape(self.mat)) - self.dt * self.mat

        for i in range(1, self.n):
            t += self.dt
            rhs = np.copy(un[-1])

            for g, idx in zip(funcs, self.idxs[1:]):
                xc = [c[idx] for c in self.phi.xc]
                rhs[idx] = g(t, *xc)

            un.append(np.linalg.solve(mat, rhs))

        return un
