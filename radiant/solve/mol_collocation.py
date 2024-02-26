from .collocation import CollocationSolver
import numpy as np


class MOLCollocation(CollocationSolver):
    def __init__(self, phi, operators, idx_funcs, tf, tn):
        super().__init__(phi, operators, idx_funcs)
        self.dt = 1 / tn
        self.tn = int(tn * tf)

    def gen_mat(self):
        return super().gen_mat() @ np.linalg.inv(self.phi(*self.phi.xc))

    def solve(self, *funcs):
        funcs = list(funcs)
        f = funcs.pop(0)

        if self.mat is None:
            self.mat = self.gen_mat()

        ts = [0.]
        un = [f(*self.phi.xc)]

        for i in range(1, self.tn):
            ts.append(ts[-1] + self.dt)

            if i == 1:
                rhs = np.copy(un[-1])
                mat = np.eye(*np.shape(self.mat)) - self.dt * self.mat
            else:
                rhs = 4 * un[-1] / 3 - un[-2] / 3
                mat = np.eye(*np.shape(self.mat)) - 2 * self.dt * self.mat / 3

            for g, idx in zip(funcs, self.idxs[1:]):
                if idx is None and g == "periodic":
                    rhs[0] = rhs[-1]
                else:
                    xc = [c[idx] for c in self.phi.xc]
                    rhs[idx] = g(ts[-1], *xc)

            un.append(np.linalg.solve(mat, rhs))

        return un, ts
