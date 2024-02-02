import cupy as cp


class MultilevelFunction(list):
    def __init__(self):
        super().__init__()

    def __call__(self, x, *args, end=None, **kwargs):
        if end is None:
            end = len(self)

        val = cp.zeros_like(x)
        for s in self[:end]:
            val += s(x, *args, **kwargs)

        return val


class MultilevelSolver:
    def __init__(self, phi, centres, delta, solver, *solver_args, outer=1):
        self.solvers = [
            solver(phi, c, d, *solver_args) for c, d in zip(centres, delta)
        ]
        self.outer = outer

    def solve(self, func):
        guess = MultilevelFunction()

        for _ in range(self.outer):
            for solver in self.solvers:
                guess.append(solver.solve(func, guess))

        return guess

    def cond(self):
        return [s.cond() for s in self.solvers]
