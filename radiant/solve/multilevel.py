from ..function import MultilevelCompositeFunction


class MultilevelSolver:
    def __init__(self, solver, phi, delta, xcs, *solver_args, outer=1):
        self.solvers = [
            solver(*solver_args, phi, d, *c) for c, d in zip(xcs, delta)
        ]
        self.outer = outer

    def solve(self, *funcs):
        guess = MultilevelCompositeFunction()

        for _ in range(self.outer):
            for solver in self.solvers:
                guess.append(solver.solve(*funcs, guess=guess))

        return guess

    def cond(self):
        return [s.cond() for s in self.solvers]
