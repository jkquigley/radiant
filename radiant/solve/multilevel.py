from ..function import CompositeFunction


class MultilevelSolver:
    def __init__(self, d, k, deltas, xcs, outer, solver, *solver_args):
        self.solvers = [
            solver(d, k, delta, xc, *solver_args)
            for xc, delta in zip(xcs, deltas)
        ]
        self.outer = outer

    def solve(self, *funcs):
        solution = CompositeFunction()

        for _ in range(self.outer):
            for solver in self.solvers:
                solution.append(solver.solve(*funcs, solution))

        return solution

    def cond(self):
        return [s.cond() for s in self.solvers]
