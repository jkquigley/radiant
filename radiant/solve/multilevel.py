from ..function import CompositeFunction


class MultilevelSolver:
    def __init__(self, phis, solver, *args, outer=1):
        self.solvers = [solver(phi, *args) for phi in phis]
        self.outer = outer

    def solve(self, *funcs):
        solution = CompositeFunction()

        for _ in range(self.outer):
            for solver in self.solvers:
                solution += solver.solve(*funcs, solution)

        return solution

    def cond(self):
        return [s.cond() for s in self.solvers]

    def bandwidth(self):
        return [s.bandwidth() for s in self.solvers]
