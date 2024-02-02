from .base import BaseSolver


class InterpolationSolver(BaseSolver):
    def __init__(self, phi, centres, delta):
        super().__init__(phi, centres, delta)

    def gen_mat(self):
        self.mat = self.phi(self.centres, self.centres, self.delta)

    def gen_rhs(self, func, guess):
        if guess is None:
            self.b = func(self.centres)
        else:
            self.b = func(self.centres) - guess(self.centres)
