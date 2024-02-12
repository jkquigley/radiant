from .base import BaseSolver


class InterpolationSolver(BaseSolver):
    def __init__(self, phi, delta, *xc):
        super().__init__(phi, delta, *xc)

    def gen_mat(self):
        self.mat = self.phi(self.delta, *self.xc, *self.xc)

    def gen_rhs(self, func, guess):
        if guess is None:
            self.b = func(*self.xc)
        else:
            self.b = func(*self.xc) - guess(*self.xc)
