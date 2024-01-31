from numpy.polynomial import legendre
import numpy as np


class BaseIntegrator:
    def __init__(self, a, b, accuracy):
        self.a = a
        self.b = b
        self.accuracy = accuracy

    def __call__(self, func, *fargs):
        raise NotImplementedError


def _rescale(x, a, b):
    return ((b - a) * x + (b + a)) / 2


class LeggaussIntegrator(BaseIntegrator):
    def __init__(self, a, b, accuracy):
        super().__init__(a, b, accuracy)

    def __call__(self, func, *fargs):
        x, w = legendre.leggauss(self.accuracy)
        x_scaled = _rescale(x, self.a, self.b)

        return (self.b - self.a) / 2 * sum(w * func(x_scaled))


class TrapezoidIntegrator(BaseIntegrator):
    def __init__(self, a, b, accuracy):
        super().__init__(a, b, accuracy)

    def __call__(self, func, *fargs):
        xs = np.linspace(self.a, self.b, self.accuracy * int(self.b - self.a))
        ys = func(xs, *fargs)

        return np.trapz(ys, xs)
