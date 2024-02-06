from .base import BaseIntegrator
from numpy.polynomial import legendre


def _rescale(x, a, b):
    return ((b - a) * x + (b + a)) / 2


class LeggaussIntegrator(BaseIntegrator):
    def __init__(self, a, b, accuracy):
        super().__init__(a, b, accuracy)

    def __call__(self, func):
        x, w = legendre.leggauss(self.accuracy)
        x_scaled = _rescale(x, self.a, self.b)

        return (self.b - self.a) / 2 * sum(w * func(x_scaled))
