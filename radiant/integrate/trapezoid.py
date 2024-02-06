from .base import BaseIntegrator
import numpy as np

class TrapezoidIntegrator(BaseIntegrator):
    def __init__(self, a, b, accuracy):
        super().__init__(a, b, accuracy)

    def __call__(self, func):
        xs = np.linspace(self.a, self.b, self.accuracy * int(self.b - self.a))
        ys = func(xs)

        return np.trapz(ys, xs)
