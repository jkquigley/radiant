from .base import BaseIntegrator
import numpy as np
from ..util import grid


class TrapezoidIntegrator(BaseIntegrator):
    def __init__(self, ranges, accuracy):
        super().__init__(ranges)
        self.accuracy = accuracy

        self.x = np.meshgrid(*[
            np.linspace(a, b, self.accuracy * int(b - a))
            for a, b in self.ranges
        ])

    def __call__(self, func):
        xs = np.meshgrid(*[
            np.linspace(a, b, self.accuracy * int(b - a))
            for a, b in self.ranges
        ])

        return self.measure * np.mean(func(*xs))
