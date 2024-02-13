from .base import BaseIntegrator
import numpy as np
from ..util import gridn


class TrapezoidIntegrator(BaseIntegrator):
    def __init__(self, ranges, accuracy):
        super().__init__(ranges)
        self.accuracy = accuracy

        self.x = gridn(self.ranges, self.accuracy)

    def __call__(self, func):
        val = func(*self.x)
        for p in self.x:
            val = np.trapz(val, p, axis=0)
        return val
