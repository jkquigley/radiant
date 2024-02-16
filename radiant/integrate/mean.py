from .base import BaseIntegrator
# import numpy as np
import cupy as cp
from ..util import gridn


class MeanIntegrator(BaseIntegrator):
    def __init__(self, ranges, accuracy):
        super().__init__(ranges)
        self.accuracy = accuracy

        self.x = gridn(self.ranges, self.accuracy)

    def __call__(self, func):
        return self.measure * (cp.mean(cp.array(func(*self.x)))).get()
