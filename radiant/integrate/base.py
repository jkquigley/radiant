import numpy as np


class BaseIntegrator:
    def __init__(self, ranges):
        self.ranges = ranges
        self.measure = np.prod([b - a for a, b in ranges])

    def __call__(self, func):
        pass
