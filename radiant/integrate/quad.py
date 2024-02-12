from .base import BaseIntegrator
from scipy import integrate


class QuadIntegrator(BaseIntegrator):
    def __init__(self, ranges):
        super().__init__(ranges)

    def __call__(self, func):
        return integrate.nquad(func, self.ranges)[0]
