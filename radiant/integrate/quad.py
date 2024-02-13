from .base import BaseIntegrator
from scipy import integrate


class QuadIntegrator(BaseIntegrator):
    def __init__(self, ranges, epsabs=1.49e-8, epsrel=1.49e-8):
        super().__init__(ranges)
        self.opts = {'epsabs': epsabs, 'epsrel': epsrel}

    def __call__(self, func):
        return integrate.nquad(func, self.ranges, opts=self.opts)[0]
