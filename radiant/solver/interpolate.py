from .approximant import Approximant
from .approximant import RBFParams


def solve(f, centres, delta, phi, *args, combine=True, approx=None, **kwargs):
    if approx is None:
        approx = Approximant()

    weights = f(centres) - approx(centres)
    params = RBFParams(phi, centres, delta, weights)

    if combine:
        return Approximant(params), []
    else:
        return params, []
