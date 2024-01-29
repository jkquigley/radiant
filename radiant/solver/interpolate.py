from .approximant import Approximant
from .approximant import RBFParams


def solve(f, centres, delta, phi, *args, combine=True, **kwargs):
    weights = f(centres)
    params = RBFParams(phi, centres, delta, weights)

    if combine:
        return Approximant(params),
    else:
        return params,
