from ..util import Approximant
from ..util import RBFParams


def solve(f, centres, delta, phi, *args, combine=True, **kwargs):
    weights = f(centres)
    params = RBFParams(phi, centres, weights, delta)

    if combine:
        return Approximant(params),
    else:
        return params,
