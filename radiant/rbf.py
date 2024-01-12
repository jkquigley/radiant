from math import comb
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from .typing import Function


def phi_factory(d: int, k: int) -> Function:
    if d <= 0:
        raise ValueError(
            "Dimension 'd' must be greater or equal to 1."
        )

    l = int(np.floor(d / 2)) + k + 1

    prefix = np.polynomial.Polynomial(
        [(-1) ** i * comb(l + k, i) for i in range(l + k + 1)],
        symbol='r',
    )

    if k == 0:
        suffix = np.polynomial.Polynomial([
            1,
        ], symbol='r')

    elif k == 1:
        suffix = np.polynomial.Polynomial([
            1,
            l + 1,
        ], symbol='r')

    elif k == 2:
        suffix = np.polynomial.Polynomial([
            3,
            3 * l + 6,
            l ** 2 + 4 * l + 3,
        ], symbol='r')

    elif k == 3:
        suffix = np.polynomial.Polynomial([
            15,
            15 * l + 45,
            6 * l ** 2 + 36 * l + 45,
            l ** 3 + 9 ** 2 + 23 * l + 15,
        ], symbol='r')

    else:
        raise ValueError(
            "Smoothness 'k' must be one of 0, 1, 2, or 3."
        )

    poly = prefix * suffix

    def func(r: NDArray, m: int = 0) -> NDArray:
        return np.where(prefix(r) >= 0, 1, 0) * poly.deriv(m)(r)

    return func
