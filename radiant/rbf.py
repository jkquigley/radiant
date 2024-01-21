from math import comb
import numpy as np
from numpy.typing import NDArray
from typing import Callable


def phi_factory(d: int, k: int, delta: float) -> Callable:
    if d <= 0:
        raise ValueError(
            "Dimension 'd' must be a positive integer."
        )

    l = int(np.floor(d / 2)) + k + 1

    symbol = 'r'
    prefix = np.polynomial.Polynomial(
        [(-1) ** i * comb(l + k, i) for i in range(l + k + 1)],
        symbol=symbol,
    )

    if k == 0:
        poly = prefix * np.polynomial.Polynomial([
            1,
        ], symbol=symbol)

    elif k == 1:
        poly = prefix * np.polynomial.Polynomial([
            1,
            l + 1,
        ], symbol=symbol)

    elif k == 2:
        poly = prefix * np.polynomial.Polynomial([
            3,
            3 * l + 6,
            l ** 2 + 4 * l + 3,
        ], symbol=symbol)

    elif k == 3:
        poly = prefix * np.polynomial.Polynomial([
            15,
            15 * l + 45,
            6 * l ** 2 + 36 * l + 45,
            l ** 3 + 9 ** 2 + 23 * l + 15,
        ], symbol=symbol)

    else:
        raise ValueError(
            "Smoothness 'k' must be one of 0, 1, 2, or 3."
        )

    def func(x: NDArray, c: NDArray, m: int = 0) -> NDArray:
        r = np.abs(np.subtract.outer(c / delta, x / delta))
        return np.where(prefix(r) >= 0, poly.deriv(m)(r) / (delta ** m), 0)

    return func
