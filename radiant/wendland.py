from math import comb
import numpy as np


class Wendland:
    def __init__(self, d, k):
        if d <= 0:
            raise ValueError(
                "Dimension 'd' must be a positive integer."
            )

        l = int(np.floor(d / 2)) + k + 1

        symbol = 'r'
        self.prefix = np.polynomial.Polynomial(
            [(-1) ** i * comb(l + k, i) for i in range(l + k + 1)],
            symbol=symbol,
        )

        if k == 0:
            self.poly = self.prefix * np.polynomial.Polynomial([
                1,
            ], symbol=symbol)

        elif k == 1:
            self.poly = self.prefix * np.polynomial.Polynomial([
                1,
                l + 1,
            ], symbol=symbol)

        elif k == 2:
            self.poly = self.prefix * np.polynomial.Polynomial([
                3,
                3 * l + 6,
                l ** 2 + 4 * l + 3,
            ], symbol=symbol)

        elif k == 3:
            self.poly = self.prefix * np.polynomial.Polynomial([
                15,
                15 * l + 45,
                6 * l ** 2 + 36 * l + 45,
                l ** 3 + 9 ** 2 + 23 * l + 15,
            ], symbol=symbol)

        else:
            raise ValueError(
                "Smoothness 'k' must be one of 0, 1, 2, or 3."
            )

        if d >= 2:
            def norm(x):
                return np.linalg.norm(x, axis=0)

            self.norm = norm
        else:
            self.norm = np.abs

    def __call__(self, x, c, delta, w=None, m=0):
        if w is None:
            w = np.ones_like(c)

        if w.shape != ():
            w = w[:, None]

        diff = - np.subtract.outer(c, x)
        normed_diff = np.abs(diff)
        r = normed_diff / delta
        unsupported = np.multiply(
            w,
            (diff / (delta * normed_diff + 1e-6)) ** m * self.poly.deriv(m)(r),
        )

        return np.where(1 - r >= 0, unsupported, 0)
