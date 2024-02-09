from math import comb
import numpy as np
from numpy.polynomial import polynomial


class Wendland:
    def __init__(self, d, k):
        if d <= 0:
            raise ValueError(
                "Dimension 'd' must be a positive integer."
            )

        l = d // 2 + k + 1

        prefix = [(-1) ** i * comb(l + k, i) for i in range(l + k + 1)]

        if k == 0:
            coef = [
                1,
            ]

        elif k == 1:
            coef = [
                1,
                l + 1,
            ]

        elif k == 2:
            coef = [
                3,
                3 * l + 6,
                l ** 2 + 4 * l + 3,
            ]

        elif k == 3:
            coef = [
                15,
                15 * l + 45,
                6 * l ** 2 + 36 * l + 45,
                l ** 3 + 9 ** 2 + 23 * l + 15,
            ]

        else:
            raise ValueError(
                "Smoothness 'k' must be one of 0, 1, 2, or 3."
            )

        self.coefs = polynomial.polymul(prefix, coef)

    def __call__(self, x, c, delta, w=1., m=None):
        if np.shape(w) != ():
            w = w[:, None]

        diff = - np.subtract.outer(c, x)
        normed_diff = np.abs(diff)
        r = normed_diff / delta

        if m is None:
            unsupported = w * polynomial.polyval(r, self.coefs)
        else:
            deriv_coefs = polynomial.polyder(self.coefs)
            deriv_scale = diff / (delta * normed_diff + 1e-6)
            unsupported = w * deriv_scale * polynomial.polyval(r, deriv_coefs)

        return np.where(1 - r >= 0, unsupported, 0)
