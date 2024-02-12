from math import comb
import numpy as np
from numpy.polynomial import polynomial
from .util import flatten


class Wendland:
    def __init__(self, d, k):
        if d <= 0:
            raise ValueError(
                f"Dimension 'd' must be a positive integer but {d} was given."
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
                f"Smoothness 'k' must be one of 0, 1, 2, or 3 but {k} was "
                f"given."
            )

        self.d = d
        self.coefs = polynomial.polymul(prefix, coef)

    def __call__(self, delta, *args, w=1., m=None):
        if len(args) != self.d * 2:
            raise ValueError(
                f"Function requires {2 * self.d} coordinate "
                f"arguments but {len(args)} were given."
            )

        shape = (*np.shape(args[self.d]), *np.shape(args[0]))
        args = list(map(flatten, args))

        diffs = [
            np.subtract.outer(args[i], args[i + self.d]).T
            for i in range(self.d)
        ]
        normed_diff = np.sqrt(np.sum([diff ** 2 for diff in diffs], axis=0))
        r = normed_diff / delta

        if m is None:
            unsupported = polynomial.polyval(r, self.coefs)
        elif m in range(self.d):
            deriv_coefs = polynomial.polyder(self.coefs)
            deriv_scale = diffs[m] / (delta * normed_diff + 1e-6)
            unsupported = deriv_scale * polynomial.polyval(r, deriv_coefs)
        else:
            raise ValueError(f"Unsupported derivative m = {m}.")

        return np.reshape(
            np.where(1 - r >= 0, unsupported, 0),
            shape,
        )

    def grad(self, delta, *args, w=1.):
        return np.array([
            self.__call__(delta, *args, w=w, m=i) for i in range(self.d)
        ])
