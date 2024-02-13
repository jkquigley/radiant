from math import comb
from itertools import product
import numpy as np
from numpy.polynomial import polynomial
from .util import flatten


def epsdiv(a, b, eps=1e-6):
    return a / (b + eps)


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

        self.allowed_first_derivatives = set(range(self.d))
        self.allowed_second_derivatives = set(
            product(range(self.d), range(self.d))
        )
        self.allowed_derivatives = (self.allowed_first_derivatives.union(
            self.allowed_second_derivatives
        ))
        self.allowed_derivatives.add(None)

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
        elif m in self.allowed_first_derivatives:
            deriv_coefs = polynomial.polyder(self.coefs)
            deriv_scale = epsdiv(diffs[m], delta * normed_diff)
            unsupported = deriv_scale * polynomial.polyval(r, deriv_coefs)
        elif m in self.allowed_second_derivatives:
            xi, xj = m
            first_deriv_coefs = polynomial.polyder(self.coefs)
            second_deriv_coefs = polynomial.polyder(first_deriv_coefs)

            term1 = diffs[xi] * diffs[xj] * epsdiv(
                polynomial.polyval(r, second_deriv_coefs),
                (delta * normed_diff) ** 2
            )

            term2 = - diffs[xi] * diffs[xj] * epsdiv(
                polynomial.polyval(r, first_deriv_coefs),
                delta * normed_diff ** 3
            )

            unsupported = term1 + term2

            if xi == xj:
                unsupported += epsdiv(
                    polynomial.polyval(r, first_deriv_coefs),
                    delta * normed_diff,
                )

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

    def div(self, delta, *args, w=1.):
        return np.sum(self.grad(delta, *args, w=w), axis=0)

    def hessian(self, delta, *args, w=1.):
        mat = np.zeros((self.d, self.d, *np.shape(args[0])))

        for i in range(self.d):
            mat[i, i] = self.__call__(delta, *args, w=w, m=(i, i))

            for j in range(i):
                mat[i, j] = self.__call__(delta, *args, w=w, m=(i, j))
                mat[j, i] = mat[i, j]

        return mat

    def laplacian(self, delta, *args, w=1.):
        return np.sum([
            self.__call__(delta, *args, w=w, m=(i, i))
            for i in range(self.d)
        ], axis=0)
