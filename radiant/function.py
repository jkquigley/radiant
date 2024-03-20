from math import comb
from itertools import product
import numpy as np
from numpy.polynomial import polynomial
from .util import flatten


def epsdiv(a, b, eps=1e-10):
    return a / (b + eps)


class Wendland:
    def __init__(self, d, k, delta, xc):
        if d <= 0:
            raise ValueError(
                f"Dimension 'd' must be a positive integer but {d} was given."
            )

        if d != len(xc):
            raise ValueError(
                f"Expected {d} elements in argument 'xc' but got {len(xc)}."
            )

        l = d // 2 + k + 1

        prefix = [(-1) ** i * comb(l + k, i) for i in range(l + k + 1)]

        if k == 0:
            coefs = [
                1,
            ]

        elif k == 1:
            coefs = [
                1,
                l + 1,
            ]

        elif k == 2:
            coefs = [
                3,
                3 * l + 6,
                l ** 2 + 4 * l + 3,
            ]

        elif k == 3:
            coefs = [
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
        self.k = k
        if np.shape(delta) == ():
            self.delta = delta * np.ones(self.d)
        else:
            self.delta = delta
        self.xc = tuple(map(flatten, xc))
        self.n = np.size(self.xc[0])

        self.coefs = polynomial.polymul(prefix, coefs)

        self.allowed_first_derivatives = set(range(self.d))
        self.allowed_second_derivatives = set(
            product(range(self.d), range(self.d))
        )
        self.allowed_derivatives = (self.allowed_first_derivatives.union(
            self.allowed_second_derivatives
        ))
        self.allowed_derivatives.add(None)

        self.nargs = d

    def __getitem__(self, item):
        xc = [c[item] for c in self.xc]
        return self.__class__(self.d, self.k, self.delta, xc)

    def __call__(self, *x, w=None, m=None):
        if len(x) != self.nargs:
            raise ValueError(
                f"Function requires {self.nargs} coordinate arguments but "
                f"{len(x)} were given."
            )

        shape = (*np.shape(self.xc[0]), *np.shape(x[0]))
        x = tuple(map(flatten, x))

        diffs = [  # .T to put xc on axis 0.
            np.subtract.outer(x[i], self.xc[i]).T
            for i in range(self.d)
        ]
        r = np.sqrt(np.sum(
            [(diff / delta) ** 2 for diff, delta in zip(diffs, self.delta)],
            axis=0,
        ))

        if m is None:
            unsupported = polynomial.polyval(r, self.coefs)
        elif m in self.allowed_first_derivatives:
            deriv_coefs = polynomial.polyder(self.coefs)
            unsupported = epsdiv(
                diffs[m], (self.delta[m] ** 2) * r
            ) * polynomial.polyval(r, deriv_coefs)
        elif m in self.allowed_second_derivatives:
            xi, xj = m
            first_deriv_coefs = polynomial.polyder(self.coefs)
            second_deriv_coefs = polynomial.polyder(first_deriv_coefs)

            term1 = diffs[xi] * diffs[xj] * epsdiv(
                polynomial.polyval(r, second_deriv_coefs),
                (self.delta[xi] * self.delta[xj] * r) ** 2
            )

            term2 = - diffs[xi] * diffs[xj] * epsdiv(
                polynomial.polyval(r, first_deriv_coefs),
                (self.delta[xi] * self.delta[xj]) ** 2 * r ** 3
            )

            unsupported = term1 + term2

            if xi == xj:
                unsupported += epsdiv(
                    polynomial.polyval(r, first_deriv_coefs),
                    self.delta[xi] ** 2 * r
                )

        else:
            raise ValueError(f"Unsupported derivative m = {m}.")

        if w is None:
            return np.reshape(np.where(1 - r >= 0, unsupported, 0), shape)
        else:
            return np.einsum(
                'i,i...->...',
                w,
                np.reshape(np.where(1 - r >= 0, unsupported, 0), shape)
            )

    def grad(self, *x, w=None):
        return np.array([
            self.__call__(*x, w=w, m=i) for i in range(self.d)
        ])

    def hessian(self, *x, w=None):
        mat = np.zeros((self.d, self.d, *np.shape(x[0])))

        for i in range(self.d):
            mat[i, i] = self.__call__(*x, w=w, m=(i, i))

            for j in range(i):
                mat[i, j] = self.__call__(*x, w=w, m=(i, j))
                mat[j, i] = mat[i, j]

        return mat

    def laplacian(self, *x, w=None):
        return np.sum([
            self.__call__(*x, w=w, m=(i, i))
            for i in range(self.d)
        ], axis=0)


class CompositeFunction(list):
    def __init__(self, *args):
        super().__init__(*args)

    def __getitem__(self, item):
        return self.__class__(super().__getitem__(item))

    def __call__(self, *x, m=None):
        return np.sum([f(*x, w=w, m=m) for w, f in self], axis=0)

    def grad(self, *x):
        return np.sum([f.grad(*x, w=w) for w, f in self], axis=0)

    def hessian(self, *x):
        return np.sum([f.hessian(*x, w=w) for w, f in self], axis=0)

    def laplacian(self, *x):
        return np.sum([f.laplacian(*x, w=w) for w, f in self], axis=0)


class TemporalFunction(list):
    def __init__(self, phi, *args):
        super().__init__(*args)
        self.phi = phi

    def __call__(self, t, *x, m=None):
        return self.phi(*x, w=self[t], m=m)
