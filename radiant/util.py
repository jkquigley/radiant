import numpy as np
from collections.abc import Iterable


def dot(a, b):
    return np.sum(a * b, axis=0)


def error(u, approx, integrator):
    numerator = integrator(lambda *x: (u(*x) - approx(*x)) ** 2)
    denominator = integrator(lambda *x: u(*x) ** 2)

    return np.sqrt(numerator / denominator)


def flatten(x):
    if isinstance(x, np.ndarray):
        return x.flatten()
    else:
        return x


def gridinc(ranges, incs, flat=False, unitary=False):
    if not isinstance(incs, Iterable):
        incs = (incs,) * len(ranges)

    if unitary:
        incs = (inc * (b - a) for inc, (a, b) in zip(incs, ranges))

    grid = np.meshgrid(*[
            np.arange(a, b + inc / 2, inc)
            for inc, (a, b) in zip(incs, ranges)
        ])

    if flat:
        return list(map(flatten, grid))
    else:
        return grid


def gridn(ranges, ns, flat=False, endpoint=True, unitary=False):
    if not isinstance(ns, Iterable):
        ns = (ns,) * len(ranges)

    if unitary:
        ns = (int(n * (b - a)) for n, (a, b) in zip(ns, ranges))

    grid = np.meshgrid(*[
            np.linspace(a, b, n, endpoint=endpoint)
            for n, (a, b) in zip(ns, ranges)
        ])

    if flat:
        return list(map(flatten, grid))
    else:
        return grid