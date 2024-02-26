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


def gridinc(ranges, incs, flat=False):
    if not isinstance(incs, Iterable):
        incs = (incs,) * len(ranges)

    if flat:
        return list(map(flatten, np.meshgrid(*[
            np.arange(a, b + (inc * (b - a)) / 2, inc * (b - a))
            for inc, (a, b) in zip(incs, ranges)
        ])))
    else:
        return np.meshgrid(*[
            np.arange(a, b + (inc * (b - a)) / 2, inc * (b - a))
            for inc, (a, b) in zip(incs, ranges)
        ])


def gridn(ranges, ns, flat=False, endpoint=True):
    if not isinstance(ns, Iterable):
        ns = (ns,) * len(ranges)

    if flat:
        return list(map(flatten, np.meshgrid(*[
            np.linspace(a, b, int(n * (b - a)), endpoint=endpoint)
            for n, (a, b) in zip(ns, ranges)
        ])))
    else:
        return np.meshgrid(*[
            np.linspace(a, b, int(n * (b - a)), endpoint=endpoint)
            for n, (a, b) in zip(ns, ranges)
        ])
