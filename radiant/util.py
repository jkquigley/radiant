import numpy as np


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


def gridinc(ranges, inc, flat=False):
    if flat:
        return list(map(flatten, np.meshgrid(*[
            np.arange(a, b + inc * int(b - a) / 2, inc * int(b - a))
            for a, b in ranges
        ])))
    else:
        return np.meshgrid(*[
            np.arange(a, b + inc * int(b - a) / 2, inc * int(b - a))
            for a, b in ranges
        ])


def gridn(ranges, n, flat=False):
    if flat:
        return list(map(flatten, np.meshgrid(*[
            np.linspace(a, b, n * int(b - a))
            for a, b in ranges
        ])))
    else:
        return np.meshgrid(*[
            np.linspace(a, b, n * int(b - a))
            for a, b in ranges
        ])
