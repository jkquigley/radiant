import numpy as np


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
            np.arange(a, b + inc * int(b - a), inc * int(b - a))
            for a, b in ranges
        ])))
    else:
        return np.meshgrid(*[
            np.arange(a, b + inc * int(b - a), inc * int(b - a))
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
