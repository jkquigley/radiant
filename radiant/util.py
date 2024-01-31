import numpy as np


def error(u, approx, integrator):
    numerator = integrator(lambda x: (u(x) - approx(x)) ** 2)
    denominator = integrator(lambda x: u(x) ** 2)

    return np.sqrt(numerator / denominator)
