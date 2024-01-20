from .integrate import integrate
from numpy.typing import NDArray
import numpy as np
from typing import Callable


def combine(
        phi: Callable, centres: NDArray, alphas: NDArray
) -> Callable:
    def func(x: NDArray):
        return np.sum(
            [a * phi(x, c) for a, c in zip(alphas, centres)],
            axis=0,
        )

    return func


def error(u: Callable, u_approx: Callable, a: float, b: float):
    numerator = integrate.leggauss(
        lambda x: (u(x) - u_approx(x)) ** 2,
        a, b, 2500
    )

    denominator = integrate.leggauss(
        lambda x: u(x) ** 2,
        a, b, 2500
    )

    return np.sqrt(numerator / denominator)
