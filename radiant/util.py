from .integrate import integrate
import numpy as np
from typing import Callable


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


def fill_distance(points):
    n = points.shape[0]
    minfill = - np.inf

    for i in range(n):
        for j in range(i+1, n):
            dist = np.abs(points[i] - points[j])
            if dist < minfill:
                minfill = dist

    return minfill / 2
