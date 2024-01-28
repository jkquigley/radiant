from .integrate import integrate
from numpy.typing import NDArray
import numpy as np
from typing import Callable
from typing import List
from dataclasses import dataclass


@dataclass
class RBFParams:
    phi: Callable
    cs: NDArray
    ws: NDArray
    d: float


class Approximant:
    def __init__(self, params: None | RBFParams | List[RBFParams] = None):
        if params is None:
            params = []
        elif isinstance(params, RBFParams):
            params = [params]

        self.params = params

    def __call__(self, x, end=None):
        if end is None:
            end = len(self.params)
        val = np.zeros_like(x)
        for p in self.params[:end]:
            val += np.sum(
                np.multiply(p.ws[:, None], p.phi(x, p.cs, p.d)),
                axis=0,
            )

        return val

    def __add__(self, other):
        if isinstance(other, RBFParams):
            return Approximant(self.params.append(other))
        elif isinstance(other, Approximant):
            return Approximant(self.params + other.params)
        else:
            raise NotImplementedError

    def __iadd__(self, other):
        if isinstance(other, RBFParams):
            self.params.append(other)
        elif isinstance(other, Approximant):
            self.params += other.params
        else:
            raise NotImplementedError

        return self


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
