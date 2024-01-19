from .rbf import phi_factory
from numpy.typing import NDArray
import numpy as np
from scipy import integrate
from typing import Callable


def combine(
        phi: Callable, centres: NDArray, alphas: NDArray
) -> Callable:
    def func(x: NDArray):
        return np.sum([a * phi(x, c) for a, c in zip(alphas, centres)], axis=0)

    return func


def generate(a, b, f, n, d, k, delta):
    centres = np.linspace(a, b, n)
    phi = phi_factory(d, k, delta)

    A = np.zeros((centres.size, centres.size))
    fs = np.zeros_like(centres)
    for i, xi in enumerate(centres):
        fs[i] = integrate.quad(
            lambda x:
            f(x) * phi(x, xi),
            a, b
        )[0]

        A[i, i] = integrate.quad(
            lambda x:
            phi(x, xi, m=1) * phi(x, xi, m=1) + phi(x, xi) * phi(x, xi),
            a, b
        )[0]

        for j, xj in enumerate(centres[:i]):
            A[i, j] = integrate.quad(
                lambda x:
                phi(x, xi, m=1) * phi(x, xj, m=1) + phi(x, xi) * phi(x, xj),
                a, b
            )[0]

            A[j, i] = A[i, j]

    return phi, centres, A, fs


def solve(
        phi: Callable, centres: NDArray, a: NDArray, b: NDArray, *,
        precondition: bool = False
) -> [Callable, NDArray]:
    if precondition:
        # Jacobi Preconditioning
        pinv = np.diag(1 / np.sqrt(np.diag(a)))

        alphas = pinv @ np.linalg.solve(pinv @ a @ pinv, pinv @ b)
    else:
        alphas = np.linalg.solve(a, b)

    return combine(phi, centres, alphas), alphas


def error(u: Callable, u_approx: Callable, a: float, b: float):
    numerator = integrate.quad(
        lambda x: (u(x) - u_approx(x)) ** 2,
        a, b
    )[0]

    denominator = integrate.quad(
        lambda x: u(x) ** 2,
        a, b
    )[0]

    return np.sqrt(numerator / denominator)
