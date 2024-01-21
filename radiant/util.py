from .integrate import integrate
from .rbf import phi_factory
from numpy.typing import NDArray
import numpy as np
from typing import Callable


def combine(
        phi: Callable, centres: NDArray, weights: NDArray
) -> Callable:
    if centres.size != weights.size:
        raise ValueError("Every centre must have a corresponding weight.")

    def func(x: NDArray):
        return np.sum(np.multiply(weights[:, None], phi(x, centres)), axis=0)

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


def lhs_integrand_factory(phi, xi, xj):
    def func(x):
        return phi(x, xi, m=1) * phi(x, xj, m=1) + phi(x, xi) * phi(x, xj)

    return func


def rhs_integrand_factory(phi, f, xi):
    def func(x):
        return f(x) * phi(x, xi)

    return func


def solve(a, b, f, n, d, k, delta, precond=False):
    centres = np.linspace(a, b, n)
    phi = phi_factory(d, k, delta)

    A = np.zeros((centres.size, centres.size))
    fs = np.zeros_like(centres)
    for i, xi in enumerate(centres):
        fs[i] = integrate.trapezoid(
            rhs_integrand_factory(phi, f, xi),
            a, b, 2500
        )

        A[i, i] = integrate.trapezoid(
            lhs_integrand_factory(phi, xi, xi),
            a, b, 2500
        )

        for j, xj in enumerate(centres[:i]):
            A[i, j] = integrate.trapezoid(
                lhs_integrand_factory(phi, xi, xj),
                a, b, 2500
            )

            A[j, i] = A[i, j]

    if precond:
        # Jacobi Preconditioning
        pinv = np.diag(1 / np.sqrt(np.diag(A)))

        alphas = pinv @ np.linalg.solve(pinv @ A @ pinv, pinv @ fs)
    else:
        alphas = np.linalg.solve(A, fs)

    return combine(phi, centres, alphas), A, fs, centres, alphas
