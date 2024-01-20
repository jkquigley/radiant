import numpy as np
from scipy import integrate
from numpy.polynomial import legendre
from typing import Callable


# TODO: Fix bugs causing large errors with various ns.
def leggauss(
        f: Callable, a: float, b: float, n: int
) -> float:
    """
    Approximate the integral of :math:`f` from :math:`a` to :math:`b` using
    the Gauss-Legendre quadrature method.

    Parameters
    ----------
    f: callable
    The function to integrate.

    a: float
    The left boundary point.

    b: float
    The right boundary point.

    n: int
    The Legendre polynomial degree.

    Returns
    -------
    y: float
    The approximated value of the integral of :math:`f` from :math:`a` to
    :math:`b`.
    """
    # Get the Gauss-Legendre quadrature points and weights.
    x, w = legendre.leggauss(n)

    # Compute the integral approximation with an interval rescaling.
    y = (b - a) / 2 * sum(w * f(((b - a) * x + (b + a)) / 2))

    return y


def trapezoid(f: Callable, a: float, b: float, n: int):
    """
    Approximate the integral of :math:`f` from :math:`a` to :math:`b` using
    the trapezoid method.

    Parameters
    ----------
    f: callable
    The function to integrate.

    a: float
    The left boundary point.

    b: float
    The right boundary point.

    n: int
    The number of integration points in an interval of length 1.

    Returns
    -------
    y: float
    The approximated value of the integral of :math:`f` from :math:`a` to
    :math:`b`.
    """
    xs = np.linspace(a, b, n * int(b - a))
    ys = f(xs)

    return integrate.trapezoid(ys, xs)

def quad(f: Callable, a: float, b: float):
    """
    Approximate the integral of :math:`f` from :math:`a` to :math:`b` using
    the trapezoid method.

    Parameters
    ----------
    f: callable
    The function to integrate.

    a: float
    The left boundary point.

    b: float
    The right boundary point.

    n: int
    The number of integration points in an interval of length 1.

    Returns
    -------
    y: float
    The approximated value of the integral of :math:`f` from :math:`a` to
    :math:`b`.
    """
    return integrate.quad(f, a, b)[0]
