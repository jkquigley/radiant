from numpy.polynomial.legendre import leggauss
from typing import Callable


# TODO: Fix bugs causing large errors with various ns.
def gauss_legendre(
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
    x, w = leggauss(n)

    # Compute the integral approximation with an interval rescaling.
    y = (b - a) / 2 * sum(w * f(((b - a) * x + (b + a)) / 2))

    return y
