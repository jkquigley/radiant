from numpy.typing import NDArray
import numpy as np
from typing import Callable
from typing import List


class RBFParams:
    """
    A data structure to store a radial basis function and its parameters; i.e.
    its centres, support width, and weights at corresponding to the centres.

    Arithmetic operations are exposed to the weights member for ease of
    manipulating these values.
    """
    def __init__(
            self, phi: Callable, cs: NDArray, d: float, ws: NDArray = None
    ):
        if ws is None:
            ws = np.ones_like(cs)

        self.phi = phi
        self.cs = cs
        self.d = d
        self.ws = ws

    def __add__(self, other):
        return self.ws + other

    def __iadd__(self, other):
        self.ws += other
        return self

    def __sub__(self, other):
        return self.ws - other

    def __isub__(self, other):
        self.ws -= other
        return self


class Approximant:
    """
    Function approximator using the parameters provided by `RBFParams`.
    """
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
        if isinstance(other, Approximant):
            return Approximant(self.params + other.params)
        else:
            raise NotImplementedError

    def __iadd__(self, other):
        if isinstance(other, Approximant):
            self.params += other.params
        else:
            raise NotImplementedError

        return self

    def __len__(self):
        return len(self.params)

    def __iter__(self):
        return iter(self.params)

    def append(self, params: RBFParams | List[RBFParams]):
        if isinstance(params, RBFParams):
            self.params.append(params)
        else:
            self.params += params
