import numpy as np
from typing import List


class RBFParams:
    def __init__(self, phi, cs, d, ws=None):
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
    def __init__(self, params: None | RBFParams | List[RBFParams] = None):
        if params is None:
            params = []
        elif isinstance(params, RBFParams):
            params = [params]

        self.params = params

    def __call__(self, x, m=0, end=None):
        if end is None:
            end = len(self.params)
        val = np.zeros_like(x)
        for p in self.params[:end]:
            val += np.sum(
                np.multiply(p.ws[:, None], p.phi(x, p.cs, p.d, m)),
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
