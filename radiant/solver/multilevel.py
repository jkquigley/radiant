import numpy as np
from .approximant import Approximant


def thin(centres, delta, nlevels, min_npoints=2):
    if nlevels == 1:
        yield centres, delta
    else:
        remaining = np.arange(centres.shape[0])
        idx = np.empty(0, dtype=int)
        ratio = (centres.shape[0] / min_npoints) ** (1 / (nlevels - 1))

        for level in range(nlevels):
            npoints = min(
                centres.shape[0],
                np.floor(
                    min_npoints * ratio ** level - idx.shape[0]
                ).astype(int),
            )

            np.random.shuffle(remaining)
            idx = np.append(idx, remaining[:npoints])
            idx = np.sort(idx)
            remaining = remaining[npoints:]

            yield centres[idx], delta / (ratio ** level)


def solve(
        f, centres, delta, phi, nlevels, solver, *solver_args,
        combine=True, **solver_kwargs,
):
    approx = Approximant()

    for c, d in thin(centres, delta, nlevels):
        result = solver(
            f, c, d, phi, *solver_args, combine=False, **solver_kwargs
        )
        params = result[0]
        # TODO: extract results

        params -= approx(c)
        approx.append(params)

    if combine:
        return approx,
    else:
        return approx.params,
