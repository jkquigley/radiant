from .approximant import Approximant


def solve(
        f, centres, delta, phi, solver, *solver_args,
        combine=True, approx=None, **solver_kwargs,
):
    if approx is None:
        approx = Approximant()

    data = []

    for c, d in zip(centres, delta):
        params, datum = solver(
            f, c, d, phi, *solver_args,
            combine=False, approx=approx, **solver_kwargs,
        )

        approx.append(params)
        data.append(datum)

    if combine:
        return approx, data
    else:
        return approx.params, data
