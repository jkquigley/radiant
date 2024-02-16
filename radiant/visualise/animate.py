import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ..util import gridn


def ml_animate(ranges, exact, approx, n=1000, interval=1000, **kwargs):
    fig = plt.figure(**kwargs)
    x = gridn(ranges, n)

    if len(ranges) == 1:
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        plt.margins(x=0.)

        ax0.set_ylim(
            bottom=np.min(approx(*x)) - plt.rcParams['axes.ymargin'],
            top=np.max(approx(*x)) + plt.rcParams['axes.ymargin'],
        )
        ax1.set_ylim(
            bottom=0.,
            top=np.max(
                np.abs(exact(*x) - approx(*x))
            ) + plt.rcParams['axes.ymargin'],
        )

        exact_val = exact(*x)
        error = np.abs(exact_val)
        exact_line, = ax0.plot(*x, exact_val, label="Exact")
        approx_line, = ax0.plot(*x, np.zeros_like(x[0]), label="Approximate")
        error_line, = ax1.plot(*x, error)
        n_label = ax0.text(
            0, 1, "$n = 0$", transform=ax0.transAxes, fontsize=13
        )

        ax0.set_title("Approximation")
        ax1.set_title("Abs. Error")
        ax0.legend()

        def func(i):
            approx_val = approx[:i](*x)
            approx_line.set_ydata(approx_val)
            error_line.set_ydata(np.abs(exact_val - approx_val))
            n_label.set_text(f"$n = {i}$")

            return approx_line, error_line, n_label,

    # elif len(ranges) == 2:
    #     ax0 = fig.add_subplot(121)
    #     ax1 = fig.add_subplot(122)
    #
    #     extent = (np.min(x[0]), np.max(x[0]), np.min(x[1]), np.max(x[1]))
    #
    #     exact_val = exact(*x)
    #     approx_val = approx(*x, end=0)
    #     error = np.abs(exact_val - approx_val)
    #
    #     approx_line = ax0.imshow(
    #         approx_val, extent=extent
    #     )
    #     error_line = ax1.imshow(
    #         error, extent=extent
    #     )
    #     n_label = ax0.text(
    #         0, 1, "$n = 0$", transform=ax0.transAxes, fontsize=13
    #     )
    #
    #     def func(i):
    #         approx_val = approx(*x, end=i)
    #         error = np.abs(exact_val - approx_val)
    #
    #         approx_line.set_data(approx_val)
    #         error_line.set_data(error)
    #         n_label.set_text(f"$n = {i}$")
    #
    #         return approx_line, error_line, n_label,

    else:
        raise ValueError(f"Cannot plot of {len(ranges)} ranges.")

    anim = animation.FuncAnimation(
        fig,
        func,
        frames=range(0, len(approx) + 1),
        interval=interval,
        blit=True,
        repeat=False
    )

    plt.close()

    return anim
