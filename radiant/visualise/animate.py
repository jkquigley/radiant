import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def ml_animate(u, approx, a, b, n=1000, interval=1000, **kwargs):
    fig, ax = plt.subplots(**kwargs)
    ax.margins(x=0.)
    xs = cp.linspace(a, b, n * int(b - a))

    true_line, = ax.plot(xs.get(), u(xs).get(), label="Exact")
    approx_line, = ax.plot(xs.get(), approx(xs, end=0).get(), label="Approx")
    n_label = ax.text(0, 1, "$n = 0$", transform=ax.transAxes, fontsize=13)

    plt.ylim(
        np.min(
            np.minimum(u(xs).get(), approx(xs, end=0).get())
        ) - plt.rcParams['axes.ymargin'],
        np.max(
            np.maximum(u(xs).get(), approx(xs).get())
        ) + plt.rcParams['axes.ymargin'],
    )

    plt.legend()

    def func(i):
        approx_line.set_ydata(approx(xs, end=i).get())
        n_label.set_text(f"$n = {i}$")

        return true_line, approx_line, n_label,

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
