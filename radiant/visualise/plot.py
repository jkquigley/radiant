import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
from ..util import gridn


def overlay(ranges, *funcs, n=1000, labels=None, **kwargs):
    if labels is None:
        labels = [None] * len(funcs)

    legend = any([label is not None for label in labels])

    fig = plt.figure(**kwargs)
    x = gridn(ranges, n)

    if len(ranges) == 1:
        ax = fig.add_subplot(111)
        plt.margins(x=0.)

        for f, label in zip(funcs, labels):
            ax.plot(*x, f(*x), label=label)
    elif len(ranges) == 2:
        ax = fig.add_subplot(111, projection='3d')
        plt.margins(x=0., y=0.)

        for f, label in zip(funcs, labels):
            ax.plot_surface(*x, f(*x), label=label, cmap='spring')
    else:
        raise ValueError(f"Cannot plot {len(ranges)} dimensions.")

    if legend:
        plt.legend()
    plt.show()


def spread(
        ranges,
        *funcs,
        n=1000,
        ncols=None,
        wspace=0.,
        hspace=0.,
        titles=None,
        filename=None,
        **kwargs
):
    if ncols is None:
        ncols = len(funcs)

    if titles is None:
        titles = [None] * len(funcs)

    fig = plt.figure(**kwargs)
    x = gridn(ranges, n)
    nrows = len(funcs) // ncols

    if ncols % len(funcs) != 0:
        nrows += 1

    if len(ranges) == 1:
        for i, (f, title) in enumerate(zip(funcs, titles)):
            ax = fig.add_subplot(nrows, ncols, i+1)
            plt.margins(x=0.)
            ax.plot(*x, f(*x))
            ax.set_title(title)
    elif len(ranges) == 2:
        for i, (f, title) in enumerate(zip(funcs, titles)):
            ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
            plt.margins(x=0., y=0.)
            ax.plot_surface(*x, f(*x), cmap='spring')
            ax.set_title(title)
    else:
        raise ValueError(f"Cannot plot {len(ranges)} dimensions.")

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def thinning(deltas, xcs, d, **kwargs):
    fig = plt.figure(**kwargs)

    if d == 1:
        ax = fig.add_subplot(111)
    # elif d == 2:
    #     ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError(f"Cannot plot {d} dimensions.")

    for delta, xc in zip(deltas, xcs):
        ax.scatter(*xc, delta * np.ones_like(xc[0]))

    plt.yscale('log')
    plt.show()
