import numpy as np
import matplotlib.pylab as plt


def many(a, b, *funcs, n=1000, labels=None, **kwargs):
    leg = True
    if labels is None:
        leg = False
        labels = [None] * len(funcs)

    plt.figure(**kwargs)
    plt.margins(x=0.)
    xs = np.linspace(a, b, n * int(b - a))

    for func, label in zip(funcs, labels):
        plt.plot(xs, func(xs), label=label)

    if leg:
        plt.legend()
    plt.show()


def thinning(centres, delta, **kwargs):
    plt.figure(**kwargs)
    for c, d in zip(centres, delta):
        plt.plot(c, (d * np.ones_like(c)), 'b.')

    plt.yscale('log')
    plt.show()
