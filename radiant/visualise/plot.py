import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
from ..util import grid


def difference(a, b, f, g, n=1000, **kwargs):
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    plt.margins(x=0.)
    xs = np.linspace(a, b, n * int(b - a))
    ax.plot(xs, np.abs(f(xs) - g(xs)))
    plt.show()


def difference3d(a, b, f, g, n=1000, **kwargs):
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111, projection='3d')
    plt.margins(x=0., y=0.)
    xs = grid(np.linspace(a, b, n * int(b - a)), 2)
    ax.plot_surface(*xs, np.abs(f(*xs) - g(*xs)))
    plt.show()


def many(a, b, *funcs, n=1000, labels=None, **kwargs):
    legend = True
    if labels is None:
        legend = False
        labels = [None] * len(funcs)

    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    plt.margins(x=0.)
    xs = np.linspace(a, b, n * int(b - a))

    for func, label in zip(funcs, labels):
        ax.plot(xs, func(xs), label=label)

    if legend:
        plt.legend()
    plt.show()


def many3d(a, b, *funcs, n=1000, labels=None, **kwargs):
    if labels is None:
        labels = [None] * len(funcs)

    legend = any([label is not None for label in labels])

    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111, projection='3d')
    plt.margins(x=0., y=0.)
    xs = grid(np.linspace(a, b, n * int(b - a)), 2)

    for func, label in zip(funcs, labels):
        ax.plot_surface(*xs, func(*xs), label=label, cmap='spring')

    if legend:
        plt.legend()
    plt.show()


def triplet(a, b, f, g, h, n=1000, titles=None, **kwargs):
    if titles is None:
        titles = [None] * 3

    fig = plt.figure(**kwargs)
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    xs = np.linspace(a, b, n * int(b - a))

    ax0.plot(xs, f(xs))
    ax0.set_title(titles[0])
    ax0.margins(x=0.)

    ax1.plot(xs, g(xs))
    ax1.set_title(titles[1])
    ax1.margins(x=0.)

    ax2.plot(xs, h(xs))
    ax2.set_title(titles[2])
    ax2.margins(x=0.)

    plt.subplots_adjust(wspace=0.25)
    plt.show()


def triplet3d(a, b, f, g, h, n=1000, titles=None, **kwargs):
    if titles is None:
        titles = [None] * 3

    fig = plt.figure(**kwargs)
    ax0 = fig.add_subplot(131, projection='3d')
    ax1 = fig.add_subplot(132, projection='3d')
    ax2 = fig.add_subplot(133, projection='3d')
    xs = grid(np.linspace(a, b, n * int(b - a)), 2)

    ax0.plot_surface(*xs, f(*xs), cmap='spring')
    ax0.set_title(titles[0])
    ax0.margins(x=0., y=0.)

    ax1.plot_surface(*xs, g(*xs), cmap='spring')
    ax1.set_title(titles[1])
    ax1.margins(x=0., y=0.)

    ax2.plot_surface(*xs, h(*xs), cmap='spring')
    ax2.set_title(titles[2])
    ax2.margins(x=0., y=0.)

    plt.subplots_adjust(wspace=0.25)
    plt.show()


def thinning(centres, delta, **kwargs):
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    for c, d in zip(centres, delta):
        ax.plot(c, (d * np.ones_like(c)), 'b.')

    plt.yscale('log')
    plt.show()
