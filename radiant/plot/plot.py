import numpy as np
import matplotlib.pylab as plt


def overlay(u, approx, a, b, n=1000, **kwargs):
    plt.figure(**kwargs)
    xs = np.linspace(a, b, n * int(b - a))
    plt.plot(xs, u(xs), label="Exact")
    plt.plot(xs, approx(xs), label="Approx")
    plt.legend()
    plt.show()
