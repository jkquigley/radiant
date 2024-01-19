import numpy as np
import matplotlib.pylab as plt


def overlay(u, u_approx, a, b, **kwargs):
    plt.figure(**kwargs)
    xs = np.linspace(a, b, 100)
    plt.plot(xs, u(xs), label="Exact")
    plt.plot(xs, u_approx(xs), label="Approx")
    plt.legend()
    plt.show()
