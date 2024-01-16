###################
# Solves -u'' = f #
###################
from radiant import phi_factory
from radiant import gauss_legendre
import numpy as np
import matplotlib.pylab as plt

# Problem Parameters
a = 0
b = 1


def u(x):
    return - np.sin(x * (2 * np.pi) / (b - a))


def f(x):
    return ((2 * np.pi) / (b - a)) ** 2 * u(x)


# Algorithm Parameters
N = 4
delta = 4 * (b - a) / N
points = np.linspace(a, b, N)
A = np.zeros((points.size, points.size))
fs = np.zeros_like(points)

# RBF Parameters
d = 1
k = 1
phi = phi_factory(d, k)

for i, xi in enumerate(points):
    fs[i] = gauss_legendre(
        lambda x: f(x) * phi(np.abs(x - xi) / delta),
        a, b, 150
    )

    for j, xj in enumerate(points[:i+1]):
        A[i, j] = gauss_legendre(
            lambda x: phi(np.abs(x - xi) / delta, 1) * phi(
                np.abs(x - xj) / delta, 1),
            a, b, 4
        )

A += np.tril(A, -1).T

alphas = np.linalg.solve(A, fs)


def u_approx(x):
    val = 0
    for a, xi in zip(alphas, points):
        val += a * phi(np.abs(x - xi) / delta)

    return val


xs = np.linspace(a, b, 100)
plt.plot(xs, u(xs), label="Exact")
plt.plot(xs, u_approx(xs), label="Approx")
plt.legend()
plt.show()

print(np.linalg.cond(A))
print(alphas)