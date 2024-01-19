##############################
# Solves -u'' + u = f        #
# With zero Neumann boundary #
##############################
import radiant as rad
import numpy as np


# Problem Parameters
a = 0
b = 2 * np.pi


def u(x):
    return np.cos(x * 2 * np.pi / (b - a))


def f(x):
    return ((2 * np.pi / (b - a)) ** 2 + 1) * np.cos(x * 2 * np.pi / (b - a))


# Parameters
N = 21
d = 1
k = 1
delta = 2 * (b - a) / np.pi

phi, points, A, fs = rad.generate(a, b, f, N, d, k, delta)

# Solve for approximate solution
u_approx, alphas = rad.solve(phi, points, A, fs)
error = rad.error(u, u_approx, a, b)

print("L2 Relative Error:", error)
print("System condition number:", np.linalg.cond(A))


# TODO:
# Look at cond number
#   - Fix delta change N
#   - Fix N look at delta
# Multilevel algorithm for plain vanilla function approx
