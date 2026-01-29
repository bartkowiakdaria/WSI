import numpy as np

def ackley_1d(x):
    x = np.asarray(x, dtype=float)
    return -20*np.exp(-0.2*np.sqrt(x**2)) - np.exp(np.cos(2*np.pi*x)) + 20 + np.e

def ackley_1d_grad(x):
    x = np.asarray(x, dtype=float)
    xx = np.sqrt(x**2)
    first = 4 * np.exp(-0.2 * xx) * x / xx
    second = 2 * np.pi * np.exp(np.cos(2*np.pi*x)) * np.sin(2*np.pi*x)
    return first + second

def ackley_2d(x):
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    first = -20*np.exp(-0.2*np.sqrt((x1**2 + x2**2) / 2.0))
    second = -np.exp((np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2)) / 2.0)
    return first + second + 20 + np.e


def ackley_2d_grad(x):
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    exp1 = np.exp(-0.2*np.sqrt((x1**2 + x2**2) / 2.0))
    exp2 = np.exp((np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2)) / 2.0)

    dfdx = 2 * x1*exp1 / np.sqrt((x1**2 + x2**2) / 2.0) + np.pi*exp2*np.sin(2*np.pi*x1)
    dfdy = 2 * x2*exp1 / np.sqrt((x1**2 + x2**2) / 2.0) + np.pi*exp2*np.sin(2*np.pi*x2)

    return np.array([dfdx, dfdy])
