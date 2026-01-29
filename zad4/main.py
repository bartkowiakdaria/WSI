from gradientDecent import gradient_descent
import numpy as np
from ackleyFunctions import ackley_2d, ackley_1d, ackley_2d_grad, ackley_1d_grad
import matplotlib.pyplot as plt


xmin, fmin, path, _ = gradient_descent(ackley_1d, ackley_1d_grad, [1], alpha=0.01)
print(f"iteracje={len(path):4}, x*={xmin[0]: .4f}, f(x*)={fmin: .4f}")
xmin, fmin, path, _ = gradient_descent(ackley_2d, ackley_2d_grad, [1, -1], alpha=0.01)
print(f"iteracje={len(path):4}, x*={xmin[0]: .4f}, f(x*)={fmin: .4f}")

# TESTOWANIE DŁUGOŚCI KROKU ALFA
print("----- 1D -----")
alphas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2]

for i in [0.01, 1]:
    print("\n ----- Punkt startowy: " + str(i) + " lub " + str([i, -i]) + "-----\n")
    x0 = np.array([i])
    for alpha in alphas:
        xmin, fmin, path, end = gradient_descent(ackley_1d, ackley_1d_grad, x0, alpha=alpha)
        if end:
            print(f"alpha={alpha:>6}: iteracje={len(path):4}, x*={xmin[0]: .4f}, f(x*)={fmin: .4f}")
        else:
            print(f"alpha={alpha:>6}: rozbiegło")

    print("----- 2D -----")

    alphas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2]
    x0 = np.array([i, -i])
    for alpha in alphas:
        xmin, fmin, path, end = gradient_descent(ackley_2d, ackley_2d_grad, x0, alpha=alpha)
        if end:
            print(f"alpha={alpha:>6}: iteracje={len(path):4}, x*={xmin}, f(x*)={fmin: .4f}")
        else:
            print(f"alpha={alpha:>6}: rozbiegło")


# WYKRES ZBIEŻNOŚCI

alphas = [0.001, 0.01]
alphas2 = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2]
x0 = np.array([1])

for alpha in alphas:
    xmin, fmin, path, _ = gradient_descent(ackley_1d, ackley_1d_grad,x0, alpha=alpha)
    plt.plot(path, label=f"alpha={alpha}")

plt.yscale("log")
plt.xlabel("iteracja")
plt.ylabel("f(x_k)")
plt.legend()
plt.grid(True)
plt.show()

for alpha in alphas2:
    xmin, fmin, path, _ = gradient_descent(ackley_1d, ackley_1d_grad,x0, alpha=alpha)
    plt.plot(path, label=f"alpha={alpha}")

plt.yscale("log")
plt.xlabel("iteracja")
plt.ylabel("f(x_k)")
plt.legend()
plt.grid(True)
plt.show()

