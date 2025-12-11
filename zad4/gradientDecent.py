import numpy as np

def gradient_descent(f, grad_f, x0, alpha, max_iter=10000, tol=1e-6):
    x = np.array(x0, dtype=float)
    path = []

    for k in range(max_iter):
        fx = float(f(x))
        path.append(fx)

        p = grad_f(x)
        if np.linalg.norm(p) < tol:
            break

        x = x - alpha * p

        if not np.isfinite(x).all():
            return x, fx, path, False

    return x, float(f(x)), path, True

