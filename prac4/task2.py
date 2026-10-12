import time
from math import sqrt

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg, gmres, splu


def build(n):
    N = n * n
    h = 1 / n
    off1 = -np.ones(N - 1)
    off1[n - 1 :: n] = 0
    A = diags(
        [-np.ones(N - n), off1, 4 * np.ones(N), off1, -np.ones(N - n)],
        [-n, -1, 0, 1, n],
        format="csr",
    )

    x = np.arange(n) * h + h
    X, Y = np.meshgrid(x, x, indexing="ij")
    b = (h**2 * 20 * X * Y * (X**2 + Y**2) * (1 - X**2 * Y**2)).ravel()
    return A, b


def test(A, b):
    print(f"n = {int(sqrt(len(b)))}:")
    start = time.perf_counter()
    x, info = cg(A, b)
    end = time.perf_counter()
    print(f"Сопряженные градиенты: {end - start:.6f} сек")

    start = time.perf_counter()
    x, info = gmres(A, b)
    end = time.perf_counter()
    print(f"Обобщенные минимальные невязки: {end - start:.6f} сек")

    start = time.perf_counter()
    lu = splu(A.tocsc())
    x = lu.solve(b)
    end = time.perf_counter()
    print(f"LU разложение: {end - start:.6f} сек\n")


def task2():
    A50, b50 = build(50)
    A100, b100 = build(100)
    A200, b200 = build(200)

    test(A50, b50)
    test(A100, b100)
    test(A200, b200)


if __name__ == "__main__":
    task2()
