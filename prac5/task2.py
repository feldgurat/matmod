import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a + b * x + c * x**2


def task2():
    x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    y = np.array([3.9, 4.3, 2.5, 2.75, 3.4, 3.6])

    popt, pcov = curve_fit(func, x, y)

    print(popt)

    for i in (7, 8):
        print(f"Прогноз на {i} год: {func(i, *popt):.3f}")

    xs = np.linspace(x.min(), 8, 200)
    names = func.__code__.co_varnames[1 : func.__code__.co_argcount]
    params = ", ".join(f"{n}={v:.4g}" for n, v in zip(names, popt))

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="black", zorder=3, label="Исходные данные")
    plt.plot(xs, func(xs, *popt), label=f"Аппроксимация: {params}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Аппроксимация методом наименьших квадратов")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    task2()
