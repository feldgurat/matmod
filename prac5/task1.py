import numpy
from matplotlib import pyplot as plt
from scipy.linalg import lstsq


def task4():
    x = numpy.array([1, 3, 4.5, 7, 9])
    y = numpy.array([1.3, 0.8, 0, 1.5, 1.2])
    n = len(x)
    A = numpy.array(
        [
            [n, x.sum(), (x**2).sum()],
            [x.sum(), (x**2).sum(), (x**3).sum()],
            [(x**2).sum(), (x**3).sum(), (x**4).sum()],
        ]
    )
    B = numpy.array([y.sum(), (y * x).sum(), (y * x**2).sum()])
    c0, c1, c2 = numpy.linalg.solve(A, B)
    print(f"c:\t{c0}\t{c1}\t{c2}")

    A1 = numpy.array(
        [[n, numpy.log(x).sum()], [numpy.log(x).sum(), (numpy.log(x) ** 2).sum()]]
    )
    B1 = numpy.array(
        [
            y.sum(),
            (y * numpy.log(x)).sum(),
        ]
    )
    a, b = numpy.linalg.solve(A1, B1)
    print(f"a, b:\t{a}\t{b}")

    M = numpy.column_stack([numpy.ones_like(x), x, x**2])
    c, res, rank, sv = lstsq(M, y)
    M1 = numpy.column_stack([numpy.ones_like(x), numpy.log(x)])
    ab, res1, rank1, sv1 = lstsq(M1, y)
    print("lstsq:")
    print(f"c:\t{c[0]}\t{c[1]}\t{c[2]}")
    print(f"a, b:\t{ab[0]}\t{ab[1]}")

    f_parab = c0 + c1 * x + c2 * x**2
    f_log = a + b * numpy.log(x)

    S_parab = ((y - f_parab) ** 2).sum()
    S_log = ((y - f_log) ** 2).sum()

    print(f"S парабола:\t{S_parab}")
    print(f"S логарифм:\t{S_log}")
    print(f"res из lstsq:\t{res}\n\t\t{res1}")

    xs = numpy.linspace(x.min(), x.max(), 200)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="black", zorder=3, label="Исходные данные")
    plt.plot(
        xs,
        c0 + c1 * xs + c2 * xs**2,
        label=f"Парабола: f = {c0:.3f} {c1:+.3f}x {c2:+.4f}x²",
    )
    plt.plot(xs, a + b * numpy.log(xs), label=f"Логарифм: f = {a:.3f} {b:+.4f}·ln x")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Аппроксимация методом наименьших квадратов")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    task4()
