import matplotlib.pyplot as plt
import numpy
from matplotlib.patches import Patch


def is_diag_dominant(A) -> bool:
    strict = False
    for i in range(len(A)):
        diag = abs(A[i, i])
        rest = sum(abs(A[i, j]) for j in range(len(A)) if j != i)
        if diag < rest:
            return False
        if diag > rest:
            strict = True
    return strict


def jacobs_method(
    A, b: list[float], eps: float = 1e-6, max_iter: int = 1000
) -> tuple[list[float], list[list[float]]]:
    if len(A) != len(A[0]):
        raise ValueError("Матрица должна быть квадратной")
    if not is_diag_dominant(A):
        print("Сходимость не гарантирована")
    n = len(A)
    sol_x = [0.0] * n
    iterations = [sol_x[:]]
    for _ in range(max_iter):
        new_x = [0.0] * n
        for i in range(n):
            s = b[i]
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    s -= A[i, j] * sol_x[j]
            new_x[i] = s / A[i, i]
        iterations.append(new_x)
        if max(abs(new_x[i] - sol_x[i]) for i in range(n)) < eps:
            return (new_x, iterations)
        sol_x = new_x
    print("Не получилось(")
    return (sol_x, iterations)


def seidel_method(
    A, b: list[float], eps: float = 1e-6, max_iter: int = 1000
) -> tuple[list[float], list[list[float]]]:
    if len(A) != len(A[0]):
        raise ValueError("Матрица должна быть квадратной")
    if not is_diag_dominant(A):
        print("Сходимость не гарантирована")
    n = len(A)
    sol_x = [0.0] * n
    iterations = [sol_x[:]]
    for _ in range(max_iter):
        old_x = sol_x[:]
        for i in range(n):
            s = b[i]
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    s -= A[i, j] * sol_x[j]
            sol_x[i] = s / A[i, i]
        iterations.append(sol_x[:])
        if max(abs(old_x[i] - sol_x[i]) for i in range(n)) < eps:
            return (sol_x, iterations)
    print("Не получилось(")
    return (sol_x, iterations)


def build_system(n: int, alpha: float):
    A = (
        2 * numpy.eye(n + 1)
        + (-1 - alpha) * numpy.eye(n + 1, k=1)
        + (-1 + alpha) * numpy.eye(n + 1, k=-1)
    )
    f = [0.0] * (n + 1)
    f[0] = 1 - alpha
    f[n] = 1 + alpha
    return A, f


def count_iterations(n: int, alpha: float, eps: float = 1e-6) -> tuple[int, int]:
    A, f = build_system(n, alpha)
    _, it_j = jacobs_method(A, f, eps, MAX_ITER)
    _, it_s = seidel_method(A, f, eps, MAX_ITER)
    return len(it_j) - 1, len(it_s) - 1


MAX_ITER = 100000


# 3D-графики зависимости числа итераций от n и alpha
ns = numpy.arange(4, 101, 4)
alphas = numpy.linspace(0, 1, 21)
N, ALPHA = numpy.meshgrid(ns, alphas)
K_j = numpy.zeros(N.shape)
K_s = numpy.zeros(N.shape)
for i in range(len(alphas)):
    for j in range(len(ns)):
        K_j[i, j], K_s[i, j] = count_iterations(int(N[i, j]), ALPHA[i, j])

fig = plt.figure(figsize=(14, 6))
for pos, (K, title) in enumerate(((K_j, "Метод Якоби"), (K_s, "Метод Зейделя")), 1):
    ax = fig.add_subplot(1, 2, pos, projection="3d")
    ax.plot_surface(N, ALPHA, numpy.log10(K), cmap="viridis", edgecolor="k", lw=0.2)
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\alpha$")
    ax.set_zlabel("число итераций")
    powers = numpy.arange(int(numpy.log10(K.min())), int(numpy.log10(K.max())) + 1)
    ax.set_zticks(powers)
    ax.set_zticklabels([str(10**p) for p in powers])
    ax.view_init(elev=25, azim=-130)
plt.tight_layout()

# Сравнение методов на одном графике
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_surface(N, ALPHA, numpy.log10(K_j), color="tab:red", alpha=0.6)
ax.plot_surface(N, ALPHA, numpy.log10(K_s), color="tab:blue", alpha=0.6)
ax.set_title("Число итераций")
ax.set_xlabel("n")
ax.set_ylabel(r"$\alpha$")
ax.set_zlabel("число итераций")
powers = numpy.arange(int(numpy.log10(K_s.min())), int(numpy.log10(K_j.max())) + 1)
ax.set_zticks(powers)
ax.set_zticklabels([str(10**p) for p in powers])
ax.view_init(elev=25, azim=-130)
ax.legend(
    handles=[
        Patch(color="tab:red", label="Якоби"),
        Patch(color="tab:blue", label="Зейдель"),
    ]
)

ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.plot_surface(N, ALPHA, K_j / K_s, cmap="coolwarm", edgecolor="k", lw=0.2)
ax.set_title("Во сколько раз Зейдель быстрее Якоби")
ax.set_xlabel("n")
ax.set_ylabel(r"$\alpha$")
ax.set_zlabel("итерации Якоби / итерации Зейделя")
ax.view_init(elev=25, azim=-130)
plt.tight_layout()
plt.show()
