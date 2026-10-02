import numpy
import scipy.linalg as sl

numpy.set_printoptions(suppress=True)



def dumat(A, b):
    P, L, U = sl.lu(A)
    print("P:\n", P)
    print("L:\n", L)
    print("U:\n", U)
    ylu = sl.solve_triangular(L, P.T @ b, lower=True)
    xlu = sl.solve_triangular(U, ylu)
    print("x (LU):", xlu)

    Q, R = sl.qr(A)
    print("Q:\n", Q)
    print("R:\n", R)
    xqr = sl.solve_triangular(R, Q.T @ b)
    print("x (QR):", xqr)

    if numpy.allclose(A, A.T) and all(numpy.linalg.eigvalsh(A) > 0):
        L = sl.cholesky(A, lower=True)
        print("L (Холецкий):\n", L)
        yhol = sl.solve_triangular(L, b, lower=True)
        xhol = sl.solve_triangular(L.T, yhol)
        print("x (Холецкий):", xhol)
    else:
        print("Холецкий невозможен")

    if numpy.allclose(A, numpy.triu(numpy.tril(A, 1), -1)):
        xpr = progonka(A, b)
        print("x (прогонка):", xpr)
    else:
        print("Прогонка невозможна: матрица не трёхдиагональная")


A_a = numpy.array(
    [[7, 1, 1, 0],
     [1, 5, 2, 1],
     [2, 3, -3, 3],
     [3, 4, 5, 5]], dtype=float
)
b_a = numpy.array([7, 0, -1, -2], dtype=float)

A_b = numpy.array(
    [
        [4, -6, 0, 8, 0, 0],
        [-6, 8, -12, 0, 16, 0],
        [0, -12, 16, 0, 0, 10],
        [8, 0, 0, 20, -8, 0],
        [0, 16, 0, -8, 24, -8],
        [0, 0, 10, 0, -8, 28],
    ],
    dtype=float,
)
b_b = numpy.array([24, 54, 84, 48, 72, 158], dtype=float)

n = 8
A_v = 2 * numpy.eye(n) - numpy.eye(n, k=1) - numpy.eye(n, k=-1)
b_v = numpy.array([4 * (-1) ** i for i in range(n)], dtype=float)
b_v[0], b_v[-1] = 3, -3

A_g = numpy.array(
    [
        [10, -1, 0, 0, 0, 0],
        [1, -10, 3, 0, 0, 0],
        [0, 1, -10, -2, 0, 0],
        [0, 0, 2, -10, -1, 0],
        [0, 0, 0, 1, -10, 1],
        [0, 0, 0, 0, 2, -10],
    ],
    dtype=float,
)
b_g = numpy.array([10, 4, -10, 1, -10, 2], dtype=float)

print("а)\nA:\n", A_a)
dumat(A_a, b_a)

print("б)\nA:\n", A_b)
dumat(A_b, b_b)

print("в)\nA:\n", A_v)
dumat(A_v, b_v)

print("г)\nA:\n", A_g)
dumat(A_g, b_g)
