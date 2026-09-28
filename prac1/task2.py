import numpy


def dumat(A, B):
    print("A:", A.shape, " B:", B.shape)
    if A.shape == B.shape:
        print("A + B =\n", A + B)
    else:
        print("сложение невозможно")

    if A.shape[1] == B.shape[0]:
        print("A * B =\n", A @ B)
    else:
        print(f"A * B: невозможно {A.shape[1]} != {B.shape[0]}")

    if B.shape[1] == A.shape[0]:
        print("B * A =\n", B @ A)
    else:
        print(f"B * A: невозможно {B.shape[1]} != {A.shape[0]}")
    print()


A1 = numpy.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
B1 = numpy.array([[0, 2, -3], [-2, 0, 6], [3, -6, 0]])
print("a)")
dumat(A1, B1)

A2 = numpy.array([[1, 3 + 2j], [3 - 2j, 5]])
B2 = numpy.array([[3, 2, 1], [7, 1, 4]])
print("б)")
dumat(A2, B2)

A3 = numpy.array([[1, 2, 3], [5, 4, 3], [6, 7, 8]])
B3 = numpy.array([[1, 4, 8], [0, 3, 6]])
print("в)")
dumat(A3, B3)
