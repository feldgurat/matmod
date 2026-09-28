import numpy


n = int(input("n: "))
A = numpy.random.default_rng().integers(0, 10, size=(n, n))

print("A:\n", A)
print("|A| = ", numpy.linalg.det(A))
try:
    print("A^-1:\n", numpy.linalg.inv(A))
except numpy.linalg.LinAlgError:
    print("Обратной матрицы не существует")
print("A^T:\n", numpy.linalg.matrix_transpose(A))
print("Собственные значения и векторы:\n", numpy.linalg.eig(A))
print("Cпектральный радиус: \n", numpy.max(numpy.abs(numpy.linalg.eigvals(A))))
