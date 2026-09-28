import numpy
import scipy.linalg as sl

m = int(input("m: "))
r = numpy.random.default_rng()

A = r.uniform(-10, 10, (m, m))
b = numpy.ones(m)
print(A)
if numpy.linalg.matrix_rank(A) == m:
    y_i = sl.inv(A) @ b
    y_s = sl.solve(A, b)
    print(y_i)
    print(y_s)
else:
    print("Решений нет")