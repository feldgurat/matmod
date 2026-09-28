from math import sqrt

import numpy
import scipy.linalg as sl

n = int(input("n: "))

r = numpy.random.default_rng()
x = r.uniform(-10, 10, size=(n))
x_1 = sum(abs(i) for i in x)
x_2 = sqrt(sum(i**2 for i in x))
x_inf = max(abs(i) for i in x)

print("x: " + str(x))
print(str(x_1) + "\t" + str(sl.norm(x, 1)))
print(str(x_2) + "\t" + str(sl.norm(x, 2)))
print(str(x_inf) + "\t" + str(sl.norm(x, numpy.inf)))

A = r.uniform(-10, 10, size=(n, n))

A_1 = max(sum(abs(i) for i in j) for j in A.T)
A_fro = sqrt(sum(sum(i**2 for i in j) for j in A))
A_inf = max(sum(abs(i) for i in j) for j in A)

print("A:\n" + str(A))
print(str(A_1) + "\t" + str(sl.norm(A, 1)))
print(str(A_fro) + "\t" + str(sl.norm(A, "fro")))
print(str(A_inf) + "\t" + str(sl.norm(A, numpy.inf)))
