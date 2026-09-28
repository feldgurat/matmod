import numpy
import scipy.linalg as sl

A = numpy.array([[8, 5, 2], [21, 19, 16], [39, 48, 53]], dtype=float)
b_a = numpy.array([14, 56, 140], dtype=float)
b_b = numpy.array([15, 56, 140], dtype=float)

B = numpy.array([[6, 3, 1], [6, 9, 2], [3, 11, 15]], dtype=float)
b_c = numpy.array([10, 17, 29], dtype=float)
b_d = numpy.array([10, 17, 28], dtype=float)

y_a = sl.solve(A, b_a)
y_b = sl.solve(A, b_b)
y_c = sl.solve(B, b_c)
y_d = sl.solve(B, b_d)

print(y_a)
print(y_b)
print(y_c)
print(y_d)

A_inv = sl.inv(A)
B_inv = sl.inv(B)

cond_A = sl.norm(A, 2) * sl.norm(A_inv, 2)
cond_B = sl.norm(B, 2) * sl.norm(B_inv, 2)
print(str(cond_A) + "\t" + str(numpy.linalg.cond(A)))
print(str(cond_B) + "\t" + str(numpy.linalg.cond(B)))
