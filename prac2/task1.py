from math import sqrt

import numpy

n = int(input("n: "))

r = numpy.random.default_rng()
x = r.integers(-10, 10, size=(n))
x_1 = sum(abs(i) for i in x)
x_2 = sqrt(sum(i**2 for i in x))
x_inf = max(abs(i) for i in x)

print("x: " + str(x))
print(x_1)
print(x_2)
print(x_inf)