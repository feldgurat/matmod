import numpy
import matplotlib.pyplot as plt

x = numpy.linspace(-50, 50, 500)
y = numpy.linspace(-50, 50, 500)
X, Y = numpy.meshgrid(x, y)


Z = X**2 + Y**2


ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()