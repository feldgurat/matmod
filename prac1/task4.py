import numpy
import matplotlib.pyplot as plt

a, b = -10, 10
x = numpy.linspace(a, b, 400)

y1 = numpy.sin(x)
y2 = numpy.log10(x)
y3 = x**2 / 10
y4 = numpy.exp(x)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))

axes[0, 0].plot(x, y1, color="red", linestyle="-", lw=2, label="f(x) = sin(x)")
axes[0, 0].set_title("Синусоида")

axes[0, 1].plot(x, y2, color="blue", linestyle="--", lw=2, label="f(x) = log10(x)")
axes[0, 1].set_title("Десятичный логарифм")

axes[1, 0].plot(x, y3, color="green", linestyle="-.", lw=2, label="f(x) = x^2/10")
axes[1, 0].set_title("Парабола")

axes[1, 1].plot(x, y4, color="magenta", linestyle=":", lw=2.5, label="f(x) = e^x")
axes[1, 1].set_title("Экспонента")

for ax in axes.flat:
    ax.set_xlabel("Ось X")
    ax.set_ylabel("Ось Y")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)

plt.show()
