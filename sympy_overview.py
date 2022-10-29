from sympy import *
import numpy as np
import scipy.misc
import scipy.integrate
import matplotlib.pyplot as plt

def main():
    # define the variables
    x, y, z = symbols('x y z')

    # expand an expression
    print(expand((x + 1)** 2))
    print(expand((x + 3) * (x + 4)))

    # factor an expression
    print(factor(x**3 -x**2 + x - 1))

    # simplify trig expression
    print(trigsimp(sin(x)**2 + cos(x)**2))

    # exapnd trig expression
    print(expand_trig(sin(x + y)))

    # get derivatives
    print(diff(cos(x), x))

    # create a list
    a_list = np.arange(0, 10)
    print(a_list)
    print(np.diff(a_list))
    print()

    b_list = np.linspace(0, 1, 6)
    print(b_list)
    print(np.diff(b_list))
    print()

    x = np.linspace(0, 1, 6)
    dx = x[1] - x[0]
    y = x**2
    dy = 2 * x

    # exact values of derivative
    print("exact values of derviative: \n", dy, '\n')

    # vales from np.diff()
    print("values form no.diff()dx: \n", np.diff(y), '\n')

    # values from np.diff() / dx
    print("values form no.diff()dx: \n", np.diff(y) / dx, '\n')

    # compute derivative using scipy
    f = lambda x: x**2
    x = np.linspace(1, 5, 5)
    df = scipy.misc.derivative(f, x, dx=1e-6)
    print(df, '\n')

    # get first and second derivative
    f = lambda x: np.sin(x) * x - np.log(x)
    x = np.linspace(1, 5, 100)
    df = scipy.misc.derivative(f, x, dx=1e-6)
    df2 = scipy.misc.derivative(f, x, dx=1e-6, n=2)
    plt.plot(x, f(x), 'b', x, df, 'r--', x, df2, 'k--')
    plt.legend(["f(x)", "f'(x)", "f''(x)"])
    plt.grid()
    plt.show()

    # get integral
    expr = x**2 + x + 1
    print(integrate(expr, x))

    expr = x / (x**2 + 2*x +1)
    print(integrate(expr, x))

    expr = log(x)**2
    print(integrate(expr, x))

    expr = sin(x) * tan(x)
    print(integrate(expr, x))

    # definite integrals
    expr = exp(-x**2)
    print(integrate(expr, (x, 0, oo)))

    expr = exp(-x**2 - y**2)
    print(integrate(expr, (x, 0, oo), (y, 0, oo)))

    # integrate with trapezoidal Reimann sums
    x = np.linspace(-2, 2, 100)
    dx = x[1] - x[0]
    y = x**2 
    print('Integral of x^2 = ', np.trapz(y) * dx)

    x = np.linspace(0, 2 * np.pi, 100)
    dx = x[1] - x[0]
    y = np.sin(x)
    print('Integral of sin(x) = ', np.trapz(y) * dx)

    # integrate with Simpson's Rule (quadrature)
    f = lambda x: x**2
    print('Integral of x^2 = ', scipy.integrate.quad(f, -2, 2))

main()
