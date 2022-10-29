from sympy import *
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.misc
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# 3.102
f1 = lambda x: np.e ** (-x**2 / 2)
f2 = lambda x: np.cos(x**2)

exact1 = scipy.integrate.quad(f1, -2, 2) # 2.392576026645216
exact2 = scipy.integrate.quad(f2, 0, 1) # 0.904524237900272

h1 = {}
h2 = {}

def trapezoidal_sum(f, a, b, h, my_dict):
    my_dict[h] = []
    n = (b - a) / h
    current_sum = 0

    for i in range(h - 1):
        current_sum += 0.5 * (f(a + i*n) + f(a + (i+1)*n)) * n
    
    my_dict[h] += [current_sum]

for h in range(1, 100):
    trapezoidal_sum(f1, -2, 2, h, h1)
    trapezoidal_sum(f2, 0, 1, h, h2)

x = [log(h) for h in range(1, 100)]
y1 = []
y2 = []

for key, value in h1.items():
    y1.append(np.log10(abs(exact1[0] - value[0])))
for key, value in h2.items():
    y2.append(np.log10(abs(exact2[0] - value[0])))

plt.ylabel('base 10 log of absolute error')
plt.xlabel('base 10 log of h')
plt.scatter(x, y1)
plt.show()

plt.ylabel('base 10 log of absolute error')
plt.xlabel('base 10 log of h')
plt.scatter(x, y2)
plt.show()

# The behavior of the loglog plots is as expected:
# The slope of the linear regression is approximately -1
# This indicates that error is reduced with a higher number of trapezoids
# This regression is a better fit at higher iterations
# The reduction in error diminishes with each iteration
# Essentially, absolute error is logarithmically related to h
