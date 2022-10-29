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

# 3.104
f = lambda x: x / (1 + x**4) 
# F(x) = arctan(x**2) / 2 + C
# F(x) | -1, 2 = arctan(4) / 2 - pi / 8 
# ~= 0.2702097501
g = lambda x: (x - 1)**3 * (x - 2)**2 
# G(x) = (x - 1)**6 / 6 - 2(x - 1)**5 / 5 + (x - 1)**4 / 4
# G(x) | -1, 2 = -27.45
h = lambda x: np.sin(x**2)
# H(x) |-1, 2 ~= 1.1150447911
F = scipy.integrate.quad(f, -1, 2)
G = scipy.integrate.quad(g, -1, 2)
H = scipy.integrate.quad(h, -1, 2)

print(abs(0.2702097501 - F[0]) < 1e-10)
print(abs(-27.45 - G[0]) < 1e-10)
print(abs(1.1150447911 - H[0]) < 1e-10)

x = np.linspace(-1, 2, 100)
f_prime = scipy.misc.derivative(f, x, dx=1e-6)
g_prime = scipy.misc.derivative(g, x, 1e-6)
h_prime = scipy.misc.derivative(h, x, 1e-6)

plt.plot(x, f(x), label='f(x)')
plt.plot(x, f_prime, label="f'(x)")
plt.legend()
plt.show()

plt.plot(x, f(x), label='g(x)')
plt.plot(x, f_prime, label="g'(x)")
plt.legend()
plt.show()

plt.plot(x, f(x), label='h(x)')
plt.plot(x, f_prime, label="h'(x)")
plt.legend()
plt.show()

# 3.105
URL1 = 'https://raw.githubusercontent.com/NumericalMethodsSullivan'
URL2 = '/NumericalMethodsSullivan.github.io/master/data/'
URL = URL1+URL2
data = np.array(pd.read_csv(URL+'Exercise3_bikespeed.csv'))

times = [int(x[0]) for x in data]
speeds = [int(x[1]) for x in data]

trapezoidal_reimann_sum = 0

for n in range(len(data) - 1):
    trapezoidal_reimann_sum += 0.5 * (times[n + 1] - times[n]) * (speeds[n + 1] + (speeds[n]))

print(trapezoidal_reimann_sum, 'ft')

# 3.106
f = lambda x: x / (1 + x**4) + np.sin(x)
g = lambda x: (x - 1)**3 * (x - 2)**2 + np.e**(-0.5*x)
f_min = minimize(f, 0)
g_min = minimize(g, 0)

f2 = lambda x: - f(x)
g2 = lambda x: -g(x)

f_max = minimize(f2, 0)
g_max = minimize(g2, 0)

if abs(f_min.x[0]) < abs(f_max.x[0]):
    print('Extrema of f(x) closest to 0: x =', f_min.x[0], ', y =', f_min.fun)
else:
    print('Extrema of f(x) closest to 0: x =', f_max.x[0], ', y =', -1 * f_max.fun)

if abs(g_min.x[0]) < abs(g_max.x[0]):
    print('Extrema of g(x) closest to 0: x =', g_min.x[0], ', y =', g_min.fun)
else:
    print('Extrema of g(x) closest to 0: x =', g_max.x[0], ', y =', -1 * g_max.fun)
