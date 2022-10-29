# 2.57

import numpy as np

f = lambda x: -(x**2) + np.cos(x) + 3*np.sin(x) + 9
f_prime = lambda x: -2*x - np.sin(x) + np.cos(x)

def bisection(f, a, b, tol):
    
    m = (a + b) / 2
    
    if np.abs(f(m)) < tol:
        return m
    elif f(a) * f(m) > 0:
        return bisection(f, m, b, tol)
    else:
        return bisection(f, a, m, tol)
    
print("bisection:", bisection(f, 2, 4, 10**-6))

def regula_falsi(f, a, b, tol):
    
    m = (f(a) - f(b)) / (a - b)
    c = (-f(a) / m) + a
    
    if np.abs(f(c)) < tol:
        return c
    elif f(a) * f(c) > 0:
        return regula_falsi(f, c, b, tol)
    else:
        return regula_falsi(f, a, c, tol)
    
print("regula falsi:", regula_falsi(f, 2, 4, 10**-6))

def newton(f, f_prime, x, tol):
    
    if np.abs(f(x)) < tol:
        return x
    else:
        new_x = x - (f(x))/(f_prime(x))
        return newton(f, f_prime, new_x, tol)
    
print("newton:", newton(f, f_prime, 3, 10**-6))

def secant(f, x, x2, tol):
    
    d = (f(x) - f(x2)) / (x - x2)
    
    if np.abs(f(x)) < tol:
        return x
    else:
        new_x = x - (f(x))/(d)
        return secant(f, new_x, d, tol)
    
print("secant:", secant(f, 2, 4, 10**-6))  
    
# 2.60
# a
# for N = 2, x = (-f'(x0)+-sqrt((f'(x0))**2-2f''(x0)(f(x0)-f'(x0)x0+f''(x0)(x0-x0**2/2))))/f''(x0)

import numpy as np

f = lambda x: -(x**2) + np.cos(x) + 3*np.sin(x) + 9
f_prime = lambda x: -2*x - np.sin(x) + 3*np.cos(x)
f_prime2 = lambda x: -2 - np.cos(x) - 3*np.sin(x)

g = lambda x: np.sin(x)
g_prime = lambda x: np.cos(x)
g_prime2 = lambda x: -np.sin(x)

h = lambda x: np.log(x) - 1
h_prime = lambda x: 1/x
h_prime2 = lambda x: -1/x**2

def taylor(f, f_prime, f_prime2, x, tol):
    
    a = f_prime2(x) / 2
    b = f_prime(x) - f_prime2(x)*x
    c = f(x) - f_prime(x)*x + (f_prime2(x)/2)*x**2

    if np.abs(f(x)) < tol:
        return x
    else:
        root1 = (-b + (b**2 - 4*a*c)**0.5) / (2*a)
        root2 = (-b - (b**2 - 4*a*c)**0.5) / (2*a)
        if np.abs(root1 - x) < np.abs(root2 - x):
            new_x = root1
        else:
            new_x = root2
        return taylor(f, f_prime, f_prime2, new_x, tol)
    
# b
print(taylor(f, f_prime, f_prime2, 3, 10**-6)) # expect ~ 2.937
print(taylor(g, g_prime, g_prime2, 3, 10**-6)) # expect pi
print(taylor(h, h_prime, h_prime2, 3, 10**-6)) # expect e

# c

import math
error_measure = []

def error(f, f_prime, f_prime2, x, tol):
    
    a = f_prime2(x) / 2
    b = f_prime(x) - f_prime2(x)*x
    c = f(x) - f_prime(x)*x + (f_prime2(x)/2)*x**2

    if np.abs(f(x)) < tol:
        return x
    else:
        root1 = (-b + (b**2 - 4*a*c)**0.5) / (2*a)
        root2 = (-b - (b**2 - 4*a*c)**0.5) / (2*a)
        if np.abs(root1 - x) < np.abs(root2 - x):
            new_x = root1
        else:
            new_x = root2
        error_measure.append(math.pi - new_x)
        return error(f, f_prime, f_prime2, new_x, tol)
    
final = error(g, g_prime, g_prime2, 3, 10**-6)

import matplotlib as mpl
import matplotlib.pyplot as plt

print(error_measure)

y = error_measure
x = range(1, len(error_measure) + 1)

plt.ylabel('absolute error')
plt.xlabel('iteration')
plt.scatter(x, y)
plt.show()

y = [np.log2(x) for x in error_measure]
x = range(1, len(error_measure) + 1)

plt.ylabel('base 2 log of absolute error')
plt.xlabel('iteration')
plt.scatter(x, y)
plt.show()

# d : Pros of this method include that it finds a solution in very few iterations.
# Cons include that it only works on differentiable functions, that is has an added margin of error
# due to the error associated with Taylor series approximations, and that it requires more work than
# other solutions to be done by hand
