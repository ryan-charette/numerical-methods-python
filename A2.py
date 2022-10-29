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
    
