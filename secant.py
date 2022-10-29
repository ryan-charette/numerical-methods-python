import numpy as np

f = lambda x: -(x**2) + np.cos(x) + 3*np.sin(x) + 9
f_prime = lambda x: -2*x - np.sin(x) + 3*np.cos(x)
f_prime2 = lambda x: -2 - np.cos(x) - 3*np.sin(x)

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
    
print(taylor(f, f_prime, f_prime2, 3, 10**-6))

# test 1, x0 = 1, f(x) = cos(x), f'(x) = -sin(x), f''(x) = -cos(x)
# expect roots at n*pi / 2 where n is an odd integer
