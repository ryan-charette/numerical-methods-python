import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
c = 1
dx = 0.1
dt = 0.1
nx = 50
nt = 50

# Set up grid
x = np.arange(0, nx * dx, dx)
t = np.arange(0, nt * dt, dt)

# Initial conditions
u0 = np.exp(-(x-4)**2)
uL = 0

# Compute the coefficient a
a = c * dt / dx

# Create the solution array
u = np.zeros((nx, nt))
u[:, 0] = u0

# Implement the finite difference scheme
for n in range(nt-1):
    for i in range(1, nx-1):
        u[i, n+1] = u[i, n] - a * (u[i, n] - u[i-1, n])
    u[0, n+1] = 0
    u[nx-1, n+1] = 0

# Compute the analytical solution
u_exact = np.zeros((nx, nt))
for n in range(nt):
    for i in range(nx):
        u_exact[i, n] = np.exp(-(x[i] - c * t[n] - 4)**2)

# Plot the numerical and analytical solutions
for n in range(0, nt, 10):
    plt.plot(x, u[:, n], 'bo', label='numerical')
    plt.plot(x, u_exact[:, n], 'r--', label='analytical')
    plt.title('Numerical and analytical solutions at t = {}'.format(t[n]))
    plt.legend()
    plt.show()
