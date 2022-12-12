import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
nx = 10
ny = 10
x = np.linspace(-1,1,nx)
y = np.linspace(-1,1,ny)

# Set up the matrices
A = np.zeros((nx*ny, nx*ny))
b = np.zeros(nx*ny)

# Initial conditions
f = lambda x, y: -20*np.exp(-((x-0.5)**2 + (y-0.5)**2)/0.05)

# Implement the finite difference scheme
for i in range(nx):
    for j in range(ny):

        A[i*ny + j, i*ny + j] = -4
        if i > 0:
            A[i*ny + j, (i-1)*ny + j] = 1
        if i < nx - 1:
            A[i*ny + j, (i+1)*ny + j] = 1
        if j > 0:
            A[i*ny + j, i*ny + j - 1] = 1
        if j < ny - 1:
            A[i*ny + j, i*ny + j + 1] = 1

        # Set the boundary conditions
        if i == 0:
            A[i*ny + j, i*ny + j] = 1
            b[i*ny + j] = 0
        elif i == nx - 1:
            A[i*ny + j, i*ny + j] = 1
            b[i*ny + j] = 0
        elif j == 0:
            A[i*ny + j, i*ny + j] = 1
            b[i*ny + j] = -np.sin(np.pi*i)
        elif j == ny - 1:
            A[i*ny + j, i*ny + j] = 1
            b[i*ny + j] = 0
        else:
            # Compute the right-hand side for the grid point (x,y)
            b[i*ny + j] = f(x[i], y[j])

# Solve the system of linear equations
u = np.linalg.solve(A, b)

# Reshape the solution array into a 2D grid
u_grid = u.reshape((nx,ny))

# Generate the contour plot
plt.contourf(x, y, u_grid)
plt.colorbar()

plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution of the 2D Poisson equation')

plt.show()