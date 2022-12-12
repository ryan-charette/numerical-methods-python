import numpy as np
import matplotlib.pyplot as plt

def solve_heat_diffusion(nx, ny, nt, dx, dy, alpha, dt, T0):
    # Define the grid points
    x = np.linspace(0, nx * dx, nx)
    y = np.linspace(0, ny * dy, ny)

    # Create a NumPy array with the initial temperature values
    T = np.array(T0)

    # Compute the coefficient A
    A = alpha * dt / dx**2

    # Loop over the number of time steps
    for i in range(nt):
        # Update the temperature values using the finite difference scheme
        T[1:-1, 1:-1] = (T[1:-1, 1:-1] +
                         A * (T[1:-1, :-2] + T[1:-1, 2:] +
                              T[:-2, 1:-1] + T[2:, 1:-1]))

    # Return the final temperature values
    return T

# Define the parameters
nx = 100
ny = 100
nt = 100
dx = 0.01
dy = 0.01
alpha = 1.22e-3
dt = 1

# Define the initial temperature distribution
T0 = np.zeros((nx, ny))
T0[:, 0] = 100  # Left boundary
T0[:, -1] = 50  # Right boundary

# Solve the heat diffusion equation
T = solve_heat_diffusion(nx, ny, nt, dx, dy, alpha, dt, T0)

# Plot the final temperature distribution
plt.imshow(T, cmap='jet')

# Add axis labels and a title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Temperature Distribution')

# Show the plot
plt.show()
