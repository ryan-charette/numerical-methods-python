import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the parameters
nx = 100
ny = 100
nt = 100
dx = 0.01
dy = 0.01
dt = 0.01
D = 2.5e-3
k1 = 0.1
k2 = 1000

# Compute the coefficient a
a = D * dt / (dx * dy)

# Set up grid
x = np.linspace(0, nx * dx, nx)
y = np.linspace(0, ny * dy, ny)
X, Y = np.meshgrid(x, y)

# Initial conditions
P = np.exp(-((X-0.5)**2+(Y-0.5)**2)/0.05)
P[nx//2, ny//2] = 100

# Set up plot
fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Population Diffusion')

# Function to animate solution
def update(i):
    # Implement the finite difference scheme
    for i in range(1, nt-1):
        for j in range(1, nt-1):
            P[i, j] = P[i, j] + k1 * P[i, j] * (1 - P[i, j] / k2) + a * \
            (P[i+1, j] + P[i-1, j] + P[i, j+1] + P[i, j-1] - 4 * P[i, j])

    # Update the plot
    im = ax.imshow(P, cmap='viridis')

    return im,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()
