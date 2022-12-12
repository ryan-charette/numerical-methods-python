import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.animation import FuncAnimation


# Define the parameters
c = 1.0      
dx = 0.1      
dy = 0.1     
dt = 0.025  # the solution is unstable for dt=0.1
nx = 10      
ny = 10
nt = 1000      

# Set up grid
x = np.arange(0, nx * dx, dx)
y = np.arange(0, ny * dy, dy)
X, Y = np.meshgrid(x, y)

# Initial conditions
u = np.sin(2*np.pi*(X-0.5))*np.sin(2*np.pi*(Y-0.5))  # initial displacement
v = np.zeros((nx, ny))  # initial velocity

# Create the solutiona array
u_steps = []

# Implement the finite difference scheme
for t in range(nt):
    u_n = u.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u_xx = (u_n[i + 1, j] - 2 * u_n[i, j] + u_n[i - 1, j]) / dx ** 2
            u_yy = (u_n[i, j + 1] - 2 * u_n[i, j] + u_n[i, j - 1]) / dy ** 2
            u_tt = (u_n[i, j] - 2 * u[i, j] + v[i, j]) / dt ** 2
            u[i, j] = 2 * u[i, j] - v[i, j] + c ** 2 * dt ** 2 * (u_xx + u_yy)
            v[i, j] = u_n[i, j]
    u_steps.append(u.copy())

# Set up the figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('displacement')
ax.plot_surface(X, Y, u_steps[0], cmap=cm.coolwarm)  # plot the initial condition

# Function to animate the solution
def animate(i):
    ax.clear()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('displacement')
    ax.set_zlim3d(-10, 10)
    ax.plot_surface(X, Y, u_steps[i], cmap=cm.coolwarm)

# Create the animation
anim = FuncAnimation(fig, animate, frames=range(len(u_steps)), interval=50)
plt.show()

# Draw the finite difference stencil
plt.subplot(1, 2, 1)
plt.title("Time Step $t_n$")

plt.plot(0, -1, "bo")
plt.text(0, -1, "$U_{i,j-1}^n$")

plt.plot(-1, 0, "bo")
plt.text(-1, 0, "$U_{i-1,j}^n$")

plt.plot(0, 0, "bo")
plt.text(0, 0, "$U_{i,j}^n$")

plt.plot(1, 0, "bo")
plt.text(1, 0, "$U_{i+1,j}^n$")

plt.plot(0, 1, "bo")
plt.text(0, 1, "$U_{i,j+1}^n$")

plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])

plt.grid()

plt.subplot(1, 2, 2)
plt.title("Time Step $t_{n+1}$")

plt.plot(0, 0, "ro")
plt.text(0, 0, "$U_{i,j}^{n+1}$")

plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])

plt.grid()

plt.tight_layout()

plt.show()
