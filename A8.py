# 6.52

# a. 
# u_xx = (x[n-1]-2*x[n]+x[n+1]) / (dx**2)
# u_yy = (y[n-1]-2*y[n]+y[n+1]) / (dy**2)
# u_tt = (t[n-1]-2*t[n]+t[n+1]) / (dt**2)

# b. 
# Let h = dx = dy
# Then via substition into the equation u_tt = c*(u_xx+u_yy)
# (t[n-1]-2*t[n]+t[n+1]) / (dt**2) = c*((x[n-1]-2*x[n]+x[n+1]) / (h**2) + (y[n-1]-2*y[n]+y[n+1]) / (h**2))
# (t[n-1]-2*t[n]+t[n+1]) = (c*(dt**2)/(h**2))*(x[n-1]-2*x[n]+x[n+1]+y[n-1]-2*y[n]+y[n+1])
# t[n+1] = (c*(dt**2)/(h**2))*(x[n-1]-2*x[n]+x[n+1]+y[n-1]-2*y[n]+y[n+1]) - (t[n-1]-2*t[n])

# c. 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

c = 1.0       # wave velocity
dx = 0.1      # grid spacing in x direction
dy = 0.1      # grid spacing in y direction
dt = 0.025     # time step (note that the solution is unstable for dt=0.1)
nx = 10       # grid size in x irection
ny = 10       # grid size in y direction

# set up grid
x = np.arange(0, nx * dx, dx)
y = np.arange(0, ny * dy, dy)
X, Y = np.meshgrid(x, y)

# initial conditions
u = np.sin(2 * np.pi * (X - 0.5)) * np.sin(2 * np.pi * (Y - 0.5))    # initial displacement
v = np.zeros((nx, ny))     # initial velocity

# array to store the solution at each time step
u_steps = []

for t in range(1000):
    u_n = u.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # approximate second partial derivatives using finite difference formulas
            u_xx = (u_n[i + 1, j] - 2 * u_n[i, j] + u_n[i - 1, j]) / dx ** 2
            u_yy = (u_n[i, j + 1] - 2 * u_n[i, j] + u_n[i, j - 1]) / dy ** 2
            u_tt = (u_n[i, j] - 2 * u[i, j] + v[i, j]) / dt ** 2
            # update u and v
            u[i, j] = 2 * u[i, j] - v[i, j] + c ** 2 * dt ** 2 * (u_xx + u_yy)
            v[i, j] = u_n[i, j]
    # append the current solution to the array of time steps
    u_steps.append(u.copy())

# set up the figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('displacement')
# plot the initial condition
ax.plot_surface(X, Y, u_steps[0], cmap=cm.coolwarm)

# function to animate the solution
def animate(i):
    ax.clear()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('displacement')
    ax.set_zlim3d(-10, 10)
    ax.plot_surface(X, Y, u_steps[i], cmap=cm.coolwarm)

# create the animation using matplotlib's FuncAnimation
from matplotlib.animation import FuncAnimation
anim = FuncAnimation(fig, animate, frames=range(len(u_steps)), interval=50)
plt.show()

# d.
fig, ax = plt.subplots()
ax.set_xlim(0, 6)
ax.set_ylim(0, 6)
ax.xaxis.set_ticks(range(0, 7))
ax.yaxis.set_ticks(range(0, 7))
ax.set_xticklabels(['i-3', 'i-2', 'i-1', 'i', 'i+1', 'i+2', 'i+3'])
ax.set_yticklabels(['j-3', 'j-2', 'j-1', 'j', 'j+1', 'j+2', 'j+3'])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Finite Difference Stencil')

ax.scatter(3, 2, color='b')
ax.scatter(2, 3, color='b')
ax.scatter(3, 3, color='b')
ax.scatter(4, 3, color='b')
ax.scatter(3, 4, color='r')
ax.grid()

plt.show()