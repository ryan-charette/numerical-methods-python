import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make the X, Y meshgrid.
xs = np.linspace(-1, 1, 50)
ys = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(xs, ys)

print(X)
print(Y)

# Set the z axis limits so they aren't recalculated each frame.
ax.set_zlim(-2, 2)

# Begin plotting.
wframe = None
for phi in np.linspace(0, 180. / np.pi, 1000):
    # If a line collection is already remove it before drawing.
    if wframe:
        wframe.remove()
    # Generate data.
    Z = np.sin(2*np.pi*(X-0.5*phi))*np.sin(2*np.pi*(Y-0.5*phi))
    # Plot the new wireframe and pause briefly before continuing.
    wframe = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, color='k')
    plt.pause(.01)