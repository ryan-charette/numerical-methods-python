import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3.14, 100)
fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.plot(x, np.sin(x))
ax.plot(x, (0) + (x) + (0) - (x**3 / 6) + (0) + (x**5 / 120))
