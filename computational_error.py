import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3.14, 100)
fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.plot(x, np.sin(x))
ax.plot(x, (0) + (x) + (0) - (x**3 / 6) + (0) + (x**5 / 120))

((2 ** 0.5) ** 2) - 2 # analytically equal to 0

def f (x, nmax = 100):
  for i in range (nmax):
    x = (x) ** 0.5
  for i in range (nmax):
    x = x**2
  return x

for xin in (5.0, 0.5):
  xout = f (xin)
  print (xin, xout)

a = 1
b = 10 ** 8
c = 1

(-b - (((b)**2) - (4*a*c))**0.5) / (2*a)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math

x = np.linspace(0.5, 1.5, 100)
fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.plot(x, x**6 + 0.1 * np.log(abs(1 + 3 * (1-x))))

# should always be finite
large = 2.0 ** 1021
for i in range(3):
    large = large * 2
    print(large)
    
# should always be > 0
small = 1.0 / 2 ** 1000
for i in range(10):
    small = small / 500
    print(i, 1+small, small)
