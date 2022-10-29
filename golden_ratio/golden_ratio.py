import weierstrass
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: weierstrass.standard(x)
g = lambda x: weierstrass.sinusoidal(x)
phi = (5**.5 + 1) / 2

intervals = []

def golden_ratio(f, x1, x4, tol=1e-6):

    interval_element = (len(intervals), (x1, x4))
    intervals.append(interval_element)

    (x1, x4) = (min(x1, x4), max(x1, x4))
    z = x4 - x1
    if z < tol:
        return (x1, x4)
    else:
        x2 = x4 - z / phi
        x3 = x1 + z / phi

    (fx1, fx2, fx3, fx4) = (f(x1), f(x2), f(x3), f(x4))
    min_fx2 = min((fx1, fx2, fx4))
    min_fx3 = min((fx1, fx3, fx4))

    if min_fx2 == fx2 and min_fx3 == fx3:
        return golden_ratio(f, x2, x3)
    elif min_fx2 == fx2:
        return golden_ratio(f, x1, x3)
    elif min_fx3 == fx3:
        return golden_ratio(f, x2, x4)
    else:
        print('No minimum found on the interval within the specified tolerance.')
        print('Best approximation found:', end=' ')
        return(x1, x4)

print(golden_ratio(g, 2.5, 3.5))

x1 = np.linspace(-2, 2, 100)
y1 = f(x1)
x2 = np.linspace(-1, 6, 100)
y2 = g(x2)

plt.plot(x1, y1)
plt.show()
plt.plot(x2, y2)
plt.show()
