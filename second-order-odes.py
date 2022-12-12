import numpy as np
import matplotlib.pyplot as plt

t0 = 0
tmax = 1e-3
dt = 1e-6
x0 = np.array([[1], [1]])

f = lambda t: np.array([[1, dt], 
               [-16*dt, 1]])

def euler(f, x0, t0, tmax, dt):
    t = np.arange(t0, tmax, dt)
    x = np.zeros((len(t), 2))
    x[0, :] = x0.T
    for j in range(0, len(t)-1):
        x[j+1, :] = np.dot(f(t[j]),x[j, :])
    return t, x

t, x = euler(f, x0, t0, tmax, dt)    
plt.plot(t,x[:,0],'b-')
plt.show()