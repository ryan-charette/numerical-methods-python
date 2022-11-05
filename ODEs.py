import numpy as np
import matplotlib.pyplot as plt

# Euler's Method

def euler1d(f,x0,t0,tmax,dt):
    t = np.arange(t0, tmax, dt) # set up the domain based on t0, tmax, and dt
    # next set up an array for x that is the same size as t
    x = np.zeros_like(t)
    x[0] = x0 # fill in the initial condition
    for n in range(len(x)-1): # think about how far we should loop
        x[n+1] = x[n] + dt*f(t[n], x[n]) # advance the solution forward in time with Euler
    return t, x

# put the f(t,x) function on the next line 
# (be sure to specify t even if it doesn't show up in your ODE)
f = lambda t, x: -x/3+np.sin(t) # your function goes here
x0 = 1 # initial condition
t0 = 0 # initial time
tmax = 10 # final time (your choice)
dt = .1 # Delta t (your choice, but make it small)
t, x = euler1d(f,x0,t0,tmax,dt)
plt.plot(t,x,'b-')
# plt.grid()
# plt.show()

# Midpoint Method

def midpoint1d(f,x0,t0,tmax,dt):
    t = np.arange(t0, tmax, dt) # build the times
    x = np.zeros_like(t) # build an array for the x values
    x[0] = x0 # build the initial condition
    # On the next line: be careful about how far you're looping
    for n in range(len(x)-1): 
        x[n+1] = x[n] + dt*f(t[n]+dt/2, x[n]+(dt/2)*f(t[n], x[n])) # The interesting part of the code goes here.
    return t, x

f = lambda t, x: -x/3+np.sin(t) # your ODE right hand side goes here
x0 = 1 # initial condition
t0 = 0
tmax = 10 # ending time (up to you)
dt = .1 # pick something small
t, x = midpoint1d(f,x0,t0,tmax,dt)
plt.plot(t,x,'r-')
# plt.grid()
# plt.show()

# Runge-Kutta 4 Method

def rk41d(f,x0,t0,tmax,dt):
    t = np.arange(t0,tmax+dt,dt)
    x = np.zeros_like(t)
    x[0] = x0
    for n in range(len(x)-1):
        # the interesting bits of the code go here
        k1 = f(t[n], x[n])
        k2 = f(t[n]+(dt/2), x[n]+(dt/2)*k1)
        k3 = f(t[n]+(dt/2), x[n]+(dt/2)*k2)
        k4 = f(t[n]+dt, x[n]+dt*k3)
        x[n+1] = x[n] + (dt/6)*(k1+2*(k2+k3)+k4)
    return t, x

f = lambda t, x: -x/3+np.sin(t)
x0 = 1 # initial condition
t0 = 0
tmax = 10 # your choice
dt = .1 # pick something reasonable
t, x = rk41d(f,x0,t0,tmax,dt)
plt.plot(t,x,'g-')
plt.grid()
plt.show()
