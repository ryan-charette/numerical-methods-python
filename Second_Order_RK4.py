# Solving Second Order Differential Equations with RK4

import numpy as np
import matplotlib.pyplot as plt

# dtheta/dt = omega

# computation of omega_dot
def w_dot (L, mu, theta, w):
    g = 9.8
    return -mu * w - (g / L) * np.sin(theta)

def rk4_solve(L, mu, time, theta_0, w_0):
    # define the initial conditions
    theta = theta_0
    w = w_0
    
    # define a time step
    h = 0.01

    # list to store the time steps
    time_steps = []

    # list to store angles
    angle = []

    # list to store the angular velocity
    ang_vel = []

    # compute the values of theta and w at each time step
    for t in np.arange(0, time, h):
        k1t = w
        k1w = w_dot(L, mu, theta, w)
        
        k2t = w + (h * k1w) / 2
        k2w = w_dot(L, mu, theta + (h * k1t) / 2, w + (h * k1w) / 2)

        k3t = w + (h * k2w) / 2
        k3w = w_dot(L, mu, theta + (h * k2t) / 2, w + (h * k2w) / 2)

        k4t = w + (h * k3w)
        k4w = w_dot(L, mu, theta + (h * k3t), w + (h * k3w))

        # update the values of theta and w
        theta += (h/6) * (k1t + 2 * (k2t + k3t) + k4t)
        w += (h/6) * (k1w + 2 * (k2w + k3w) + k4w)

        # append the results
        time_steps.append(t)
        angle.append(theta)
        ang_vel.append(w)

    # return the results
    return time_steps, angle, ang_vel

def main():
    # define initial conditions
    L = 2.0
    mu = 0.1
    theta_0 = np.pi / 3
    w_0 = 0
    time_interval = 20

    time, theta, w = rk4_solve(L, mu, time_interval, theta_0, w_0)

    # plot theta vs time
    x = np.array(time)
    y = np.array(theta)

    plt.subplot(1, 3, 1)
    plt.title('Theta vs Time')
    plt.xlabel('Time')
    plt.ylabel('Theta')
    plt.plot(x, y)

    # plot w vs time
    x = np.array(time)
    y = np.array(w)

    plt.subplot(1, 3, 2)
    plt.title('Angular Velocity vs Time')
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity')
    plt.plot(x, y)

    # plot w vs theta

    x = np.array(theta)
    y = np.array(w)

    plt.subplot(1, 3, 3)
    plt.title('Angular Velocity vs Theta')
    plt.xlabel('Theta')
    plt.ylabel('Angular Velocity')
    plt.plot(x, y)

    plt.show()

main()