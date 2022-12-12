import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# definition of the ode
def calc_theta_double_dot (theta, theta_dot):
    g = 9.8 # acceleration due to gravity
    L = 2.0 # length of the pendulum
    mu = 0.1 # resistance due to air resistance

    return -mu * theta_dot - (g/L) * np.sin(theta)

# solution of the differential equation
def theta_solve (t, theta_0, theta_dot_0):
    # define the initial conditions
    theta = theta_0
    theta_dot = theta_dot_0

    # define time step
    delta_t = 0.01

    # list to store time interval
    time_int = []

    # list to store angle
    angle = []

    # list to store angular velocity
    ang_vel = []

    # compute the values of theta and theta_dot at each time step
    for time in np.arange(0, t, delta_t):
        theta_double_dot = calc_theta_double_dot(theta, theta_dot)
        theta_dot += theta_double_dot * delta_t
        theta += theta_dot * delta_t

        # append the results
        time_int.append(time)
        angle.append(theta)
        ang_vel.append(theta_dot)

    # return the results
    return time_int, angle, ang_vel

def main():
    # define inital conditions
    theta_0 = np.pi / 3
    theta_dot_0 = 0
    t = 10

    time, theta, theta_dot = theta_solve(t, theta_0, theta_dot_0)

    # plt.plot(time, theta)
    # plt.show()

    # plt.plot(theta, theta_dot)
    # plt.show()

    fig, ax = plt.subplots()
    line, = ax.plot(theta, theta_dot)

    def animate(i):
        line.set_xdata(theta[i])
        return line,

    ani = animation.FuncAnimation(fig, animate, interval=20, blit=True)

    plt.show()


main()

