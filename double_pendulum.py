import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# define global constants (initial angle and velocity are set in main)
g = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
m1 = 0.1  # mass of pendulum 1 in kg
m2 = 0.1  # mass of pendulum 2 in kg

t_min = 0 # start time
t_max = 30 # end time
dt = 0.05 # number of steps for runge-kutta 4th order method
t = np.arange(t_min, t_max+dt, dt)

# computation of first and second derivates of theta
def dots(initial_conditions, t):

    theta_1, theta_2, omega_1, omega_2 = initial_conditions

    # calculate angular velocity
    theta_dot_1 = omega_1
    theta_dot_2 = omega_2

    # calculate angular acceleration
    omega_dot_1 = (m2*L1*omega_1*omega_1*sin(theta_2 - theta_1)*
                cos(theta_2 - theta_1) + m2*g*sin(theta_2)*
                cos(theta_2 - theta_1) + m2*L2*omega_2*omega_2*
                sin(theta_2 - theta_1) - (m1 + m2)*g*sin(theta_1))\
                /((m1 + m2)*L1 - m2*L1*cos(theta_2 - theta_1)*
                cos(theta_2 - theta_1))

    omega_dot_2 = (-m2*L2*omega_2*omega_2*sin(theta_2 - theta_1)*
                cos(theta_2 - theta_1) + (m1 + m2)*g*sin(theta_1)*
                cos(theta_2 - theta_1) - (m1 + m2)*L1*omega_1*omega_1*
                sin(theta_2 - theta_1) - (m1 + m2)*g*sin(theta_2))\
                /((L2/L1)*(m1 + m2)*L1 - m2*L1*cos(theta_2 - theta_1)*
                cos(theta_2 - theta_1))

    return [theta_dot_1, theta_dot_2, omega_dot_1, omega_dot_2]

# numerical solution of ODEs with Runge-Kutta Method 4th Order
def runge_kutta(initial_conditions, t, dt):

    # unpack initial conditions
    theta_1, theta_2, omega_1, omega_2 = initial_conditions

    # lists to store values at each time interval
    theta_1_list = []
    theta_2_list = []
    omega_1_list = []
    omega_2_list = []

    for _ in t:

        # compute values of theta and omega at each time interval
        k1_theta_1 = omega_1
        k1_theta_2 = omega_2
        k1_omega_1 = dots(initial_conditions, t)[2]
        k1_omega_2 = dots(initial_conditions, t)[3]

        k2_theta_1 = omega_1 + (dt * k1_omega_1) / 2
        k2_theta_2 = omega_2 + (dt * k1_omega_2) / 2
        k2_omega_1 = dots([theta_1 + (dt * k1_theta_1) / 2, theta_2 + (dt * k1_theta_2) / 2, omega_1 + (dt * k1_omega_1) / 2, omega_2 + (dt * k1_omega_2) / 2], t)[2]
        k2_omega_2 = dots([theta_1 + (dt * k1_theta_1) / 2, theta_2 + (dt * k1_theta_2) / 2, omega_1 + (dt * k1_omega_1) / 2, omega_2 + (dt * k1_omega_2) / 2], t)[3]

        k3_theta_1 = omega_1 + (dt * k2_omega_1) / 2
        k3_theta_2 = omega_2 + (dt * k2_omega_2) / 2
        k3_omega_1 = dots([theta_1 + (dt * k2_theta_1) / 2, theta_2 + (dt * k2_theta_2) / 2, omega_1 + (dt * k2_omega_1) / 2, omega_2 + (dt * k2_omega_2) / 2], t)[2]
        k3_omega_2 = dots([theta_1 + (dt * k2_theta_1) / 2, theta_2 + (dt * k2_theta_2) / 2, omega_1 + (dt * k2_omega_1) / 2, omega_2 + (dt * k2_omega_2) / 2], t)[3]

        k4_theta_1 = omega_1 + (dt * k3_omega_1)
        k4_theta_2 = omega_2 + (dt * k3_omega_2)
        k4_omega_1 = dots([theta_1 + (dt * k3_theta_1), theta_2 + (dt * k3_theta_2), omega_1 + (dt * k3_omega_1), omega_2 + (dt * k3_omega_2)], t)[2]
        k4_omega_2 = dots([theta_1 + (dt * k3_theta_1), theta_2 + (dt * k3_theta_2), omega_1 + (dt * k3_omega_1), omega_2 + (dt * k3_omega_2)], t)[3]

        # update the values
        theta_1 += (dt/6) * (k1_theta_1 + 2 * (k2_theta_1 + k3_theta_1) + k4_theta_1)
        theta_2 += (dt/6) * (k1_theta_2 + 2 * (k2_theta_2 + k3_theta_2) + k4_theta_2)
        omega_1 += (dt/6) * (k1_omega_1 + 2 * (k2_omega_1 + k3_omega_1) + k4_omega_1)
        omega_2 += (dt/6) * (k1_omega_2 + 2 * (k2_omega_2 + k3_omega_2) + k4_omega_2)

        # restrict the domain to [-2pi, 2pi] (improves graph readability) 
        while abs(theta_1) > 2*pi:
            if theta_1 > 0:
                theta_1 -= 2*pi
            else:
                theta_1 += 2*pi
        while abs(theta_2) > 2*pi:
            if theta_2 > 0:
                theta_2 -= 2*pi
            else:
                theta_2 += 2*pi

        # append the results
        theta_1_list.append(theta_1)
        theta_2_list.append(theta_2)
        omega_1_list.append(omega_1)
        omega_2_list.append(omega_2)

    return theta_1_list, theta_2_list, omega_1_list, omega_2_list

def main():

    # theta_1 and theta_2 are the initial angles (radians)
    # omega_1 and omega_2 are the initial angular velocities (radians per second)

    theta_1i = (1/3)*pi 
    theta_2i = (-1/3)*pi
    omega_1i = 1.0
    omega_2i = 0.0

    initial_conditions = [theta_1i, theta_2i, omega_1i, omega_2i]

    # solution to the system of ODEs
    theta_1, theta_2, omega_1, omega_2 = runge_kutta(initial_conditions, t, dt)

    # plot theta_dot_1 vs theta_1
    x1 = np.array(theta_1)
    y1 = np.array(omega_1)

    plt.subplot(1, 3, 1)
    plt.title('$\\dot\\theta_1$ vs $\\theta_1$')
    plt.xlabel('$\\theta_1$')
    plt.ylabel('$\\dot\\theta_1$')
    plt.plot(x1, y1)

    # plot theta_dot_2 vs theta_2
    x2 = np.array(theta_2)
    y2 = np.array(omega_2)

    plt.subplot(1, 3, 2)
    plt.title('$\\dot\\theta_2$ vs $\\theta_2$')
    plt.xlabel('$\\theta_2$')
    plt.ylabel('$\\dot\\theta_2$')
    plt.plot(x2, y2)

    x3 = np.array(t)
    y3a = np.array(theta_1)
    y3b = np.array(theta_2)

    plt.subplot(1, 3, 3)
    plt.title('$\\theta_1, \\theta_2$ vs Time')
    plt.xlabel('Time')
    plt.ylabel('$\\theta_1$ vs $\\theta_2$')
    plt.plot(x3, y3b)
    plt.plot(x3, y3a, color='indianred')


    plt.tight_layout()
    # plt.show()

    # animation
    x1 = L1*sin(theta_1)
    y1 = -L1*cos(theta_1)

    x2 = L2*sin(theta_2) + x1
    y2 = -L2*cos(theta_2) + y1

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2, color='black')
    trace1, = ax.plot([], [], '.-', lw=1, ms=0, color='indianred')
    trace2, = ax.plot([], [], '.-', lw=1, ms=0)
    history_len = 1000
    time_template = 'Time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    history_x1, history_y1 = deque(maxlen=history_len), deque(maxlen=history_len)
    history_x2, history_y2 = deque(maxlen=history_len), deque(maxlen=history_len)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        if i == 1:
            history_x1.clear()
            history_y1.clear()
            history_x2.clear()
            history_y2.clear()

        history_x1.appendleft(thisx[1])
        history_y1.appendleft(thisy[1])

        history_x2.appendleft(thisx[2])
        history_y2.appendleft(thisy[2])

        line.set_data(thisx, thisy)
        trace1.set_data(history_x1, history_y1)
        trace2.set_data(history_x2, history_y2)
        time_text.set_text(time_template % (i*dt))
        return line, trace1, trace2, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(t)),
                                interval=30, blit=True, init_func=init)

    ani.save('double_pendulum.gif', fps=15)
    plt.show()

main()