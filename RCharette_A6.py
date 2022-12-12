import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def Q1():

    # YOUR CODE and PLOTS HERE
    import numpy as np
    import matplotlib.pyplot as plt

    def rk41d(f,x0,t0,tmax,dt,L):
        t = np.arange(t0,tmax+dt,dt)
        x = np.zeros_like(t)
        x[0] = x0
        for n in range(len(x)-1):
            k1 = f(t[n], x[n], L)
            k2 = f(t[n]+(dt/2), x[n]+(dt/2)*k1, L)
            k3 = f(t[n]+(dt/2), x[n]+(dt/2)*k2, L)
            k4 = f(t[n]+dt, x[n]+dt*k3, L)
            x[n+1] = x[n] + (dt/6)*(k1+2*(k2+k3)+k4)
        return t, x

    def euler1d(f,x0,t0,tmax,dt,L):
        t = np.arange(t0, tmax+dt, dt)
        x = np.zeros_like(t)
        x[0] = x0
        for n in range(len(x)-1):
            x[n+1] = x[n] + dt*f(t[n], x[n], L) 
        return t, x

    def midpoint1d(f,x0,t0,tmax,dt,L):
        t = np.arange(t0, tmax+dt, dt)
        x = np.zeros_like(t)
        x[0] = x0
        for n in range(len(x)-1): 
            x[n+1] = x[n] + dt*f(t[n]+dt/2, x[n]+(dt/2)*f(t[n], x[n], L), L)
        return t, x

    f = lambda t, x, L: L*(x-np.cos(t)) - np.sin(t)
    exact = lambda t, L: (1/2)*np.exp(L*t) + np.cos(t)
    x0 = 1.5
    t0 = 0
    tmax = 1
    Llist = -1*10.0**(np.arange(0,7,1))
    DTlist = 10.0**(-np.arange(1,5))
    for L in Llist:
        counter = 0
        error_euler = []
        error_midpoint = []
        error_rk4 = []
        #counter = 0
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
        for dt in DTlist:
        # Here I'm gonna start off by defining how/where I want my plots to be. I got help from a classmate with some of this part
        # ACKNOWLEDGEMENT: THANK YOU TO MY PEERS WHO HELPED ME FIGURE OUT HOW TO DO THE GRAPHS!
        # The given template had 2 rows and 2 columns, and so the plan was to make a sort of 4x4 grid of plots.
        # Where dt of 0.1 would be i 0 j 0 (top left) and then dt 0.01 would be top right (i 0 j 1)
        # and we follow the same pattern for the next dt's just bottom left and bottom right
        # So this loop takes care of the positionings of the columns and rows for the plots
            if counter%4 == 0:
                i, j = 0,0
            elif counter%4 == 1:
                i, j = 0,1
            elif counter%4 == 2:
                i, j = 1,0
            elif counter%4 == 3:
                i,j = 1 ,1

            # Here we get the return arrays from our functions and put them into two variables
            # In reality our time_steps will be the same for all the functions, so no need to do it for all of them but it looks
            # cleaner to do it for all of them
            time_steps, numerical_euler = euler1d(f,x0,t0,tmax,dt,L)
            #print('time_steps', time_steps)
            #print('time_steps.size', time_steps.size)
            error_euler = np.abs(numerical_euler - exact(time_steps,L))
            #print('error_euler', error_euler)
            #print('error_euler.size', error_euler.size)
            time_steps, numerical_midpoint = midpoint1d(f,x0,t0,tmax,dt,L)
            error_midpoint = np.abs(numerical_midpoint - exact(time_steps,L))
            #print('time_steps', time_steps)
            #print('time_steps.size', time_steps.size)
            time_steps, numerical_rk4 = rk41d(f,x0,t0,tmax,dt,L)
            error_rk4 = np.abs(numerical_rk4 - exact(time_steps,L))

            #print('time_steps', time_steps) 
            #print('error_euler', error_euler)
            
            # Here we set the titles and labels for all the plots!
            ax[i][j].set_title("LogLog Plot with lambda L = "+ str(L) +", dt = " +str(dt))
            ax[i][j].set_ylabel("Error axis")
            ax[i][j].set_xlabel("Time steps t")
            ax[i][j].grid()
            
            # Here we use the built in loglog function from matplotlib
            ax[i][j].loglog(time_steps, error_euler, time_steps ,error_midpoint, time_steps,error_rk4)
            ax[i][j].legend(['euler','midpoint','rk4'])


Q1()

def Q2():

    def x2_dot (t, x1, x2):
        return np.cos(t)*x2 + np.sin(t)*x1

    def rk4(F,x0, t0,tmax,dt):
        # define the initial conditions
        x1 = x0
        x2 = 0.56873 # value determined computationally to satisify initial conditions
        
        # define a time step
        h = dt

        # list to store the time steps
        time_steps = []

        # list to store x1s
        x1_list = []

        # list to store x2s
        x2_list = []

        # compute the values of x1 and x2 at each time step
        for t in np.arange(t0, tmax, h):
            k1x1 = x2
            k1x2 = x2_dot(t, x1, x2)
            
            k2x1 = x2 + (h * k1x2) / 2
            k2x2 = x2_dot(t, x1 + (h * k1x1) / 2, x2 + (h * k1x2) / 2)

            k3x1 = x2 + (h * k2x2) / 2
            k3x2 = x2_dot(t, x1 + (h * k2x1) / 2, x2 + (h * k2x2) / 2)

            k4x1 = x2 + (h * k3x2)
            k4x2 = x2_dot(t, x1 + (h * k3x1), x2 + (h * k3x2))

            # update the values of x1 and x2
            x1 += (h/6) * (k1x1 + 2 * (k2x1 + k3x1) + k4x1)
            x2 += (h/6) * (k1x2 + 2 * (k2x2 + k3x2) + k4x2)

            # append the results
            time_steps.append(t)
            x1_list.append(x1)
            x2_list.append(x2)

        # return the results
        return time_steps, x1_list, x2_list

    t, x1, x2 = rk4(x2_dot, 0, 0, 1, 1e-3)

    plt.plot(t, x1)
    plt.show()


def Q3():

     # x''(t) = -x (x^2+y^2)^(-3/2) and y''(t) = -y (x^2+y^2)^(-3/2)
    
    # Initial Conditions:
    # x(0) = 4
    # x'(0) = 0
    # y(0) = 0
    # y'(0) = 0.5

    # (x^2+y^2)^(-3/2) is a constant = 64
    # x(t) = sqrt(16 - y(t)^2)

    # x''(t) = -x/64
    # y''(t) = -y/64

    # x'(t) = -tx/64
    # v_x'(t) = -x/64

    # y'(t) = -ty/64 + 1/2
    # v_y'(t) = -y/64
    
    # x'(t) = -t/64 * sqrt(16 - y(t)^2)
    # v_x'(t) = -1/64 * sqrt(16 - y(t)^2)

    # y'(t) = -ty/64 + 1/2
    # v_y'(t) = -y/64


    # code from lecture

    def x2_dot (t, x1, x2):
        g = 9.8
        return -mu * x2 - (g / L) * np.sin(x1)

    def rk4_solve(t, time, theta_0, w_0):
        # define the initial conditions
        x1 = theta_0
        x2 = w_0
        
        # define a time step
        h = 0.01

        # list to store the time steps
        time_steps = []

        # list to store angles
        x1_list = []

        # list to store the angular velocity
        x2_list = []

        # compute the values of x1 and x2 at each time step
        for t in np.arange(0, time, h):
            k1x1 = x2
            k1x2 = x2_dot(t, x1, x2)
            
            k2x1 = x2 + (h * k1x2) / 2
            k2x2 = x2_dot(t, x1 + (h * k1x1) / 2, x2 + (h * k1x2) / 2)

            k3x1 = x2 + (h * k2x2) / 2
            k3x2 = x2_dot(t, x1 + (h * k2x1) / 2, x2 + (h * k2x2) / 2)

            k4x1 = x2 + (h * k3x2)
            k4x2 = x2_dot(t, x1 + (h * k3x1), x2 + (h * k3x2))

            # update the values of x1 and x2
            x1 += (h/6) * (k1x1 + 2 * (k2x1 + k3x1) + k4x1)
            x2 += (h/6) * (k1x2 + 2 * (k2x2 + k3x2) + k4x2)

            # append the results
            time_steps.append(t)
            x1_list.append(x1)
            x2_list.append(x2)

        # return the results
        return time_steps, x1_list, x2_list

