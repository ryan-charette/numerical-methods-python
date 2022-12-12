
#   A differential equation is an equation that relates the derivative(s)
#   of a function to itself.

#   An analytical solution of a differential equation is a function which
#   when substituted into the differential equation creates a tautology.

#   A numerical solution to a differential equation is a list of ordered 
#   pairs that gives a point-wise approximation to the actual solution.

#   Euler's Method for solving Ordinary Differential Equations:
#   Consider an ordinary linear differential equation of the form
#   y'(x) = f(x, y)
#   We will replace the derivative as a difference equation
#   (y(x + h) - y(x)) / h = f(x, y)
#   y(x + h) = y(x) + h * f(x, y)

#   We will start with an initial value of (x0, y0) and then compute
#   x1 = x0 + h
#   y1 = y0 + h * f(x0, y0)

#   x2 = x1 + h
#   y2 = y1 + h * f(x1, y1)

#   We then compute subsequent pairs of values similarly.

#   Write code to implement the Euler method. The template is given below.
#   Please fill in the code,
 
  import numpy as np
  import matplotlib.pyplot as plt
  
  def euler_1d (f, x0, y0, xmax, h): 
    # set up the domain of the function between x0, xmax, and h
    x = x0 + h*(xmax / x0)

    # zero out the range of the function for the given domain
    y = np.zeros_like (x)

    # fill in the initial conditions
    x[0] = x0
    y[0] = y0

    # now compute the range using Euler's approximation
    for i in range (h*(xmax / x0)):
      y[i] = f(x)

    # return the solution
    return x, y

  def main():
    # define your function f(x, y) as a lambda function
    # specify x even if x does not show up in f(x, y)
    f = 

    # specify the initial conditions
    x0 =
    y0 =

    # specify the max in your domain
    xmax = 

    # specify the increment in x
    dx = 

    # get the solution of the differential equation
    x, y = euler_1d (f, x0, y0, xmax, dx)

    # display the  numerical solution
    plt.plot (x, y, 'red')

    # for the vector (domain) x get the analytical range y_actual
    y_actual = 

    # display the analytical solution
    plt.plot (x, y_actual, 'blue')

    plt.grid()
    plt.show()

    # estimate the error of your numerical solution
    # obtain the root-mean-square of your residuals (y - y_actual)
    std = 

    # print both dx and std to the same precision for comparison

  main()

#   Run the above code on the differential equation
#   y' = -(1/3) y + sin(x) where y(0) = 1

#   The analytical solution is
#   y(x) = (1/10) * (19 * e^(-x/3) + 3 * sin(x) - 9 * cos(x))

#   How does the error of your numerical solution scale with the size
#   of the increment in x? Can you either confirm or not numerically
#   that the error will scale as the square root of the increment?
#   That is if the increment is reduced by a factor of 100 your error
#   decreases by a factor of 10.

#   Euler's method of numerically solving differential equations is both
#   direct and simple. Can we improve upon Euler's method with just a
#   little amount of effort?

#   In Euler's method, we start at (x0, y0) obtain the slope at (x0, y0)
#   and use that slope to obtain (x1, y1). f the function is changing
#   rapidly at that point then this method will not give an accurate value
#   of (x1, y1).

#   Instead we take half a step and obtain the slope at the mid-point. Then
#   use that slope to obtain (x1, y1).
#   m_n = f(x_n, y_n)
#   y_temp = y_n + (h/2) * m_n
#   y_n+1 = y_n + h * f(x_n + h/2, y_temp)
 
#   Implement this method as a new function

  def euler_midpoint (f, x0, y0, xmax, h): 
    # set up the domain of the function between x0, xmax, and h
    x = 

    # zero out the range of the function for the given domain
    y = np.zeros_like (x)

    # fill in the initial conditions
    x[0] = x0
    y[0] = y0

    # now compute the range using Euler's approximation
    for i in range (?):
      y[i] = ?

    # return the solution
    return x, y

#  Test your code. Is your mid-point method giving you smaller errors?
