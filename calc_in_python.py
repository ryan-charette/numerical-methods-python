from cmath import pi
import numpy as np
import matplotlib.pyplot as plt

def first_deriv (f, a, b, n):
  # divide the interval into n equal parts
  x = np.linspace (a, b, n + 1)

  # the incremental value is
  h = x[1] - x[0]

  # store the derivatives
  df = []

  # use centered difference f'(x) = (f(x + h) - f(x - h)) / (2 * h)
  # set up the loop correctly
  for value in x:
    derivative = (f(value + h) - f(value - h)) / (2 * h)
    df.append(derivative)

  # return df
  return df

def main():
  # define the function whose derivative is required
  f = lambda x: np.sin(x)

  # exact derivative for the above function for comparison
  exact_df = lambda x: np.cos(x)

  # define the interval [0, 2 * pi]
  a = 0
  b = 2 * np.pi
  
  # change this parameter to see how close you can get to the exact solution
  n = 100

  # get the first derivative numerically
  df = first_deriv (f, a, b, n)

  # now plot 3 curves
  # f(x) = sin(x)
  # f'(x) = cos(x)
  # numerical approximation of f'(x)
  x = np.linspace(a, b, n + 2)
  plt.plot (x, f(x), 'blue', x, exact_df(x), 'red', x[0:-1], df, 'black')
  plt.grid()
  plt.legend(['f(x) = sin(x)', 'exact first deriv', 'approx first deriv'])
  plt.show()

main()

# Now add a function that computes the 2nd derivative

def second_deriv (f, a, b, n):
  # divide the interval into n equal parts
  x = np.linspace (a, b, n + 1)

  # the incremental value is
  h = x[1] - x[0]

  # store the second derivatives
  df2 = []

  # use centered difference f''(x) = (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
  # set up the loop correctly
  for value in x:
    derivative = (f(value + h) - 2*f(value) + f(value - h)) / (h * h)
    df2.append(derivative)

  # return df2
  return df2

# Plot the second derivative

def main2():

  f = lambda x: np.sin(x)

  # exact second derivative for the above function for comparison
  exact_df2 = lambda x: (-1) * np.sin(x)
 
  # define the interval [0, 2 * pi]
  a = 0
  b = 2 * np.pi
  
  # change this parameter to see how close you can get to the exact solution
  n = 100

  # get the second derivative numerically
  df2 = second_deriv (f, a, b, n)

  # now plot 3 curves
  # f(x) = sin(x)
  # f''(x) = -sin(x)
  # numerical approximation of f''(x)
  x = np.linspace(a, b, n + 2)
  plt.plot (x, f(x), 'blue', x, exact_df2(x), 'red', x[0:-1], df2, 'black')
  plt.grid()
  plt.legend(['f(x) = sin(x)', 'exact second deriv', 'approx second deriv'])
  plt.show()

main2()
