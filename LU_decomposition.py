import numpy as np

# function takes input a square matrix A and returns two matrices L and U.
def LU (A):
  # get the dimension of A
  n = A.shape[0]

  # create the identity matrix for L
  L = np.matrix (np.identity(n))

  # start a placeholder for the matrix U
  U = A

  # build L and U
  for j in range (0, n - 1):
    for i in range (j + 1, n):
      mult = A[i,j] / A[j,j]
      A[i, j+1:n] = A[i, j+1:n] - mult * A[j, j+1:n]
      U[i, j+1:n] = A[i, j+1:n]
      L[i,j] = mult
      U[i,j] = 0

  # return the matrices L and U
  return L, U

# * Test this code for the matrices that we worked on in class
La = np.matrix([[2, 4, 3, 5], [-4, -7, -5, -8], [6, 8, 2, 9], [4, 9, -2, 14]])

Lb = np.matrix([[2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]])

Lc = np.matrix([[1, 1, -1], [1, -2, 3], [2, 3, 1]])

print(LU(La))
print(LU(Lb))
print(LU(Lc))

# * Here is the code for the forward substitution. This solves the equation
#   L * Y = B

def lsolve (L, b):
  # make sure that L is the right data type
  L = np.matrix (L)

  # get the size of array b
  n = b.size

  # create a placeholder for y
  y = np.matrix (np.zeros((n, 1)))

  # solve for y
  for i in range (n):
    y[i] = b[i]
    for j in range (i):
      y[i] = y[i] - L[i,j] * y[j]

  # return the solution for y
  return y

# * Test your code for the values of b corresponding to the matrices
#   defined above:
a = np.matrix([1, 2, 3, 4])
b = np.matrix([-4, 3, 9, 7])
c = np.matrix([4, -6, 7])
