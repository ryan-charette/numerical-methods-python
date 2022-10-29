# Part 1: Some basics
import numpy as np
import numpy.linalg as la

# What is a vector and how can it be represented in python?
u = np.array([1,2,3])

print("We are working on u: \n", u)
print("The shape of u is: ", u.shape)
print("The len of u is: ", len(u))

# What are the defining properties of a vector?
#1 
print("First property is: ____, which is computed in python to be ", " for u")
#2
print("Second property is: ____, which is computed in python to be ", " for u")

# So what is the shape of u after we transpose it?
uT = u.transpose()
print("If I transpose u it is now: \n", uT)
print("The shape of u is: ", uT.shape)

v = np.array([[1,2,3]])
print("\nNow we are working on v: \n", v)
print("The shape of v is: ", v.shape)
print("The len of v is: ", len(v))

vT = v.transpose()
print("If I transpose v it is now: \n", vT)
print("With shape", vT.shape)

# What is the first column of this matrix?
M = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
print("\nNow we are working on the matrix M: \n", M)
print("The shape of M is: ", M.shape)
print("The len of M is: ", len(M))

# What is the first column of this matrix?
MT = M.transpose()
print("If I transpose M it is now: \n", MT)
print("With shape", MT.shape)

# Part 2: Matrices, indexing, slicing, and at least one nuance I wish I knew
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(A)
print("The first column of A is \n",A[:,0])
print("The second row of A is \n",A[1,:])
print("The top left 2x2 sub matrix of A is \n",A[:-1,:-1])
print("The bottom right 2x2 sub matrix of A is \n",A[1:,1:])
u = np.array([1,2,3,4,5,6])
print("The first 3 entries of the vector u are \n",u[:3])
print("The last entry of the vector u is \n",u[-1])
print("The last two entries of the vector u are \n",u[-2:])

# What do you expect to happen if we do the operation below?
A + 1

# So what is A now?
print(A)

# What is A * A and A * v?
v = np.array([[1, 2, 3]])
print("A * A is: \n", A * A)
print("A * v is: \n", A * v)

# If we convert A to a matrix, or define it that way from the start, what is A * A and A * v now?
print(type(A))
A = np.matrix(A)
print(type(A))
print("A * A is: ", A * A)
print("A * v is: ", A * v)

# What went wrong?
print("A: ", A)
print("v: ", v)
print("")
print("Size of A: ", A.shape)
print("Size of v: ", v.shape)
# So what if we now transpose v?
A * v.transpose()

# Part 3: Geometric interpretations and 3 potential ways to get vector dot products
import matplotlib.pyplot as plt

u = np.array([1, 3])
v = np.array([5, 0])
origin = np.array([0,0])

# Plot the vectors
plt.quiver(*origin, *u, color=['r'], scale=21)
plt.quiver(*origin, *v, color=['b'], scale=21)
