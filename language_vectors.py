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
