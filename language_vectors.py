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

# What does it mean to project one vector onto another? 
# What would be the visual representation of the projection of the red vector onto the blue vector?
print("Shape of u is: ", u.shape)
print("First dot product method: ", u[0] * v[0] + u[1] * v[1] + u[2] * v[2])
print("Second dot product method: ", u.dot(v))

u = np.matrix([1, 2, 3])
v = np.matrix([1, 2, 3])
print("Shape of u is: ", u.shape)

print("Third dot product method: ", u * v.transpose())

# Part 4: Lets play with some word vectors
# More reading here: https://nlp.stanford.edu/projects/glove/
# ASSUMPTIONS: All word embeddings are unit vectors 

# The loading of the glove model is shamelessly reproduced from here: 
# https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
import numpy as np
import numpy.linalg as la
import random

# Load in the glove model
def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model
gloveModel = load_glove_model("glove.6B.50d-relativized.txt")

# Make an array containing all the glove word embeddings
all_embed_matrix = np.array(list(gloveModel.values()))

# Ensure our embeddings are all unit vectors
all_embed_matrix = all_embed_matrix / np.linalg.norm(all_embed_matrix, axis=-1)[:, np.newaxis]

print("Shape of all_embed_matrix is: ", all_embed_matrix.shape)

# Make a dictionary of keys and their indices for the word embeddings
all_keys = list(gloveModel.keys())
name2ind = {k: v for v, k in enumerate(all_keys)}
ind2name = {v: k for k, v in name2ind.items()}


# Examples:
print("The index of red in the word embedding matrix is: ", name2ind['red'])
print("The index of apple in the word embedding matrix is: ", name2ind['apple'])
print("The magnitude of the red word embedding vector is: ", la.norm(all_embed_matrix[name2ind['red']]))
print("The magnitude of the apple word embedding vector is: ", la.norm(all_embed_matrix[name2ind['apple']]))
print("The index 545 maps to the word: ", ind2name[545])
print("The index 2385 maps to the word: ", ind2name[2385])

# Colors
red_vec = all_embed_matrix[name2ind['red']]
green_vec = all_embed_matrix[name2ind['green']]
yellow_vec = all_embed_matrix[name2ind['yellow']]
orange_vec = all_embed_matrix[name2ind['orange']]

# Fruits
apple_vec = all_embed_matrix[name2ind['apple'], :]
lemon_vec = all_embed_matrix[name2ind['lemon'], :]

# More random
computer_vec = all_embed_matrix[name2ind['computer'], :]

print(orange_vec.shape)

# Example of dot product to find similarity between two words:
print("Angle between red and yellow vec is: ", np.arccos(red_vec.dot(yellow_vec)), " radians")
print("Angle between red and lemon vec is: ", np.arccos(red_vec.dot(lemon_vec)), " radians")
print("Angle between lemon and computer vec is: ", np.arccos(lemon_vec.dot(computer_vec)), " radians")

