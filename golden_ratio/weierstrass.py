import numpy as np

def standard(x, n=21):
    W = 0
    for i in range(n):
        W += 0.5**i * (np.cos(3**i * np.pi * x))
    return W

def sinusoidal(x, n=21):
    W = 0
    for i in range(n):
        W += 0.5**i * (np.sin(2**i * x))
    return W
