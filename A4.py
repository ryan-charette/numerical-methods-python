# Question 1

import math
import random
from matplotlib import pyplot as plt

f = lambda x: (math.sin(x)) ** 2

def integrate(num_throws):
    under_curve = 0
    total = 0
    for i in range(num_throws):
        xPos = random.uniform(0, 2*math.pi)
        yPos = random.uniform(0, 1.0)
        if f(xPos) <= yPos:
            under_curve += 1
        total += 1
    return 2*math.pi * (under_curve / total)

def graph():

    w = [10**a for a in range(7)]
    x = [integrate(b) for b in w]
    y = [math.log10(c) for c in w]
    z = [math.log10(abs((d - math.pi) / math.pi)) for d in x]

    plt.ylabel('base 10 log of error')
    plt.xlabel('base 10 log of number of points generated')
    plt.scatter(y, z)
    plt.show()

graph()
