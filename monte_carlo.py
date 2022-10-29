import math
import random

# This function creates a circumscribed square around a circle of radius 1
# Points are randomly placed within the square
# The ratio of the two shapes' areas is approximated by counting the number of points in each
# Area of a circle = pi*r^2 and area of the square = 4*r^2
# Thus, A_circle/A_square = pi/4 --> pi = 4 * A_circle/A_square

def computePI(num_throws):
    in_or_on_circle = 0
    total = 0
    for i in range(num_throws):
        xPos = random.uniform(-1.0, 1.0)
        yPos = random.uniform(-1.0, 1.0)
        if math.hypot(xPos, + yPos) <= 1:
            in_or_on_circle += 1
        total += 1
    return 4 * (in_or_on_circle / total)

def main():
    p1 = computePI(100)
    p2 = computePI(1_000)
    p3 = computePI(10_000)
    p4 = computePI(100_000)
    p5 = computePI(1_000_000)

    print('Num = 100, , calculated pi = ,', p1, 'error =', (p1 - math.pi)/math.pi)
    print('Num = 1,000, calculated pi = ,', p2, 'error =', (p2 - math.pi)/math.pi)
    print('Num = 10,000, calculated pi = ,', p3, 'error =', (p3 - math.pi)/math.pi)
    print('Num = 100,000, calculated pi = ,', p4, 'error =', (p4 - math.pi)/math.pi)
    print('Num = 1,000,000, calculated pi = ,', p5, 'error =', (p5 - math.pi)/math.pi)

main()
