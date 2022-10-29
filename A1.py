# 1.53

# a
# 0.00011001100110011001100 = 2**-4 + 2**-5 + 2**-8 + 2**-9 + 2**-12 + 2**-13 + 2**-16 + 
# 2**-17 + 2**-20 + 2**-21
# = (2**17)/(2**21) + (2**16)/(2**21) + (2**13)/(2**21) + (2**12)/(2**21) + (2**9)/(2**21) +
# (2**8)/(2**21) + (2**5)/(2**21) + (2**4)/(2**21) + (2**1)/(2**21) + (2**0)/(2**21)
# = (131072 + 65536 + 8192 + 4096 + 512 + 256 + 32 + 16 + 2 + 1)/(2097152)
# = 209715 / 2097152 ~= 
print('209715') # numerator
print('2097152') # denominator

# b
# (209715 / 2097152) - (1 / 10) = (2097150 / 20971520) - (2097152 / 20971520) = -2 / 20971520
# = -1 / 10485760
print(-1 / 10485760) # approximate decimal represenation

# c
# 100 hours = 6,000 minutes = 360,000 seconds = 3,600,000 tenths of a second
# -1 / 10485760 * 3600000 = -3600000 / 10485760 tenths of a second = -360000 / 10485760 seconds
# = -1125 / 32768 seconds
print(-1125 / 32768) # approximate decimal representation

# d 
# 3750 mph = 62.5 mi/min = 25/24 mi/second
# 25 / 24 * 1125 / 32768 = 28125 / 786432
# = 9375 / 262144 miles
print(9375 / 262144) # approximate decimal representation

import math
# 1.54

# f(x) = 1/ln(x)
# f(e) / 0! * (x - e)**0 = 1
f0 = 1

# f'(x) = -1 / x(ln(x))**2
# f'(e) / 1! * (x - e)**1 = (e - x) / e
f1 = (math.e - 3) / math.e

# f''(x) = [ln(x) + 2] / x**2(ln(x))**3
# f''(e) / 2! * (x - e)**2 = {[ln(e) + 2] / 2(e**2)} * (x - e)**2
f2 = ((math.log(math.e) + 2) / (2*(math.e)**2)) * (3 - math.e)**2

# f'''(x) = (-2*(ln(x))**2 + 3*ln(x) + 3) / ((x**3)*(ln(x)**4))
# f'''(e) / 3! * (x - e)**3 = (-14 / (6 * e**4)) * (x - e)**3
f3 = (-7/(3*math.e**3)) * (3 - math.e)**3

# f''''(x) = ((6*(ln(x))**3) + (22*(ln(x))**2) + (36*(ln(x))) + 24) / ((x**4) * ((ln(x))**5))
# f''''(e) / 4! * (x - e)**4 = (88 / (24 * e**4)) * (x - e)**4
f4 = (11/(3*math.e**4)) * (3 - math.e)**4

print(f0 + f1 + f2 + f3 + f4)

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#1.55

# a
# f(x) = 1 / (1 + x)
# Taylor series: Sum from k=1 to n of (-1)**(k+1) * (1 / x)**k

# b
# Taylor series of f(x ** 2) = Sum from k=1 to n of (-1)**(k+1) * (1 / x**2)**k
x = np.linspace(0, 5, 100)
fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.plot(x, 1/(x+2), label='0th order')
plt.plot(x, (x+2)**-1 - (x+2)**-2 + (x+2)**-3, label='2nd order')
plt.plot(x, (x+2)**-1 - (x+2)**-2 + (x+2)**-3 - (x+2)**-4 + (x+2)**-5, label='4th order')
plt.plot(x, (x+2)**-1 - (x+2)**-2 + (x+2)**-3 - (x+2)**-4 + (x+2)**-5 - (x+2)**-6 + (x+2)**-7, label='6th order')
plt.legend()
plt.show()

#c
# Taylor series for the integral of 1 / (x + 2) = 
# Sum from k=1 to n of (-1)**(k+1) * (x**(2k - 1))/(2k - 1)

# d
true_pi = 3.141593
estimated_pi = 0
n = 1
m = 1

while round(4*estimated_pi, 6) != true_pi:
    estimated_pi += (m/n)
    n += 2
    m *= -1
    
print(round(4*estimated_pi, 6))
