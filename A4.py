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

# Question 2

import numpy as np
from scipy.optimize import minimize 
import pandas as pd

URL1 = 'https://raw.githubusercontent.com/NumericalMethodsSullivan'
URL2 = '/NumericalMethodsSullivan.github.io/master/data/'
URL = URL1+URL2

datasetA = np.array( pd.read_csv(URL+'Exercise3_datafit5.csv') )

xdata_list = [datasetA[n][0] for n in range(len(datasetA))]
ydata_list = [datasetA[n][1] for n in range(len(datasetA))]
xdata = np.asarray(xdata_list)
ydata = np.asarray(ydata_list)

# Dataset A: exponential

def SSRes(parameters):
  yapprox = parameters[0]* np.e ** (xdata * parameters[1])
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(-2, 4, 100)
y = BestParameters.x[0] * np.e ** (x * BestParameters.x[1])
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset A, Best Fit Exponential')
plt.show()

print('Dataset A, Best Fit Exponential: y =', BestParameters.x[0], 'e ^ x(', BestParameters.x[1], ')')

# Dataset A: quadratic

def SSRes(parameters):
  yapprox = parameters[0]*xdata**2 + parameters[1]*xdata + parameters[2]
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(-2, 4, 100)
y = BestParameters.x[0]*x**2 + BestParameters.x[1]*x + BestParameters.x[2]
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset A, Best Fit Quadratic')
plt.show()

print('Dataset A, Best Fit Quadratic: y =', BestParameters.x[0], 'x^2 +', BestParameters.x[1], 'x +', BestParameters.x[2])

datasetB = np.array( pd.read_csv(URL+'Exercise3_datafit6.csv') )
xdata_list = [datasetB[n][0] for n in range(len(datasetB))]
ydata_list = [datasetB[n][1] for n in range(len(datasetB))]
xdata = np.asarray(xdata_list)
ydata = np.asarray(ydata_list)

# Dataset B: linear

def SSRes(parameters):
  yapprox = parameters[0]* xdata + parameters[1]
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(2.1, 3.4, 100)
y = BestParameters.x[0] * x + BestParameters.x[1]
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset B, Best Fit Linear')
plt.show()

print('Dataset B, Best Fit Linear: y =', BestParameters.x[0], 'x +', BestParameters.x[1])

# Dataset B: Quadratic

def SSRes(parameters):
  yapprox = parameters[0]*xdata**2 + parameters[1]*xdata + parameters[2]
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(2.1, 3.4, 100)
y = BestParameters.x[0]*x**2 + BestParameters.x[1]*x + BestParameters.x[2]
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset B, Best Fit Quadratic')
plt.show()

print('Dataset B, Best Fit Quadratic: y =', BestParameters.x[0], 'x^2 +', BestParameters.x[1], 'x +', BestParameters.x[2])

datasetC = np.array( pd.read_csv(URL+'Exercise3_datafit7.csv') )
xdata_list = [datasetC[n][0] for n in range(len(datasetC))]
ydata_list = [datasetC[n][1] for n in range(len(datasetC))]
xdata = np.asarray(xdata_list)
ydata = np.asarray(ydata_list)

print(xdata)

# Dataset C: Quadratic

def SSRes(parameters):
  yapprox = parameters[0]*xdata**2 + parameters[1]*xdata + parameters[2]
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(0, 2.9, 100)
y = BestParameters.x[0]*x**2 + BestParameters.x[1]*x + BestParameters.x[2]
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset B, Best Fit Quadratic')
plt.show()

print('Dataset B, Best Fit Quadratic: y =', BestParameters.x[0], 'x^2 +', BestParameters.x[1], 'x +', BestParameters.x[2])

# Dataset C: exponential

def SSRes(parameters):
  yapprox = parameters[0]* np.e ** (xdata * parameters[1])
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(0, 2.9, 100)
y = BestParameters.x[0] * np.e ** (x * BestParameters.x[1])
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset C, Best Fit Exponential')
plt.show()

print('Dataset C, Best Fit Exponential: y =', BestParameters.x[0], 'e ^ x(', BestParameters.x[1], ')')

datasetD = np.array( pd.read_csv(URL+'Exercise3_datafit8.csv') )
xdata_list = [datasetD[n][0] for n in range(len(datasetD))]
ydata_list = [datasetD[n][1] for n in range(len(datasetD))]
xdata = np.asarray(xdata_list)
ydata = np.asarray(ydata_list)

print(xdata)

# Dataset D: log

def SSRes(parameters):
  yapprox = parameters[0]* np.log (xdata * parameters[1])
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(1.5, 27.5, 100)
y = BestParameters.x[0] * np.log (x * BestParameters.x[1])
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset D, Best Fit Logarithmic')
plt.show()

print('Dataset D, Best Fit Logarithmic: y =', BestParameters.x[0], 'log(x(', BestParameters.x[1], '))')

# Dataset D: power

def SSRes(parameters):
  yapprox = parameters[0]* xdata ** (parameters[1])
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(1.5, 27.5, 100)
y = BestParameters.x[0] * x ** (BestParameters.x[1])
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset D, Best Fit Power')
plt.show()

print('Dataset D, Best Fit Power: y =', BestParameters.x[0], 'x ^', BestParameters.x[1], '))')

datasetE = np.array( pd.read_csv(URL+'Exercise3_datafit9.csv') )
xdata_list = [datasetE[n][0] for n in range(len(datasetE))]
ydata_list = [datasetE[n][1] for n in range(len(datasetE))]
xdata = np.asarray(xdata_list)
ydata = np.asarray(ydata_list)

print(xdata)

# Dataset E: Sinusoidal

def SSRes(parameters):
  yapprox = parameters[0]* np.sin(parameters[1]*xdata) + parameters[2]
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[3.5, .1, 2])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(-40, 105, 100)
y = BestParameters.x[0] * np.sin( BestParameters.x[1] * x) + BestParameters.x[2]
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset E, Best Fit Sinusoidal')
plt.show()

print('Dataset E, Best Fit Sinusoidal: y =', BestParameters.x[0], 'sin(', BestParameters.x[1], 'x) +', BestParameters.x[2])

# Dataset E: Quartic

def SSRes(parameters):
  yapprox = parameters[0] + parameters[1]*xdata + parameters[2]*xdata**2 + parameters[3]*xdata**3 + parameters[4]*xdata**4 
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1, 1, 1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(-40, 105, 100)
y = BestParameters.x[0] + BestParameters.x[1]*x + BestParameters.x[2]*x**2 + BestParameters.x[3]*x**3 + BestParameters.x[4]*x**4 
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset E, Best Fit Quartic')
plt.show()

print('Dataset E, Best Fit Quartic: y =', BestParameters.x[0], '+', BestParameters.x[1], 'x +', \
    BestParameters.x[2], 'x^2 +', BestParameters.x[3], 'x^3 +', BestParameters.x[4], 'x^4')

datasetF = np.array( pd.read_csv(URL+'Exercise3_datafit10.csv') )
xdata_list = [datasetF[n][0] for n in range(len(datasetF))]
ydata_list = [datasetF[n][1] for n in range(len(datasetF))]
xdata = np.asarray(xdata_list)
ydata = np.asarray(ydata_list)

# Dataset F: Sinusoidal

def SSRes(parameters):
  yapprox = parameters[0]* np.sin(parameters[1]*xdata) + parameters[2]
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[4, 1, -1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(0, 7.5, 100)
y = BestParameters.x[0] * np.sin( BestParameters.x[1] * x) + BestParameters.x[2]
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset F, Best Fit Sinusoidal')
plt.show()

print('Dataset F, Best Fit Sinusoidal: y =', BestParameters.x[0], 'sin(', BestParameters.x[1], 'x) +', BestParameters.x[2])

# Dataset F: Sin^2(x)

def SSRes(parameters):
  yapprox = parameters[0]* (np.sin(parameters[1]*xdata))**2 + parameters[2]
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[4, 1, -1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(0, 7.5, 100)
y = BestParameters.x[0] * (np.sin( BestParameters.x[1] * x))**2 + BestParameters.x[2]
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset F, Best Fit Sin^2(x)')
plt.show()

print('Dataset F, Best Fit Sin^2(x): y =', BestParameters.x[0], 'sin^2(', BestParameters.x[1], 'x) +', BestParameters.x[2])

datasetG = np.array( pd.read_csv(URL+'Exercise3_datafit11.csv') )
xdata_list = [datasetG[n][0] for n in range(len(datasetG))]
ydata_list = [datasetG[n][1] for n in range(len(datasetG))]
xdata = np.asarray(xdata_list)
ydata = np.asarray(ydata_list)

print(xdata)

# Dataset G: -x + sin(x)

def SSRes(parameters):
  yapprox = parameters[0]*xdata + parameters[1] * np.sin(parameters[2]*xdata) + parameters[3]
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[-2, 200, 0, 300])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(10, 750, 100)
y = BestParameters.x[0]*x + BestParameters.x[1] * np.sin( BestParameters.x[2] * x) + BestParameters.x[3]
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset G, Best Fit -x + sin(x)')
plt.show()

print('Dataset G, Best Fit -x + sin(x): y =', BestParameters.x[0], 'x +', BestParameters.x[1], 'sin(', BestParameters.x[2], 'x) +', BestParameters.x[3])

# Dataset G: linear

def SSRes(parameters):
  yapprox = parameters[0]* xdata + parameters[1]
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(10, 750, 100)
y = BestParameters.x[0] * x + BestParameters.x[1]
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset G, Best Fit Linear')
plt.show()

print('Dataset G, Best Fit Linear: y =', BestParameters.x[0], 'x +', BestParameters.x[1])

datasetH = np.array( pd.read_csv(URL+'Exercise3_datafit12.csv') )
xdata_list = [datasetH[n][0] for n in range(len(datasetH))]
ydata_list = [datasetH[n][1] for n in range(len(datasetH))]
xdata = np.asarray(xdata_list)
ydata = np.asarray(ydata_list)

# Dataset H: x * cos(x)

def SSRes(parameters):
  yapprox = (parameters[0]*xdata) * np.cos(parameters[1]*xdata) + parameters[2]
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(0, 35, 100)
y = BestParameters.x[0] * x * np.cos( BestParameters.x[1] * x) + BestParameters.x[2]
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset H, Best Fit x * cos(x)')
plt.show()

print('Dataset H, Best Fit x * cos(x): y =', BestParameters.x[0], 'x * cos(x *', BestParameters.x[1], ') +', BestParameters.x[2])

# Dataset H: Periodic

def SSRes(parameters):
  yapprox = parameters[0] + parameters[1] * np.sin(xdata) + parameters[2] * np.cos(xdata)
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(0, 35, 100)
y = BestParameters.x[0] + BestParameters.x[1] * np.sin(x) + BestParameters.x[2] * np.cos(x)
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset H, Best Fit Periodic')
plt.show()

print('Dataset H, Best Fit Periodic: y =', BestParameters.x[0], '+', BestParameters.x[1], 'sin(x) +', BestParameters.x[2], 'np.cos(x)')
