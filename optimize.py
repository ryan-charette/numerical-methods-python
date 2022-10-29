import numpy as np
import matplotlib.pyplot as plt
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

def SSRes(parameters):
  yapprox = parameters[0]* np.e ** (xdata * parameters[1])
  residuals = np.abs(ydata-yapprox)
  return np.sum(residuals**2)

BestParameters = minimize(SSRes,[1, 1, 1])

plt.plot(xdata,ydata,'bo',markersize=5)
x = np.linspace(-2, 4, 100)
y = BestParameters.x[0] * np.e ** (x * BestParameters.x[1])
print(x)
print(y)
plt.plot(x,y,'r--')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset A: Best Fit Exponential')
plt.show()
