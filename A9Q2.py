import numpy as np
from matplotlib import pyplot as plt
from numpy import loadtxt

x_max = lambda n: 2**int(np.log2(n))

def main():
  x_string = loadtxt ("./sunspots.txt", str)
  x = [float(x_string[i][1]) for i in range(len(x_string))]
  plt.plot (x)
  plt.xlim (0, len(x))
  plt.xlabel("Time (months)")
  plt.ylabel("Number of Sunspots")
  plt.tight_layout()
  plt.show()

  # values are approximated peaks determined by looking at the graph
  period_s = np.average([148-10, 249-148, 352-249, 467-352, 669-467, 818-669, 975-818, 1055-975,
                    1185-1055, 1338-1185, 1456-1338, 1598-1456, 1735-1598, 1897-1735, 2023-1897, 
                    2171-2023, 2274-2171, 2380-2274, 2505-2380, 2642-2505, 2768-2642, 2899-2768, 
                    3018-2899])
  print(period_s)
  # The average distance between peaks is 130.7826086956522
  # Thus, we estimate the sunspot cycle to be 130-131 months long.

  plt.plot(abs(np.fft.fft(x)))
  plt.xlim (0, x_max(len(x)/2))
  plt.xlabel("k")
  plt.ylabel("X[k]")
  plt.tight_layout()
  plt.show()

  # k is approximately 24 at the nonzero peak of the power spectrum
  # sampling frequency is 1 observation/month
  f = (24) / len(x)
  period = f**-1
  print(period)
  # The period of the sin wave for k=24 is 130.95833333333331
  # This value corresponds to our estimated periodicity

main()