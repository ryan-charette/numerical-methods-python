import numpy as np

def heat_equation():

  D = 0.5 # diffusion coefficient
  x = np.linspace(0, 1, 100)
  t = np.linspace(0, 1, 100)
  dx = x[1]-x[0]
  dt = t[1]-t[0]
  IC = np.sin(2*np.pi*(x)) # initial conditions
  BC = [0,0] # Dirichlet boundary conditions

  u = np.zeros((len(x), len(t)))
  # enforcing boundary and initial conditions
  u[0:] = BC[0]
  u[-1:] = BC[1]
  u[:,0] = IC

  # define parameter r as
  r = D*dt/dx**2

  # finite difference scheme
  A = np.diag([1+2*r]*(len(x)-2),0) + np.diag([-r]*(len(x)-3),-1) + np.diag([-r]*(len(x)-3),1)
  print(t)
  print(A)
  for n in range(1,len(t)):
    b = u[1:-1,n-1].copy()
    b[0] = b[0] + r*u[0,n]
    b[-1] = b[-1] + r*u[-1,n]
    u[1:-1,n] = np.linalg.solve(A,b)