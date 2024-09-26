# import libraries
import numpy as np
import matplotlib.pyplot as plt

def driver():

# use routines    
    f = lambda x: np.e**(x**2 + 7*x - 30) - 1
    fp = lambda x: (2*x + 7)*np.e**(x**2 + 7*x - 30)
    fpp = lambda x: 2*np.e**(x**2 + 7*x - 30) + (2*x + 7)**2*np.exp(x**2 + 7*x - 30)
    g = lambda x: x - f(x)/fp(x)
    gp = lambda x: 1 - (fp(x)**2 - f(x)*fpp(x)) / fp(x)**2
    a = 1
    b = 4

    tol = 1e-7
    [mstar,ier] = bisection(f,gp,a,b,tol)
    print('the point in the basin of convergence is',mstar)
    print('the error message reads:',ier)
    print("g'(mstar) =", gp(mstar))

# define routines
def bisection(f,gp,a,b,tol):
    
#    Inputs:
#     f,gp,a,b - function, the first derivative of g(x) = x - f(x) / f'(x), and endpoints of initial interval
#      tol - bisection stops when interval length < tol

#    Returns:
#      mstar - approximation of midpoint
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
       ier = 1
       mstar = a
       return [mstar, ier]

#   verify end points are not a root 
    if (fa == 0):
      mstar = a
      ier = 0
      return [mstar, ier]

    if (fb ==0):
      mstar = b
      ier = 0
      return [mstar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(gp(d)) > 1):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
      
    astar = d
    ier = 0
    return [astar, ier]
      
driver()               

