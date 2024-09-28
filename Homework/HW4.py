import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.special

# Define Routines
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       count = 0
       return [astar, ier, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier = 0
      count = 0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      count = 0
      return [astar, ier, count]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier, count]

def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1)
  p[0] = p0
  for it in range(Nmax):

      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [np.trim_zeros(p),pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [np.trim_zeros(p),pstar,info,it]

def scaled_newton(m,f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    m    - multiplicity of root
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1)
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-m*f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [np.trim_zeros(p),pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [np.trim_zeros(p),pstar,info,it]

def secant(f,p0,p1,tol,Nmax):
  """
  Secant iteration.
  
  Inputs:
    f    - function
    p0   - first guess for root
    p1   - second guess for root 
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1)
  p[0] = p0
  p[1] = p1
  it = 0
  if (f(p0) - f(p1)) == 0:
     info = 1
     pstar = p1
     return [np.trim_zeros(p),pstar,info,it]
  
  for it in range(1, Nmax):
    p2 = p1 - f(p1)*(p1 - p0)/(f(p1) - f(p0))
    p[it + 1] = p2
    if abs(p2 - p1) < tol:
     pstar = p2
     info = 0
     return [np.trim_zeros(p),pstar,info,it] 
    p0 = p1
    p1 = p2
  pstar = p2
  info = 1
  return [np.trim_zeros(p),pstar,info,it]

# Question 1
def driver_1 ():
    x = np.linspace(0,3)
    f = 35*scipy.special.erf(x / (2*np.sqrt(0.138 * 5.184))) - 15
    plt.plot(x, f)
    plt.xlabel("Depth (x) in meters")
    plt.ylabel("Temperature (f(x)) in degrees Celcius")
    plt.show()

    f = lambda x: 35*scipy.special.erf(x / (2*np.sqrt(0.138 * 5.184))) - 15
    a = 0
    b = 2

    tol = 1e-13

    [astar,ier, count] = bisection(f,a,b,tol)
    print('the results from the bisection method are:')
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('Number of iterations:', count)

    fp = lambda x: 35*np.exp(-(x**2)/(4*0.138*5.184))/np.sqrt(np.pi*0.138*5.184)
    p0 = 0.01
    p1 = 2
    Nmax = 100
    tol = 1e-13

    (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
    print('Results from Newton using p0 = 0.01 are:')
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)

    (p,pstar,info,it) = newton(f,fp,p1,tol, Nmax)
    print('Results from Newton using p0 = 2 are:')
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)

# driver_1()

# Question 4
def driver_4 ():
    f = lambda x: np.exp(3*x) - 27*x**6 + 27*x**4*np.exp(x) - 9*x**2*np.exp(2*x)
    fp = lambda x: 3*(np.exp(x) - 6*x)*(np.exp(x) - 3*x**2)**2
    g = lambda x: (np.exp(x) - 3*x**2)/(3*np.exp(x) - 18*x)
    gp = lambda x: (6*x**2 + np.exp(x)*(2 - 4*x + x**2))/(np.exp(x) - 6*x)**2
    a = 3
    b = 5

    p0 = 3.01
    Nmax = 100
    tol = 1e-13

    (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
    print('Results from Vanilla Newton:')
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    print('The iterates are', p)

    (p,pstar,info,it) = newton(g,gp,p0,tol, Nmax)
    print('Results from Modified Newton:')
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    print('The iterates are', p)

    m = 2 # multiplicity of the root we are finding
    (p,pstar,info,it) = scaled_newton(m,f,fp,p0,tol, Nmax)
    print('Results from Modified Newton from Question 2:')
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    print('The iterates are', p)
    
driver_4()

# Question 5
def driver_5 ():
    f = lambda x: x**6 - x - 1
    fp = lambda x: 6*x**5 - 1
    a = 0
    b = 3

    p0 = 2
    p1 = 1
    Nmax = 100
    tol = 1e-13

    (p,pstar,info,it1) = newton(f,fp,p0,tol, Nmax)
    print('Results from Newton:')
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it1)
    print('The iterates are', p)
    error1 = abs(p - pstar)
    print('The list of error values is', error1)

    (p,pstar,info,it2) = secant(f,p0,p1,tol, Nmax)
    print('Results from Secant:')
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it2)
    print('The iterates are', p)
    error2 = abs(p - pstar)
    print('The list of error values is', error2)
    
    plt.plot(error1[0:(it2-1)],error1[1:it2])
    plt.plot(error2[0:(it2-1)],error2[1:it2])
    plt.legend(['Newton', 'Secant'])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('|xk+1 - alpha|')
    plt.ylabel('|xk - alpha|')
    plt.show()

driver_5()


