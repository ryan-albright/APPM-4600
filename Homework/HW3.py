import numpy as np
import matplotlib.pyplot as plt

# Question 1
def driver_1():
    f = lambda x: np.sin(x) - 2*x + 1
    a = -np.pi/2
    b = np.pi/2
    tol = 0.5*1e-8

    [astar, ier, count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'{count + 1} iterations were performed')

# define bisection function to be used
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
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
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

# driver_1() # commented out to run other parts of the code

# Question 2 
def driver_2(part): # indicate "part a" or "part b"
    
    if part == "part a":
        f = lambda x: (x - 5)**9
    elif part == "part b":
        f = lambda x: x**9-45*x**8+900*x**7-10500*x**6+78750*x**5-393750*x**4+1312500*x**3-2812500*x**2+3515625*x-1953125
    a = 4.82
    b = 5.2
    tol = 1e-4

    [astar, ier, count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))

# driver_2("part b") # commented out to run other parts of the code

# Question 3
def driver_3(): 
    f = lambda x: x**3 + x - 4
    a = 1
    b = 4
    tol = 1e-3

    [astar, ier, count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'{count} iterations were performed')

# driver_3() # commented out to run other parts of the code

# Question 5
x = np.linspace(-2,8,100)
y = x - 4*np.sin(2*x) - 3
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

def driver_4():
     
     f1 = lambda x: -np.sin(2*x) - 5*x/4 - 3/4

#fixed point  

     Nmax = 20
     tol = 5*1e-10
    
#test f1 '''
     x0 = -1
     [xstar, x_list, ier] = fixedpt(f1,x0,tol,Nmax)
     [_lambda, alpha] = compute_order(x_list, xstar)
     print('the approximate fixed point is:',xstar)
     print('the list of iterates is:', x_list)
     print('f1(xstar):',f1(xstar))
     print('Error message reads:',ier)
     print(f"The order of convergence is {np.round(alpha, 1)}")

# define fixed point routine
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    x_list = np.array([x0])
    while (count <Nmax):
       count = count + 1
       x1 = f(x0)
       x_list = np.append(x_list, x1)
       if (abs(x1-x0) < tol):
          xstar = x1
          ier = 0
          return [xstar,x_list, ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, x_list, ier]
    
def compute_order(x_list, xstar):
    """
    Approximates order of convergence given 
    x_list: list of iterates
    x_star: fixed point/solution
    """
    x_list = x_list[0:-1]
    diff1 = np.abs(x_list[1::] - xstar)
    print(diff1)
    diff2 = np.abs(x_list[0:-1] - xstar)
    print(diff2)
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)

    _lambda = np.exp(fit[1])
    alpha = fit[0]
    """
    lamba  = e^intercept
    alpha = slope
    """

    return [_lambda, alpha]

driver_4()
