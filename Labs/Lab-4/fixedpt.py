# import libraries
import numpy as np
    
def driver():

# test functions 
     f1 = lambda x: np.sqrt(10 / (x + 4))

#fixed point  

     Nmax = 15
     tol = 1e-10
    
#test f1 '''
     x0 = 1.5
     [xstar, x_list, ier] = fixedpt(f1,x0,tol,Nmax)
     [_lambda, alpha] = compute_order(x_list, xstar)
     print('the approximate fixed point is:',xstar)
     print('the list of iterates is:', x_list)
     print('f1(xstar):',f1(xstar))
     print('Error message reads:',ier)
     print(f"The order of convergence is {np.round(alpha, 1)}")

# define routines
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

driver()