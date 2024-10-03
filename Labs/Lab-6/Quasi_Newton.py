import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv 
from numpy.linalg import norm 

def pre_lab():
    f = lambda x: np.cos(x)

    h_list = 0.01 * 2. ** (-np.arange(0, 10))
    s = np.pi / 2
    fd = np.empty(len(h_list))
    cd = np.empty(len(h_list))

    i = 0
    for h in h_list:
        fd[i] = (f(s+h) - f(s)) / h
        cd[i] = (f(s+h) - f(s-h)) / (2*h)
        i+= 1
    
    fd_dif = fd[:-1] - fd[1:]    
    cd_dif = cd[:-1] - cd[1:]
    x = list(range(len(h_list) - 1))
    plt.plot(x,fd_dif)
    plt.plot(x,cd_dif)
    plt.yscale('log')
    plt.show()

# pre_lab()

def evalF(x): 
    F = np.zeros(2)
    
    F[0] = 4*x[0]**2 + x[1]**2 - 4
    F[1] = x[0] + x[1] - np.sin(x[0] - x[1])
    return F
    
def evalJ(x):     
    J = np.array([[8*x[0], 2*x[1]],[1 - np.sin(x[0] - x[1]), 1 + np.cos(x[0] - x[1])]])
    return J

def slackerNewton(x0,tol,tol_2,Nmax):
    ''' Slacker Newton = updates the inverse of the Jacobian occassionally based on a specific condition '''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)

       if (norm(x1 - x0) < tol):
           xstar = x1
           ier = 0
           return[xstar, ier,its]
       if (norm(x1 - x0) > tol_2):
           J = evalJ(x1)
           Jinv = inv(J)

       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   

def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier = 0
           return[xstar, ier,its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

def driver():
    x0 = np.array([2,0])
    tol = 1e-10
    tol_2 = 0.1
    nmax = 20

    [xstar,ier,its] = LazyNewton(x0,tol,nmax)    
    print('The results from the Lazy Newton are:')
    print(f'After {its} iterations, the root found was {xstar} and the error message is {ier}')

    [xstar,ier,its] = slackerNewton(x0,tol,tol_2,nmax)
    print('The reuslts from the slacker newton method are:')
    print(f'After {its} iterations, the root found was {xstar} and the error message is {ier}')

driver()



