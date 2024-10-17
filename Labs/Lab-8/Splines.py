import numpy as np

# Pre Lab
def line_eval(a,x0,y0,x1,y1):
    return (y1 - y0)/(x1 - x0)*(a - x0) + y0

# 3.1
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
def driver():
    f = lambda x: 1 / (1 + 100*x**2)
    a = -1
    b = 1
    
    # create points you want to evaluate at
    Neval = 100
    xeval = np.linspace(a,b,Neval)

    # number of intervals
    Nint = 10

    # evaluate the linear spline
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)

    # evaluate f at the evaluation points
    fex = f(xeval)
    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval,'bs-')
    plt.legend(['f(x)','p(x)'])
    plt.show()

    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval,err,'ro-')
    plt.show()

def eval_lin_spline(xeval,Neval,a,b,f,Nint):
    # create the intervals for piecewise approximations
    xint = np.linspace(a,b,Nint+1)

    #create vector to store the evaluation of the linear splines
    yeval = np.zeros(Neval)

    for jint in range(Nint):
        # find indices of xeval in interval (xint(jint),xint(jint+1))
        # let ind denote the indices in the intervals
        atmp = xint[jint]
        btmp= xint[jint+1]

        # find indices of values of xeval in the interval
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
        n = len(xloc)
        
        #temporarily store your info for creating a line in the interval of interest
        fa = f(atmp)
        fb = f(btmp)
        yloc = np.zeros(len(xloc))

        for k in range(n):
            #use your line evaluator to evaluate the spline at each location
            yloc[k] = line_eval(xloc[k],atmp,fa,btmp,fb) #Call your line evaluator with points (atmp,fa) and (btmp,fb)

            # Copy yloc into the final vector
            yeval[ind] = yloc

    return yeval

driver()