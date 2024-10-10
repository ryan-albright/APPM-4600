import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm

# Pre Lab
def Vandermonde(x):
    '''
    Inputs:
    x - an n x 1 array of x values for which we want to create a vandermonde matrix
    Outputs:
    V - a vandermonde matrix of these x values
    '''
    n = x.size
    
    V = np.empty([n,n])

    for i in range(n):
        c = np.power(x, i)
        V[:,i] = c
    return V

# 3.1
def monomial_driver(): 

    f = lambda x: 2*x +4
    
    N = 10
    a = 0
    b = 1
    
    ''' Create interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    # print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
    # print('yint =',yint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint)
    # print('V = ',V)

    ''' Invert the Vandermonde matrix'''    
    Vinv = inv(V)
    # print('Vinv = ' , Vinv)
    
    ''' Apply inverse to rhs'''
    ''' to create the coefficients'''
    # coef = Vinv @ yint
    coef = np.dot(Vinv, yint)
    
    # print('coef = ', coef)

# No validate the code
    Neval = 100    
    xeval = np.linspace(a,b,Neval+1)
    yeval = eval_monomial(xeval,coef,N,Neval)

# exact function
    yex = f(xeval)
    
    err =  norm(yex-yeval) 
    print('err = ', err)
    
    return

def eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
#    print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
#        print('yeval[i] = ', yeval[i])
#        print('a[j] = ', a[j])
#        print('i = ', i)
#        print('xeval[i] = ', xeval[i])
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

def driver_311():

    f = lambda x: (1 + (10*x)**2)**-1

    ''' interval'''
    a = -1
    b = 1
   
    ''' create points for evaluating the Monomial/Lagrange/Newton interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)

    for N in range(2, 11):
        ''' create equispaced interpolation nodes'''
        xint = np.linspace(a,b,N+1)
    
        ''' create interpolation data'''
        yint = f(xint)
        # Runs the monomial approximation
        ''' Create the Vandermonde matrix'''
        V = Vandermonde(xint)

        ''' Invert the Vandermonde matrix'''    
        Vinv = inv(V)
        
        ''' Apply inverse to rhs'''
        ''' to create the coefficients'''
        coef = Vinv @ yint
        
        # No validate the code
        yeval_m = eval_monomial(xeval,coef,N,Neval)

        # exact function
        # yex = f(xeval)
        
        # err =  norm(yex-yeval_m) 
        # print('err = ', err)


        # Runs lagrange and newton and graphs the all the approximations
        '''Initialize and populate the first columns of the 
        divided difference matrix. We will pass the x vector'''
        y = np.zeros( (N+1, N+1) )
        
        for j in range(N+1):
            y[j][0]  = yint[j]

        y = dividedDiffTable(xint, y, N+1)
        ''' evaluate lagrange poly '''
        for kk in range(Neval+1):
            yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
            yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)
        
        ''' create vector with exact values'''
        fex = f(xeval)
        

        plt.figure()
        plt.plot(xeval,fex,'ro-')
        plt.plot(xeval, yeval_m, 'md--')
        plt.plot(xeval,yeval_l,'bs--') 
        plt.plot(xeval,yeval_dd,'c.--')
        plt.legend(['True Function', 'Monomial Approximation','Langrage Approximation', 'Newton Approximation'])

        plt.figure() 
        err_m = abs(yeval_m - fex)
        err_l = abs(yeval_l-fex)
        err_dd = abs(yeval_dd-fex)
        plt.semilogy(xeval,err_l,'ro--',label='lagrange')
        plt.semilogy(xeval,err_dd,'bs--',label='Newton DD')
        plt.semilogy(xeval, err_m, 'md--',label='monomial')
        plt.legend()
        plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)

''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval

driver_311()