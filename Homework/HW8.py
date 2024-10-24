import matplotlib.pyplot as plt
import numpy as np
import math
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm

# Question 1
def driver_1(interpolationType,N_interp,pointType=None):
    '''
    Inputs:
    InterpolationType - type of interpolation: lagrange, hermite, natural cubic spline, clamped cubic spline  
    N_interp - the number of interpolation points as a list Ex: [2,4,6]
    Outputs:
    Graph of the error between the actual function and the interpolation(s) as well as a graph of the interpolation(s) 
    and the actual function
    '''
    # initialize problem
    f = lambda x: 1 / (1+x**2)
    fp = lambda x: -2*x / (1+x**2)**2
    a = -5
    b = 5

    # Evaluation Points
    Neval = 500
    xeval = np.linspace(a,b,Neval+1)
    y_eval = np.empty((Neval+1,len(N_interp)))

    # loop and corresponding logic waterfall plots the specified interpolation method
    for j,N in enumerate(N_interp):
        # interpolation points

        if pointType == 'cheb':
            xinterp = np.empty(N)
            for i in range(N):
                x = (2*i + 1)*np.pi
                y = 2*N
                xinterp[i] = 5*np.cos(x/y)
            xinterp = np.sort(xinterp)
        else:
            xinterp = np.linspace(a,b,N)
        yinterp = f(xinterp)
    
        if interpolationType == 'lagrange':
            # evaluate lagrange poly
            for i in range(Neval+1):
                y_eval[i,j] = eval_lagrange(xeval[i],xinterp,yinterp,N-1)
    
        elif interpolationType == 'hermite':
            # evaluate the Hermite polynomial
            ypinterp = fp(xinterp)
            for i in range(Neval+1):
                y_eval[i,j] = eval_hermite(xeval[i],xinterp,yinterp,ypinterp,N-1)

        elif interpolationType == 'natural cubic spline':
            (M,C,D) = create_natural_spline(yinterp,xinterp,N-1)
            y_eval[:,j] = eval_cubic_spline(xeval,Neval,xinterp,N-1,M,C,D)

        elif interpolationType == 'clamped cubic spline':
            (M,C,D) = create_clamped_spline(yinterp,xinterp,N-1)
            y_eval[:,j] = eval_cubic_spline(xeval,Neval,xinterp,N-1,M,C,D)
    # Computing Actual Values
    fex = f(xeval)
    fex_array = np.array([fex]*len(N_interp))
    fex_array = fex_array.T
    error = abs(y_eval - fex_array)

    # Graphing Approximations
    plt.figure()
    plt.plot(xeval,fex,'o')
    legend = ['f(x)']
    for i,N in enumerate(N_interp):
        plt.plot(xeval,y_eval[:,i],'o')
        legend.append(f'p_{N}(x)')
    plt.legend(legend)

    # Graphing the Error
    plt.figure() 
    for i,N in enumerate(N_interp):
        plt.semilogy(xeval,error[:,i],'o--')
    plt.legend(legend[1:])
    plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)       

def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              

    yeval = 0.
    
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)

def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6
    A[N][N] = 1

    Ainv = inv(A)
    
    M  = Ainv.dot(b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)
       
def create_clamped_spline(yint,xint,N):
    #   create the right  hand side for the linear system
    b = np.zeros(N+1)
    #  vector values
    h = np.zeros(N+1)
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip

#  create matrix so you can solve for the M values  
    d0 = [h[0]/3] + [(h[j]+h[j-1])/3 for j in range(1,N)] + [h[N-1]/3]
    d_1 = [h[j]/6 for j in range(N)]
    A1 = np.diag(d0)
    A2 = np.diag(d_1,k=1)
    A3 = np.diag(d_1,k=-1)
    A = A1 + A2 + A3

    M = np.linalg.solve(A,b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)

def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) \
            + C*(xip-xeval) + D*(xeval-xi)
    return yeval 
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#        print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)

# driver_1('clamped cubic spline',[5,10,15,20])  
# InterpolationType - type of interpolation: lagrange, hermite, natural cubic spline, clamped cubic spline  
# N_interp - the number of interpolation points as a list Ex: [2,4,6]

# Question 2
driver_1('clamped cubic spline',[5,10,15,20],'cheb')
# same inputs as for question 1 driver
