import numpy as np
import matplotlib.pyplot as plt

# Question 1
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

def coeff(f, N):
    h = 2 / (N - 1)
    i = np.array(range(1,N+1))
    xi = np.array(-1 + (i - 1)*h)
    yi = f(xi)

    # Assemble Linear System
    V = Vandermonde(xi)
    c = np.linalg.inv(V) @ yi
    return [c, xi, yi]

def eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    for j in range(1,N):
      for i in range(Neval+1):
         yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

def driver_1(plot_type = None):
    if isinstance(plot_type, int):
        f = lambda x: 1 / (1 + 100*x**2)
        N = 4

        [c,xi,yi] = coeff(f, N)

        # plot f(x) and p(x)
        x = np.linspace(-1,1,num=1001)
        y = f(x)
        p_x = eval_monomial(x,c,N,len(x)-1)
        
        plt.plot(x,y,'o')
        plt.plot(x,p_x,'o')

        # plot points
        plt.plot(xi, yi, 'o')

        plt.legend(['f(x)', 'p(x)','Data Points'])
        plt.show()
    else: 
        for N in [4,8,12,16]:
            f = lambda x: 1 / (1 + 100*x**2)
            
            [c,xi,yi] = coeff(f, N)

            # plot f(x) and p(x)
            x = np.linspace(-1,1,num=1001)
            y = f(x)
            p_x = eval_monomial(x,c,N,len(x)-1)
            
            plt.plot(x,p_x,'-')

            # plot points
            # plt.plot(xi, yi, 'o')
        y = f(x)
        plt.plot(x,y,'-')
        plt.legend(['p_4(x)','p_8(x)', 'p_12(x)','p_16(x)','f(x)'])
        plt.show()

# driver_1() # either input an N to test or the program will run through a predetermined sequences of N to display the Runge Phenomena

# Question 2
def driver_2():
    f = lambda x: 1/(1+100*x**2)

    # interpolation points
    N = 100
    a = -1
    b = 1
    xinterp = np.linspace(a,b,N)
    y = f(xinterp)

    # evaluation points
    aeval = -0.6
    beval = 0.6
    N_eval = 200
    xeval = np.linspace(aeval,beval,N_eval)

    # Evaluate the Polynomial
    w = weight_vector(xinterp,N)
    p_x = np.empty(N_eval)
    for i in range(N_eval):
        phi = phi_n(xeval[i],xinterp,N)

        sum_x = 0
        for j in range(N):
            sum_x = sum_x + w[j] * y[j] / (xeval[i] - xinterp[j])
        p_x[i] = phi * sum_x
    print(p_x)

    plt.plot(xinterp,y,'o')
    plt.plot(xeval,p_x,'o')
    plt.legend(['f(x)','p(x)'])
    plt.show()

def weight_vector(xinterp,N):
    '''
    Inputs:
    xinterp - an array of interpolation points
    N - the number of interpolation points
    '''
    W = np.empty(N)  
    for j in range(N):
        xj = xinterp[j]
        
        w = 1

        for i in range(N):
            xi = xinterp[i]
            if i == j:
                w_p = 1
            else:
                w_p = (xj - xi)

            w = w * w_p
        W[j] = 1 / w
    return W

def phi_n(x, xinterp, N):
    '''
    Inputs:
    x - an x value that is not in xinterp
    xinterp - an array of interpolation points
    N - the number of interpolation points
    Outputs:
    phi - value of phi for N and x
    '''
    phi = 1

    for i in range(N):
        phi_p = (x - xinterp[i])
        phi = phi*phi_p
    return phi
    
# driver_2()

# Question 3
def driver_3():
    f = lambda x: 1/(1+100*x**2)
    # f = lambda x: np.exp(x)

    # interpolation points
    N = 100
    a = -1
    b = 1
    xinterp = np.linspace(a,b,N)
    # xinterp = np.empty(N)
    # for i in range(N):
    #     xinterp[i] = np.cos((2*i - 1)*np.pi / (2*N))
    y = f(xinterp)

    # evaluation points
    aeval = -0.6
    beval = 0.6
    N_eval = 200
    xeval = np.linspace(aeval,beval,N_eval)

    # Evaluate the Polynomial
    w = weight_vector(xinterp,N)
    p_x = np.empty(N_eval)
    for i in range(N_eval):
        phi = phi_n(xeval[i],xinterp,N)

        sum_x = 0
        for j in range(N):
            sum_x = sum_x + w[j] * y[j] / (xeval[i] - xinterp[j])
        p_x[i] = phi * sum_x
    print(p_x)

    plt.plot(xinterp,y,'o')
    plt.plot(xeval,p_x,'o')
    plt.legend(['f(x)','p(x)'])
    plt.show()

driver_3()