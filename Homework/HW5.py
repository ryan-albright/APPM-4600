import numpy as np
import matplotlib.pyplot as plt

def driver_1():
    f = lambda x,y: 3*x**2 - y**2 
    g = lambda x,y: 3*x*y**2 - x**3 - 1

    x0 = 1
    y0 = 1
    nmax = 20
    tol = 1e-13

    [xn, yn] = non_linear_solver(f, g, x0, y0, nmax)
    print(f'After {nmax} iterations, xn and yn converged to {xn} and {yn} respectively.')


def non_linear_solver (f, g, x0, y0, nmax):
    """
    Inputs: 
    f - function #1
    g - function #2
    x0 - starting value for x
    y0 - starting value for y
    nmax - maximum number of iterations
    Outputs:

    """
    
    J = np.array([[1/6,1/18], [0,1/6]])
    y1 = np.zeros(nmax)
    y2 = np.zeros(nmax)
    x = np.zeros(nmax)
    y1[0] = x0
    y2[0] = y0

    for i in range(1,nmax):
        B = np.array([[f(x0, y0)], [g(x0, y0)]])
        p1 = np.array([[x0], [y0]]) - np.dot(J,B)             
        x0 = p1[0][0]
        y0 = p1[1][0]

        y1[i] = x0
        y2[i] = y0
        x[i] = i
    
    plt.plot(x,y1, 'o')
    plt.plot(x,y2, 'o')
    plt.legend(["xn", "yn"])
    plt.xlabel("Iterations")
    plt.show()

    return [x0, y0]



driver_1()


