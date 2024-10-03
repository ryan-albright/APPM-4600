import numpy as np
import matplotlib.pyplot as plt

# Question 1
def driver_1():
    f = lambda x,y: 3*x**2 - y**2 
    fpx = lambda x,y: 6*x
    fpy = lambda x,y: -2*y
    g = lambda x,y: 3*x*y**2 - x**3 - 1
    gpx = lambda x,y: 3*y**2 - 3*x**2
    gpy = lambda x,y: 6*x*y
    

    x0 = 1
    y0 = 1
    nmax = 40
    tol = 1e-13

    [xn, yn] = non_linear_solver(f, g, x0, y0, nmax)
    print('Here are the results after simply iterating through the given equation')
    print(f'After {nmax} iterations, xn and yn converged to {xn} and {yn} respectively.')

    [xn,yn,y1,y2,info,i] = newton_2D(f,g,fpx,fpy,gpx,gpy,x0,y0,tol,nmax)
    print('The results from the Newton Method are:')
    print(f'After {i} iterations, xn and yn were found to be {xn} and {yn}')  

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

    y1 = np.zeros(nmax+1)
    y2 = np.zeros(nmax+1)
    x = list(range(nmax + 1))
    y1[0] = x0
    y2[0] = y0

    for i in range(nmax):
        F = np.array([[f(x0, y0)], [g(x0, y0)]])
        p1 = np.array([[x0], [y0]]) - np.dot(J,F)             
        x0 = p1[0][0]
        y0 = p1[1][0]

        y1[i+1] = x0
        y2[i+1] = y0
    
    plt.plot(x,y1, 'o')
    plt.plot(x,y2, 'o')
    plt.legend(["xn", "yn"])
    plt.xlabel("Iterations")
    plt.show()

    return [x0, y0]

def newton_2D(f,g,fpx,fpy,gpx,gpy,x0,y0,tol,nmax):
    """
    Inputs:
    f - function #1 and its derivative
    g - function #2 and its derivative
    x0 - initial guess for x
    y0 - initial guess for y
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
    Returns:
    x0    - the last x point
    y0    - the last y point
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
    """

    y1 = np.zeros(nmax+1)
    y2 = np.zeros(nmax+1)
    y1[0] = x0
    y2[0] = y0

    for i in range(nmax):
        F = np.array([[f(x0, y0)], [g(x0, y0)]])
        J = np.array([[fpx(x0,y0), fpy(x0,y0)], [gpx(x0,y0), gpy(x0,y0)]])
        p = np.linalg.solve(J, -F )
        pn = np.array([[x0], [y0]]) + p

        x0 = pn[0][0]
        y0 = pn[1][0]

        y1[i+1] = x0
        y2[i+1] = y0
        if (abs(f(x0,y0) - g(x0,y0)) < tol):
            info = 0
            return [x0,y0,y1,y2,info,i]
    
    return [x0,y0,y1,y2,info,i]

# driver_1()

# Question 3
def driver_2 ():
    f = lambda x,y,z: x**2 + 4*y**2 + 4*z**2 - 16
    fx = lambda x,y,z: 2*x
    fy = lambda x,y,z: 8*y
    fz = lambda x,y,z: 8*z

    fp = [fx, fy, fz]

    p0 = np.array([1,1,1]).reshape([3,1])
    nmax = 40
    tol = 1e-13

    [pn,x,y,z,info, i] = newton_3D(f,fx,fy,fz,p0,tol,nmax)
    print(f'After {i} iterations, the method finds a point on the surface to be (x,y,z) = ({pn[0][0]}, {pn[1][0]}, {pn[2][0]})')
    
    # defining variables to graph error
    x_dif = x[:-1] - x[1:]
    y_dif = y[:-1] - y[1:]
    z_dif = z[:-1] - z[1:]

    # making all of them positive
    x_dif = [abs(x) for x in x_dif]
    y_dif = [abs(x) for x in y_dif]
    z_dif = [abs(x) for x in z_dif]

    x = list(range(i+1)) 

    plt.plot(x, x_dif)
    plt.plot(x, y_dif)
    plt.plot(x, z_dif)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('|xn+1 - xn|')
    plt.legend(['x','y','z'])
    plt.show()

def newton_3D (f,fx,fy,fz,p0,tol,nmax):
    """
    Inputs:
    f - function #1 and its derivative
    p0 - initial guess for points
    tol  - iteration stops when p_n,p_{n+1} are within tol
    nmax - max number of iterations
    Returns:
    pn    - the calculated points
    x     - list of iterates of first coordinate
    y     - list of iterates of second coordinate
    z     - list of iterates of third coordinate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
    """
    x = np.zeros(nmax)
    y = np.zeros(nmax)
    z = np.zeros(nmax)

    x[0] = p0[0][0]
    y[0] = p0[1][0]
    z[0] = p0[2][0]

    for i in range(nmax):
        a = np.array([fx(p0[0],p0[1], p0[2]), fy(p0[0],p0[1], p0[2]), fz(p0[0],p0[1], p0[2])]).reshape(3,1)
        d = f(p0[0],p0[1], p0[2]) / (fx(p0[0],p0[1], p0[2])**2 + fy(p0[0],p0[1], p0[2])**2 + fz(p0[0],p0[1], p0[2])**2)
        pn = p0 - d * a

        x[i+1] = pn[0][0]
        y[i+1] = pn[1][0]
        z[i+1] = pn[2][0]

        if (abs(f(pn[0],pn[1],pn[2])) < tol):
            info = 0
            return [pn,np.trim_zeros(x),np.trim_zeros(y),np.trim_zeros(z),info,i]
        p0 = pn
    
    return [pn,x,y,z,info,i]  

driver_2()






