import numpy as np
import matplotlib.pyplot as plt

def driver_1 ():
    x = np.linspace(0,5,200)

    y = lambda x: np.sin(x)
    T6 = lambda x: x - x**3/6 + x**5/120
    P33 = lambda x: (x - (14/120)*x**3) / (1 + (1/20)*x**2) 
    P24 = lambda x: x / (1 + (1/6)*x + (1/36 - 1/120)*x**4)
    P42 = lambda x: (x - (14/120)*x**4) / (1 + (1/20)*x**2)

    # Error between sine and its 5th degree taylor series
    error_0 = abs(T6(x) - y(x))
    plt.figure()
    plt.semilogy(x,error_0,'o--')

    # Error between sine and each Pade polynomial
    error_1 = abs(P33(x) - y(x))
    error_2 = abs(P24(x) - y(x))
    error_3 = abs(P42(x) - y(x))

    plt.figure()
    # plt.semilogy(x,error_0,'o--')
    plt.semilogy(x,error_1,'o--')
    plt.semilogy(x,error_2,'o--')
    plt.semilogy(x,error_3,'o--')
    plt.legend(['T6(x)','P_3^3', 'P_4^2', 'P_2^4'])

    plt.show()

driver_1()

