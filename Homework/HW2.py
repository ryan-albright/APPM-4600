import numpy as np
import matplotlib.pyplot as plt
import math 
import random

# Question 3

def func (x):
    y = math.e**x
    return y - 1
print("The function outputs ",func(9.999999995000000*10**-10))

def taylor_series (x):
    return x + x**2/2 + x**3/6
print("The Taylor Series Approximation outputs", taylor_series(9.999999995000000*10**-10))

# Question 4
def sum_vector ():
    t = np.arange(0, np.pi, np.pi/30)
    y = np.cos(t)
    return np.sum(np.multiply(t, y))
print("The sum is",sum_vector())

def fig_1 ():
    theta = np.linspace(0, 2*np.pi, 200)
    x = 1.2*(1 + 0.1*np.sin(15*theta))*np.cos(theta)
    y = 1.2*(1 + 0.1*np.sin(15*theta))*np.sin(theta)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

fig_1()

def fig_2 ():
    theta = np.linspace(0, 2*np.pi, 200)
    p = random.uniform(0,2)

    for i in range(1,11):
        print(i)
        x = i*(1 + 0.05*np.sin((2+i)*theta + p))*np.cos(theta)
        y = i*(1 + 0.05*np.sin((2+i)*theta + p))*np.sin(theta)
        plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

fig_2()



