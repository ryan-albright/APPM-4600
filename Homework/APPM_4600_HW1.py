import numpy as np
import matplotlib.pyplot as plt

def question_1 (part = "i"):

    x = np.arange(1.920, 2.080, 0.001)

    y1 = x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 +5376*x**3 - 4608*x**2 +2304*x - 512
    y2 = (x - 2)**9
    if part == "i":
        plt.plot(x, y1)
    elif part == "ii":
        plt.plot(x, y2)
    else: 
        plt.plot(x, y1)
        plt.plot(x, y2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

question_1()
print('Hi')
def question_5 (x_mag = "small"):
    if x_mag == "small":
        x = np.pi
    else:
        x = 10**6
    delta = np.array([10**-16, 10**-15, 10**-14, 10**-13, 10**-12, 10**-11, 10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0])
    y = -2*np.sin((2*x + delta)/2)*np.sin(delta/2)
    return y

y_x_small = question_5 ("small")
y_x_big = question_5 ("big")

def new_question_5 (x_mag = "small"):
    if x_mag == "small":
        x = np.pi
        y_bad = y_x_small
    else:
        x = 10**6
        y_bad = y_x_big
    delta = np.array([10**-16, 10**-15, 10**-14, 10**-13, 10**-12, 10**-11, 10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0])
    y = delta*np.sin(x) - delta**2/2

    rel_error = np.abs((y - y_bad) / y)

    return rel_error

print(new_question_5("small"))
print(new_question_5("big"))





    
