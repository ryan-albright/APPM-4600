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

#question_1()

#x = np.arange(-0.02, 0.02, 0.001)
#y = np.sqrt(x + 1) - 1

x = 1.000000000000001
y = 1

print(np.sin(x) - np.sin(y))
print(2 * np.cos((x + y) / 2) * np.sin((x - y) / 2))
