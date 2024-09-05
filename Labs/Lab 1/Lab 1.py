import numpy as np
import matplotlib.pyplot as plt

# 3.2 Exercises: The Basics

# 3.2.1
x = np.linspace(0, 4, 20)
y = np.arange(0,20)

# 3.2.2
print(x[0])

# 3.2.3
print(f"the first three entries of x are {x[0]}, {x[1]}, and {x[2]}")

# 3.2.4
w = 10**(-np.linspace(1,10,10))
x = np.arange(0, len(w))
len(x) == len(w)
plt.plot(x, w)
plt.yscale('log')
plt.xlabel("x")
plt.ylabel("w")

# 3.2.5
s = 3 * w
plt.plot(s)
plt.show()

# Sample Code for 4.2 Exercises
def driver():
    n = 100
    x = np.linspace(0,np.pi,n)
# this is a function handle. You can use it to define
# functions instead of using a subroutine like you
# have to in a true low level language.
    f = lambda x: x**2 + 4*x + 2*np.exp(x)
    g = lambda x: 6*x**3 + 2*np.sin(x)
    y = np.array([0,1])
    w = np.array([1,0])
# evaluate the dot product of y and w
    dp = dotProduct(y,w,n)
# print the output
    print("the dot product is : ", dp)
    
    return

def dotProduct(x,y,n):
# Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp

driver()

# 4.2.1
