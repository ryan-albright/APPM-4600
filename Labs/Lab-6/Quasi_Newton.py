import numpy as np
import matplotlib.pyplot as plt

def pre_lab():
    f = lambda x: np.cos(x)

    h_list = 0.01 * 2. ** (-np.arange(0, 10))
    s = np.pi / 2
    fd = np.empty(len(h_list))
    cd = np.empty(len(h_list))

    i = 0
    for h in h_list:
        fd[i] = (f(s+h) - f(s)) / h
        cd[i] = (f(s+h) - f(s-h)) / (2*h)
        i+= 1
    
    fd_dif = fd[:-1] - fd[1:]    
    cd_dif = cd[:-1] - cd[1:]
    x = list(range(len(h_list) - 1))
    plt.plot(x,fd_dif)
    plt.plot(x,cd_dif)
    plt.yscale('log')
    plt.show()

pre_lab()
