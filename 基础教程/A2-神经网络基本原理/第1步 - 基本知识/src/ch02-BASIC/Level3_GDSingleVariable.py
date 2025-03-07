# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    y = x**2+2
    return y

def derivative_function(x):
    return 2*x

def draw_function():
    x = np.linspace(-100,100)
    y = target_function(x)
    plt.plot(x,y)

def draw_gd(X):
    Y = []
    for i in range(len(X)):
        Y.append(target_function(X[i]))
    
    plt.plot(X,Y)

if __name__ == '__main__':
    x = 90
    eta = 0.3
    error = 1e-3
    X = []
    X.append(x)
    y = target_function(x)
    y_last = y
    while True:
        x = x - eta * derivative_function(x)
        X.append(x)
        y = target_function(x)
        print("x=%f, y=%f" %(x,y))
        if abs(y - y_last) < error:
            break
        y_last = y


    draw_function()
    draw_gd(X)
    plt.show()

