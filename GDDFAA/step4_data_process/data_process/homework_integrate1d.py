#!/usr/bin/env python
# -*- coding:utf-8 -*-

##############################################
# Two integration method for 1D
##############################################

import numpy as np
import matplotlib.pyplot as plt



## the integrated function, 1D
def func(x):
    if x==0:
        return 0.
    else:
        # t = abs(x)
        # return t**(1./2) * np.log(t)
        return x**(1./2) * np.log(x)

## the primitive function of the integrated function, 1D
def Func(x):
    if x==0:
        return 0.
    else:
        return 2./3 * x**(2./3) * ( np.log(x) - 2./3 )



## compound trapezoidal rule method for 1D
def CTR(f, down, up, n=100):
    h = 1.*(up-down)/n
    Ak = np.zeros(n)
    for k in np.arange(n):
        xk = down+k*h
        Ak[k] = f(xk)+f(xk+h)
    return h/2 * np.sum(Ak)

## compound Simpson rule method for 1D
def CSR(f, down, up, n=100):
    h = 1.*(up-down)/n
    Ak = np.zeros(n)
    for k in np.arange(n):
        xk = down+k*h
        Ak[k] = f(xk) + 4*f(xk+h/2) + f(xk+h)
    return h/6 * np.sum(Ak)



if __name__ == '__main__':

    down = 0.
    up = 1.
    n = 1000
    I0 = Func(up)-Func(down)
    I1 = CTR(func, down, up, n=n)
    I2 = CSR(func, down, up, n=n)
    
    print("python3 homework_integrate1d.py #on shell")
    print("Results comparasion:")
    print("analytical result:  \t\t\t%f\n" \
        "compound trapezoidal rule method:  \t%f\n" \
        "compound Simpson rule method: \t\t%f\n" % (I0, I1, I2))
