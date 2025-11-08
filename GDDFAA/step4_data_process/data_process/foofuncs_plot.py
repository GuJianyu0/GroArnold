#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def foo(x):
    y = x
    if 0<=x<=0.5:
        y = 1.-6.*pow(x,2)+6.*pow(x,3)
    elif 0.5<x<=1.:
        y = 2*pow(1.-x,3)
    else:
        y = 0.
    return y

if __name__ == '__main__':

    x = np.linspace(0.,1.,1000)
    y = np.zeros(x.shape)
    l = len(x)

    for i in range(l):
        y[i] = foo(x[i])

    plt.plot(x,y)
    plt.show()