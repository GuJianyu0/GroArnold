#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def func(x):
    return x**2*0.1
    
if __name__ == '__main__':

    x = np.linspace(0.,265.,266)
    print(x)
    y = func(x)
    func_interp_cubic = interpolate.interp1d(x, y, kind='cubic') #cubic spline
    func_interp_US =interpolate.UnivariateSpline(x,y,s=0) #s=0: compulsively passing all data points

    x0 = 5.6
    y01 = func_interp_cubic(x0)
    y02 = func_interp_US(x0)
    print(y01, y02)
