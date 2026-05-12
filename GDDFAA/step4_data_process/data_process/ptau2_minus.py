#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def norm_l(a,l=2):
    x=0
    for e in a:
        x += e**l
    return x**(1./l)

if __name__ == '__main__':
    
    Alpha = -1.1; Gamma = -1
    tau = 1.00001; taubargl = 1.00766; Deltagl = 0.00765638
    I = [-282697, 8589.93, -1.87584e+06]
    phi = -1.87386e+06

    a = I[0]*(tau+Gamma)
    b = -I[1]*(tau+Gamma)/(tau+Alpha)
    c = -I[2]
    d = phi
    e = 2*(tau+Gamma)*(tau+Alpha)

    JintA = a+b+c+d
    Jint = (a+b+c+d)/e
    print(a,b,c,d,e,JintA, Jint)
