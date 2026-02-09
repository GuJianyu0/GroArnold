#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tables
import matplotlib.pyplot as plt

if __name__ == '__main__':

    filename="./Category+/pieces/kdtree2/nearest_statistics.txt"
    A = np.loadtxt(filename)
    x=np.linspace(0,9999,10000)
    y=np.zeros(len(x))
    print(len(A))
    for i in range(0,9999):
        y[i]=len(A[A==x[i]])
    print(len(x), len(y))
    print(sum(y))

    fig=plt.figure()
    plt.plot(x,y)
    plt.savefig("number_neareastneibours_of_each_particle")
    plt.show()