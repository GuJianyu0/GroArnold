#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    #Lz
    #filename = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galaxy_PM17/Plummer_20201111/txt/snap_0.txt"
    #filename = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galaxy_PM17/txt/snap_0.halo.txt"
    filename = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galaxy_PM17/txt/snap_0.disk.txt"
    data = np.loadtxt(filename)
    Lz = data[:,0]*data[:,4] -data[:,1]*data[:,3]
    Lz = abs(Lz)

    nmesh = 100
    nmeshmesh = range(nmesh)
    N_eff = len(Lz)
    Jmedian = np.median(Lz) #as scale
    Jmesh = np.linspace(0., max(Lz)/Jmedian, nmesh) #line
    print(Lz, Jmedian)

    Jdistribution = np.zeros(nmesh)
    for j in Lz:
        for n in range(nmesh-1):
            if j/Jmedian>Jmesh[n] and j/Jmedian<=Jmesh[n+1]:
                Jdistribution[n] += 1
    Jdistribution = Jdistribution/N_eff
    plt.scatter(nmeshmesh, Jdistribution, s=2.)
    plt.show()