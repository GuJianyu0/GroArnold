#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

import galaxy_models
import analysis_data_distribution

if __name__ == '__main__':

    ## orbits data
    galbox = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"
    # galbox = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galsbox/"
    # galbox = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galsbox/gals_20210903_actions_spherical_1e4_centered/"
    model = "galsbox/galaxy_general_1_NFW"
    # model = "galaxy_general" #centerno
    # model = "galaxy_general_1_NFW"
    # model = "galaxy_general_2_Hernquist"
    # model = "galaxy_general_3_Burkert"
    # model = "galaxy_general_4_Einasto"
    # model = "galaxy_general_5_isothermal"
    # model = "galaxy_general_6_Plummer"
    whatkindofdata = "/snaps/txt/snapshot" #gadget output to txt
    # whatkindofdata = "/snaps/aa/allID" #tact action data output
    
    snapshots_count = 300 #period??
    particles_count = 10 #10000
    particle_target = 0
    dim = 3
    orbits_data = np.zeros((particles_count, dim, snapshots_count))
    for snapid in range(snapshots_count):
        ff = galbox+model
        filename = ff+whatkindofdata+"_%03d.txt" % (snapid)
        dataread = np.array(np.loadtxt(filename, dtype=float))
        orbits_data[:,:,snapid] = dataread[particle_target:particle_target+1,0:dim]
    print("snapshots done")

    ## F DFT
    # #:: example
    # N_sample = 140 #should it > max data freq
    # x = np.linspace(0,1,N_sample)
    # y = 7*np.sin(2*np.pi*18*x) + 1.5*np.sin(2*np.pi*39*x)+5.1*np.sin(2*np.pi*60*x) #3 kind of freqs, each wave number is less than 1./1400
    # yy = np.array(fft(y)) #DFT

    #::gadget orbits
    N_sample = snapshots_count
    time_step = 0.01 #by gadget running params
    x = np.arange(N_sample)*time_step #time
    y = orbits_data[0,0,:] #orbits points on x-axis of particle-1 #T=2*pi*300??
    print(y)
    print("len: ", len(x),len(y))
    yy = np.array(fft(y))

    xf = np.arange(N_sample) #freq *2*pi spectrum from 0 to len(data)
    yf = abs(yy) #norm as freq??
    xf1 = xf
    yf1 = yf/((N_sample/2)) #normalization

    # print(yf)
    plt.subplot(211)
    plt.plot(x,y,'k') #original signal
    plt.title('original signal',fontsize=9,color='k')
    plt.subplot(212)
    plt.plot(xf1,yf1,'r') #normalized freq spectrum 
    plt.title('FFT (normalized)',fontsize=9,color='r')
    plt.show()