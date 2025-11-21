#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    tag_plot = 2
    filenames = [
        "many_MG_params_fit.txt", 
        "many_xv_params_fit.txt", 
        "many_AA_params_fit.txt"
    ]
    filename = filenames[tag_plot]
    data = np.loadtxt(filename)

    N_pm, N_ss = np.shape(data)
    xplot = np.linspace(0., 0.7, 8)
    yplot = data[5]
    plt.plot(xplot, yplot)
    plt.show()
