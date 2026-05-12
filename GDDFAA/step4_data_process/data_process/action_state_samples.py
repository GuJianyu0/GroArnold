#!/usr/bin/env python
# -*- coding:utf-8 -*-

# In[]
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as spopt
from scipy.optimize import curve_fit
# from re import DEBUG
# from IPyhton.display import Latex

import affine_trans_group as atg
import galaxy_models as gm
import analysis_data_distribution as add
import fit_rho_fJ as fff

# In[]
# calculator



# In[]
if __name__ == '__main__':

    ##data debug
    path_base = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/"
    suffix = ""
    # filename = path_base+"aa/snapshot_5000.action.method_all"+suffix+".txt"
    filename = path_base+"aa/snapshot_5000.action_samples.method_all"+suffix+".txt"
    add.DEBUG_PRINT_V(1, "read ing ...")
    data = np.array(np.loadtxt(filename, dtype=float))
    add.DEBUG_PRINT_V(1, "read done")
    add.DEBUG_PRINT_V(1, data[0], len(data))

    xv = data[:,0:6]
    J = data[:,32:35]
    add.DEBUG_PRINT_V(1, "pointto done")

    ##plot
    bd = 1e5
    # bd = 1e5/2
    # bd = 2e6
    PLOT = fff.Plot_model_fit()
    ddl = [ [J, None, "label"] ]
    add.DEBUG_PRINT_V(1, "plot prepare done")
    PLOT.plot_x_scatter3d_dd(ddl, bd=bd, is_show=1)
    add.DEBUG_PRINT_V(1, "plot done")
