#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from sklearn.neighbors import KDTree
from scipy.interpolate import RBFInterpolator

import analysis_data_distribution as ads
import galaxy_models as gm
import KDTree_python as kdtp
import triaxialize_galaxy as tg



#### column index of actions data file
Dim = 3
mask_select_type = [1, 2, 0, 3] #[1]
col_x = 0
col_v = 3
col_particle_IDs=6
col_particle_mass=7
col_actions = 78 #Staeckel Fudge
col_frequencies = col_actions+7
col_particle_type=-6
col_potential = -4

#### path to data
galaxy_data_folder = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"
galaxy_name = sys.argv[1]
# galaxy_name = "galaxy_general"
aa_data = "aa/snapshot_%d.action.method_all.txt"
plot_folder = "fit/" #next to aa/
snapshot_ID = int(sys.argv[2])
# snapshot_ID = 10
snapshot_list = [snapshot_ID-2, snapshot_ID-1, snapshot_ID, snapshot_ID+1, snapshot_ID+2]
# snapshot_list = [8, 9, 10, 11, 12]
# is_show = True
is_show = False

#### running example
# $ python3 analysis_actions_output.py galaxy_general 10



#### functions
def plot_potential_contour():
    # ...
    return 0

def plot_actions_variaiton():
    # ...
    return 0



#### main
if __name__ == '__main__':

    for snapshot_ID in snapshot_list:
        aa_output_file = galaxy_data_folder+galaxy_name+aa_data%(snapshot_ID)

    plot_potential_contour()
    plot_actions_variaiton()
