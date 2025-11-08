#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

# from ....data_process import analysis_data_distribution as add
# from ....data_process import galaxy_models as gm
# from ....data_process import fit_rho_fJ as fff



def read_IC_settings(filename):

    Mass_vir = 0.
    frac_mass_comp1 = 0.
    mass_comp1 = 0.
    N_comp1 = 0
    v_sigma1 = 30.0 #??
    cold_alpha1   = 1.0
    cold_alphamax1 = 3.0
    seed1 = 487543

    file_handle = open(filename, mode="r")
    st = file_handle.readlines()
    # print(st)
    # print(st[-1])
    print(np.shape(st))

    for s in st:
        if ("Mass_vir" in s) and s[0]!="#":
            n = float(s.split()[1])
            print(n)
            Mass_vir = n
        if ("frac_mass_comp1" in s) and s[0]!="#":
            n = float(s.split()[1])
            print(n)
            frac_mass_comp1 = n
        if ("N_comp1" in s) and s[0]!="#":
            n = int(s.split()[1])
            print(n)
            N_comp1 = n
        if ("v_sigma1" in s) and s[0]!="#":
            n = float(s.split()[1])
            print(n)
            v_sigma1 = n
        if ("cold_alpha1" in s) and s[0]!="#":
            n = float(s.split()[1])
            print(n)
            cold_alpha1 = n
        if ("cold_alphamax1" in s) and s[0]!="#":
            n = float(s.split()[1])
            print(n)
            cold_alphamax1 = n
        if ("seed1" in s) and s[0]!="#":
            n = int(s.split()[1])
            print(n)
            seed1 = n

    mass1 = Mass_vir*frac_mass_comp1/N_comp1
    
    return mass1, N_comp1, v_sigma1, cold_alpha1, cold_alphamax1, seed1



if __name__ == '__main__':

    a = 1
