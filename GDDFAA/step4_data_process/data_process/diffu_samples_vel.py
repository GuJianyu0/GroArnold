#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import warnings
import pandas as pd
import tikzplotlib
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors

import analysis_data_distribution as ads
import diffu_basic as dbc
import diffusion_coef_integration as dci
import observed_data_process as odp
import generate_fractal_sample as gfs
import fit_speed_DF_tail as fst
import diffu_calling_cpp as dcc



## main
if __name__ == '__main__':

    #### Step 1. settings
    R0 = dbc.R0_scale
    Rm = R0/dbc.lambda3
    M_total = dbc.M_total_gal_1e10MSun
    v0 = np.sqrt(dbc.G*M_total/dbc.frac_mass/R0)
    
    vmean_3 = np.array([ #the setted tree mean velocity of velocity DF
        0., 0., 0.
    ])
    v2m_sqrt = v0 #the setted rms of speed, as the typical speed
    v2m = v2m_sqrt*v2m_sqrt
    dispersion_3_iso_ratio = np.array([1., 1., 1.]) #the setted iso diag-component ratio
    dispersion_3_high_ratio = np.array([8., 6., 1.]) #the setted aniso diag-component ratio

    sig_iso = dbc.calculate_dispersion_iso(v2m, vmean_3)
    dispersion_3_iso = ads.rescale_dispersion_keep_ratio(vmean_3, dispersion_3_iso_ratio, v2m_sqrt)
    dispersion_3_high = ads.rescale_dispersion_keep_ratio(vmean_3, dispersion_3_high_ratio, v2m_sqrt)
    beta_z = 1.-(dispersion_3_high[0]**2+dispersion_3_high[1]**2)/(2.*dispersion_3_high[2]**2)
    ads.DEBUG_PRINT_V(1, M_total, R0, v0, v2m_sqrt, "galaxy_info basic")
    ads.DEBUG_PRINT_V(1, dispersion_3_iso, dispersion_3_high, beta_z, "galaxy_info dispersion")



    #### Step 2. samples and diffusion coefficients
    #: settings
    path_load = "../data/samples_vel/"
    path_save = "../data/examples_vel/"
    # is_run_exe = True
    is_run_exe = False
    # is_enable_plot_each = True
    is_enable_plot_each = False
    is_enable_plot_eta = True
    # is_enable_plot_eta = False

    # np.random.seed(75)
    vel_DFs = [
        "vel_iso", "vel_aniso", "vel_tail", "vel_noised", "vel_composite", 
        # "vel_simulated", "vel_obs", 
    ]

    vel_iso = None
    vel_aniso = None
    vel_tail = None
    vel_noised = None
    vel_composite = None
    vel_simu = None
    vel_obs = None
    vel_info_store_file = "../data/vel_info_store.csv"
    # fst.fit_speed_DF_tail_each_workflow()
    # fst.load_speed_fitted_paramters()

    #: run
    # N_particles_list = [10000]
    # N_particles_list = [10000, 30000]
    # N_particles_list = [100000]
    N_particles_list = [10000, 30000, 100000, 300000]
    # N_particles_list = [10000, 30000, 100000, 300000, 1000000]
    diffueff_iso_median_list, diffueff_iso_mean_list = None, None
    diffueff_aniso_median_list, diffueff_aniso_mean_list = None, None
    diffueff_tail_median_list, diffueff_tail_mean_list = None, None
    diffueff_noised_median_list, diffueff_noised_mean_list = None, None
    diffueff_composite_median_list, diffueff_composite_mean_list = None, None
    diffueff_referencevalue_list = None
    diffueff_simulated_list = None
    diffueff_observed_list = None
    diffueff_median_pt = list(range(len(N_particles_list)))
    diffueff_mean_pt = list(range(len(N_particles_list)))
    Dim_frac_list_pt = list(range(len(N_particles_list)))
    
    for (i_np, N_particles) in enumerate(N_particles_list):
        #: prepare arguments
        print("N_particles {}, begin.".format(N_particles))

        dict_consts = dbc.calculate_fixed_const_coefs(N_particles, M_total, R0, v0)
        print("dict_consts: \n", dict_consts)

        #: call C++ prog to generate samples and calculate diffusion coef
        #() note that one should make before calling C++
        exe_path = "../diffu_r_simple_each/./out_vel.exe"
        snapshot_name = "snapshot_for_"
        N_particles = N_particles #user set
        M_total_set = M_total #user set
        R0_set = R0 #user set
        is_by_generating = 1
        is_diffu = 1
        N_iter = 12 #user set
        # N_iter = 14 #user set
        vmean_3_x = vmean_3[0] #user set
        vmean_3_y = vmean_3[1] #user set
        vmean_3_z = vmean_3[2] #user set
        ratio_sigma_xx = dispersion_3_high_ratio[0] #user set
        ratio_sigma_yy = dispersion_3_high_ratio[1] #user set
        ratio_sigma_zz = dispersion_3_high_ratio[2] #user set
        if is_run_exe:
            dcc.run_cpp_program_vel(
                exe_path, snapshot_name, N_particles, M_total_set, R0_set, is_by_generating, is_diffu, N_iter, 
                vmean_3_x, vmean_3_y, vmean_3_z, ratio_sigma_xx, ratio_sigma_yy, ratio_sigma_zz, 
                path_save+"nohup.out"
            )

        #: load files and plot each sample
        tag_list = ["iso", "aniso", "tail", "noised", "composite"]
        vel_list = list(range(len(tag_list)))
        label_list = list(range(len(tag_list)))
        label_N = "N{}".format(N_particles)
        label_components = ["D_vel", "D_11", "D_22", "D_33", "D_12", "D_13", "D_23"]
        diffueff_median_list = np.zeros(len(tag_list))
        diffueff_mean_list = np.zeros(len(tag_list))
        Dim_frac_list = np.zeros(len(tag_list))
        for (i_tag, tag) in enumerate(tag_list):
            print("tag {}: {}".format(i_tag, tag))

            file_vel        = path_load+"vel_{}_N{}.txt".format(tag, N_particles)
            file_nr         = path_load+"nr_vel_{}_N{}.txt".format(tag, N_particles)
            file_diffu_each = path_load+"Diffu_vel_{}_N{}.txt".format(tag, N_particles)
            file_diffu_sta  = path_load+"DiffuStatistics_vel_{}_N{}.txt".format(tag, N_particles)
            file_diffu_pers = path_load+"DiffuPers_vel_{}_N{}.txt".format(tag, N_particles)
            suffix = "vel_{}_N{}".format(tag, N_particles)

            data_diffu_eff_each = None
            if is_enable_plot_each and N_particles<=100000: #much time to load
                vel = np.loadtxt(file_vel, dtype=float)[:, 3:6]
                vel_list[i_tag] = vel
                data_diffu_each = np.loadtxt(file_diffu_each, dtype=float)
                data_diffu_eff_each = data_diffu_each[:, 1]
                
            data_nr = np.loadtxt(file_nr, dtype=float)
            radii, inside_counts = data_nr[:, 0], data_nr[:, 1]
            h_frac, Dim_frac = gfs.calculate_mean_neareast_count_load(
                radii, inside_counts, save_path=path_save, suffix=suffix, is_plot=is_enable_plot_each
            )
            
            data_diffu_sta = np.loadtxt(file_diffu_sta, dtype=float)
            diffu_eff_mean = data_diffu_sta[0, 0]
            diffu_eff_median = data_diffu_sta[0, 1]
            diffu_eff_0 = data_diffu_sta[7, 1]
            data_diffu_pers = np.loadtxt(file_diffu_pers, dtype=float)
            label_list[i_tag] = suffix
            diffueff_median_list[i_tag] = diffu_eff_median
            diffueff_mean_list[i_tag] = diffu_eff_mean
            Dim_frac_list[i_tag] = Dim_frac
            ads.DEBUG_PRINT_V(1, diffu_eff_median, Dim_frac, "Dim_frac")

            if is_enable_plot_each and N_particles<=100000: #much time to plot
                k_neighbors = int(32*N_particles/1e4)
                dbc.plot_sample_3D_pos_or_vel(vel, path_save, Dim_frac=Dim_frac, suffix=suffix)
                dbc.plot_velocity_DF_contour_compare([vel], [suffix], save_path=path_save)
                dbc.plot_normalized_pdf_components(
                    data_diffu_each, label_components, diffu_eff_0, suffix=suffix, k_neighbors=k_neighbors
                )
                dbc.plot_normalized_pdf_from_eachpoints_histogram_vel_data(
                    data_diffu_eff_each, diffu_eff_0, suffix=suffix, vel_data=vel
                )
                # dbc.plot_normalized_pdf_from_eachpoints_KDE(
                #     data_diffu_eff_each, diffu_eff_0, suffix=suffix, k_neighbors=k_neighbors
                # )
                # dbc.plot_normalized_pdf_from_percentile(
                #     data_diffu_pers, diffu_eff_mean, diffu_eff_median, diffu_eff_0, suffix=suffix
                # )
        
        if is_enable_plot_each and N_particles<=100000: #much time to plot
            k_neighbors = 1000
            bandwidth = v2m_sqrt/10.
            dbc.plot_velocity_speed_distribution(
                vel_list, label_list, bandwidth=bandwidth, k_neighbors=k_neighbors, 
                save_path=path_save, suffix="{}".format(N_particles)
            )
        
        diffueff_median_pt[i_np] = diffueff_median_list
        diffueff_mean_pt[i_np] = diffueff_mean_list
        Dim_frac_list_pt[i_np] = Dim_frac_list
        print("N_particles {}, end.".format(N_particles))
        # exit(0) #debug #xxx compare tree, xxx eta and zeta

    #: plot eta_N about relaxation time for N_particles
    diffueff_median_pt = np.array(diffueff_median_pt)
    diffueff_mean_pt = np.array(diffueff_mean_pt)
    diffueff_median_p_iso = diffueff_median_pt[:, 0]
    diffueff_median_p_aniso = diffueff_median_pt[:, 1]
    diffueff_median_p_tail = diffueff_median_pt[:, 2]
    diffueff_median_p_noised = diffueff_median_pt[:, 3]
    diffueff_median_p_composite = diffueff_median_pt[:, 4]
    diffueff_mean_p_iso = diffueff_mean_pt[:, 0]
    diffueff_mean_p_aniso = diffueff_mean_pt[:, 1]
    diffueff_mean_p_tail = diffueff_mean_pt[:, 2]
    diffueff_mean_p_noised = diffueff_mean_pt[:, 3]
    diffueff_mean_p_composite = diffueff_mean_pt[:, 4]

    Dim_frac_list_pt = np.array(Dim_frac_list_pt)
    Dim_frac_mean_iso = np.mean(Dim_frac_list_pt[:, 0])
    Dim_frac_mean_noised = np.mean(Dim_frac_list_pt[:, 3])
    Dim_frac_mean_composite = np.mean(Dim_frac_list_pt[:, 4])
    ads.DEBUG_PRINT_V(1, Dim_frac_mean_iso, Dim_frac_mean_noised, Dim_frac_mean_composite, "Dim_frac_list_pt")
    
    if is_enable_plot_eta:
        N_particles_list = np.array(N_particles_list)
        hu = None
        du = Dim_frac_mean_iso
        hn = None
        dn = Dim_frac_mean_noised
        h0 = None
        d0 = 3.
        Dim_frac_lb = [dn, du, d0]

        diffueff_iso_median_list = [ #note: a pointer for array, need to copy
            N_particles_list, diffueff_median_p_iso, hu, du, "vel_iso", 
        ]
        diffueff_aniso_median_list = [
            N_particles_list, diffueff_median_p_aniso, hu, du, "vel_aniso", 
        ]
        diffueff_tail_median_list = [
            N_particles_list, diffueff_median_p_tail, hu, du, "vel_tail", 
        ]
        diffueff_noised_median_list = [
            N_particles_list, diffueff_median_p_noised, hn, dn, "vel_noised", 
        ]
        diffueff_composite_median_list = [
            N_particles_list, diffueff_median_p_composite, hn, dn, "vel_composite", 
        ]

        diffueff_iso_mean_list = [ #note: a pointer for array, need to copy
            N_particles_list, diffueff_mean_p_iso, hu, du, "vel_iso", 
        ]
        diffueff_aniso_mean_list = [
            N_particles_list, diffueff_mean_p_aniso, hu, du, "vel_aniso", 
        ]
        diffueff_tail_mean_list = [
            N_particles_list, diffueff_mean_p_tail, hu, du, "vel_tail", 
        ]
        diffueff_noised_mean_list = [
            N_particles_list, diffueff_mean_p_noised, hn, dn, "vel_noised", 
        ]
        diffueff_composite_mean_list = [
            N_particles_list, diffueff_mean_p_composite, hn, dn, "vel_composite", 
        ]

        suffix = "modified_vel"
        dbc.plot_relaxation_time_with_N_and_dim_vel(
            R0, v0, M_total, Dim_frac_lb, 
            diffueff_iso_median_list, diffueff_iso_mean_list, 
            diffueff_aniso_median_list, diffueff_aniso_mean_list, 
            diffueff_tail_median_list, diffueff_tail_mean_list, 
            diffueff_noised_median_list, diffueff_noised_mean_list, 
            diffueff_composite_median_list, diffueff_composite_mean_list, 
            diffueff_observed_list, diffueff_simulated_list, 
            diffueff_referencevalue_list, suffix
        )

    #: calculate amplification ratio zeta for vel
    # zeta_vel = diffueff_median_p_aniso/diffueff_median_p_iso
    # zeta_vel = diffueff_median_p_noised/diffueff_median_p_iso
    zeta_vel = diffueff_median_p_composite/diffueff_median_p_iso
    zeta_versus_N = np.hstack((np.array([N_particles_list]).T, np.array([zeta_vel]).T))
    np.savetxt(path_save+"zeta_vel.txt", zeta_versus_N)
    ads.DEBUG_PRINT_V(1, zeta_versus_N, "zeta_vel")
