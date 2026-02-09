#!/usr/bin/env python
# -*- coding:utf-8 -*-

# ============================================================================================
# Description: A wrapper to fit galaxy models of mass density and action probability density.
# Author: Jianyu Gu
# ============================================================================================



import sys
import os
import re, copy
import json, yaml
# import pdb
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import analysis_data_distribution as ads
import galaxy_models as gm
import change_params_galaxy_init as cpgi
import action_state_samples as asa
import transformation_some as ts
import KDTree_python as kdtp
import fit_galaxy_wrapper as fgw
import plot_galaxy_wrapper as pgw
import RW_data_CMGD as rdc
import actions_error_with_time as aet
import triaxialize_galaxy as tg



####[] step0: settings
filenme_IC = "../../step1_galaxy_IC_preprocess/step1_set_IC_DDFA/IC_param.txt"
mass1, N_comp1, v_sigma, cold_alpha, cold_alphamax, ls, seed = cpgi.read_IC_settings(filenme_IC) #??!! One should manually change gm.Mass_total
# output_folder_name = "fit1/"
# os.makedirs("savefig/"+output_folder_name, exist_ok=True)
# output_folder_name = cpgi.read_output_folder_name()
# os.makedirs("savefig/"+output_folder_name, exist_ok=True)
output_folder_name = cpgi.read_output_folder_name_from_user_settings()[0] #model1 name
# ads.DEBUG_PRINT_V(0, output_folder_name)

model_ID = 0 #no use
M = mass1*N_comp1
N_ptcs = N_comp1 #int(1e4)
ls = ls
ds = M/(4./3*np.pi*ls**3)
Js = (gm.G*M*ls)**0.5
ar = np.array([1., 0.6, 0.3])

N_neighbour = 1024
bd1 = np.inf
# bd2 = 1e4
# bd2 = 5e4
# bd2 = 1e5
bd2 = 1e6
# bd2 = 1e8 #bd_fit
# is_show = True
is_show = False

tag_task = int(sys.argv[1])
# snapshot_ID = 0
snapshot_ID = int(sys.argv[2])
is_actionerror = int(sys.argv[3])
name_MG = sys.argv[4] #to be the "default" or multi strings
print("is_actionerror:", is_actionerror)
# name_change = sys.argv[4]
suffix_action = ""
# suffix_action = ".variation"
gadget_type_name = None ##Gadget format: 0=Gas,1=Halo,2=Disk,3=Bulge,4=Stars; -1 is for all particles
mask_select_type = [1, 2, 0, 3] #[1]
mass_select_type = [1., 0., 0., 0.] #?? read

fitmassdensity_tag, fitAA_tag, is_preprocess_rotation = cpgi.read_fitmassdensity_tag(name_MG)
#\() Enable is_preprocess_rotation to be Fault for triaxialize file in galaxy_general/aa/ with [0.,0.,0.]
ads.DEBUG_PRINT_V(1, fitmassdensity_tag, fitAA_tag, is_preprocess_rotation, "fitmassdensity_tag, fitAA_tag, is_preprocess_rotation")

folder_name = "../../step2_Nbody_simulation/gadget/Gadget-2.0.7/"
folder_params_statistics = folder_name+"params_statistics/"
gm_name = "galaxy_general"
# gm_name = "galaxy_general"
# gm_name = "galaxy_general_4_EinastoUsual_triaxial_soft5.0_count1e4"
gm_names_inp = [""]
# if name_MG!="default":
#     gm_names_inp = re.split(r"[ ]+", name_MG)
#     # gm_name = gm_name+gm_names_inp[0] #wrong
mgs_name = None #read_and_compare_params_vectors
# if name_MG!="default":
#     mgs_name = cpgi.read_output_folder_name_from_user_settings()[0]
# else:
#     mgs_name = cpgi.read_output_folder_name_from_user_settings()
mgs_name = cpgi.read_output_folder_name_from_user_settings()
suffix = ""
# suffix = ".SCF"
folder_many_params_fit = "../../step2_Nbody_simulation/gadget/paramters_fit/"
galaxymodel_name = folder_name+gm_name+"/"
galaxyfit_name = folder_name+gm_name+"/fit/"

paramsresults_name = gm.paramsresults_name



####[] step1: functions
def main_step0_set_small():
    return 

def main_step1_set_MG(M, N_ptcs, ls, ds, ar):
    MG = fgw.Model_galaxy(M, ls, ds) #write MG
    Js = (gm.G*M*ls)**0.5
    # coef_boundary = 1.e2
    # coef_boundary = 1.e8
    coef_boundary = 1.e40
    scale_boundary = 1.e1
    scale_boundary = 1.e4
    # scale_boundary = 1.e2
    # power_boundary = 1.e2
    # power_boundary = 1.e3
    power_boundary = 1.e4
    axisratio_boundary = 1.e2
    MG.set_value("length_scale",    np.array([1., 1., 1./coef_boundary, 1.*coef_boundary])*ls)
    MG.set_value("density_scale",   np.array([1., 1., 1./coef_boundary, 1.*coef_boundary])*ds)
    MG.set_value("axis_ratio_x",    np.array([1., 1., 1./coef_boundary, 1.*coef_boundary])*ar[0])
    MG.set_value("axis_ratio_y",    np.array([1., 1., 1./coef_boundary, 1.*coef_boundary])*ar[1])
    MG.set_value("axis_ratio_z",    np.array([1., 1., 1./coef_boundary, 1.*coef_boundary])*ar[2])
    MG.set_value("rotate_angle_x",  np.array([0., 2*np.pi, 0., 2*np.pi])) # np.pi #divide by zero
    MG.set_value("rotate_angle_y",  np.array([0., 2*np.pi, 0., 2*np.pi]))
    MG.set_value("rotate_angle_z",  np.array([0., 2*np.pi, 0., 2*np.pi]))
    MG.set_value("action_scale",    np.array([Js, Js, Js/scale_boundary, Js*scale_boundary]))
    # MG.set_value("scale_free_1",    np.array([Js, Js, Js/scale_boundary, Js*scale_boundary]))
    MG.set_value("scale_free_1",    np.array([Js, Js, Js/scale_boundary, Js*scale_boundary]))
    MG.set_value("scale_free_2",    np.array([Js/1000., Js/1000., 0., Js]))
    MG.set_value("scale_free_3",    np.array([Js/10., Js/10., Js/10./scale_boundary, Js/10.*scale_boundary]))
    # MG.set_value("scale_free_3",    np.array([Js/100., Js/100., 0., Js/2.]))
    MG.set_value("scale_free_4",    np.array([Js, Js, Js/scale_boundary, Js*scale_boundary]))
    MG.set_value("log_penalty",     np.array([-10., -10., -100., 1.]))
    # MG.set_value("coef_total",      np.array([1., 1., 1.-0.1, 1.+0.1]))
    # MG.set_value("coef_total",      np.array([1., 1., 1.e-3, 1.e3]))
    MG.set_value("coef_total",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_n3",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_n2",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_n1",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_0",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_1",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_2",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_3",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_4",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_5",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_6",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_7",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_8",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_9",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p0",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p1",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p2",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p3",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p4",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p5",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p6",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p7",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p8",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p9",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_exp_1",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_exp_2",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_exp_3",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_axis_1",     np.array([1., 1., 1./axisratio_boundary, axisratio_boundary]))
    MG.set_value("coef_axis_2",     np.array([1., 1., 1./axisratio_boundary, axisratio_boundary]))
    MG.set_value("coef_axis_3",     np.array([1., 1., 1./axisratio_boundary, axisratio_boundary]))
    MG.set_value("power_alpha",     np.array([1., 1., 0.,               power_boundary]))
    MG.set_value("power_beta",      np.array([3., 3., 0.,               power_boundary]))
    MG.set_value("power_total",     np.array([1., 1., 1e-2,               1e1]))
    MG.set_value("power_free_1",    np.array([1., 1., 1e-2,               1e1]))
    MG.set_value("power_free_2",    np.array([1., 1., 1e-2,               1e1]))
    MG.set_value("power_free_3",    np.array([1., 1., 1e-2,               1e1]))
    MG.set_value("power_free_4",    np.array([1., 1., 1e-2,               1e1]))
    MG.set_value("power_Einasto",   np.array([1., 1., 1e-2,               1e1]))
    # MG.set_value("power_Einasto",   np.array([1., 0.5, 1./power_boundary,               0.5])) #larger action is less

    return MG

def main_step2_DF(tag, snapshot_ID, is_read=False, is_grid=False):
    ## 0. values
    ps, mass = None, None
    tgts, DF = None, None
    xe, ye = None, None
    cl, meds = None, None
    DF_name = None
    spin_L_old_direct = None
    spin_L_processed_direct = None
    particle_type = None

    cols = [0,1,2]
    pers = [0.5, 20., 50., 80., 99.5]

    ## 2.1. ps
    if int(tag)==1:
        path_file = galaxymodel_name+"txt/snapshot_%03d.txt"%(snapshot_ID)
        DF_name = "DF_x_mass"

        RG = rdc.Read_galaxy_data(path_file)
        # RG.data = RG.data[0:100] #cut data
        RG.AAAA_set_particle_variables(
            col_particle_x_coordinates=0, col_particle_v_velocities=3, 
            col_particle_IDs=6, col_particle_mass=8, col_particle_potential=14, col_particle_type=7
        )

        x = RG.particle_x_coordinates # copy.deepcopy()
        v = RG.particle_v_velocities
        mass = RG.particle_mass
        pot_old = RG.particle_potential
        particle_type = RG.particle_type ##particle_type of gadget setting
        
        N_dim = RG.system_space_dimension
        N_ptc = RG.system_particles_count
        ads.DEBUG_PRINT_V(1, np.shape(x), np.shape(v))
        # spin_L_old_direct = ads.calculate_spin_L_before_preprocess(np.hstack((x, v)), mass, pot_old)
        spin_L_old_direct = tg.spin_lambda_Nbody(x, v, mass, pot_old)

        frac_unbounded_old = tg.frac_unbounded_particles(v, pot_old)
        print("The fraction of unbounded particles before preprocess is ", frac_unbounded_old)

        x, v, operators = main_step3_triaxialize(x, v, mass, snapshot_ID, is_preprocess_rotation=is_preprocess_rotation)
        # ps = np.hstack((x,v)) #new xv

        # vv:
        xv = np.hstack((x, v))
        bd = bd1
        xv_cl, cl, cln = ads.screen_boundary_some_cols(xv, cols, -bd, bd, value_discard=None)
        ads.check_x_un_finite(xv, "xv after screen_boundary_some_cols")
        ps = xv #now: ps, mass, cl, meds

    ## 2.2. ps
    if int(tag)==2:
        DF_name = "DF_AA_one"
        path_file = None
        if is_actionerror:
            path_file = galaxymodel_name+"aa/snapshot_%d.action.method_all%s.txt"%(snapshot_ID, ".variation")
        else:
            path_file = galaxymodel_name+"aa/snapshot_%d.action.method_all%s.txt"%(snapshot_ID, "")
        
        RG = rdc.Read_galaxy_data(path_file)
        # RG.data = RG.data[0:100] #cut data
        RG.AAAA_set_particle_variables(
            col_particle_IDs=7-1, col_particle_mass=8-1, col_particle_type=-6
        )

        data = RG.data
        mass = RG.particle_mass
        IDs = RG.particle_IDs
        particle_type = RG.particle_type ##particle_type of gadget setting
        
        Dim = gm.Dim #3
        iast = 28
        adur = 10
        AA_TF_FP = data[:, iast+adur*0:iast+adur*0+adur]
        AA_OD_FP = data[:, iast+adur*1:iast+adur*1+adur] #none
        AA_GF_FP = data[:, iast+adur*2:iast+adur*2+adur] #none
        iast += adur*5 # = 78
        AA_TF_DP = data[:, iast+adur*0:iast+adur*0+adur]
        AA_OD_DP = data[:, iast+adur*1:iast+adur*1+adur]
        AA_GF_DP = data[:, iast+adur*2:iast+adur*2+adur] #none

        # vv:
        AA_method = AA_TF_DP
        Act = AA_method[:, 0:3]
        Ang = AA_method[:, 3+1:7]
        Fre = AA_method[:, 7:10]
        ads.DEBUG_PRINT_V(1, AA_TF_DP.shape, Act.shape, Fre.shape)
        AA = np.hstack((Act, Fre))
        bd = bd2
        AA_cl, cl, cln = ads.screen_boundary_some_cols(AA, cols, 1e-2, bd, value_discard=None) #the AA has been changed after this function
        # AA_cl, cl, cln = ads.screen_boundary_some_cols(AA, cols, 1e-2, bd, value_discard=bd*1e20) #the AA has been changed after this function
        # AA_cl, cl, cln = ads.screen_boundary_some_cols(AA, cols, 1./bd, bd, value_discard=bd*1e4)
        ps = AA #now: ps, mass, cl, meds
        # spin_L_processed_direct = ads.calculate_spin_L_before_preprocess(xv=data[:,0:6], mass=data[:,7], pot=data[:,-4])
        spin_L_processed_direct = tg.spin_lambda_Nbody(data[:,0:3], data[:,3:6], mass=data[:,7], pot=data[:,-4]) #TTF, no eli rot vel
        # ads.DEBUG_PRINT_V(1, np.mean(data[:, 0]), np.median(np.abs(data[:, 0])), "mean and median of data[0]")
        # ads.DEBUG_PRINT_V(0, np.mean(data[:, 3]), np.median(np.abs(data[:, 3])), "mean and median of data[0]")
        # ads.DEBUG_PRINT_V(1, np.shape(data))
        # ads.DEBUG_PRINT_V(1, 1e-2, bd)
        # ads.DEBUG_PRINT_V(1, np.max(Act))
        # i_Act = np.argmax(np.sum(Act, axis=1))
        # ads.DEBUG_PRINT_V(1, np.max(Fre))
        # ads.DEBUG_PRINT_V(1, np.max(AA))
        # i_bad = np.argmax(np.sum(AA, axis=1))
        # ads.DEBUG_PRINT_V(1, data[i_bad], data[i_bad, 78:78+10])
        # ads.DEBUG_PRINT_V(1, i_Act, i_bad, Act[i_bad], Fre[i_bad], AA[i_bad])
        # ads.DEBUG_PRINT_V(0, np.max(AA_cl), len(AA_cl))

    else:
        print("No such tag provided. None.")

    ## 3. DF
    if is_read:
        path_write = galaxymodel_name+"aa/snapshot_%d_%s.txt"%(snapshot_ID, DF_name)
        data_DF = np.loadtxt(path_write)
        tgts = data_DF[:,0:6] #[adjust] fit
        DF = data_DF[:,6]
    else:
        if not is_grid:
            ## [select]: samples: grid
            # tgts = (ps[:, cols])[cl] #note: used cl
            # tgts = ps[cl] #note: used cl
            tgts = np.abs(ps[cl]) #note: used cl
            if xe is not None:
                xe = xe[cl]
            if ye is not None:
                ye = ye[cl]
            # ads.DEBUG_PRINT_V(0, np.max(AA_cl), np.max(tgts), "max_AA")
        else:
            ## [select]: samples: grid
            N_grid = 100000
            # tag_func = 0 #[adjust] fit
            tag_func = 1
            # tag_func = 2
            # tag_func = 3
            tgts = ads.generate_actions_grid_3d_3d(bd, N_grid, tag_func=tag_func)
            xe = tgts #??
            ye = DF

            ##: if nn point
            # from sklearn.neighbors import KDTree
            # tree_tgts = KDTree(AA_col)
            # distances, indices = tree_tgts.query(A3, k=1)
            # AA = AA_col[indices]
            ##: if knn Fre
            # [] SPH Fre by Act

        # density:
        KD = kdtp.KDTree_galaxy_particles(tgts[:,cols], weight_extern_instinct=mass)
        # targets = [[0.,0.,0.], [1e2, 1e2, 1e2]]
        ads.DEBUG_PRINT_V(1, np.shape(tgts), "tgts")
        DF = KD.density_SPH(tgts[:,cols]) #some are None #?? debug
        print("%s: "%(DF_name), DF) #now: tgts, DF

        # #?? read xe by stde function, xe +- and  xe to ye, less foci orbit time, action error DF, write, fractal local, to 4o fractal resonance
        # if is_actionerror and int(tag)==2:
        #     fne = galaxymodel_name+"aa/snapshot_%d.action.method_all%s.txt"%(snapshot_ID, ".error")
        #     if os.path.exists(fne):
        #         data_fne = np.loadtxt(fne)
        #         xe = data_fne[cl] #used cl
        #         ye = aet.roughly_yerror_and_mask(xe, tgts, np.log10(DF)) #used cl
        #         #\this ye is after log10 #?? DF_error_log10 = k*DF_log10 or DF_error_log10 = k_log10+DF_log10
        #         # ads.DEBUG_PRINT_V(0, np.shape(xe), np.shape(ye), np.mean(ye), np.mean(np.log10(DF)), ye)
        #     else:
        #         # print("The error file `%s` do not exist. Set the default error be 0."%(fne))
        #         # xe = None #ps*0. #before cl
        #         raise ValueError("The error file `%s` do not exist. Exit.")

        # record: 
        path_write = galaxymodel_name+"aa/snapshot_%d_%s.txt"%(snapshot_ID, DF_name)
        data_write = np.hstack(( tgts, np.array([ DF ]).T ))
        RG.write_numpy_savetxt(path_write, data_write)

    ## 4. returned
    if int(tag)==1:
        meds = [
            np.percentile(ps[cl,0], pers), 
            np.percentile(ps[cl,1], pers), 
            np.percentile(ps[cl,2], pers), 
            np.percentile(ads.norm_l(ps[cl,0:3], axis=1), pers), 
            np.percentile(ps[cl,3], pers), 
            np.percentile(ps[cl,4], pers), 
            np.percentile(ps[cl,5], pers), 
            np.percentile(ads.norm_l(ps[cl,3:6], axis=1), pers), 
            np.percentile(DF, pers)
        ]
    if int(tag)==2:
        meds = [
            np.percentile(ps[cl,0], pers), 
            np.percentile(ps[cl,1], pers), 
            np.percentile(ps[cl,2], pers), 
            np.percentile(np.sum(ps[cl,0:3], axis=1), pers), 
            np.percentile(ps[cl,3], pers), 
            np.percentile(ps[cl,4], pers), 
            np.percentile(ps[cl,5], pers), 
            np.percentile(ads.norm_l(ps[cl,3:6], axis=1), pers), 
            np.percentile(DF, pers)
        ]
    ads.DEBUG_PRINT_V(1, np.shape(ps), np.shape(cl), np.shape(tgts), "shapes: ps, cl, tgts")
    print("medians of data ps and DF, tag=%d (for actions and frequencies, the total is summation instead of norm_l2): "%(tag), meds)
    
    # param00 = [spin_L_old_direct, spin_L_processed_direct]
    param00 = [spin_L_old_direct, spin_L_processed_direct, particle_type]
    return ps, mass, tgts, DF, xe, ye, cl, meds, param00

def main_step3_triaxialize(x, v, mass, snapshot_ID, is_preprocess_rotation=True):
    # triaxialize:
    # boundary_rotate_refer = None
    # boundary_rotate_refer = [1., 1e6] #kpc, km/s
    boundary_rotate_refer = [20., 1e6] #kpc, km/s
    # boundary_rotate_refer = [60., 1000.] #kpc, km/s #old
    # boundary_rotate_refer = [80., 1000.] #kpc, km/s #usually
    # x, v, operators = tg.all_triaxial_process( #only move
    #     x, v, mass, is_centralize_coordinate=True, 
    #     is_rotate_mainAxisDirection=False, r_range=boundary_rotate_refer, 
    #     is_eliminate_totalRotation=False, is_by_DF=False, DF=None
    # )

    # mask = (component_type == 0)  # "halo" type or any selection
    # x_sel, v_sel, m_sel = x[mask], v[mask], mass[mask]
    # x_rot, v_rot, ops = tg.all_triaxial_process(x_sel, v_sel, m_sel, True, True, r_range=None, is_eliminate_totalRotation=False)

    x, v, operators = tg.all_triaxial_process(
        x, v, mass, is_centralize_coordinate=True, 
        is_rotate_mainAxisDirection=True, r_range=boundary_rotate_refer, 
        is_eliminate_totalRotation=is_preprocess_rotation, is_by_DF=False, DF=None
    )

    # unpack the triaxialize parameters and write:
    x_mean_old, v_mean_old = operators[0][0], operators[0][1] #arrays 3 and 3 for translation
    T = operators[1][0] #array 3*3 for rotation
    OA = operators[2][0] #array 3 for elimination #only in IC before simulation
    print("The triaxialization process values:")
    print(x_mean_old, v_mean_old)
    print(T)
    print(OA)

    galaxymodel_name = folder_name+gm_name+"/"
    path_file = galaxymodel_name+"aa/snapshot_%d_triaxialize.txt"%(snapshot_ID) #preprocess_snapshot
    
    file_handle = open(path_file, mode="w")
    file_handle.write("## operater values for actions under triaxial potential\n")
    file_handle.write("%e %e %e %e %e %e %s \n"%(x_mean_old[0], x_mean_old[1], x_mean_old[2], 
        v_mean_old[0], v_mean_old[1], v_mean_old[2], "#arrays 3 and 3 for translation centers of xv Cartesian"))
    file_handle.write("%e %e %e %e %e %e %e %e %e %s \n"%(T[0,0], T[0,1], T[0,2], 
        T[1,0], T[1,1], T[1,2], T[2,0], T[2,1], T[2,2], 
        "#array 3*3 for rotation main axis of interia moment"))
    file_handle.write("%e %e %e %s \n"%(OA[0], OA[1], OA[2], 
        "#array 3 for elimination total angular velocity #only in IC before simulation"))
    file_handle.close()
    print("Write `%s` ... Done."%(path_file))

    # write particles what SCF reads
    path_SCF_usual = folder_name+gm_name+"/"+"aa/galaxy_general.SCF.txt"
    tg.write_preprocessed_SCF(x, v, mass, path_file=path_SCF_usual)
    # ads.DEBUG_PRINT_V(0, tg.centralize_coordinate(x, v, mass))
    return x, v, operators

def get_arguments_auto_from_MG(names, params_dict, col=1):
    n = len(names)
    p0 = list(range(n))
    for i in np.arange(n):
        p0[i] = params_dict[names[i]][col]
    return p0

def assign_fitvalues_auto_to_MG(values_order, names, MG, col=0): #note: change value
    n = len(names)
    for i in np.arange(n):
        # MG.params_dict[names[i]][col] = values_order[i]
        nm = names[i]
        # lazily create/extend per-parameter slots in MG.params_dict
        if nm not in MG.params_dict:
            # infer a typical row length from any existing entry; fallback to at least col+1
            _len = next((len(v) for v in MG.params_dict.values() if hasattr(v, "__len__")), 0)
            _len = max(_len, col + 1, 1)
            MG.params_dict[nm] = [0.0] * _len
        row = MG.params_dict[nm]
        if col >= len(row):
            row.extend([0.0] * (col + 1 - len(row)))
        row[col] = float(values_order[i])
    return 0

def mask_particle_type_and_compute_density(
    tgts,                   # (N, d) array of target points (already pre-masked for Nones/OOB upstream)
    mass,                   # (N,) mass array (same length as tgts)
    type_each_particle,     # (N,) int array of particle types
    certaintype             # int (the particle type to keep)
):
    """
    Filter particles by `certaintype` and compute a mass-weighted SPH KDE density
    for the filtered subset.

    Returns
    -------
    targets_certaintype : (N_sel, d) array
    DF_certaintype      : (N_sel,)   array  (mass-weighted density at each target)
    mass_certaintype    : (N_sel,)   array

    Failure behavior
    ----------------
    - If inputs are inconsistent or selection is empty, raise SystemExit with a clear message
      (so downstream won't silently proceed with wrong data).
    """
    # --- basic checks
    if tgts is None or mass is None or type_each_particle is None:
        raise SystemExit("Input arrays must not be None.")

    tgts = np.asarray(tgts)
    mass = np.asarray(mass).reshape(-1)
    type_each_particle = np.asarray(type_each_particle).reshape(-1)

    if tgts.shape[0] != mass.shape[0] or mass.shape[0] != type_each_particle.shape[0]:
        raise SystemExit(
            "Length mismatch: "
            f"N_tgts={tgts.shape[0]}, N_mass={mass.shape[0]}, N_type={type_each_particle.shape[0]}"
        )

    # --- sanitize NaNs quietly (very small, conservative drop)
    good = np.isfinite(mass) & np.all(np.isfinite(tgts), axis=1) & np.isfinite(type_each_particle)
    if not np.all(good):
        tgts = tgts[good]
        mass = mass[good]
        type_each_particle = type_each_particle[good]
        if tgts.size == 0:
            raise SystemExit("All entries filtered out as non-finite.")

    # --- select the requested type
    try:
        tsel = int(certaintype)
    except Exception:
        raise SystemExit(f"certaintype '{certaintype}' is not an int.")

    sel = (type_each_particle == tsel)
    if sel.sum() == 0:
        raise SystemExit(f"No particles for type={tsel}. Exit.")

    targets_certaintype = tgts[sel]
    mass_certaintype    = mass[sel]

    # --- compute mass-weighted density (SPH-style) at the subset locations
    KD = kdtp.KDTree_galaxy_particles(targets_certaintype, weight_extern_instinct=mass_certaintype)
    DF_certaintype = KD.density_SPH(targets_certaintype)   # same points as tree nodes

    # basic sanity
    if DF_certaintype is None or len(DF_certaintype) != len(targets_certaintype):
        raise SystemExit("Wrong length of input arguments.")

    # optional lightweight debug
    # ads.DEBUG_PRINT_V(1, targets_certaintype.shape, DF_certaintype.shape, mass_certaintype.shape,
    #                   f"type={tsel} subset shapes")

    return targets_certaintype, DF_certaintype, mass_certaintype

def main_step4_mpfit_fit_DF_x_mass(
    ps, mass, tgts, DF, xe, ye, cl, meds, MG, param00, 
    is_fit_1d=True, #no use
    fitmassdensity_tag=-1, #no use
    type_each_particle=None, #type of each particle, same length with tgts
    particle_type_select=1 #halo
):
    ####[] tables for params, with axisRatio, ds, rs, 
    #: processed data
    tgts, DF, mass = mask_particle_type_and_compute_density(
        tgts, mass[cl], type_each_particle[cl], particle_type_select
    ) #cl
    
    vinp, vout = np.abs(tgts), np.log10(DF) #xdata, ydata
    Js_data = meds[3][2]
    Js_expected = np.sqrt(gm.G*M*ls)
    axisratio_direct = ads.axis_ratio_by_configuration_data(ps[:,0:3]) #tgts
    
    vinp_err, vout_err = xe, ye
    if ye is not None:
        vout_err = np.log10(ye)
    # binsxy = pgw.plot_1d_DF_by_bins(vinp[:, 0:3], is_show=is_show)

    #: parameters by direct calculation
    paramsresult1 = {} #paramsresults
    paramsresult1.update({"scale_length_comp1_median_direct": np.median( ads.norm_l(ps[:,0:3], axis=1) )})
    paramsresult1.update({"flatx_comp1_direct": axisratio_direct[0]})
    paramsresult1.update({"flaty_comp1_direct": axisratio_direct[1]})
    paramsresult1.update({"flatz_comp1_direct": axisratio_direct[2]})
    paramsresult1.update({"spin_L_old_direct": param00[0]})
    paramsresult1.update({"beta_sigma_total_direct": ads.calculate_beta_sigma_total_from_xv(ps)})

    #: data process before fitting
    combmodel = [
        None
    ]
    h = gm.AA_combination_freeCoef(vinp, axisratio_direct[1], axisratio_direct[2]) #might be zero
    # np.savetxt("tmp_data/1.txt", ads.merge_array_by_hstack([h,vout]))
    # ads.DEBUG_PRINT_V(0, np.mean(vinp, axis=0))
    # bd = bd1
    # h, vout = ads.screen_remove_far_back(h, vout, False) #optional if back bad points and write
    hbd, fbd, febd, xsbs, nxss = ads.split_various_bin_1d_percentile(h, vout, None, [0., 0.01, 0.99, 1.], [10, 100, 10], True) #curve
    # hbd, fbd, febd, xsbs, nxss = ads.split_various_bin_1d_percentile(h, vout, None, [0., 1.], [1000], False) #curve
    # hbd, fbd, febd, xsbs, nxss = ads.split_various_bin_1d_percentile(h, vout, None, [0., 1.], [1000], True) #curve
    # plt.figure()
    # plt.scatter(h, vout)
    # plt.scatter(hbd, fbd)
    # plt.xscale("log")
    # plt.show()
    # plt.close()
    ads.DEBUG_PRINT_V(1, np.max(h), np.max(h))

    #: fit model
    fitmodelfunc = None
    if is_fit_1d:
        fitmodelfunc = [
            [
                gm.rhorq_DPL_log10, 
                "rhorq_DPL_log10", 
                [
                    "scale_free_1", "scale_free_2", 
                    "power_free_1", "power_free_2", 
                    "log_penalty"
                ], 
                None, #the fixed params
                None, #the value of unfixed fit params
                None  #the spetical settings in mpfit
            ]
            , 
            [
                gm.rhorq_Einasto_more_log10, 
                "rhorq_Einasto_more_log10", 
                # gm.rhorq_Einasto_simplify_log10, 
                # "rhorq_Einasto_simplify_log10", 
                [
                    "scale_free_1", "scale_free_2", 
                    "power_free_1", "power_free_2", 
                    "log_penalty"
                ], 
                None, #the fixed params
                None, #the value of unfixed fit params
                None  #the spetical settings in mpfit
            ]
            , 
            [
                gm.rhorq_MDPL_core_and_cuspy_log10, 
                "rhorq_MDPL_core_and_cuspy_log10", 
                [
                    "scale_free_1", "scale_free_2", 
                    "power_free_1", "power_free_2", 
                    "coef_free_1", 
                    "log_penalty"
                ], 
                None, #the fixed params
                None, #the value of unfixed fit params
                None  #the spetical settings in mpfit
            ]
        ]
    else:
        fitmodelfunc = [
            [
                gm.DFXV_fCombinationFixed_DPL_log, #this function will be modifed to fix some params, the two axistatios
                "DFXV_fCombinationFixed_DPL_log", 
                [
                    "power_free_1", "power_free_2", 
                    "length_scale", "density_scale", 
                    "log_penalty"
                ], 
                axisratio_direct[1:3], #the fixed params
                None, #the value of unfixed fit params
                None  #the spetical settings in mpfit
            ]
        ]

    N_combmodel = len(combmodel)
    N_fitmodelfunc = len(fitmodelfunc)
    DL = [[0 for i in range(N_combmodel)] for j in range(N_fitmodelfunc)]

    #: fit arguments for each fitmodelfunc
    rs0 = paramsresult1["scale_length_comp1_median_direct"]
    rhos0 = M/rs0**3
    ads.DEBUG_PRINT_V(1, rs0, rhos0, "rs0, rhos0")
    power1_0 = 1.
    power2_0 = 3.

    p0_0 = np.array([ rs0, rhos0, power1_0, power2_0 ])
    bounds_0 = [ p0_0*1e-4, p0_0*10. ]

    p0_1 = np.array([ rs0, rhos0, power1_0, 1. ])
    bounds_1 = [ p0_1*1e-4, p0_1*100. ]
    bounds_1[0][0] = -10.
    bounds_1[1][1] = 1e5
    
    p0_2 = np.array([ rs0, rhos0, power1_0, power2_0, 1. ])
    bounds_2 = [ p0_2*1e-4, p0_2*10. ]
    
    p0_pack = [p0_0, p0_1, p0_2]
    bounds_pack = [bounds_0, bounds_1, bounds_2]

    #: select fitmodelfunc and fit (here the combmodel is absorbed)
    for j in [fitmassdensity_tag]:
        p0 = p0_pack[j]
        bounds = bounds_pack[j]
        ads.DEBUG_PRINT_V(1, fitmassdensity_tag, fitmodelfunc[j][0], p0, bounds[0], bounds[1])
        MF = fgw.MP_fit(fitmodelfunc[j][0], p0, 
            xdata=hbd, ydata=fbd, errdata=None, bounds=bounds, 
            quiet=0)

        mparams = MF.run()
        fitmodelfunc[j][4] = mparams
        assign_fitvalues_auto_to_MG(fitmodelfunc[j][4], fitmodelfunc[j][2][:-1], MG)
        # ads.DEBUG_PRINT_V(0, mparams)

    #: after fitting, send value from the fit MG to total param #paramsresults
    paramsresult1.update({"scale_length_comp1_fit": MG.params_dict["length_scale"][0]})
    paramsresult1.update({"scale_density_comp1_fit": MG.params_dict["density_scale"][0]})
    paramsresult1.update({"fitmodelfunc_name_xv_fit": fitmodelfunc[0][1]})
    paramsresult1.update({"powerS1_fit": -1.})
    paramsresult1.update({"powerA1_fit": MG.params_dict["power_free_1"][0]})
    paramsresult1.update({"powerB1_fit": MG.params_dict["power_free_2"][0]})
    paramsresult1.update({"powerA2_fit": -1.})
    paramsresult1.update({"powerB2_fit": -1.})

    return DL, combmodel, fitmodelfunc, paramsresult1, hbd, fbd, h, DF

def main_step4_mpfit_fit_DF_x_mass_brief(r_data, DF_data, mass):
    #: data
    q_axisratio = ads.axis_ratio_by_configuration_data(r_data)
    rq_data = ( (r_data[:,0])**2+(r_data[:,1]/q_axisratio[1])**2+(r_data[:,2]/q_axisratio[2])**2 )**0.5
    DF_data_log10 = np.log10(DF_data)
    M = np.sum(mass)
    rs = np.median(r_data) #a reference value of scale length
    rhos = M/(4./3.*np.pi*rs**3) #a reference value of scale density

    #: fit setting
    fitfunction_1d = gm.rhorq_DPL_rhos_rs_log10
    p0 = [rs, rhos, 1., 3.]
    bounds = [
        [rs*1e-2, rhos*1e-2, 0.01, 0.03], 
        [rs*1e2, rhos*1e2,   10.,  10. ]
    ]

    # fitfunction_1d = gm.rhorq_TPL_rhos_rs_log10
    # p0 = [rs, rs*10., rhos, 1., 3., 1]
    # bounds = [
    #     [rs*1e-2, rs*1e-1, rhos*1e-2, 0.01, 0.03, 0.01], 
    #     [rs*1e2,  rs*1e3,  rhos*1e2,  10.,  10.,  0.01]
    # ]
    
    #: prepare bins, note that here we ignroed the forward-model bins
    percentile_split_used = [0., 0.01, 0.99, 1.]
    N_bins_arr_used      = [10, 100, 10]
    is_geomspace_used    = True
    rq_data_bin, DF_data_bin_log10, _, _, _ = ads.split_various_bin_1d_percentile(
        rq_data, DF_data_log10, None, percentile_split_used, N_bins_arr_used, is_geomspace_used
    ) #no forward-model bins

    #: fit
    # quiet = 0 #print least
    quiet = 1 #not print least
    MF = fgw.MP_fit(fitfunction_1d, p0, 
        xdata=rq_data_bin, ydata=DF_data_bin_log10, errdata=None, bounds=bounds, 
        maxiter=1000, quiet=quiet)
    fit_params = MF.run(is_fit_log=False) #mpfit CHI-SQUARE should be low, influenced by initial value
    DF_fitvalue_log10 = fitfunction_1d(rq_data_bin, *fit_params)
    return rq_data, DF_data_log10, rq_data_bin, DF_data_bin_log10, DF_fitvalue_log10, fit_params, q_axisratio

def main_step5_mpfit_fit_DF_AA_one(
    ps, mass, tgts, DF, xe, ye, cl, meds, MG, param00=None, 
    fitmodelfunc=None, is_fit_1d=True, value_vectors_extern=None, 
    particle_type=None, particle_type_select=None, 
    recompute_df_for_selected=True
):
    """
    Here recalculate actions density for each type and then fit.
    """
    #: per-type filtering (we added earlier)
    do_type_filter = (particle_type is not None) and (particle_type_select is not None)
    counts_type = ads.check_count_of_particle_type(particle_type, mask_select_type)
    ads.DEBUG_PRINT_V(1, mask_select_type, counts_type, "counts_type")
    if do_type_filter: #mask_select_type
        ptype_cl = particle_type[cl]
        if ptype_cl is not None:
            sel = (ptype_cl == int(particle_type_select))
            if sel.sum() == 0: #??
                # return None, None, None, {"note": "empty selection"}, None, None
                raise SystemExit(f"No particles for type={particle_type_select}. Wrong settings, please check particle_type_select. Exit.")
            tgts = tgts[sel]
            DF   = DF[sel] if DF is not None else DF
            if xe is not None: xe = xe[sel]
            if ye is not None: ye = ye[sel]
            #build a matching mass array for the filtered set
            mass_cl = mass[cl] if (mass is not None and cl is not None) else mass
            mass = mass_cl[sel] if mass_cl is not None else None
            #recompute meds on filtered subset (best-effort)
            pers = [0.5, 20., 50., 80., 99.5]
            meds = ads.percentiles_by_angle_action_data(tgts, DF, pers)
            ads.DEBUG_PRINT_V(1, np.shape(ps), np.shape(cl), np.shape(tgts), "shapes: ps, cl, tgts")
            print("medians of data ps and DF of a component type (for actions and frequencies, the total is summation instead of norm_l2): ", meds)

            #per-type KDE (mass-weighted) to recompute DF of the subset
            if recompute_df_for_selected:
                # DF_new = _recompute_df_kde_per_type(tgts, mass)
                KD = kdtp.KDTree_galaxy_particles(tgts, weight_extern_instinct=mass)
                DF_new = KD.density_SPH(tgts)
                if DF_new is not None:
                    DF = DF_new

            #record
            DF_name = "DF_AA_one"
            path_write = galaxymodel_name+"aa/snapshot_%d_%s_type_%d.txt"%(snapshot_ID, DF_name, int(particle_type_select))
            data_write = np.hstack(( tgts, np.array([ DF ]).T ))
            np.savetxt(path_write, data_write)

    #: processed data, tables for params, with axisRatio, ds, rs, etc.
    vinp, vout_data = tgts, np.log10(DF) #xdata and ydata after log10
    AA_sum = np.sum(vinp[:, 0:3], axis=1)
    Js_data = meds[3][2]
    Js_expected = np.sqrt(gm.G*M*ls)
    axisratio_action_direct = ads.axis_ratio_by_configuration_data(vinp) #tgts

    vinp_err, vout_err = xe, ye #yerror_data after log10
    # if ye is not None:
    #     vout_err = np.log10(ye)
    # binsxy = pgw.plot_1d_DF_by_bins(vinp[:, 0:3], is_show=is_show)

    paramsresult2 = {} #paramsresults
    paramsresult2.update({"Jsum_median_direct": meds[3][2]})
    paramsresult2.update({"Jl_median_direct": meds[0][2]})
    paramsresult2.update({"Jm_median_direct": meds[1][2]})
    paramsresult2.update({"Jn_median_direct": meds[2][2]})
    paramsresult2.update({"Ol_median_direct": meds[4][2]})
    paramsresult2.update({"Om_median_direct": meds[5][2]})
    paramsresult2.update({"On_median_direct": meds[6][2]})
    paramsresult2.update({"actions_ratio_m_direct": axisratio_action_direct[1]})
    paramsresult2.update({"actions_ratio_n_direct": axisratio_action_direct[2]})
    paramsresult2.update({"spin_L_processed_direct": param00[1]})

    #: models
    combmodel = [
        None
    ]
    h = gm.AA_combination_sumWeightFrequency_rateF1(vinp) #might be zero# for halo
    # h = np.abs(gm.AA_combination_sumWeightFrequency_rateF1(vinp)) #might be zero
    # ads.DEBUG_PRINT_V(0, np.mean(vinp, axis=0)) #[8e3, 4e3, 2e3, 14., 13., 11.]
    # bd = bd2
    h, vout = ads.screen_remove_far_back(h, vout_data, h_far=5e5, is_mask_log10f=False) #optional if back bad points and write
    # h, vout = ads.screen_remove_far_back(h, vout_data, h_far=1e6, is_mask_log10f=False) #optional if back bad points and write
    
    ## action state density 1d: forward-model bins, store the exact binning recipe so model and data use *identical* bins
    percentile_split_used = [0., 0.01, 0.99, 1.]
    N_bins_arr_used      = [10, 100, 10]
    is_geomspace_used    = True
    hbd, fbd, febd, xsbs, nxss = ads.split_various_bin_1d_percentile(
        h, vout, vout_err, percentile_split_used, N_bins_arr_used, is_geomspace_used
    ) #curve
    
    # ads.DEBUG_PRINT_V(0, np.shape(fbd), np.shape(febd))
    # hbd, fbd, febd, xsbs, nxss = ads.split_various_bin_1d_percentile(h, vout, vout_err, [0., 0.01, 0.99, 1.], [20, 60, 20], True) #curve
    # hbd, fbd, febd, xsbs, nxss = ads.split_various_bin_1d_percentile(h, vout, vout_err, [0., 0.01, 0.99, 1.], [100, 100, 3], True) #curve #sometimes for Einasto
    # hbd, fbd, febd, xsbs, nxss = ads.split_various_bin_1d_percentile(h, vout, vout_err, [0., 0.01, 0.95, 1.], [10, 100, 10], False) #curve
    # plt.figure()
    # plt.scatter(h, vout)
    # plt.scatter(hbd, fbd)
    # plt.xscale("log")
    # plt.show()
    # plt.close()
    ads.DEBUG_PRINT_V(1, np.max(h), np.max(h))

    #if caller supplies a fitmodelfunc, deep-copy and use it; otherwise build default
    if fitmodelfunc is not None:
        fitmodelfunc = copy.deepcopy(fitmodelfunc)  # do not mutate caller's list
    elif is_fit_1d:
        fitmodelfunc = [
            [
                gm.fh_MPLTF_log10, 
                "fh_MPLTF_log10", 
                [
                    "scale_free_1", "scale_free_2", "scale_free_3", "scale_free_4", 
                    "power_free_1", "power_free_2", "power_free_3", 
                    "coef_free_1", 
                    "log_penalty"
                ], 
                None, #the fixed params
                None, #the value of unfixed fit params
                None  #the spetical settings in mpfit
            ]
        ]
    else: #not is_fit_1d
        fitmodelfunc = [
            [
                gm.DFAA_fCombinationFreq_MTPL_log, 
                "DFAA_fCombinationFreq_MTPL_log", 
                [
                    "power_free_1", "power_free_2", "power_free_3", 
                    "scale_free_1", "scale_free_2", 
                    "log_penalty"
                ], 
                None, #the fixed params
                None, #the value of unfixed fit params
                None  #the spetical settings in mpfit
            ]
        ]

    N_combmodel = len(combmodel)
    N_fitmodelfunc = len(fitmodelfunc) #here only 1
    DL = [[0 for i in range(N_combmodel)] for j in range(N_fitmodelfunc)]

    #: fit
    for j in np.arange(N_fitmodelfunc):
        #: if user provided [p0, bounds] in fitmodelfunc[j][5], prefer it; else keep legacy defaults below
        _settings = fitmodelfunc[j][5] if len(fitmodelfunc[j]) > 5 else None
        # ads.DEBUG_PRINT_V(0, _settings, "_settings")
        if _settings is not None:
            p0 = np.array(_settings[0], dtype=float)
            bounds = [np.array(_settings[1][0], dtype=float), np.array(_settings[1][1], dtype=float)]
        else:
            raise SystemExit(f"[settings] ERROR: No p0 and bounds setted.")

        # #: EinastoUsual 2e5, fh_MPLTF_log10
        # p0 = np.array([
        #     1.02165602e+04, 2.00000000e+04, 6.06867397e+04, 8.06867397e+05, \
        #     1.e+00, 2.e+00, 1.2e+00, \
        #     -0.5
        # ])
        # bounds = [ -p0*100, p0*100 ]
        # bounds[0][0] = 0.
        # bounds[0][1] = 0.
        # bounds[0][2] = 0.
        # bounds[0][3] = 0.
        # bounds[0][7] = -100.
        # bounds[1][7] = 100.
        # # bounds[1][-3] = 0.
        # # bounds[1][-3] = 100.

        # #: EinastoUsual 1e6, fh_SPL_log10
        # p0 = np.array([
        #     1.02165602e+04, 6.06867397e+04, 8.06867397e+05, \
        #     2.75e+00, 1.09e-01, \
        #     5.0e-1, 5.e-1, 5.e-1, 5.e-1, 5.e-1
        # ])
        # # p0 = np.array([
        # #     1.02165602e+04, 2.00000000e+04, 6.06867397e+04, 8.06867397e+05, \
        # #     1.1e-01, 2.75e+00, 1.09e-01, \
        # #     5.0e-1, 5.e-1, 5.e-1, 5.e-1, 5.e-1
        # # ])
        # bounds = [ -p0*100, p0*100 ]
        # bounds[0][0] = 0.
        # bounds[0][1] = 0.
        # bounds[0][2] = 0.
        # bounds[0][3] = 0.
        # bounds[0][4] = -100.
        # bounds[0][5] = -100.
        # bounds[0][6] = -100.
        # bounds[0][7] = -1000. #?? k23 too low
        # bounds[1][8] = 100.
        # # bounds = [ p0/10000000, p0*100 ]
        # # bounds[0][6] = -10.
        # # bounds[0][7] = -10.
        # # bounds[0][-3] = -100.
        # # bounds[0][-2] = -10.

        # #: EinastoUsual 5e5, fh_TPL_log10
        # p0 = np.array([3.0e4, 2.0e3, 5.1e4, 1.1, 1.90, 0.1, 0.15])
        # bounds = [ p0*0., p0*1000 ]
        # # bounds[1][1] = 2e10
        # # bounds[1][2] = 2e10
        # # bounds[1][3] = 1e5
        # bounds[0][3] = -10.
        # bounds[0][4] = -10.
        # bounds[0][5] = -10.
        # # bounds[1][5] = 10.

        # #: DPL
        # p0 = np.array([3.0e4, 2.0e3, 5.1e4, 1.1, 1.90, 0.1, 0.15])
        # # p0 = np.array([2.5e4, 2.0e3, 4.1e4, 0.5, 1.60, 0.01, 0.19])
        # bounds = [ p0/1000, p0*100 ]
        # # bounds = [ p0/100, p0*10 ]
        # bounds[0][5] = -1.
        # bounds[1][5] = 1.
        ads.DEBUG_PRINT_V(1, fitmodelfunc[j][0], p0, bounds[0], bounds[1])
        
        MF = None
        # quiet = 0 #print least
        quiet = 1 #not print least
        if is_fit_1d:

            # ## [select] no forward-model bins
            # if is_actionerror:
            #     MF = fgw.MP_fit(fitmodelfunc[j][0], p0, 
            #         xdata=hbd, ydata=fbd, errdata=febd, bounds=bounds, 
            #         maxiter=1000, quiet=quiet) #?? bad fit result
            # else:
            #     MF = fgw.MP_fit(fitmodelfunc[j][0], p0, 
            #         xdata=hbd, ydata=fbd, errdata=None, bounds=bounds, 
            #         maxiter=1000, quiet=quiet)

            ## [select] action state density 1d: forward-model bins, evaluate the DF on *each particle's* h(J),
            #\ then aggregate with the *same bins* used to build (hbd, fbd).
            _base_df_fn = fitmodelfunc[j][0]
            def _fh_forward1d_log10(
                _hbd_ignored, *theta, _fn=_base_df_fn, _h=h, _ps=percentile_split_used,
                _nb=N_bins_arr_used, _geo=is_geomspace_used
            ):
                # model log10 f for every particle (vectorized)
                y_model_i = _fn(_h, *theta)
                # bin with the exact same scheme used for the data
                _hbd_m, _fbd_m, _, _, _ = ads.split_various_bin_1d_percentile(
                    _h, y_model_i, None, _ps, _nb, _geo
                )
                return _fbd_m  # same shape as fbd
            if is_actionerror:
                MF = fgw.MP_fit(_fh_forward1d_log10, p0,
                    xdata=hbd, ydata=fbd, errdata=febd, bounds=bounds,
                    maxiter=1000, quiet=quiet)
            else:
                MF = fgw.MP_fit(_fh_forward1d_log10, p0,
                    xdata=hbd, ydata=fbd, errdata=None, bounds=bounds,
                    maxiter=1000, quiet=quiet)

            ads.DEBUG_PRINT_V(1, np.shape(hbd), "np.shape(hbd)")
            ads.DEBUG_PRINT_V(1, fitmodelfunc[j][0], p0, bounds, "p0")
        
        else:
            MF = fgw.MP_fit(fitmodelfunc[j][0], p0, 
                xdata=vinp, ydata=vout_data, errdata=None, bounds=bounds, 
                maxiter=1000, quiet=quiet)

        mparams = MF.run(is_fit_log=False) #mpfit CHI-SQUARE should be low, influenced by initial value
        # mparams = MF.run(is_fit_log=True)
        # ads.DEBUG_PRINT_V(0, mparams, fitmodelfunc[j][2])
        fitmodelfunc[j][4] = mparams #to record
        assign_fitvalues_auto_to_MG(fitmodelfunc[j][4], fitmodelfunc[j][2][:-1], MG)

    # update paramsresult2 #paramsresults
    paramsresult2.update({"fitmodelfunc_name_AA_fit": fitmodelfunc[0][1]})
    paramsresult2.update({"J1_scale_fit": MG.params_dict["scale_free_1"][0]})
    paramsresult2.update({"J2_scale_fit": MG.params_dict["scale_free_2"][0]})
    paramsresult2.update({"J3_scale_fit": MG.params_dict["scale_free_3"][0]}) #might not fit
    paramsresult2.update({"J4_scale_fit": MG.params_dict["scale_free_4"][0]}) #might not fit
    paramsresult2.update({"poly_coeff_k1_fit": MG.params_dict["coef_free_1"][0]}) #might not fit
    paramsresult2.update({"poly_coeff_k2_fit": MG.params_dict["coef_free_2"][0]}) #might not fit
    # paramsresult2.update({"poly_coeff_k3_fit": MG.params_dict["coef_free_3"][0]}) #might not fit
    # paramsresult2.update({"poly_coeff_k4_fit": MG.params_dict["coef_free_4"][0]}) #might not fit
    # paramsresult2.update({"poly_coeff_k5_fit": MG.params_dict["coef_free_5"][0]}) #might not fit
    paramsresult2.update({"actions_coef_free_m_fit": MG.params_dict["coef_axis_1"][0]}) #might not fit
    paramsresult2.update({"actions_coef_free_n_fit": MG.params_dict["coef_axis_2"][0]}) #might not fit
    paramsresult2.update({"powerAA_E1_fit": -1})
    paramsresult2.update({"powerAA_P1_fit": MG.params_dict["power_free_1"][0]})
    paramsresult2.update({"powerAA_P2_fit": MG.params_dict["power_free_2"][0]})
    paramsresult2.update({"powerAA_P3_fit": MG.params_dict["power_free_3"][0]})
    paramsresult2.update({"powerAA_P4_fit": MG.params_dict["power_free_4"][0]}) #might not fit
    if value_vectors_extern is not None:
        paramsresult2.update({"spin_L_parameter_P1_direct": value_vectors_extern[0]})
        paramsresult2.update({"rotate_sig_parameter_P1_direct": value_vectors_extern[1]})
        paramsresult2.update({"beta_parameter_P1_direct": value_vectors_extern[2]})
        paramsresult2.update({"beta_z_parameter_P1_direct": value_vectors_extern[3]})
        paramsresult2.update({"spin_L_parameter_P2_direct": value_vectors_extern[4]})
        paramsresult2.update({"rotate_sig_parameter_P2_direct": value_vectors_extern[5]})
        paramsresult2.update({"beta_parameter_P2_direct": value_vectors_extern[6]})
        paramsresult2.update({"beta_z_parameter_P2_direct": value_vectors_extern[7]})

    return DL, combmodel, fitmodelfunc, paramsresult2, hbd, fbd, tgts, DF

def main_step6_record_and_plot_fitvalues(MG, DL, combmodel, fitmodelfunc, tag=2):
    ##[] unpack DL
    galaxy_basic_info = [
        MG.params_dict["mass"][1], #total mass
        N_ptcs, #total particle
        MG.params_dict["length_scale"][1], 
        MG.params_dict["density_scale"][1], 
        MG.params_dict["action_scale"][1]
    ]
    galaxy_fit_params_to_compare = DL[0][0][6]
    print("galaxy_basic_info: ", galaxy_basic_info)
    print("galaxy_basic_info: debug: ", MG.params_dict["power_Einasto"][0])
    print("galaxy_fit_params_to_compare: ", galaxy_fit_params_to_compare)

    N_combmodel = len(combmodel)
    N_fitmodelfunc = len(fitmodelfunc)
    N_pf = len(galaxy_fit_params_to_compare)
    Js_data = DL[0][0][7][2]
    Js_expected = DL[0][0][7][3]
    meds = DL[0][0][7][4]
    print("scale_action by data: ", Js_data)
    print("scale_action by expected: ", Js_expected)

    ##[] record one (select DL[j=0][i=0]) #preprocess_snapshot
    galaxymodel_name = folder_name+gm_name+"/"
    galaxy_params_fit = galaxymodel_name+"aa/"+"snapshot_%d_DF_params_fit"%(snapshot_ID)
    
    #: tag0
    galaxy_params_settings = galaxy_params_fit+".MG.txt"
    file_handle = open(galaxy_params_settings, mode="w")
    # file_handle.write("## settings \n")
    file_handle.write("%d # %s \n"%(0, "model_ID"))
    file_handle.write("%e # %s \n"%(M, "total mass"))
    file_handle.write("%e # %s \n"%(N_ptcs, "total count of particles (comp1)"))
    file_handle.write("%e # %s \n"%(ls, "expected scale length"))
    file_handle.write("%e # %s \n"%(ds, "expected scale density"))
    file_handle.write("%e # %s \n"%(Js_expected, "expected scale action"))
    file_handle.write("%e # %s \n"%(ar[0], "expected axis_ratio_x"))
    file_handle.write("%e # %s \n"%(ar[1], "expected axis_ratio_y"))
    file_handle.write("%e # %s \n"%(ar[2], "expected axis_ratio_z"))
    file_handle.close()
    
    #: tag1 or tag2
    if tag==1:
        galaxy_params_fit += ".xv.txt"
    elif tag==2:
        galaxy_params_fit += ".AA.txt"
    else:
        print("tag: ", tag)
        print("No such tag provided. Exit.")
        exit(0)
    file_handle = open(galaxy_params_fit, mode="w")
    # file_handle.write("## paramters by %s \n"%(fitmodelfunc[0][1]))
    file_handle.write("%d # %s \n"%(0, "model_ID"))
    for k in np.arange(N_pf):
        file_handle.write("%e # %s \n"%(galaxy_fit_params_to_compare[k], fitmodelfunc[0][2][k]))
    #: fit params of simple DPL: "coef_free_p1", "coef_free_p2", "power_free_1", "power_free_2", "density_scale"
    file_handle.write("%e # %s \n"%(meds[0][2], "scale: median of x or actions, coor1"))
    file_handle.write("%e # %s \n"%(meds[1][2], "scale: median of x or actions, coor2"))
    file_handle.write("%e # %s \n"%(meds[2][2], "scale: median of x or actions, coor3"))
    file_handle.write("%e # %s \n"%(meds[3][2], "scale: median of x or actions, l2_norm"))
    file_handle.write("%e # %s \n"%(meds[4][2], "scale: median of v or frequencies, coor1"))
    file_handle.write("%e # %s \n"%(meds[5][2], "scale: median of v or frequencies, coor2"))
    file_handle.write("%e # %s \n"%(meds[6][2], "scale: median of v or frequencies, coor3"))
    file_handle.write("%e # %s \n"%(meds[7][2], "scale: median of v or frequencies, l2_norm"))
    file_handle.close()
    print("Write parameters of a galaxy ... Done.")

    ##[] plot one
    # Js = Js_data
    # print("the selected plot-scale_action is by data: ", Js)
    Js = Js_expected
    print("the selected plot-scale_action is by expected: ", Js)
    for j in np.arange(N_fitmodelfunc):
        dp_fit = [0 for i in range(N_combmodel)]
        dp_err = [0 for i in range(N_combmodel)]
        dp_local_mean_var0 = [0 for i in range(N_combmodel)]
        dp_local_scatter_var0 = [0 for i in range(N_combmodel)]
        percentiles = None
        for i in np.arange(N_combmodel):
            print("plot: ", j,i)
            # Jcombine = DL[j][i][1]/Js
            Jcombine = DL[j][i][1]
            percentiles = np.percentile(Jcombine, [0.5, 99.5])
            dp_fit[i] = [
                [Jcombine, DL[j][i][2], "VB_%s versus VC_data"%(combmodel[i][1]), 0], #k=0
                [Jcombine, DL[j][i][3], "VB_%s versus VC_fit"%(combmodel[i][1]), 0]  #k=1
            ]
            dp_err[i] = [
                [Jcombine, DL[j][i][7][0], 
                    "VB_%s versus VC_residual_sigma"%(combmodel[i][1]), 0]
            ]
            dp_local_mean_var0[i] = [
                [Jcombine, DL[j][i][2], "VB_%s versus VC_data"%(combmodel[i][1]), 0], 
                [Jcombine, DL[j][i][4], "VB_localAverage_%s versus VC_data"%(combmodel[i][1]), 0]
            ]
            dp_local_scatter_var0[i] = [
                [Jcombine, DL[j][i][5], 
                    "VB_localScatter_%s versus VC_data"%(combmodel[i][1]), 0]  #k=0
            ]
        # end for i

        #: plot fit paramters:
        PLOT = pgw.Plot_model_fit()
        datapack = [dp_fit, dp_err, dp_local_mean_var0, dp_local_scatter_var0] #the plot input is this datapack
        xl = [
            "scaled combination_actions", 
            "scaled combination_actions", 
            "scaled combination_actions localAverage", 
            "scaled combination_actions localScatter"
        ]
        yl = [
            "log10 DF/1.", "residual of log10 DF/1.", 
            "log10 DF/1.", "log10 DF/1."
        ]
        sn = output_folder_name+"/"+"fitmodelfunc_%d_"%(j)+"DFAA_"+fitmodelfunc[j][1]+"_versus_combmodel_plot"\
            +"_of_"+gm_name+"_snapshot_%d"%(snapshot_ID)+suffix
        ep = str(Js)
        tx = str(galaxy_fit_params_to_compare)
        st = sn+"\ninfo = "+ep+"\nthe fitvalue: "+tx
        lim = None
        lim = [[0.,0.],[-20.,-2.]] #None
        PLOT.plot_actions_Comb_NDF_subplot(
            datapack, xl=xl, yl=yl, lim=lim, bd_much=percentiles, 
            suppertitle=st, savename=sn, is_show=is_show
        )
    # end for j

    PLOT = pgw.Plot_model_fit()
    # mcmc and plot
    # PLOT.plot_x_scatter3d_general()
    return tag

def plot_1d_fit_massdensity(rq_data, DF_data_log10, rq_data_bin, DF_data_bin_log10, DF_fitvalue_log10=None, pathfig=None):
    fig = plt.figure()
    plt.scatter(rq_data, DF_data_log10, label="data points", s=2.)
    plt.scatter(rq_data_bin, DF_data_bin_log10, label="data bins", s=2.)
    if DF_fitvalue_log10 is not None:
        plt.scatter(rq_data_bin, DF_fitvalue_log10, label="fit bins", s=0.5)
    # plt.xlim(0., 1e3)
    plt.legend()
    plt.xscale("log")
    plt.xlabel(r"axis-ratio radius, $r_q$ (kpc)")
    plt.ylabel(r"log10 value of mass density, $\rho$ ($\mathrm{1e10\, M_\odot/kpc)^3}$")
    plt.savefig(pathfig)
    print("Plot and save 1d mass density, done.")
    return 0

def plot_comb_and_DF_in_2d_and_3d(func, fitvalue, xdata, ydata, tag_comb=0):

    #[] 1-1, 2d, value
    xcomb = None #xdata is JO (actions and frequencies), whose shape is (N,6)
    if tag_comb==0:
        xcomb = gm.AA_combination_sum(xdata)
    elif tag_comb==1:
        xcomb = gm.AA_combination_sumWeightFrequency_rateF1(xdata)
    else:
        xcomb = gm.AA_combination_freeCoef(xdata, tag_comb[0], tag_comb[1])
    yfit = func(xdata, *fitvalue)

    f_list = None
    is_save = True
    PLOT = pgw.Plot_model_fit()
    
    dim = 2
    plot_list = [
        [xcomb, ydata], 
        [xcomb, yfit]
    ]
    label_list = [
        "NDF_data", 
        "NDF_fit"
    ]

    xyzlim = [
        [1., bd2], 
        None
        # [-20., -2.]
    ]
    scalevalue = [
        Js, 
        None
    ]
    xyztitle = [
        "lincomb_actions (log10 unit)", 
        "NDF (log10 unit)"
    ]
    xyzlogscale = [
        True, 
        False
    ]

    pathname = output_folder_name
    figurename = "value_of_data_and_fit_"+gm_name+"_snapshot_%d"%(snapshot_ID)+suffix
    textname = "scale: "+str(Js)+"\nfitvalue: "+str(fitvalue)
    
    PLOT.plot_scatter_2d_or_3d_with_color(
        plot_list, f_list=f_list, label_list=label_list, 
        dim=dim, xyzlogscale=xyzlogscale, xyzlim=xyzlim, xyztitle=xyztitle, scalevalue=scalevalue, 
        pathname=pathname, figurename=figurename, textname=textname, is_save=is_save, is_show=is_show
    )

    #[] 1-2, 2d, error
    # y_relative_error = yfit/ydata-1.
    y_relative_error = yfit-ydata
    plot_list = [
        [xcomb, y_relative_error]
    ]
    label_list = [
        "NDF_relative_error"
    ]

    figurename = "relative_error_of_data_and_fit_"+gm_name+"_snapshot_%d"%(snapshot_ID)+suffix
    
    PLOT.plot_scatter_2d_or_3d_with_color(
        plot_list, f_list=f_list, label_list=label_list, 
        dim=dim, xyzlogscale=xyzlogscale, xyzlim=xyzlim, xyztitle=xyztitle, scalevalue=scalevalue, 
        pathname=pathname, figurename=figurename, textname=textname, is_save=is_save, is_show=is_show
    )

    #[] 3d
    if 0:
        J_l = xdata[:,0]
        J_mn = xcomb-J_l
        yfit = func(xdata, *fitvalue)
        # ads.DEBUG_PRINT_V(0, np.min(yfit), np.max(yfit))

        dim = 3
        plot_list = [
            [J_l, J_mn, ydata], 
            [J_l, J_mn, yfit]
        ]
        label_list = [
            "NDF_data", 
            "NDF_fit"
        ]

        xyzlim = [
            [1./bd2*1e2, bd2], 
            [1./bd2*1e2, bd2], 
            None
            # [-20., -2.]
        ]
        scalevalue = [
            Js, 
            Js, 
            None
        ]
        xyztitle = [
            "lincomb_actions_l (log10 unit)", 
            "lincomb_actions_mn (log10 unit)", 
            "NDF (log10 unit)"
        ]
        xyzlogscale = [
            True, 
            True, 
            False
        ]

        pathname = output_folder_name
        figurename = "value_of_data_and_fit_3d_"+gm_name+"_snapshot_%d"%(snapshot_ID)+suffix
        textname = "scale: "+str(Js)+"\nfitvalue: "+str(fitvalue)
        
        PLOT.plot_scatter_2d_or_3d_with_color(
            plot_list, f_list=f_list, label_list=label_list, 
            dim=dim, xyzlogscale=xyzlogscale, xyzlim=xyzlim, xyztitle=xyztitle, scalevalue=scalevalue, 
            pathname=pathname, figurename=figurename, textname=textname, is_save=is_save
        )

        #[] 2-2, 3d, error
        y_relative_error = yfit/ydata-1.
        plot_list = [
            [J_l, J_mn, y_relative_error]
        ]
        label_list = [
            "NDF_relative_error"
        ]
        xyzlim = [
            [1./bd2*1e2, bd2], 
            [1./bd2*1e2, bd2], 
            None
            # [-20., -2.]
        ]
        
        figurename = "relative_error_of_data_and_fit_3d_"+gm_name+"_snapshot_%d"%(snapshot_ID)+suffix
        
        PLOT.plot_scatter_2d_or_3d_with_color(
            plot_list, f_list=f_list, label_list=label_list, 
            dim=dim, xyzlogscale=xyzlogscale, xyzlim=xyzlim, xyztitle=xyztitle, scalevalue=scalevalue, 
            pathname=pathname, figurename=figurename, textname=textname, is_save=is_save
        )

    return 0

def main_step7_write_info_and_params(gm_names):
    ####[] write galaxy info, params about xv and params about AA 
    # and write multi models and compare
    # If you want to compare snapshots of one model only, just let len(gm_names) be 1
    print("The models names are:")
    for gn in gm_names:
        print(gn)
    print("by user input.")
    
    ##[] read
    params0 = []
    params1 = []
    params2 = []
    name_p0 = None
    name_p1 = None
    name_p2 = None
    idx = 0
    for gn in gm_names:
        galaxymodel_name_folder = ""
        if gn=="" or gn==None:
            galaxymodel_name_folder = folder_name+"galaxy_general"+"/"+"aa/"
        else:
            galaxymodel_name_folder = folder_name+"galaxy_general"+"_"+gn+"/"+"aa/"
        name_ld = os.listdir(galaxymodel_name_folder)
        name_ld.sort()
        read_did = [False, False, False]
        for name_file in name_ld:
            if name_file.find("_DF_params_fit.MG.txt")!=-1:
                pm = np.loadtxt(galaxymodel_name_folder+name_file)
                params0.append(pm)
                if read_did[0]==False:
                    name_p0 = ads.read_single_comment_of_each_line(
                        galaxymodel_name_folder+name_file, comment_symbol="#")
                    read_did[0] = True
            if name_file.find("_DF_params_fit.xv.txt")!=-1 and name_file!="example_snapshot_DF_params_fit.xv.txt":
                pm = np.loadtxt(galaxymodel_name_folder+name_file)
                params1.append(pm)
                if read_did[1]==False:
                    name_p1 = ads.read_single_comment_of_each_line(
                        galaxymodel_name_folder+name_file, comment_symbol="#")
                    read_did[1] = True
            if name_file.find("_DF_params_fit.AA.txt")!=-1:
                pm = np.loadtxt(galaxymodel_name_folder+name_file)
                params2.append(pm)
                if read_did[2]==False:
                    name_p2 = ads.read_single_comment_of_each_line(
                        galaxymodel_name_folder+name_file, comment_symbol="#")
                    read_did[2] = True
            idx += 1
    N_ss = len(params0)
    if not( len(params0)==len(params1) and len(params0)==len(params2) ):
        print("len of MGs: ", len(params0))
        print("len of XVs: ", len(params1))
        print("len of AAs: ", len(params2))
        print("Error when running: The counts of galaxies in type \".MG\", \".xv\" and \".AA\" is not same. Exit.")
        exit(0)
    N_pp0 = len(params0[0])
    N_pp1 = len(params1[0])
    N_pp2 = len(params2[0])
    ads.DEBUG_PRINT_V(1, [N_pp0, N_pp1, N_pp2], N_ss, "counts of multi MG")

    ##[] write
    params0_record = np.zeros((N_pp0, N_ss))
    params1_record = np.zeros((N_pp1, N_ss))
    params2_record = np.zeros((N_pp2, N_ss))
    for j in np.arange(N_ss):
        for i in np.arange(N_pp0):
            params0_record[i,j] = params0[j][i]
        for i in np.arange(N_pp1):
            params1_record[i,j] = params1[j][i]
        for i in np.arange(N_pp2):
            params2_record[i,j] = params2[j][i]

    path_params0 = folder_many_params_fit+"many_MG_params_fit.txt"
    np.savetxt(path_params0, params0_record)
    fh = open(path_params0, mode="r")
    cs = fh.readlines()
    fh.close()
    for i in np.arange(N_pp0):
        cs[i] = cs[i][:-1]+name_p0[i]+cs[i][-1] #cs[-1] is "\n"
    fh = open(path_params0, mode="w")
    fh.writelines(cs)
    fh.close()
    path_params1 = folder_many_params_fit+"many_xv_params_fit.txt"
    np.savetxt(path_params1, params1_record)
    fh = open(path_params1, mode="r")
    cs = fh.readlines()
    fh.close()
    for i in np.arange(N_pp1):
        cs[i] = cs[i][:-1]+name_p1[i]+cs[i][-1] #cs[-1] is "\n"
    fh = open(path_params1, mode="w")
    fh.writelines(cs)
    fh.close()
    path_params2 = folder_many_params_fit+"many_AA_params_fit.txt"
    np.savetxt(path_params2, params2_record)
    fh = open(path_params2, mode="r")
    cs = fh.readlines()
    fh.close()
    for i in np.arange(N_pp2):
        cs[i] = cs[i][:-1]+name_p2[i]+cs[i][-1] #cs[-1] is "\n"
    fh = open(path_params2, mode="w")
    fh.writelines(cs)
    fh.close()
    print("Write parameters of multi galaxy ... Done.")

    return 0

def main_step8_compare_DFAA_parameters_of_multi_galaxies():
    ####[] plot and compare
    path_params0 = folder_many_params_fit+"many_MG_params_fit.txt"
    params0_record = np.loadtxt(path_params0)
    name_p0 = ads.read_single_comment_of_each_line(path_params0, comment_symbol="#")
    path_params1 = folder_many_params_fit+"many_xv_params_fit.txt"
    params1_record = np.loadtxt(path_params1)
    name_p1 = ads.read_single_comment_of_each_line(path_params1, comment_symbol="#")
    path_params2 = folder_many_params_fit+"many_AA_params_fit.txt"
    params2_record = np.loadtxt(path_params2)
    name_p2 = ads.read_single_comment_of_each_line(path_params2, comment_symbol="#")
    if not (len(params0_record)==len(name_p0) and len(params1_record)==len(name_p1) 
        and len(params2_record)==len(name_p2)):
        print("There something wrong about lines in the file reading. Exit.")
        exit(0)
    prl = [params0_record, params1_record, params2_record]
    pcl = [name_p0, name_p1, name_p2]
    snl = ["_MG", "_xv", "_AA"]
    ads.DEBUG_PRINT_V(1, np.shape(prl[0]), np.shape(prl[1]), np.shape(prl[2]), "shape of prl")

    ##: plot each fit by reading the fit params
    # fitmodelfunc, combmodel

    ##: plot multi params
    for j in np.arange(3):
        pgw.plot_parameters_mg_list(prl[j], label_list=pcl[j], sn=snl[j], is_show=is_show)
    return 0

def _recompute_df_kde_per_type(actions, weights, bandwidth=None):
    """
    Build a mass-weighted KDE on `actions` and evaluate it at the same points.
    Tries project KD-tree class first; falls back to None (caller keeps old DF).
    actions: (N, d) array, already filtered to one type
    weights: (N,) masses for those particles
    """
    if actions is None or len(actions)==0:
        return None
    KD = kdtp.KDTree_galaxy_particles(actions, weight_extern_instinct=mass)
    return KD.density_SPH(actions)
    
def plot_simply_density_fitting(
        hbd, fbd, fit_function, fit_params_list, pathfig, x, log10_DF, h
):
    # r = ads.norm_l(x, axis=1)
    r = h
    yf = fit_function(hbd, *fit_params_list)

    fig = plt.figure()
    plt.scatter(r, log10_DF, label="data points", s=2.)
    plt.scatter(hbd, fbd, label="data bins", s=2.)
    plt.scatter(hbd, yf, label="fit bins", s=0.5)
    # plt.xlim(0., 1e3)
    plt.legend()
    plt.xscale("log")
    plt.xlabel(r"axis-ratio radius, $r_q$ (kpc)")
    plt.ylabel(r"log10 value of mass density, $\rho$ ($\mathrm{1e10\, M_\odot/kpc)^3}$")
    plt.savefig(pathfig)
    print("Plot and save 1d mass densty, done.")
    return 0

def plot_simple_actions_fitting(
        fit_function, fit_params_list, pathfig, JOb=None, DF_log10=None, 
        auxiliary_function_plot2d_x=None, auxiliary_function_plot2d_args=None, 
        JO1=None, DF1_log10=None, is_fit_1d=True
):
    ads.DEBUG_PRINT_V(1, np.shape(JO1), auxiliary_function_plot2d_args, "shape")
    _aux_args = auxiliary_function_plot2d_args
    # Accept None, empty list/tuple; filter stray None entries
    if _aux_args is None:
        _aux_args = ()
    else:
        _aux_args = tuple(a for a in _aux_args if a is not None)
    xd = auxiliary_function_plot2d_x(JO1, *_aux_args)
    yd = DF1_log10

    fig = plt.figure()
    plt.scatter(xd, yd, label="data points", s=2.)
    if is_fit_1d:
        xdb = JOb
        ydb = DF_log10
        yfb = fit_function(JOb, *fit_params_list)
        plt.scatter(xdb, ydb, label="data bins", s=1.)
        plt.scatter(xdb, yfb, label="fit bins", s=1.)
    else:
        yf = fit_function(JO1, *fit_params_list) #nan value
        plt.scatter(xd, yf, label="fit", s=1.)
    plt.legend()
    plt.xscale("log")
    plt.xlabel(r"value of combination of actions and frequencies, $h_J$ ($\mathrm{kpc\, km/s/Gyr)}$")
    plt.ylabel(r"log10 value of NDFA, f(J) ($\mathrm{1/(kpc\, km/s)^3)}$")
    plt.savefig(pathfig)
    print("Plot and save 1d AA DF, done.")
    return 0

from pathlib import Path
import yaml
def get_fit_choice(yaml_path, particle_type):
    """
    Return (fitting_model_name, is_fit_1d) for the requested particle_type.

    Behaviour:
      1) Loads YAML (merges '<<:' anchors are handled by yaml.safe_load).
      2) Locates a 'fit:' block either at top-level or nested (e.g. cfg['settings']['fit']).
      3) Reads 'fit.defaults' (mapping) and 'fit.components' (list).
      4) Merges defaults -> matching component (component wins) if a component with
         c['type'] == particle_type exists; otherwise uses defaults alone.
      5) Exits(2) with clear messages if anything required is missing.

    Args:
      yaml_path: str | Path to YAML file.
      particle_type: int (e.g. Gadget type / your component selector)

    Returns:
      (fitting_model: str, is_fit_1d: bool)
    """
    p = Path(yaml_path)
    if not p.exists():
        print(f"[settings] ERROR: YAML not found: {p}", file=sys.stderr)
        sys.exit(2)

    text = p.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)

    if not isinstance(cfg, dict):
        print("[settings] ERROR: YAML root must be a mapping (dict).", file=sys.stderr)
        sys.exit(2)

    # --- locate 'fit:' block (top-level or nested) ---
    fit = None
    if "fit" in cfg and isinstance(cfg["fit"], dict):
        fit = cfg["fit"]
    else:
        # DFS over nested dicts to find the first mapping child named 'fit'
        stack = [cfg]
        visited = set()
        while stack and fit is None:
            node = stack.pop()
            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)
            if isinstance(node, dict):
                if "fit" in node and isinstance(node["fit"], dict):
                    fit = node["fit"]
                    break
                # push children that are dicts
                for v in node.values():
                    if isinstance(v, dict):
                        stack.append(v)
                    elif isinstance(v, list):
                        for t in v:
                            if isinstance(t, dict):
                                stack.append(t)

    if fit is None:
        # Show top-level keys to help debugging structure
        print("[settings] ERROR: 'fit:' section not found anywhere in YAML.\n"
              f"  Top-level keys: {list(cfg.keys())}", file=sys.stderr)
        sys.exit(2)

    # --- pull defaults and components with type checks ---
    defaults = fit.get("defaults") if isinstance(fit, dict) else None
    components = fit.get("components") if isinstance(fit, dict) else None

    if defaults is None and components is None:
        print("[settings] ERROR: 'fit:' must contain 'defaults:' and/or 'components:'.", file=sys.stderr)
        sys.exit(2)

    if defaults is None:
        defaults = {}
    if components is None:
        components = []

    if not isinstance(defaults, dict):
        print("[settings] ERROR: 'fit.defaults' must be a mapping.", file=sys.stderr)
        sys.exit(2)
    if not isinstance(components, list):
        print("[settings] ERROR: 'fit.components' must be a list.", file=sys.stderr)
        sys.exit(2)

    # --- select component entry by particle_type, then merge with defaults ---
    entry_component = None
    for c in components:
        if isinstance(c, dict) and "type" in c:
            try:
                if int(c.get("type", -999)) == int(particle_type):
                    entry_component = c
                    break
            except Exception:
                # ignore non-int convertible 'type'
                pass

    merged = {}
    # order matters: defaults first, then per-component overrides
    if isinstance(defaults, dict):
        merged.update(defaults)
    if isinstance(entry_component, dict):
        merged.update(entry_component)

    # --- validate required keys after merge ---
    if ("fitting_model" not in merged) or ("is_fit_1d" not in merged):
        def _keys(d): return list(d.keys()) if isinstance(d, dict) else "N/A"
        print("[settings] ERROR: missing 'fitting_model' or 'is_fit_1d' after merging defaults with component.\n"
              f"  particle_type={particle_type}\n"
              f"  defaults keys={_keys(defaults)}\n"
              f"  component keys={_keys(entry_component)}\n"
              f"  merged={merged}", file=sys.stderr)
        sys.exit(2)

    # --- return normalized values ---
    fitting_model = str(merged["fitting_model"])
    is_fit_1d = bool(merged["is_fit_1d"])
    return fitting_model, is_fit_1d

def write_paramsresult(result_file_info_path, paramsresult, tag, meta=None):
    """
    Write the legacy TXT AND (new) a JSON sidecar with enough metadata
    so that plotting never needs hard-coded param-name lists.
    The JSON is optional for back-compat; plotters will fall back to TXT.
    """
    GP = cpgi.Galaxy_params_to_record()
    GP.record_gm_params_result_txt(result_file_info_path, paramsresult, tag)
    print("Record gm params result to txt, done.")
    if meta is not None:
        json_path = result_file_info_path.replace(".txt", ".json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
    return 0

def calculate_value_vectors_extern(snapshot_ID, N_params_name=8):
    value_vectors_extern = np.zeros(N_params_name)
    print("Calculate some extern value_vectors of snapshot_%d ..."%(snapshot_ID))
    
    paramfiles_name = folder_name+gm_name+"/"+"txt/snapshot_%03d.txt"%(snapshot_ID)
    data = np.loadtxt(paramfiles_name)
    xv = data[:,0:6]
    x = data[:,0:3]
    mass = None
    pot = data[:,14]
    value_vectors_extern[0] = kdtp.spin_L_parameter_from_xv(xv, mass , pot) #not used
    value_vectors_extern[1] = kdtp.rotate_parameter_cylindrel_from_xv(None, xv)
    value_vectors_extern[2] = kdtp.beta_parameter_spherical_from_xv(x, xv)
    value_vectors_extern[3] = kdtp.beta_z_parameter_cylindrel_from_xv(x, xv)

    paramfiles_name = folder_name+gm_name+"/"+"aa/snapshot_%d.action.method_all%s.txt"%(snapshot_ID, suffix_action)
    data = np.loadtxt(paramfiles_name)
    xv = data[:,0:6]
    x = data[:,0:3]
    mass = None
    pot = data[:,-4]
    value_vectors_extern[4] = kdtp.spin_L_parameter_from_xv(xv, mass , pot) #not used
    value_vectors_extern[5] = kdtp.rotate_parameter_cylindrel_from_xv(None, xv)
    value_vectors_extern[6] = kdtp.beta_parameter_spherical_from_xv(x, xv)
    value_vectors_extern[7] = kdtp.beta_z_parameter_cylindrel_from_xv(x, xv)
    
    return value_vectors_extern

def read_and_compare_params_debug(prefix_name, mgs_name, suffix_name):
    N_paramfiles_name = len(mgs_name)
    paramfiles_name = list(range(N_paramfiles_name))
    N_params_name = 6
    value_vectors_1 = np.zeros((N_params_name, N_paramfiles_name))
    value_vectors_2 = np.zeros((N_params_name, N_paramfiles_name))

    for j in np.arange(N_paramfiles_name): #before preprocessed
        print("rotate and beta 1: ", j)
        paramfiles_name[j] = prefix_name+mgs_name[j]+"/txt/snapshot_%03d.txt"%(snapshot_ID)
        # ads.DEBUG_PRINT_V(0, paramfiles_name, "paramfiles_name[j]")
        data = np.loadtxt(paramfiles_name[j])
        xv = data[:,0:6]
        x = data[:,0:3]
        mass = None
        pot = data[:,14]
        # a = kdtp.velocity_dispersion_knn(None, xv, coordinate_str="Cartesian")
        # b = kdtp.velocity_dispersion_knn(None, xv, coordinate_str="Spherical")
        # c = kdtp.velocity_dispersion_knn(None, xv, coordinate_str="Cylindrel")
        # ads.DEBUG_PRINT_V(0, a, b, c)
        value_vectors_1[0,j] = kdtp.spin_L_parameter_from_xv(xv, mass , pot) #not used
        value_vectors_1[1,j] = kdtp.rotate_parameter_cylindrel_from_xv(None, xv)
        value_vectors_1[2,j] = kdtp.beta_parameter_spherical_from_xv(x, xv)
        value_vectors_1[3,j] = kdtp.beta_z_parameter_cylindrel_from_xv(x, xv)

    for j in np.arange(N_paramfiles_name): #after processed
        print("rotate and beta 2: ", j)
        paramfiles_name[j] = prefix_name+mgs_name[j]+"/aa/snapshot_%d.action.method_all%s.txt"%(snapshot_ID, suffix_action)
        data = np.loadtxt(paramfiles_name[j])
        xv = data[:,0:6]
        x = data[:,0:3]
        mass = None
        pot = data[:,-4]
        value_vectors_2[0,j] = kdtp.spin_L_parameter_from_xv(xv, mass , pot) #not used
        value_vectors_2[1,j] = kdtp.rotate_parameter_cylindrel_from_xv(None, xv)
        value_vectors_2[2,j] = kdtp.beta_parameter_spherical_from_xv(x, xv)
        value_vectors_2[3,j] = kdtp.beta_z_parameter_cylindrel_from_xv(x, xv)
    # ads.DEBUG_PRINT_V(1, value_vectors_1, value_vectors_2)
    vv = np.hstack((value_vectors_1.T, value_vectors_2.T)).T
    np.savetxt(folder_params_statistics+suffix_name+".someparam.txt", vv)
    return value_vectors_1, value_vectors_2

def read_and_compare_params_vectors(paramfiles_name, params_name, suffix_name, x_name=None, plot_index_of_params=None):
    
    #: read parameters value from files
    N_paramfiles_name = len(paramfiles_name) #models
    N_params_name = len(params_name) #params of each model
    value_vectors = np.zeros((N_params_name, N_paramfiles_name))
    x = np.arange(N_paramfiles_name).astype(float)

    for j in np.arange(N_paramfiles_name):
        file_handle = open(paramfiles_name[j], mode="r")
        st = file_handle.readlines()
        file_handle.close()
        for i in np.arange(N_params_name):
            value_vectors[i,j] = cpgi.read_by_first_lineword_from_text(params_name[i], st)
            if x_name is not None and x_name == params_name[i]:
                x[j] = value_vectors[i,j]

    header = "# each line of params: "
    for i in np.arange(N_params_name):
        header += ( params_name[i]+", " )
    savepath = folder_params_statistics+suffix_name+".txt"
    np.savetxt(savepath, value_vectors, header=header)
    print("Save to %s. Done."%(savepath))

    #: plot some of parameters
    if plot_index_of_params is not None:
        fig = None
        x_value = value_vectors[plot_index_of_params[0]]
        x_label = params_name[plot_index_of_params[0]]
        y_value = value_vectors[plot_index_of_params[1]]
        y_label = params_name[plot_index_of_params[1]]
        z_value = value_vectors[plot_index_of_params[2]]
        z_label = params_name[plot_index_of_params[2]]
        if 1: #len(plot_index_of_params)==2:
            fig = plt.figure()
            ax = fig.add_subplot(2,2,1)
            ax.scatter(x_value[0], y_value[0])
            ax.plot(x_value, y_value)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax = fig.add_subplot(2,2,2)
            ax.scatter(y_value[0], z_value[0])
            ax.plot(y_value, z_value)
            ax.set_xlabel(y_label)
            ax.set_ylabel(z_label)
            ax = fig.add_subplot(2,2,3)
            ax.scatter(z_value[0], x_value[0])
            ax.plot(z_value, x_value)
            ax.set_xlabel(z_label)
            ax.set_ylabel(x_label)
            plt.tight_layout()
        # if len(plot_index_of_params)==3:
        #     fig = plt.figure() #(projection="3d")
        plt.savefig(folder_params_statistics+suffix_name+".png")
        plt.close()
    
    fig = None
    as_x = np.argsort(x)
    # ads.DEBUG_PRINT_V(0, as_x, x[as_x], x[as_x][0], x[as_x][-1])
    plt.figure(dpi=300)
    plt.plot([x[as_x][0]*1., x[as_x][0]*1.+(x[as_x][-1]-x[as_x][0])*1.4], [np.tanh(1.), np.tanh(1.)], lw=1., color="k", linestyle="--")
    plt.plot([x[as_x][0]*1., x[as_x][0]*1.+(x[as_x][-1]-x[as_x][0])*1.4], [np.tanh(0.), np.tanh(0.)], lw=1., color="k", linestyle="--")
    plt.plot([x[as_x][0]*1., x[as_x][0]*1.+(x[as_x][-1]-x[as_x][0])*1.4], [np.tanh(-1.), np.tanh(-1.)], lw=1., color="k", linestyle="--")
    # plt.plot([np.min(x)*1., np.min(x)*1.+(np.max(x)-np.min(x))*1.2], [np.tanh(1.), np.tanh(1.)], lw=1., color="k", linestyle="--")
    # plt.plot([np.min(x)*1., np.min(x)*1.+(np.max(x)-np.min(x))*1.2], [np.tanh(0.), np.tanh(0.)], lw=1., color="k", linestyle="--")
    # plt.plot([np.min(x)*1., np.min(x)*1.+(np.max(x)-np.min(x))*1.2], [np.tanh(-1.), np.tanh(-1.)], lw=1., color="k", linestyle="--")
    for i in np.arange(N_params_name):
        y = value_vectors[i,:]
        x1 = x[as_x]/1
        # y1 = np.tanh(y/y[0]-1.)
        y1 = np.tanh(y[as_x]/1.)
        # y1 = ads.log_abs_P1(y/y[0]-1.)
        x0 = x1[0]
        y0 = y1[0] #np.tanh(y0/1.)
        if x_name is None:
            plt.scatter(x0, y0)
        plt.plot(x1, y1, label="%s"%(params_name[i]))
        # if y[0]<10.:
        #     plt.scatter(x[0], y[0])
        #     plt.plot(x, y, label="%s"%(params_name[i]))
        # plt.xlabel("value of %s in xv space, frac to the first substract one and activate value by tanh"%(suffix_name))
        plt.legend(fontsize=4.)
    x_name2 = x_name
    if x_name is not None:
        x_name2 = "init_number"
    text = "The x-axis param: %s; the init order: "%(x_name2)
    for i in np.arange(N_paramfiles_name):
        text += " %d"%(as_x[i])
    plt.title(text)
    plt.xlabel("value of %s in xv space, activate value by tanh"%(suffix_name))
    plt.ylabel("value of multi varaibles in AA space")
    # plt.yscale("log")
    plt.savefig(folder_params_statistics+suffix_name+".multi.png")
    plt.close()
    return 0



####[] step2: the main
if __name__ == '__main__':
    
    ####[] settings
    # [] size (AA_combination_sumWeightFrequency 
    # "np.sum(J*O,axis=1)" (997004,3) (997004,0)), 
    # KDTree leaf mean (less fit points), 
    # plot and see DF, 
    # run fit, write params of each model, 
    # more IC running and more models, 
    # server, debug PPOD, background, paper

    ##[] step0 small
    main_step0_set_small() #Nothing

    ##[] step1 MG
    MG = main_step1_set_MG(M, N_ptcs, ls, ds, ar)



    #### record a series of snapshots
    if tag_task==1: #xv triaxialization
        ps, mass, tgts, DF, xe, ye, cl, meds, param00 = main_step2_DF(1, snapshot_ID)
        ptype = param00[2]
        # ads.DEBUG_PRINT_V(0, np.shape(ps), np.shape(tgts), np.shape(mass), np.shape(ptype), len(cl), "np.shape(ptype)")

        #choose which particle types to fit
        try:
            type_list = list(mask_select_type)
        except NameError:
            type_list = [1] #default: halo only
        if not type_list:
            type_list = [1]

        type_list = [1] #now only plot halo type
        for gadget_type in type_list:
            # ## old version
            # x, v = ps[:,0:3], ps[:,3:6] #note: without cl
            # fitmassdensity_tag = fitmassdensity_tag #?? 0: DPL; 1: Einasto; 2: MDPL
            # # DL, combmodel, fitmodelfunc, paramsresult = main_step4_fit_DF_x_mass(ps, mass, tgts, DF, xe, ye, cl, meds, MG, param00)
            # DL, combmodel, fitmodelfunc, paramsresult, hbd, fbd, h_sub, DF_sub = \
            #     main_step4_mpfit_fit_DF_x_mass(
            #         ps, mass, tgts, DF, xe, ye, cl, meds, MG, param00, 
            #         True, fitmassdensity_tag, ptype, gadget_type
            #     )
            # path_file_data = galaxyfit_name+"snapshot_%d.fit.txt"%(snapshot_ID)
            # pathfig = path_file_data+".massdensity.halo.png"
            # p_rho = fitmodelfunc[fitmassdensity_tag][4]
            # # if fitmodelfunc[fitmassdensity_tag][3] is not None:
            # #     p_rho = np.append(fitmodelfunc[fitmassdensity_tag][4], fitmodelfunc[fitmassdensity_tag][3])
            # plot_simply_density_fitting(hbd, fbd, fitmodelfunc[fitmassdensity_tag][0], p_rho, pathfig, x, np.log10(DF_sub), h_sub)
            # write_paramsresult(path_file_data, paramsresult, tag=1)

            ## to fit by axis ratio
            tgts, DF, mass = mask_particle_type_and_compute_density(
                tgts, mass[cl], ptype[cl], gadget_type
            ) #cl
            rq_data, DF_data_log10, rq_data_bin, DF_data_bin_log10, DF_fitvalue_log10, fit_params, q_axisratio = \
                main_step4_mpfit_fit_DF_x_mass_brief(tgts[:,0:3], DF, mass)
            path_file_data = galaxyfit_name+"snapshot_%d.fit.txt"%(snapshot_ID)
            pathfig = path_file_data+".massdensity.type_%d.png"%(gadget_type)
            DF_fitvalue_log10 = None #if not display fit
            plot_1d_fit_massdensity(rq_data, DF_data_log10, rq_data_bin, DF_data_bin_log10, DF_fitvalue_log10, pathfig)
            # paramsresult = None #to compare paramsresult
            # write_paramsresult(path_file_data, paramsresult, tag=1)
            print("q_axisratio: ", q_axisratio)
            path_file_data_types = galaxyfit_name + f"snapshot_{snapshot_ID}.type_{int(gadget_type)}.massdensity.txt"
            np.savetxt(path_file_data_types, q_axisratio)



    elif tag_task==2: #actions fitting
        ps, mass, tgts, DF, xe, ye, cl, meds, param00 = main_step2_DF(2, snapshot_ID)
        value_vectors_extern = calculate_value_vectors_extern(snapshot_ID, N_params_name=8)
        ptype = param00[2]
        # ads.DEBUG_PRINT_V(0, np.shape(ps), np.shape(tgts), np.shape(mass), np.shape(ptype), len(cl), "np.shape(ptype)")

        #choose which particle types to fit
        try:
            type_list = list(mask_select_type)
        except NameError:
            type_list = [1] #default: halo only
        if not type_list:
            type_list = [1]

        for gadget_type in type_list:
            #(1) Pick fitting model wrapper by name (from YAML later). Default if absent.            
            path_unified_yaml = "../../../install_and_run/unified_settings.yaml"
            wrapper_name, is_fit_1d = get_fit_choice(path_unified_yaml, gadget_type)
            # wrapper_name = "fitting_model_MPLF_freq"
            # wrapper_name = "fitting_model_MPLTF_freq"
            # is_fit_1d = True
            print("wrapper_name for fitting: ", wrapper_name)

            wrapper = getattr(gm, wrapper_name, None)
            wrap = wrapper() #expected to return a dict
            fitmodelfunc  = wrap.get("fitmodelfunc", None)
            combo_rec = wrap.get("combination", None)
            if fitmodelfunc is None:
                raise SystemExit(f"[settings] ERROR: Wrong fitting_model.")
            n_params_fitfunc = ads.count_function_args(fitmodelfunc[0][0])-1 #length of target length of not-first params of fit model
            params_name_fitfunc = [f"pn{i+1}_fit" for i in range(max(0, n_params_fitfunc))] + ["log_penalty"]
            fitmodelfunc[0][2] = params_name_fitfunc
            # ads.DEBUG_PRINT_V(0, n_params_fitfunc, params_name_fitfunc)

            #(2) fit actions DF
            DL, combmodel, fitmodelfunc, paramsresult, hbd, fbd, tgts_sub, DF_sub = \
                main_step5_mpfit_fit_DF_AA_one(
                    ps, mass, tgts, DF, xe, ye, cl, meds, MG, 
                    param00, fitmodelfunc=fitmodelfunc, 
                    is_fit_1d=is_fit_1d, value_vectors_extern=value_vectors_extern, 
                    particle_type=ptype, 
                    particle_type_select=gadget_type, 
                    recompute_df_for_selected=True
                ) #note that there is only one fit function for each type or component
            fitresult = fitmodelfunc[0][4]
            if (DL is None) and (paramsresult is not None) and ("note" in paramsresult):
                print(paramsresult["note"], "type=", gadget_type)
                continue
            ads.DEBUG_PRINT_V(1, np.shape(tgts_sub), "tgts_sub")

            #(3) plot helper: derive combination used for h
            auxiliary_function_plot2d_x   = gm.AA_combination_sumWeightFrequency_rateF1 if is_fit_1d else gm.AA_combination_freeCoef
            auxiliary_function_plot2d_args = None if is_fit_1d else (fitmodelfunc[0][4][-2:] if (fitmodelfunc and fitmodelfunc[0] and fitmodelfunc[0][4] is not None and len(fitmodelfunc[0][4])>=2) else None)

            vals  = [float(v) for v in fitmodelfunc[0][4]]
            if len(vals) != len(params_name_fitfunc):
                if len(vals) < len(params_name_fitfunc):
                    vals = vals + [0.0] * (len(params_name_fitfunc) - len(vals))
                else:
                    vals = vals[:len(params_name_fitfunc)]
            meta = {
                "fit_params_names": params_name_fitfunc,
                "fit_params_values": vals,
                "snapshot_ID": int(snapshot_ID),
                "particle_type": int(gadget_type),
                "is_fit_1d": bool(is_fit_1d),
                "fit_function": fitmodelfunc[0][1] if (fitmodelfunc and fitmodelfunc[0]) else "fh_MPLF_log10",
                "combination": (
                    {"name": "AA_combination_sumWeightFrequency_rateF1", "args": []} if is_fit_1d
                    else {"name": "AA_combination_freeCoef",
                          "args": ([float(v) for v in auxiliary_function_plot2d_args] if auxiliary_function_plot2d_args is not None else [])}
                )
            }

            #(4) output per-type
            path_file_data_types = galaxyfit_name+"snapshot_%d.type_%d.fit.txt"%(snapshot_ID, int(gadget_type))
            pathfig = path_file_data_types+".png"
            #note: tgts/DF here are already filtered inside the call; pass the filtered ones:
            print("gadget_type: ", gadget_type)
            path_file_data_types = galaxyfit_name + f"snapshot_{snapshot_ID}.type_{int(gadget_type)}.fit.txt"
            write_paramsresult(path_file_data_types, paramsresult, tag=2, meta=meta)
            
            pathfig = path_file_data_types + ".png"
            plot_simple_actions_fitting(
                fitmodelfunc[0][0], fitmodelfunc[0][4], pathfig, hbd, fbd, 
                auxiliary_function_plot2d_x, auxiliary_function_plot2d_args, tgts_sub, np.log10(DF_sub), 
                is_fit_1d=is_fit_1d
            )
        # sys.exit(2)



    elif tag_task==3: #params statistics for one galaxy model
        print("mgs_name: ", mgs_name)
        paramfiles_name = list(range(len(mgs_name)))
        prefix_name = folder_name+"galaxy_general_"
        for i in np.arange(len(mgs_name)):
            gadget_type = 1 #default halo
            if name_MG!="default": #much folders in current "path/to/GroArnold_framework/install_and_run/user_settings_multi.txt"
                paramfiles_name[i] = prefix_name+mgs_name[i]+"/fit/snapshot_%d.type_%d.fit.txt"%(snapshot_ID, int(gadget_type))
            else: #only plot about current debug folder galaxy_general
                paramfiles_name[i] = folder_name+"galaxy_general"+"/fit/snapshot_%d.type_%d.fit.txt"%(snapshot_ID, int(gadget_type))
        suffix_name = mgs_name[0]+"_etal"
        ads.DEBUG_PRINT_V("suffix_name, ", suffix_name)

        # val1, val2 = read_and_compare_params_debug(prefix_name, mgs_name, suffix_name)
        # ads.DEBUG_PRINT_V(0, val1, val2, "someparam")

        params_name = [ #paramsresults
            "spin_L_parameter_P1_direct", "rotate_sig_parameter_P1_direct", 
            "beta_parameter_P1_direct", "beta_z_parameter_P1_direct", 
            "spin_L_parameter_P2_direct", "rotate_sig_parameter_P2_direct", 
            "beta_parameter_P2_direct", "beta_z_parameter_P2_direct"
        ]
        x_name = None
        plot_index_of_params = None
        read_and_compare_params_vectors(paramfiles_name, params_name, suffix_name+"_velocity", x_name, plot_index_of_params)

        params_name = gm.params_name
        x_name = None
        # x_name = "powerAA_P1_fit"
        # x_name = "powerAA_P2_fit"
        plot_index_of_params = None #one should change this each time
        # plot_index_of_params = [0,1,2] #one should change this each time
        read_and_compare_params_vectors(paramfiles_name, params_name, suffix_name, x_name, plot_index_of_params)



    else:
        print("tag_task: ", tag_task)
        raise SystemExit("No such tag_task provided. Exit.")
