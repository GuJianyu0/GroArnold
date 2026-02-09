#!/usr/bin/env python
# -*- coding:utf-8 -*-
#In[] modules
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from sklearn.cluster import DBSCAN
# from scipy.spatial import cKDTree
# # from scipy.spatial import KDTree
# # from Corrfunc.theory.DD import DD
from scipy.integrate import tplquad
from noise import pnoise3
import vegas
import pandas as pd
import warnings

import analysis_data_distribution as ads
import diffusion_coef_integration as dci

#In[] settings
Dim = 3
col_actions = 78
col_frequencies = col_actions+7
# galaxy_name = sys.argv[1]
# snapshot_ID = int(sys.argv[2])
snapshot_ID = 10 #fixed

# G = 43007.1      # Gravitational constant (kpc, km/s, 1 M_sun)
# N = 1e11         # stars count
# M = 1.37e12*0.05 # total mass of stars (1 M_sun)
# Rs = 20.0        # scale length (kpc)
# ma = M/N         # Mass (arbitrary units)
# km = 1.0         # mass DF adjustmant
# v0 = 230.0       # Initial velocity (km/s)
# t_during = 0.25  # Time (Gyr)
# Vol0 = 4.*np.pi/3.*(Rs*2.)**3 #considered total volume
# R_plus = 20.0*2    # Upper bound for R' (kpc)
# R_minus =  1e-3    # Lower bound for R' (kpc)
# # R_minus =  R_plus/N    # Lower bound for R' (kpc)

#In[] functions
def generate_vel_DF_Gaussian_kinetic_energy(N, velmean_xyz, sigma_xyz, vel_bound=None):
    vel = np.random.normal(velmean_xyz, sigma_xyz, (N, 3)) #?? kinetic energy
    if vel_bound is not None:
        vel[vel<vel_bound[0]] = vel_bound[0]
        vel[vel>vel_bound[1]] = vel_bound[1]
    return vel

def particles_3d_Perlin_motion_factor_vel(
    initial_positions, N_iter=30, is_same_mean_radius=True, is_plot=True, 
    is_plot_witin_bound=False, suffix="suffix"
):
    # Step 1: Generate Perlin noise directions to simulate Brownian or Levy motion
    # scale = 0.001
    # scale = 0.005
    # scale = 0.01
    # scale = 0.011
    # scale = 0.015
    scale = 0.02 #see
    # scale = 0.05
    # scale = 0.5
    # scale = 1.
    # scale = 5.
    # step_size = 1.
    # step_size = 5.
    step_size = 20. #see
    # step_size = 50.
    # step_size = 80.
    # step_size = 100.
    # step_size = 200.
    #: (0.1, 1.): not obvious noise, rate near 1.
    #: (0.01, 200.): seperated clusters noise, rate near 1.
    #: (0.01, 80.): seperated clusters noise, rate 50.
    #: (0.011, 50.): seperated clusters noise, rate 13.
    #: (0.015, 5.): not too large noise, rate 1.04
    #: (0.015, 50., 50): clusters noise, rate 2.
    #: (0.02, 20.): bars noise, rate 4.

    new_positions = initial_positions*1.
    # # N_iter = 10
    # N_iter = 50
    for i in range(N_iter):
        new_positions = dci.Perlin_motion(new_positions, scale=scale, octaves=4, step_size=step_size)
    new_positions -= np.mean(new_positions, axis=0) #translate to mass center

    # Step 2: Rescale positions to match the initial volume (cylindrical boundary)
    scaling_factor = 1. #when scaling_factor decrease, the ampl rate increase, with not large influence
    R_init = ads.norm_l(initial_positions, axis=1)
    R_new = ads.norm_l(new_positions, axis=1)
    # R_init_mean = np.mean(R_init) #scaled radius, 1-order
    # R_new_mean = np.mean(R_new)
    R_init_mean = np.mean(R_init**2) #scaled kinetic energy, 2-order
    R_new_mean = np.mean(R_new**2)
    scaling_factor = R_new_mean/R_init_mean
    if is_same_mean_radius:
        new_positions = new_positions/scaling_factor
    ads.DEBUG_PRINT_V(1, R_init_mean, R_new_mean, scaling_factor, is_same_mean_radius, "scaling_factor")

    # Step 3: Plot the initial and new positions of particles
    if is_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], c='blue', s=5, alpha=0.5)
        ax1.set_title("Initial Velocities of Particles (vx, vy)")
        ax1.set_xlabel("vx")
        ax1.set_ylabel("vy")
        if is_plot_witin_bound:
            ax1.set_xlim([-R_init_mean, R_init_mean])
            ax1.set_ylim([-R_init_mean, R_init_mean])
        ax1.grid(True)
        ax2.scatter(new_positions[:, 0], new_positions[:, 1], c='red', s=5, alpha=0.5)
        ax2.set_title("New Velocities of Particles (vx, vy)")
        ax2.set_xlabel("vx")
        ax2.set_ylabel("vy")
        if is_plot_witin_bound:
            ax2.set_xlim([-R_init_mean, R_init_mean])
            ax2.set_ylim([-R_init_mean, R_init_mean])
        ax2.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig("../data/Perlin_motion_xy_"+suffix+".png", format="png", bbox_inches='tight')
    return new_positions

def Diffu_direct_integrand_descrete(vel_target, vel_sample):
    delta_v_size = ads.norm_l(vel_target-vel_sample, axis=1)
    # ads.DEBUG_PRINT_V(0, np.shape(delta_v_size))
    I = np.sum(delta_v_size)
    return I

def difference_3dim_2order_descrete(x, func, func_args=None, dx=0.1, difference_type=0):
    y = func(x, *func_args)
    ym = func(x-dx, *func_args)
    yp = func(x+dx, *func_args)
    # ypp = np.diff(y)
    ydd_11 = (yp[:,0]+ym[:,0]-2.*y[:,0])/(dx*dx) #?? differene
    ydd_22 = (yp[:,0]+ym[:,0]-2.*y[:,0])/(dx*dx)
    ydd_33 = (yp[:,0]+ym[:,0]-2.*y[:,0])/(dx*dx)
    ydd_12 = (yp[:,0]+ym[:,0]-2.*y[:,0])/(dx*dx)
    ydd_13 = (yp[:,0]+ym[:,0]-2.*y[:,0])/(dx*dx)
    ydd_23 = (yp[:,0]+ym[:,0]-2.*y[:,0])/(dx*dx)
    k_D = 4.*np.pi*dci.G**2*dci.ma**2
    ydd = k_D*np.array([ydd_11, ydd_22, ydd_33, ydd_12, ydd_13, ydd_23])
    return ydd

def Diffu_mean_difference_descrete(vel_sample): #?? mean
    return 1.



#In[] main
## Main calculation
if __name__ == "__main__":

    # 1. load data
    ## simulation data
    data_path = "../data/samples_input/snapshot_%03d.txt"%(snapshot_ID)
    # data_path = "../data/snapshot_%03d_big.txt"%(snapshot_ID)
    data = np.loadtxt(data_path, dtype=float)
    N_particles_simu = len(data)
    positions_simu = data[:, 0:3]
    velocities_simu = data[:, 3:6]
    # m_simu = data[:, 8]
    positions_simu -= np.mean(positions_simu, axis=0) #translate x center
    velocities_simu -= np.mean(velocities_simu, axis=0) #translate v center
    # ads.DEBUG_PRINT_V(1, np.mean(positions_simu, axis=0), np.mean(velocities_simu, axis=0), "vel_mean_initial")
    # ads.DEBUG_PRINT_V(1, velocities_simu[0], "velocities_simu[0]")
    
    pos_size_simu = ads.norm_l(positions_simu, axis=1)
    mean_pos_size = np.mean(pos_size_simu) #for settings
    vel_size_simu = ads.norm_l(velocities_simu, axis=1)
    mean_vel_size = np.mean(vel_size_simu) #for settings
    # velmean_xyz_simu = np.mean(velocities_simu, axis=0)
    # sigma_xyz_simu = np.std(velocities_simu, axis=0)
    # sigma_total_one = np.std(velocities_simu)
    # sigma_rpt_simu = kdtp.velocity_dispersion_knn(None, velocities_simu)

    kinetic_energy_simu = 0.5*np.mean(vel_size_simu**2)
    vel_2_mean = np.mean(velocities_simu**2, axis=0) #xyz, 3 components
    vel_mean = np.mean(velocities_simu, axis=0) #xyz, 3 components
    vel_mean_2 = ( np.mean(velocities_simu, axis=0) )**2 #xyz, 3 components
    sigma = np.std(velocities_simu, axis=0) #xyz, 3 components
    sigma_2 = ( sigma )**2 #xyz, 3 components
    sigma_for_iso = np.ones(3)*np.sqrt(np.mean(sigma**2)) #xyz, 3 components
    sig_small = sigma_for_iso[0]*0.1 #-8.1
    # sig_small = sigma_for_iso[0]*0.01 #-86.
    sig_high = np.sqrt(np.sum(sigma**2)-sig_small**2)
    sigma_high = np.array([sig_high, sig_small, sig_small])
    beta_z_sigma = 1.-(sigma[0]+sigma[1])/(2.*sigma[2])
    beta_z = 1.-(sigma_high[0]+sigma_high[1])/(2.*sigma_high[2])
    quadratic_mean = np.sqrt(np.sum(vel_mean_2+sigma_2))
    ads.DEBUG_PRINT_V(0, quadratic_mean, beta_z_sigma, beta_z, vel_mean, vel_2_mean, vel_mean_2, sigma, sigma_2, "v2")

    # mean_radius_setting = mean_pos_size
    mean_radius_setting = 50.
    mean_speed_setting = mean_vel_size
    N_particles_pos = 10000
    N_particles_vel = N_particles_pos
    n0_pos = N_particles_pos/(4.*np.pi*mean_radius_setting**3)
    log_Alpha = np.log(0.2*N_particles_pos)
    ads.DEBUG_PRINT_V(1, n0_pos, log_Alpha, "log_Alpha")

    pos_simu_original = positions_simu
    vel_simu_original = velocities_simu
    # dci.print_sample_info(vel_simu_original, "vel_simu_original")

    ## obs data
    vel_obs_original = None #??
    # dci.print_sample_info(vel_obs_original, "vel_obs_original")

    # ## generated samples by custom DF
    # np.random.seed(39)
    # vel_iso = dci.generate_velocities_sample_Gaussian(N_particles_vel, vel_mean, sigma_for_iso)
    # vel_iso_noise = None
    # # vel_iso_noise = particles_3d_Perlin_motion_factor_vel(
    # #     vel_iso, N_iter=30, suffix="vel_iso_noise"
    # # )
    # dci.print_sample_info(vel_iso, "vel_iso")
    
    # vel_aniso = dci.generate_velocities_sample_Gaussian(N_particles_vel, vel_mean, sigma)
    # vel_aniso_noise = None
    # vel_simu_noise = None
    # dci.print_sample_info(vel_aniso, "vel_aniso")

    

    # 2. process data
    ## total positions
    np.random.seed(39)
    mean_radius_setting = 50. #diffusion_descrete_cyl_softenning_mean
    M_total = dci.M_total_gal_1e10MSun #convert unit to 1e10 MSun
    # r_mean = mean_radius_setting
    # v_mean = np.sqrt(dci.G*M_total/r_mean)
    N_particles_list = [10000]
    # N_particles_list = [5000, 10000]
    vel_DFs = [
        "vel_iso", "vel_iso_noise", 
        # "vel_simu", "vel_iso", "vel_iso_noise", "vel_aniso", "vel_aniso_high", 
        # "vel_speed_Gaussian", "vel_speed_tail", 
        # "vel_obs", "vel_simu_noise", "vel_aniso_noise", 
    ]
    vel_simu = None
    vel_iso = None
    vel_iso_noise = None
    vel_aniso = None
    vel_info_store_file = "../data/vel_info_store.csv"

    vel_info_store = []
    for (j, N_particles_setting) in enumerate(N_particles_list):
        for (i, typenme) in enumerate(vel_DFs):
            vel_name = typenme
            vel_class = dci.DF_particles_6D(
                vel_name, M_total, N_particles_setting, #for name
                v_mean_31=vel_mean, v_dispersions_31=sigma, N_samples_vel1=N_particles_setting, 
                N_particles_pos1=N_particles_pos, n0_pos1=n0_pos, log_Alpha1=log_Alpha
            )
            
            if vel_name=="vel_obs":
                vel_obs = vel_class.generate_vel(typenme, vel_original=vel_obs_original)
            elif vel_name=="vel_iso":
                vel_iso = vel_class.generate_vel(typenme, v_dispersions_3_reset=sigma_for_iso)
            elif vel_name=="vel_iso_noise": #must be after vel_iso in the list
                vel_iso_noise = vel_class.generate_vel(typenme, vel_original=vel_iso, N_iter_Perlin=50)
                # vel_iso_noise = vel_class.generate_vel(typenme, vel_original=vel_iso, N_iter_Perlin=30)
            elif vel_name=="vel_aniso":
                vel_aniso = vel_class.generate_vel(typenme, v_dispersions_3_reset=sigma)
            elif vel_name=="vel_aniso_high":
                vel_aniso = vel_class.generate_vel(typenme, v_dispersions_3_reset=sigma_high)
            elif vel_name=="vel_simu":
                vel_simu = vel_class.generate_vel(typenme, vel_original=vel_simu_original)
            elif vel_name=="vel_simu_noise": #must be after vel_simu in the list
                vel_simu_noise = vel_class.generate_vel(typenme, vel_original=vel_simu, N_iter_Perlin=30)
            else:
                raise ValueError("No such type of DF can be generated, please check. Exit.")

            if vel_class.state != 0:
                warnings.warn("Unexpected state of pos_class. Skip this loop.")
                continue

            vel_class.run_vel()
            vel_info = vel_class.display_vel()
            vel_info_store.append(vel_info)
            ads.DEBUG_PRINT_V(0, vel_name, "debug for the first sample")
            ads.DEBUG_PRINT_V(1, vel_info, "vel_info")

    cols_name = [
        "v_mean_3[0]", "v_mean_3[1]", "v_mean_3[2]", 
        "v_dispersions_3[0]", "v_dispersions_3[1]", "v_dispersions_3[2]", 
        "N_particles_pos", "N_samples_vel", 
        "diffu_mean", "diffu_median", "diffu_0", 
        "vel_type", "name", 
    ]
    vel_info_store = pd.DataFrame(vel_info_store, columns=cols_name)
    vel_info_store.to_csv(vel_info_store_file, index=False)

    vel_info_store = pd.read_csv(vel_info_store_file)
    # dci.plot_frac_dim_total(vel_info_store, suffix="type_4_N_20000")
    ads.DEBUG_PRINT_V(0, len(vel_info_store), "vel_info_store")






    # ## old
    # # Step 1. data
    # data_path = "../data/snapshot_%03d.txt"%(snapshot_ID)
    # # data_path = "../data/snapshot_%03d_big.txt"%(snapshot_ID)
    # data = np.loadtxt(data_path, dtype=float)
    # N_particles_data = len(data)
    # positions = data[:, 0:3]
    # velocities = data[:, 3:6]
    # # m = data[:, 8]

    # positions -= np.mean(positions, axis=0) #translate x center
    # velocities -= np.mean(velocities, axis=0) #translate v center
    # ads.DEBUG_PRINT_V(1, velocities[0], "velocities[0]")
    # pos_size_data = ads.norm_l(positions, axis=1)
    # mean_pos_size = np.mean(pos_size_data)
    # vel_size_data = ads.norm_l(velocities, axis=1)
    # mean_vel_size = np.mean(vel_size_data)
    # vel_min = np.min(velocities)
    # vel_max = np.max(velocities)
    # velmean_xyz_data = np.mean(velocities, axis=0)
    # velmean_total_one = np.mean(velocities)
    # sigma_xyz_data = np.std(velocities, axis=0)
    # sigma_total_one = np.std(velocities)
    # kinetic_energy_data = 0.5*np.sum(vel_size_data**2)
    # # sigma_rpt_data = kdtp.velocity_dispersion_knn(None, velocities)
    # ads.DEBUG_PRINT_V(1, velmean_xyz_data, sigma_xyz_data, "velmean_xyz_data, sigma_xyz_data")

    # # # is_screen_radius = True
    # # is_screen_radius = False
    # # r_data = ads.norm_l(positions, axis=1)
    # # V_data = ads.norm_l(velocities, axis=1)
    # # x_mean = np.mean(r_data)
    # # v_mean = np.mean(V_data)
    # # if is_screen_radius: #to screen particles whose radius within 0.5 ~ 1.5 r_mean
    # #     idx_range = np.where((r_data>0.5*r_mean) & (r_data<1.5*r_mean))[0]
    # #     xv_range = data[idx_range]
    # #     positions = xv_range[:, 0:3]
    # #     velocities = xv_range[:, 3:6]
    # #     r_data = ads.norm_l(positions, axis=1)
    # #     V_data = ads.norm_l(velocities, axis=1)
    # #     ads.DEBUG_PRINT_V(1, len(idx_range), "idx_range")

    # # Step 2. iso sample and aniso sample
    # # N_iter = 10
    # N_iter = 30
    # # N_iter = 50
    # # is_plot_witin_bound = True
    # is_plot_witin_bound = False
    # is_same_mean_radius = True
    # # is_same_mean_radius = False
    # velmean_xyz = velmean_total_one
    # # pos_uniform = dci.generate_particle_positions(mean_radius*2., N_particles_data, bound_shape="cylindrical") #debug: uniform
    # velocities_iso = dci.generate_vel_DF_Gaussian(N_particles_data, velmean_xyz, sigma_total_one, vel_bound=[vel_min, vel_max])
    # velocities_iso_noise = particles_3d_Perlin_motion_factor_vel(
    #     velocities_iso, N_iter=N_iter, is_same_mean_radius=is_same_mean_radius, 
    #     is_plot_witin_bound=is_plot_witin_bound, suffix="velGauss"
    # )
    # # ads.DEBUG_PRINT_V(1, np.min(velocities_iso), np.max(velocities_iso), "velocities_iso min and max")
    
    # # sigma_x_setting = sigma_total_one
    # # sigma_y_setting = sigma_total_one
    # # sigma_z_setting = sigma_total_one
    # sigma_x_setting = sigma_xyz_data[0]
    # sigma_y_setting = sigma_xyz_data[1]
    # sigma_z_setting = sigma_xyz_data[2]
    # sigma_xyz = np.array([sigma_x_setting, sigma_y_setting, sigma_z_setting])
    # # velocities_aniso = velocities
    # # velocities_aniso_noise = particles_3d_Perlin_motion_factor_vel(
    # #     velocities_aniso, N_iter=N_iter, is_same_mean_radius=is_same_mean_radius, 
    # #     is_plot_witin_bound=is_plot_witin_bound, suffix="veldata"
    # # )
    # ads.DEBUG_PRINT_V(1, np.min(velocities_aniso), np.max(velocities_aniso), "velocities_aniso min and max")

    # # ## plot
    # # dci.plot_mass_contour_postions_or_velocities(
    # #     x_input=velocities_iso, m_input=m, is_pos=False, 
    # #     savename="../data/mass_contour_vel_iso"
    # # )
    # # dci.plot_mass_contour_postions_or_velocities(
    # #     x_input=velocities_iso_noise, m_input=m, is_pos=False, 
    # #     savename="../data/mass_contour_vel_iso_noise"
    # # )

    # # Step 3. diffusion coef
    # ## target
    # vel_target = np.array([sigma_total_one, sigma_total_one, sigma_total_one])
    # # vel_target = np.array([sigma_x_setting, sigma_y_setting, sigma_z_setting])
    # # vel_target *= 10. #debug: change vel_target
    # ads.DEBUG_PRINT_V(1, sigma_xyz, vel_target, "sigma_xyz, vel_target")
    
    # ## difference then integrate
    # # Diffu1 = Diffu_using_relative_velocity_descrete(vel_target, velocities_iso)
    # # Diffu2 = Diffu_using_relative_velocity_descrete(vel_target, velocities_iso_noise)
    # # Diffu1 = Diffu_using_relative_velocity_descrete(vel_target, velocities_aniso)
    # # Diffu2 = Diffu_using_relative_velocity_descrete(vel_target, velocities_aniso_noise)

    # Diffu1 = diffusion_velocity_softenning_mean_descrete(velocities_iso, suffix="iso")
    # Diffu2 = diffusion_velocity_softenning_mean_descrete(velocities_iso_noise, suffix="iso_noise")
    # # Diffu1 = diffusion_velocity_softenning_mean_descrete(velocities_aniso, suffix="aniso")
    # # Diffu2 = diffusion_velocity_softenning_mean_descrete(velocities_aniso_noise, suffix="aniso_noise")

    # # ## integrate then difference
    # # dv = np.array([0.1, 0., 0.])
    # # # dv = np.array([0.1, 0.1, 0.1])
    # # I_diffu_n0 = Diffu_direct_integrand_descrete(vel_target, velocities_iso)
    # # I_diffu_nm1 = Diffu_direct_integrand_descrete(vel_target-dv, velocities_iso)
    # # I_diffu_np1 = Diffu_direct_integrand_descrete(vel_target+dv, velocities_iso)
    # # # I_diffu_n0 = Diffu_direct_integrand_descrete(vel_target, velocities_iso_noise)
    # # # I_diffu_nm1 = Diffu_direct_integrand_descrete(vel_target-dv, velocities_iso_noise)
    # # # I_diffu_np1 = Diffu_direct_integrand_descrete(vel_target+dv, velocities_iso_noise)
    # # # I_diffu_n0 = Diffu_direct_integrand_descrete(vel_target, velocities_aniso)
    # # # I_diffu_nm1 = Diffu_direct_integrand_descrete(vel_target-dv, velocities_aniso)
    # # # I_diffu_np1 = Diffu_direct_integrand_descrete(vel_target+dv, velocities_aniso)
    # # # I_diffu_n0 = Diffu_direct_integrand_descrete(vel_target, velocities_aniso_noise)
    # # # I_diffu_nm1 = Diffu_direct_integrand_descrete(vel_target-dv, velocities_aniso_noise)
    # # # I_diffu_np1 = Diffu_direct_integrand_descrete(vel_target+dv, velocities_aniso_noise)
    # # I_pp = (I_diffu_nm1+I_diffu_np1-2.*I_diffu_n0)/(dv[0]*dv[0])
    # # ads.DEBUG_PRINT_V(0, I_diffu_n0, I_pp, "I_diffu")

    # # Diffu1 = Diffu_mean_descrete(velocities_iso)
    # # Diffu2 = Diffu_mean_descrete(velocities_aniso)
    # ads.DEBUG_PRINT_V(1, Diffu1, Diffu2, "Diffu1, Diffu2")
    # amplification_rate = Diffu2/Diffu1
    # print("amplification_rate: \n{}".format(amplification_rate))
