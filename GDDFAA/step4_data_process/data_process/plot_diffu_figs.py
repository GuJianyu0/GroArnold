#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors

import analysis_data_distribution as ads
import generate_fractal_sample as gfs
import diffusion_coef_integration as dci
import diffu_basic as dbc



def plot_compare_tree(filename, suffix="suffix"):
    data = np.loadtxt(filename)
    Diffu_direct = data[:, 1]
    Diffu_tree = data[:, 2]

    indices = np.arange(len(Diffu_direct))
    Diffu_direct = Diffu_direct[np.argsort(Diffu_direct)]
    Diffu_tree = Diffu_tree[np.argsort(Diffu_tree)]
    # ads.DEBUG_PRINT_V(0, Diffu_direct[0], Diffu_tree[0], "Diffu[0]")
    error_tree = np.mean( np.abs(Diffu_tree/Diffu_direct-1.) )
    # error_tree = np.sqrt( np.sum( (Diffu_tree/Diffu_direct-1.)**2 ) ) / len(Diffu_tree)
    # ads.DEBUG_PRINT_V(0, error_tree, "error_tree")

    fontsize = 40
    pointsize = 3.2
    figsize = (20, 15)  # For 3x3 layout
    dpi = None
    plt.figure(figsize=figsize, dpi=dpi)

    plt.plot(indices, Diffu_direct, "-", label="direct summation", lw=pointsize)
    plt.plot(indices, Diffu_tree, "--", label="tree", lw=pointsize)

    plt.legend(fontsize=fontsize*0.6, loc=0)
    # plt.xscale("log")
    plt.yscale("log")
    # plt.title(r"diffusion coefficient DF on position space", fontsize=fontsize)
    plt.xlabel(r"particles indices $i$", fontsize=fontsize)
    plt.ylabel(r"Diffusion $D_\mathrm{main}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("../data/examples_pos/diffur_compare_tree_{}.eps".format(suffix), format="eps", bbox_inches='tight')
    plt.close()
    
    print("Plot {}, done.".format(suffix))
    return 0

def plot_diffur(filename, Diffu_0, suffix="suffix"):
    data = np.loadtxt(filename)
    Diffu_data = data[:, 2]
    # pers = np.percentile(Diffu_data, q=[1., 10., 50., 90., 99.])
    # mask = (Diffu_data>pers[0]) & (Diffu_data<pers[-1]) #1., 99.
    # indices = np.where(mask)[0]
    # Diffu_data = Diffu_data[indices]

    min_val = np.min(Diffu_data)
    max_val = np.max(Diffu_data)
    mean_val = np.mean(Diffu_data)
    median_val = np.median(Diffu_data)
    count_total = len(Diffu_data)
    # ads.DEBUG_PRINT_V(1, min_val, max_val, mean_val, median_val, count_total, "Diffu_data")
    
    fontsize = 40.
    pointsize = 3.2
    figsize = 20, 15 #for 3, 3
    dpi = None
    plt.figure(figsize=figsize, dpi=dpi)

    N_bins = 100
    bins = np.geomspace(min_val, max_val, N_bins)
    counts, bin_edges, _ = plt.hist(Diffu_data, bins=bins, density=False, alpha=0.5, edgecolor='black')
    bin_widths = np.diff(bin_edges)
    normalized_distribution = counts / (bin_widths * count_total)
    plt.clf()  # Clear the previous histogram
    plt.grid(True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    plt.plot(bin_centers, normalized_distribution, 'o-', label='Normalized Distribution', color='b', lw=pointsize)
    plt.plot([mean_val, mean_val], [0., np.max(normalized_distribution)], '--', label='Diffu_mean = {:.4f}'.format(mean_val), color='k', lw=pointsize)
    plt.plot([median_val, median_val], [0., np.max(normalized_distribution)], '-.', label='Diffu_median = {:.4f}'.format(median_val), color='k', lw=pointsize)
    if Diffu_0 is not None:
        plt.plot([Diffu_0, Diffu_0], [0., np.max(normalized_distribution)], label='Diffu_0 = {:.4f}'.format(Diffu_0), color='k', lw=pointsize)
    
    plt.xscale("log")
    plt.yscale("log")
    plt.title(r"histogram of main diffusion coefficient of each particle", fontsize=fontsize)
    plt.xlabel(r"diffusion, $D_\mathrm{position, main}$ ($\mathrm{(km/s)^3/kpc}$)", fontsize=fontsize)
    plt.ylabel(r"distribution, $f$ ($\mathrm{kpc/(km/s)^3}$)", fontsize=fontsize)
    plt.legend(fontsize=fontsize*0.6, loc=0)
    plt.savefig("../data/examples_pos/diffur_DF_{}.eps".format(suffix), format="eps", bbox_inches='tight')
    plt.close()
    
    print("Plot {}, done.".format(suffix))
    return 0

def plot_normalized_df_from_percentile(filename, Diffu_represent, suffix):
    """
    Plot the normalized distribution function (DF) based on percentile data.

    Args:
        filename (str): Path to the file containing the (N, 2)-array data. 
                        The first column is the percentiles, the second column is the values.
        suffix (str): Suffix for the saved plot filename.
        geom_q (array): Geometric sequence for percentile values (0 to 1, exclusive).
                        Defaults to np.geomspace(0.01, 0.99, 50).
    """
    # Load the percentile data (N, 2) from the file
    data = np.loadtxt(filename)
    q = data[:, 0]  # Percentiles
    values = data[:, 1]  # Corresponding values
    Diffu_0 = Diffu_represent[1]
    mean_val = Diffu_represent[2]
    median_val = Diffu_represent[3]
    ads.DEBUG_PRINT_V(1, q, values, "perentile")

    # bin_centers = np.diff(values)
    bin_centers = (values[:-1] + values[1:]) / 2.0
    # normalized_distribution = np.diff(values) / np.diff(q)  # Density is the difference in values divided by difference in q
    # normalized_distribution = df / np.sum(df)  # Normalize the DF to 1
    normalized_distribution = np.diff(q)

    # Plotting settings
    fontsize = 40
    pointsize = 3.2
    figsize = (20, 15)  # For 3x3 layout
    dpi = 400
    plt.figure(figsize=figsize, dpi=dpi)

    plt.plot(bin_centers, normalized_distribution, label="Normalized DF", marker="o", markersize=pointsize)
    plt.plot([mean_val, mean_val], [0., np.max(normalized_distribution)], '--', label='Diffu_mean = {:.4f}'.format(mean_val), color='k', lw=pointsize)
    plt.plot([median_val, median_val], [0., np.max(normalized_distribution)], '-.', label='Diffu_median = {:.4f}'.format(median_val), color='k', lw=pointsize)
    if Diffu_0 is not None:
        plt.plot([Diffu_0, Diffu_0], [0., np.max(normalized_distribution)], label='Diffu_0 = {:.4f}'.format(Diffu_0), color='k', lw=pointsize)
    
    # Set logarithmic scale for x and y axes
    plt.xscale("log")
    plt.yscale("log")

    # Title and labels
    plt.title(r"Histogram of main diffusion coefficient of each particle", fontsize=fontsize)
    plt.xlabel(r"Diffusion, $D_\mathrm{position, main}$ ($\mathrm{(km/s)^3/kpc}$)", fontsize=fontsize)
    plt.ylabel(r"Distribution, $f$ ($\mathrm{kpc/(km/s)^3}$)", fontsize=fontsize)

    # Legend and ticks
    plt.legend(fontsize=fontsize * 0.6, loc=0)
    plt.tick_params(which='major', length=0, labelsize=fontsize * 0.8)

    # Layout and save the plot
    plt.tight_layout()
    plt.savefig("../data/examples_pos/diffur_percentile_{}.eps".format(suffix), format="eps", bbox_inches='tight')
    plt.close()
    print("Saved fig of diffur_percentile.")
    return 0

def plot_relaxation_time_with_N_and_dim_pos(
    R0, v0, M_total, Dim_frac_lb, 
    diffur_uniform_median_list, diffur_uniform_meanvalue_list, 
    diffur_noised_median_list, diffur_noised_meanvalue_list, 
    diffur_referencevalue_list, diffur_simulated_list=None, 
    suffix="suffix"
):
    '''
    cols_name = ["mean_radius", "N_particles", "diffu_mean", "diffu_median", "diffu_ref", "h_frac", "Dim_frac", 
        "eta_diffu", "eta_IR2", "eta_N", "pos_type", "name", ]
    pos_DFs = ["pos_obs", "pos_uniform", "pos_uniform_noise", "pos_simu", "pos_simu_noise", "pos_extra", ]
    '''
    fontsize = 40.
    pointsize = 3.2
    figsize = 20, 15 #for 3, 3
    dpi = 400
    N_points_plot = 100

    # N_particles_arr = np.geomspace(1e3, 1e5, N_points_plot)
    N_particles_arr = np.geomspace(1e1, 1e11, N_points_plot)
    N_dimlb_plot = len(Dim_frac_lb)
    eta_N_arr_count = np.zeros((N_dimlb_plot, N_points_plot))
    diffu_N_arr_referencevalue = np.zeros(N_points_plot)
    eta_N_arr_referencevalue = np.zeros(N_points_plot)
    for i in np.arange(N_points_plot):
        diffu_N_arr_referencevalue[i] = dbc.diffusion_reference_value_Diffu_ref(N_particles=N_particles_arr[i], M_total=M_total, R0=R0, v0=v0)
        eta_N_arr_referencevalue[i] = dbc.eta_ralaxaiton_time_ratio(diffu_N_arr_referencevalue[i], R0, v0)
    for j in np.arange(N_dimlb_plot):
        for i in np.arange(N_points_plot):
            eta_N_arr_count[j, i] = dci.rate_t_relax_to_t_cross_count(N_particles_arr[i], Dim_frac=Dim_frac_lb[j]) #not updated

    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True)
    color = {
        "pos_obs": "k", "pos_uniform": "b", "pos_uniform_noise": "r", "pos_referencevalue": "k",
        "pos_simu": "g", "pos_simu_noise": "orange", "pos_extra": "purple", 
    }

    colorj = ["red", "orange", "green", "blue", "purple"]
    plt.plot(N_particles_arr, eta_N_arr_referencevalue, "-", label="by referencevalue", color="k", lw=pointsize*0.5)
    for j in np.arange(N_dimlb_plot):
        plt.plot(N_particles_arr, eta_N_arr_count[j], "-.", label="by single fractal model, Dim_frac={:.2f}".format(Dim_frac_lb[j]), color=colorj[j], lw=pointsize*0.5)

    # dldl = [diffur_referencevalue_list, ] #diffur_simulated_list, ]
    # for dl in dldl:
    #     N_list = dl[0]
    #     diffur_list = dl[1]
    #     N_N_list = len(N_list)
    #     h = dl[2]*N_list/1e4
    #     d = dl[3]
    #     pos_type_name = dl[4]
    #     eta_IR2_list = np.zeros(N_N_list)
    #     eta_diffu_list = np.zeros(N_N_list)
    #     for i in np.arange(N_N_list):
    #         eta_IR2_list[i] = dci.rate_t_relax_to_t_cross_IR2(N_list[i], h_frac=h[i], Dim_frac=d, R_p=rmean)
    #         eta_diffu_list[i] = dbc.eta_ralaxaiton_time_ratio(diffur_list[i], R0, v0)
    #     # ads.DEBUG_PRINT_V(1, diffur_list, eta_diffu_list, "diffur")
    #     # plt.plot(N_list, eta_IR2_list, "-.", label="eta_IR2, Dim_frac={:.2f}".format(d), color=color[pos_type_name], lw=pointsize*0.5)
    #     # plt.scatter(N_list, eta_diffu_list, label="eta_diffu by {}".format(pos_type_name), color=color[pos_type_name], s=pointsize*60., marker=".")
    #     plt.plot(N_list, eta_diffu_list, label="eta_diffu by {}".format(pos_type_name), color=color[pos_type_name], lw=pointsize*1.)
    
    dldl = [diffur_uniform_median_list, diffur_noised_median_list, ]
    for dl in dldl:
        N_list = dl[0]
        diffur_list = dl[1]
        N_N_list = len(N_list)
        h = dl[2]*N_list/1e4
        d = dl[3]
        pos_type_name = dl[4]
        eta_IR2_list = np.zeros(N_N_list)
        eta_diffu_list = np.zeros(N_N_list)
        for i in np.arange(N_N_list):
            eta_IR2_list[i] = dci.rate_t_relax_to_t_cross_IR2(N_list[i], h_frac=h[i], Dim_frac=d, R_p=rmean)
            eta_diffu_list[i] = dbc.eta_ralaxaiton_time_ratio(diffur_list[i], R0, v0)
        # ads.DEBUG_PRINT_V(1, diffur_list, eta_diffu_list, "diffur")
        # plt.plot(N_list, eta_IR2_list, "-.", label="eta_IR2, Dim_frac={:.2f}".format(d), color=color[pos_type_name], lw=pointsize*0.5)
        plt.scatter(N_list, eta_diffu_list, label="by median value of {}".format(pos_type_name), color=color[pos_type_name], s=pointsize*60., marker="*")
    
    dldl = [diffur_uniform_meanvalue_list, diffur_noised_meanvalue_list, ]
    for dl in dldl:
        N_list = dl[0]
        diffur_list = dl[1]
        N_N_list = len(N_list)
        h = dl[2]*N_list/1e4
        d = dl[3]
        pos_type_name = dl[4]
        eta_IR2_list = np.zeros(N_N_list)
        eta_diffu_list = np.zeros(N_N_list)
        for i in np.arange(N_N_list):
            eta_IR2_list[i] = dci.rate_t_relax_to_t_cross_IR2(N_list[i], h_frac=h[i], Dim_frac=d, R_p=rmean)
            eta_diffu_list[i] = dbc.eta_ralaxaiton_time_ratio(diffur_list[i], R0, v0)
        # ads.DEBUG_PRINT_V(1, diffur_list, eta_diffu_list, "diffur")
        # plt.plot(N_list, eta_IR2_list, "-.", label="eta_IR2, Dim_frac={:.2f}".format(d), color=color[pos_type_name], lw=pointsize*0.5)
        plt.scatter(N_list, eta_diffu_list, label="by mean value of {}".format(pos_type_name), color=color[pos_type_name], s=pointsize*60., marker="+")
    
    plt.xscale("log")
    plt.yscale("log")
    # plt.title(r"eta_relax versus N", fontsize=fontsize)
    plt.xlabel(r"particle count $N$", fontsize=fontsize)
    plt.ylabel(r"relaxation time ratio $\eta$", fontsize=fontsize)
    plt.legend(fontsize=fontsize*0.36, loc=0)
    plt.tick_params(which='major', length=0, labelsize=fontsize*0.8) #size of the number characters
    plt.tight_layout()

    plt.savefig("../data/examples_pos/relaxation_time_N_and_dim_{}.eps".format(suffix), format="eps", bbox_inches='tight')
    plt.close()
    print("Saved fig of relaxation_time_with_N_and_dim_pos.")
    return 0



def plot_each_velocity_samples_debug(N_particles, v2m_sqrt, vmean_3, dispersion_3_iso, dispersion_3_high, path_save):
    sigma = 143.02621867679355
    vc = 293.1071270852349
    gamma = 4.69149629070613
    # k = 0.08853016313884869
    k = 0.06
    v1 = 396.99512203793194
    A_pl = 0.311190198455538
    epsilon = 0.1
    dispersion_3_core = ads.rescale_dispersion_keep_ratio(vmean_3, dispersion_3_iso, sigma*np.sqrt(3.))
    dispersion_3_tail = ads.rescale_dispersion_keep_ratio(vmean_3, dispersion_3_iso, vc*np.sqrt(3.))
    # dispersion_3_core = ads.rescale_dispersion_keep_ratio(vmean_3, dispersion_3_high, sigma*np.sqrt(3.))
    # dispersion_3_tail = ads.rescale_dispersion_keep_ratio(vmean_3, dispersion_3_high, vc*np.sqrt(3.))
    vmax = v2m_sqrt*5.
    vmax_three = dispersion_3_core*5.
    vel_GP_iso = gfs.sample_velocity_isotropic_df(
        N_particles, sigma, vc, gamma, k, v1, A_pl, epsilon, vmax
    )
    vel_GP_aniso_13 = gfs.sample_isotropic_to_anisotropic(
        vel_GP_iso, dispersion_3_core
    )
    vel_GP_aniso_3D = gfs.sample_anisotropic_df(
        N_particles, dispersion_3_core, dispersion_3_tail, gamma, k, v1, A_pl, epsilon, vmax_three
    )
    
    vel_GP_iso = dci.readjust_velocities(vel_GP_iso, vmean_3, dispersion_3_high)
    vel_GP_aniso_13 = dci.readjust_velocities(vel_GP_aniso_13, vmean_3, dispersion_3_high)
    vel_GP_aniso_3D = dci.readjust_velocities(vel_GP_aniso_3D, vmean_3, dispersion_3_iso)
    # vel_GP_aniso_3D = dci.readjust_velocities(vel_GP_aniso_3D, vmean_3, dispersion_3_high)
    
    # vel_load = None
    # filename_tail = "../data/samples_vel/vel_tail_N10000.txt"
    filename_tail = "../data/samples_vel/vel_composite_N10000.txt"
    vel_load = np.loadtxt(filename_tail, dtype=float)[:, 3:6]
    
    ads.DEBUG_PRINT_V(1, 
        vmean_3, 
        np.mean(vel_GP_iso, axis=0), 
        np.mean(vel_GP_aniso_13, axis=0), 
        np.mean(vel_GP_aniso_3D, axis=0), 
        np.mean(vel_load, axis=0), 
    "vmean_3")
    ads.DEBUG_PRINT_V(1, 
        v2m_sqrt, 
        dci.get_quadratic_mean(vel_GP_iso), 
        dci.get_quadratic_mean(vel_GP_aniso_13), 
        dci.get_quadratic_mean(vel_GP_aniso_3D), 
        dci.get_quadratic_mean(vel_load), 
    "get_quadratic_mean of three kind of samples")
    
    k_neighbors = 32
    bandwidth = v2m_sqrt/10.
    plot_velocity_speed_distribution(
        vel_GP_iso, vel_GP_aniso_13, vel_GP_aniso_3D, vel_load, 
        bandwidth=bandwidth, k_neighbors=k_neighbors, save_path=path_save, 
    )
    plot_velocity_DF_contour_compare(
        vel_GP_iso, vel_GP_aniso_13, vel_GP_aniso_3D, vel_load, 
        save_path=path_save, 
    )
    return 0

def plot_velocity_speed_distribution(
    vel_GP_iso: np.ndarray, vel_GP_aniso_13: np.ndarray, vel_GP_aniso_3D: np.ndarray, vel_load=None, 
    bandwidth: float = 5.0, k_neighbors: int = None, v_range=None, 
    save_path: str = "save_path/", title: str = "Velocity Speed Distribution (KDE)",
    is_show: bool = False, debug: bool = False,
):
    """
    Plot 1D speed distribution curves for three velocity DFs (using KDE).

    Parameters:
        vel_GP_iso (np.ndarray): Shape (N, 3), isotropic DF sample
        vel_GP_aniso_13 (np.ndarray): Shape (N, 3), anisotropic 1D to 3D DF sample
        vel_GP_aniso_3D (np.ndarray): Shape (N, 3), anisotropic 3D DF sample
        vel_load: Shape (N, 3), the sample that read from file
        bandwidth (float): KDE bandwidth
        v_range (tuple): (v_min, v_max) for plotting; if None, auto-infer
        save_path (str): path to save plot image
        title (str): title of the plot
        debug (bool): whether to print debug information
    """
    # Helper function for KDE computation
    def kde_bandwidth_1d(speed_data, grid, bandwidth):
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(speed_data[:, None])
        log_dens = kde.score_samples(grid[:, None])
        return np.exp(log_dens)

    def knn_kde_1d(v_data, v_grid, k=None):
        """Estimate PDF from 1D samples using kNN."""
        N_s = len(v_data)
        if k is None:
            # k = int(N_s/10)
            k = 500
        # ads.DEBUG_PRINT_V(0, k, "k")
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(v_data[:, None])
        dists, _ = nbrs.kneighbors(v_grid[:, None])
        radius = dists[:, -1]
        pdf = k / (N_s * 2.0 * radius + 1e-16)  # 1D volume is length
        return pdf

    # Step 1: compute speed arrays
    v_iso = np.linalg.norm(vel_GP_iso, axis=1)
    v_13  = np.linalg.norm(vel_GP_aniso_13, axis=1)
    v_3D  = np.linalg.norm(vel_GP_aniso_3D, axis=1)

    # Define v range
    v_l = None
    all_speeds = None
    if vel_load is None:
        all_speeds = np.concatenate([v_iso, v_13, v_3D])
    else:
        v_l   = np.linalg.norm(vel_load, axis=1)
        all_speeds = np.concatenate([v_iso, v_13, v_3D, v_l])
    labels = ["Isotropic", "Anisotropic_1D_to_3D", "Anisotropic_3D_formula", "Loaded"]
    
    v_range = None
    if v_range is None:
        v_min, v_max = np.percentile(all_speeds, [0., 98.])
    else:
        v_min, v_max = v_range
    v_grid = np.linspace(v_min, v_max, 500)

    # Step 2: compute densities
    df_iso, df_13, df_3D, df_l = None, None, None, None
    if bandwidth is not None:
        df_iso = kde_bandwidth_1d(v_iso, v_grid, bandwidth)
        df_13 = kde_bandwidth_1d(v_13, v_grid, bandwidth)
        df_3D = kde_bandwidth_1d(v_3D, v_grid, bandwidth)
        if vel_load is not None:
            df_l = kde_bandwidth_1d(v_l, v_grid, bandwidth)
    else:
        df_iso = knn_kde_1d(v_iso, v_grid, k=k_neighbors)
        df_13  = knn_kde_1d(v_13, v_grid, k=k_neighbors)
        df_3D  = knn_kde_1d(v_3D, v_grid, k=k_neighbors)
        if vel_load is not None:
            df_l = knn_kde_1d(v_l, v_grid, k=k_neighbors)

    # Step 4: Plot
    plt.figure(figsize=(10, 6))
    plt.plot(v_grid, df_iso, label=labels[0], lw=2)
    plt.plot(v_grid, df_13,  label=labels[1], lw=2)
    plt.plot(v_grid, df_3D,  label=labels[2], lw=2)
    if vel_load is not None:
        plt.plot(v_grid, df_l,  label=labels[3], lw=2)
    plt.xlabel("Speed |v|")
    plt.ylabel("Density (KDE)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path+"velocity_DF_speed_compare.eps", format="eps", bbox_inches='tight')
    if is_show:
        plt.show()
    if debug:
        print(f"Saved speed KDE plot to: {save_path}")
    print("Plot velocity_DF_speed_compare, done.")

def plot_velocity_DF_contour_compare(
    vel_GP_iso: np.ndarray, vel_GP_aniso_13: np.ndarray, vel_GP_aniso_3D: np.ndarray, vel_load=None, 
    bandwidth: float = 10.0, k_neighbors: int = 32, 
    N_grid: tuple = (100, 80, 9), percent_clip: float = 98.0,
    levels=np.linspace(-8., 0.2, 16), save_path: str = "./",
    is_show: bool = False, debug: bool = False
):
    """
    Visualize 3D velocity DF for three samples via KDE contour slices at different vz.

    Parameters:
        vel_GP_iso (np.ndarray): shape (N,3) isotropic sample
        vel_GP_aniso_13 (np.ndarray): anisotropic 1D to 3D sample
        vel_GP_aniso_3D (np.ndarray): anisotropic 3D-formula sample
        vel_load: Shape (N, 3), the sample that read from file
        bandwidth (float): KDE bandwidth
        N_grid (tuple): (Nx, Ny, Nz) grid size
        percent_clip (float): range clipping percentile
        levels (list): contour levels (log10 density)
        save_path (str): save path prefix (will append filename)
        is_show (bool): whether to call plt.show()
        debug (bool): print debug info
    """
    fontsize = 14
    dpi = 300
    figsize = (20, 15)
    labels = ["Isotropic", "Anisotropic_1D_to_3D", "Anisotropic_3D_formula", "Loaded"]
    datasets = None
    if vel_load is None:
        datasets = [vel_GP_iso, vel_GP_aniso_13, vel_GP_aniso_3D]
    else:
        datasets = [vel_GP_iso, vel_GP_aniso_13, vel_GP_aniso_3D, vel_load]

    # Grid setup
    Nx, Ny, Nz = N_grid
    all_data = np.vstack(datasets)
    bounds = np.array([
        [-np.percentile(np.abs(all_data[:, i]), percent_clip), np.percentile(np.abs(all_data[:, i]), percent_clip)]
        for i in range(3)
    ])
    bounds[0][0] = np.min([bounds[0][0], bounds[1][0]]) #use the max bound of x and y
    bounds[1][0] = np.min([bounds[0][0], bounds[1][0]])
    bounds[0][1] = np.max([bounds[0][1], bounds[1][1]])
    bounds[1][1] = np.max([bounds[0][1], bounds[1][1]])
    grid_x = np.linspace(bounds[0][0], bounds[0][1], Nx)
    grid_y = np.linspace(bounds[1][0], bounds[1][1], Ny)
    grid_z = np.linspace(bounds[2][0], bounds[2][1], Nz)
    mg1, mg2, mg3 = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_coords = np.column_stack([mg1.ravel(), mg2.ravel(), mg3.ravel()])

    if debug:
        print("Grid bounds:", bounds)
        print("Grid shape (X,Y,Z):", mg1.shape)

    for idx, vel_data in enumerate(datasets):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        rho_logcube = np.zeros((Nx, Ny, Nz))
        N_s = len(vel_data)

        # kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        # kde.fit(vel_data)
        # log_rho = kde.score_samples(grid_coords)
        # rho_cube = np.exp(log_rho).reshape(Nx, Ny, Nz)
        # # rho_logcube = rho_cube
        # rho_logcube = np.log10(rho_cube + 1e-16)  # Avoid log(0)
        
        # Fit kNN and estimate local volumes
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(vel_data)
        dists, _ = nbrs.kneighbors(grid_coords)
        radius = dists[:, -1]  # Distance to the kth neighbor
        vol_kNN = (4./3) * np.pi * radius**3 + 1e-16  # avoid zero volume

        dists0, _ = nbrs.kneighbors([[0., 0., 0.]])
        radius0 = dists0[:, -1]  # Distance to the kth neighbor
        vol_kNN0 = (4./3) * np.pi * radius0**3 + 1e-16  # avoid zero volume
        rho0 = k_neighbors/N_s / vol_kNN0

        # Density estimate: rho ~ k / volume
        rho = k_neighbors/N_s / vol_kNN
        rho_logcube = np.log10(rho/rho0).reshape(Nx, Ny, Nz)

        # levels_min = -7.5
        levels_min = np.min(rho_logcube)
        # levels_max = 0.5
        levels_max = np.max(rho_logcube)
        # ads.DEBUG_PRINT_V(0, levels_min, levels_max, "levels")
        levels = np.linspace(levels_min, levels_max, 16)

        for k in range(Nz):
            ax = fig.add_subplot(3, 3, k+1)
            contour = ax.contourf(
                grid_x, grid_y, rho_logcube[:, :, k].T,
                levels=levels, cmap="viridis"
            )
            ax.set_aspect('equal')
            ax.set_title(f"{labels[idx]}, $v_z$ = {grid_z[k]:.2f}", fontsize=fontsize)
            ax.set_xlabel("$v_x$", fontsize=fontsize)
            ax.set_ylabel("$v_y$", fontsize=fontsize)
            ax.tick_params(labelsize=fontsize)

        cax = fig.add_axes([0.92, 0.35, 0.015, 0.3])
        cbar = fig.colorbar(contour, cax=cax)
        cbar.set_label(r"$\log_{10} f_v$", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

        fig.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.07, wspace=0.3, hspace=0.4)
        plt.savefig(save_path + f"velocity_DF_contour_compare_"+labels[idx]+".eps", format="eps", bbox_inches='tight')
        if is_show:
            plt.show()
        plt.close()
    print("Plot velocity_DF_contour_compare, done.")



## main
if __name__ == '__main__':

    # #### samples for pos
    # ## 1. compare_tree
    # suffix = "noised14_N100000"
    # filename = "../data/samples_simulated/snapshot_compare_tree_Diffur_read0_{}.txt".format(suffix)
    # plot_compare_tree(filename, suffix)
    # filename_Diffu_represent = "../data/samples_simulated/snapshot_fractal_Diffustatistics_read0_noised14_N100000.txt"
    # Diffu_represent = np.loadtxt(filename_Diffu_represent)
    # print(Diffu_represent)
    # plot_diffur(filename, Diffu_represent[1], suffix)

    # ## 2. diffu_simulated
    # suffix = "snapshot_080_N1000000"
    # filename = "../data/samples_simulated/snapshot_080_Diffur_read1_N1000000.txt"
    # data_tmp = np.loadtxt(filename)
    # data_tmp = data_tmp[:, 2]
    # filename_Diffu_represent = "../data/samples_simulated/snapshot_080_Diffustatistics_read1_N1000000.txt"
    # Diffu_represent = np.loadtxt(filename_Diffu_represent)
    # print(Diffu_represent)
    # # plot_diffur(filename, Diffu_represent, suffix)
    # dbc.plot_normalized_pdf_from_eachpoints_histogram(data_tmp, Diffu_represent[1], suffix, pos_or_vel="pos")
    # dbc.plot_normalized_pdf_from_eachpoints_KDE(data_tmp, Diffu_represent[1], suffix, pos_or_vel="pos")
    # data_tmp = None

    # ## 3. percentile
    # suffix = "noised14_N1000000000"
    # filename = "../data/samples_simulated/snapshot_fractal_Diffu_pers_read0_noised0_N1000000000.txt"
    # data_tmp = np.loadtxt(filename, dtype=float)
    # filename_Diffu_represent = "../data/samples_simulated/snapshot_fractal_Diffustatistics_read0_noised14_N1000000000.txt"
    # Diffu_represent = np.loadtxt(filename_Diffu_represent)
    # diffu_eff_mean = Diffu_represent[2]
    # diffu_eff_median = Diffu_represent[3]
    # diffu_eff_0 = Diffu_represent[1]
    # print(Diffu_represent)
    # # plot_normalized_df_from_percentile(filename, Diffu_represent, suffix)
    # dbc.plot_normalized_pdf_from_percentile(
    #     data_tmp, diffu_eff_mean, diffu_eff_median, diffu_eff_0, suffix=suffix, pos_or_vel="pos"
    # )
    # data_tmp = None

    # ## 4. diffu_with_N_and_dim
    # R0 = dbc.R0_scale
    # Rm = R0/dbc.lambda3
    # M_total = dbc.M_total_gal_1e10MSun
    # v0 = np.sqrt(dbc.G*M_total/dbc.frac_mass/R0)
    
    # N_list = np.array([10000, 100000, 1000000, 10000000, 100000000, 1000000000])
    # hu = 0.16
    # du = 2.27 #??
    # hn = 1.14
    # dn = 1.49
    # h0 = None
    # d0 = 3.
    # Dim_frac_lb = [dn, du, d0]
    # diffur_median = np.zeros(len(N_list))
    # diffur_mean = np.zeros(len(N_list))
    # diffur_median_noised = np.zeros(len(N_list))
    # diffur_mean_noised = np.zeros(len(N_list))
    # diffur_referencevalue = np.zeros(len(N_list))

    # diffur_simulated_list = None
    # diffur_referencevalue_list = None
    # diffur_referencevalue_list = [
    #     N_list, 
    #     diffur_referencevalue, 
    #     hn, 
    #     dn, 
    #     "pos_referencevalue", 
    # ]

    # diffur_uniform_median_list = None
    # diffur_uniform_meanvalue_list = None
    # for (i, N) in enumerate(N_list):
    #     filename = "../data/samples_simulated/snapshot_fractal_Diffustatistics_read0_noised0_N{}.txt".format(N)
    #     data = np.loadtxt(filename)
    #     diffur_referencevalue[i] = data[1]*1.
    #     diffur_mean[i] = data[2]*1.
    #     diffur_median[i] = data[3]*1.
    #     # print(i, N, diffur_median[i], "uniform")

    # diffur_uniform_median_list = [
    #     N_list, 
    #     diffur_median, #note: a pointer, need to copy
    #     hu, 
    #     du, 
    #     "pos_uniform", 
    # ]
    # diffur_uniform_meanvalue_list = [
    #     N_list, 
    #     diffur_mean, 
    #     hu, 
    #     du, 
    #     "pos_uniform", 
    # ]

    # diffur_noised_median_list = None
    # diffur_noised_meanvalue_list = None
    # for (i, N) in enumerate(N_list):
    #     filename = "../data/samples_simulated/snapshot_fractal_Diffustatistics_read0_noised14_N{}.txt".format(N)
    #     data = np.loadtxt(filename)
    #     diffur_referencevalue[i] = data[1]*1.
    #     diffur_mean_noised[i] = data[2]*1.
    #     diffur_median_noised[i] = data[3]*1.
    #     # print(i, N, diffur_median[i], "noised")

    # diffur_noised_median_list = [
    #     N_list, 
    #     diffur_median_noised, 
    #     hn, 
    #     dn, 
    #     "pos_uniform_noise", 
    # ]
    # diffur_noised_meanvalue_list = [
    #     N_list, 
    #     diffur_mean_noised, 
    #     hn, 
    #     dn, 
    #     "pos_uniform_noise", 
    # ]
    # # ads.DEBUG_PRINT_V(0, diffur_median, diffur_mean, "diffur")

    # suffix = "modified_pos"
    # plot_relaxation_time_with_N_and_dim_pos(
    #     R0, v0, M_total, Dim_frac_lb, 
    #     diffur_uniform_median_list=diffur_uniform_median_list, 
    #     diffur_uniform_meanvalue_list=diffur_uniform_meanvalue_list, 
    #     diffur_noised_median_list=diffur_noised_median_list, 
    #     diffur_noised_meanvalue_list=diffur_noised_meanvalue_list, 
    #     diffur_referencevalue_list=diffur_referencevalue_list, 
    #     diffur_simulated_list=diffur_simulated_list, 
    #     suffix=suffix
    # )

    # path_load = "../data/samples_pos/"
    # path_save = "../data/examples_pos/"
    # zeta_pos = diffur_median_noised/diffur_median
    # zeta_versus_N = np.hstack((np.array([N_list]).T, np.array([zeta_pos]).T))
    # np.savetxt(path_save+"zeta_pos.txt", zeta_versus_N)
    # ads.DEBUG_PRINT_V(1, zeta_versus_N, "zeta_pos")



    #### zeta by outter linear fitting
    path_save = "../data/examples_pos/"
    # counts_targets = [1e6]
    # counts_targets = [1e10]
    counts_targets = [1e11]

    suffix = "zeta_pos"
    filename = "../data/examples_pos/zeta_pos.txt"
    data_tmp = np.loadtxt(filename, dtype=float)
    counts_pos, zeta_pos = data_tmp[:, 0], data_tmp[:, 1]
    zeta_pos_targets = dbc.zeta_linear_fitting_outer(
        counts_pos, zeta_pos, counts_targets, save_path=path_save, suffix=suffix
    )
    ads.DEBUG_PRINT_V(1, zeta_pos_targets, "zeta_pos_targets")

    suffix = "zeta_vel"
    filename = "../data/examples_vel/zeta_vel.txt"
    data_tmp = np.loadtxt(filename, dtype=float)
    counts_vel, zeta_vel = data_tmp[:, 0], data_tmp[:, 1]
    zeta_vel_targets = dbc.zeta_linear_fitting_outer(
        counts_vel, zeta_vel, counts_targets, save_path=path_save, suffix=suffix
    )
    ads.DEBUG_PRINT_V(1, zeta_vel_targets, "zeta_vel_targets")

    zeta_total_targets = zeta_pos_targets*zeta_vel_targets
    zeta_versus_N_pos = np.hstack((np.array([counts_targets]).T, np.array([zeta_pos_targets]).T))
    zeta_versus_N_vel = np.hstack((np.array([counts_targets]).T, np.array([zeta_vel_targets]).T))
    zeta_versus_N_total = np.hstack((np.array([counts_targets]).T, np.array([zeta_total_targets]).T))
    zeta_versus_N = np.vstack((zeta_versus_N_pos, zeta_versus_N_vel, zeta_versus_N_total))
    np.savetxt(path_save+"zeta_total.txt", zeta_versus_N)
    print(f"zeta of violent relaxation for {counts_targets[0]:.2e} particles ~ {1e9:.0e}")
    print(f"zeta of local effect       for {counts_targets[0]:.2e} particles ~ {zeta_total_targets[0]:.4e}")
    ads.DEBUG_PRINT_V(1, zeta_total_targets, "zeta_total_targets")
