#!/usr/bin/env python
# -*- coding:utf-8 -*-
#In[] modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors

import analysis_data_distribution as ads
import observed_data_process as odp

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "text.latex.preamble": r"\usepackage{bm}"
# })



## settings
#() note that these should be consist with the C++ code
Dim_emb = 3
G = 43007.1      # Gravitational constant (kpc, km/s, 1 M_sun)
frac_mass = 1.
# frac_mass = 0.05
M_total_gal_1e10MSun = 137.
R0_scale = 75.

GXX = 0.15
lambda1 = 0.75
lambda2 = 1.5
lambda3 = 1.5
lambda4 = 0.2

# R0 = dbc.R0_scale
# Rm = R0/dbc.lambda3
# M_total = dbc.M_total_gal_1e10MSun
# v0 = np.sqrt(dbc.G*M_total/dbc.frac_mass/R0) #set the typical speed as the virial speed
# v2m_sqrt = v0 #set the mean speed as the virial speed



## consts
def calculate_speed_square_mean(vmean_3, dispersion_3):
    mean_v2 = np.sum(np.array(vmean_3)**2)
    disp_v2 = np.sum(np.array(dispersion_3)**2)
    return mean_v2 + disp_v2

def calculate_dispersion_iso(v2m_sqrt, vmean_3):
    mean_v2 = np.sum(np.array(vmean_3)**2)
    return np.sqrt((v2m_sqrt**2-mean_v2) / 3.0)

def calculate_coef_const_Diffu(M_total, N_particles, R0):
    n0 = 3.0 * N_particles / (4.0 * np.pi * R0**3)
    Lambda = lambda4 * N_particles
    log_Lambda = np.log(Lambda)
    return 4 * np.pi * G**2 * M_total**2 * n0 * log_Lambda / N_particles**3

def calculate_v_epsilon(v0, lambda4, N_particles):
    return v0 / (lambda4 * N_particles)

def calculate_coef_const_Diffu_vel(M_total, N_particles, R0):
    n0 = N_particles / (4.0 * np.pi * R0**3)
    Lambda = lambda4 * N_particles
    log_Lambda = np.log(Lambda)
    return 4 * np.pi * G**2 * M_total**2 * n0 * log_Lambda / N_particles**2 / N_particles

def calculate_Diffu_0_unit(N_particles, M_total, n0, v0):
    '''
    Return the unit value of diffusion coef.
    '''
    Diffu_0_unit = (
        4.0 * np.pi * G * G * M_total * M_total * n0
    ) / (3.0 * N_particles * N_particles * v0)
    return Diffu_0_unit

def diffusion_reference_value_Diffu_ref(N_particles, M_total, R0, v0):
    '''
    Return the reference value of diffusion coef.
    '''
    ln_Lambda = np.log(lambda4 * N_particles)
    n0        = (3.0 * N_particles) / (4.0 * np.pi * R0**3)
    Diffu_0_unit = calculate_Diffu_0_unit(N_particles, M_total, n0, v0)
    return np.sqrt(6.0 / np.pi) * GXX * ln_Lambda * Diffu_0_unit

def calculate_fixed_const_coefs(N_particles, M_total, R0, v0):
    '''
    Return a dict for consts.
    '''
    # 1) geometry & sampling
    N_samples = N_particles
    Rm        = R0 / lambda2
    R_epsilon = R0 / (lambda4 * N_particles)
    b90       = R_epsilon

    # 2) velocities & softening
    v2m_sqrt   = v0
    v2m        = v0 * v0
    v_epsilon  = v0 / (lambda4 * N_particles)

    # 3) Coulomb logarithm & number density
    ln_Lambda = np.log(lambda4 * N_particles)
    n0        = (3.0 * N_particles) / (4.0 * np.pi * R0**3)

    # 4) unit diffusion
    Diffu_0_unit = calculate_Diffu_0_unit(N_particles, M_total, n0, v0)

    # 5) reference values
    Diffu_ref = diffusion_reference_value_Diffu_ref(N_particles, M_total, n0, v0)

    coef_Diffu_parallel_iso   = (
        2.0 * lambda1 * lambda2**2 * lambda3
        * np.sqrt(6.0 / np.pi) * GXX * Diffu_0_unit
        / N_samples
    )
    coef_Diffu_tensor_uniform = 3.0 * ln_Lambda * Diffu_0_unit / N_samples

    # 6) separable & constant coefficients
    coef_Diffu_separable    = 2.0 * lambda1 * lambda2**2 * lambda3 * Diffu_0_unit
    coef_const_Diffu        = calculate_coef_const_Diffu(M_total, N_particles, R0)
    coef_const_Diffu_vel    = calculate_coef_const_Diffu_vel(M_total, N_particles, R0)

    # 7) legacy ratios (for debugging/comparison)
    ratio_Diffu_old     = Rm * Rm * coef_Diffu_parallel_iso / coef_const_Diffu
    ratio_Diffu_old_vel = v2m_sqrt * coef_Diffu_tensor_uniform / coef_const_Diffu_vel

    return {
        # as input
        "N_particles": N_particles,
        "M_total": M_total,
        "R0": R0,
        "v0": v0,

        # as output
        "N_samples": N_samples,
        "Rm": Rm,
        "R_epsilon": R_epsilon,
        # "b90": b90,
        "v2m_sqrt": v2m_sqrt,
        # "v2m": v2m,
        "v_epsilon": v_epsilon,
        "ln_Lambda": ln_Lambda,
        "n0": n0,
        "Diffu_0_unit": Diffu_0_unit,
        "Diffu_ref": Diffu_ref,
        "coef_Diffu_parallel_iso": coef_Diffu_parallel_iso,
        "coef_Diffu_tensor_uniform": coef_Diffu_tensor_uniform,
        "coef_Diffu_separable": coef_Diffu_separable,
        # "coef_const_Diffu": coef_const_Diffu,
        # "coef_const_Diffu_vel": coef_const_Diffu_vel,
        "ratio_Diffu_old": ratio_Diffu_old,
        "ratio_Diffu_old_vel": ratio_Diffu_old_vel,
    }

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



## plot functions
def plot_velocity_speed_distribution(
    vel_list, label_list, 
    bandwidth: float = 5.0, k_neighbors: int = None, v_range=None, 
    save_path: str = "save_path/", suffix: str = "suffix", 
    is_show: bool = False, pos_or_vel="vel"
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
    # all samples
    N_compare = len(vel_list)
    all_speeds = list(range(N_compare))
    all_dfs = list(range(N_compare))

    # Step 1: compute speed arrays
    for i in range(N_compare):
        all_speeds[i] = np.linalg.norm(vel_list[i], axis=1)
    all_speeds_concatenate = np.concatenate(all_speeds)

    labels = label_list
    
    v_range = None
    if v_range is None:
        v_min, v_max = np.percentile(all_speeds_concatenate, [0., 98.])
    else:
        v_min, v_max = v_range
    v_grid = np.linspace(v_min, v_max, 500)

    # Step 2: compute densities
    if bandwidth is not None:
        for i in range(N_compare):
            all_dfs[i] = kde_bandwidth_1d(all_speeds[i], v_grid, bandwidth)
    else:
        for i in range(N_compare):
            all_dfs[i] = knn_kde_1d(all_speeds[i], v_grid, k=k_neighbors)

    # Step 4: Plot
    plt.figure(figsize=(10, 6))
    for i in range(N_compare):
        plt.plot(v_grid, all_dfs[i], label=labels[i], lw=2)
    if pos_or_vel == "vel":
        plt.xlabel(r"speed $v$, $\mathrm{km/s}$")
        plt.ylabel(r"PDF (KDE) $f(v)$, $1/\mathrm{km/s}$")
    else: #pos
        plt.xlabel(r"radius $r$, $\mathrm{kpc}$")
        plt.ylabel(r"number density (KDE) $n(r)$, $1/\mathrm{kpc}$")
    # title = "Velocity Speed Distribution (KDE)"
    # plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_file = None
    if pos_or_vel == "vel":
        save_file = save_path+"velocity_DF_speed_compare_{}.eps".format(suffix)
    else: #pos
        save_file = save_path+"rho_r_compare_{}.eps".format(suffix)
    fig_tmp = plt.gcf()
    fig_tmp.savefig(save_file, format="eps", bbox_inches='tight')
    if is_show:
        plt.show()
    print("Plot velocity_DF_speed_compare, done.")

def plot_velocity_DF_contour_compare(
    vel_list, label_list, 
    bandwidth: float = 10.0, k_neighbors: int = 32, 
    N_grid: tuple = (100, 80, 9), percent_clip: float = 98.0, 
    levels=np.linspace(-8., 0.2, 16), save_path: str = "./", 
    is_show: bool = False, pos_or_vel="vel"
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
    ## calculate
    datasets = vel_list
    labels = label_list

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

    ## plot
    fontsize = 14
    figsize = (20, 15)

    for idx, vel_data in enumerate(datasets):
        fig = plt.figure(figsize=figsize)
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
        levels = np.linspace(levels_min, levels_max, 16)
        # ads.DEBUG_PRINT_V(0, levels, "levels")

        for k in range(Nz):
            ax = fig.add_subplot(3, 3, k+1)
            contour = ax.contourf(
                grid_x, grid_y, rho_logcube[:, :, k].T,
                levels=levels, cmap="viridis"
            )
            ax.set_aspect('equal')
            if pos_or_vel == "vel":
                ax.set_title(f"{labels[idx]}, $v_z$ = {grid_z[k]:.2f} km/s", fontsize=fontsize)
                ax.set_xlabel("$v_x, km/s$", fontsize=fontsize)
                ax.set_ylabel("$v_y, km/s$", fontsize=fontsize)
            else:
                ax.set_title(f"{labels[idx]}, $z$ = {grid_z[k]:.2f} kpc", fontsize=fontsize)
                ax.set_xlabel("$x$, kpc", fontsize=fontsize)
                ax.set_ylabel("$y$, kpc", fontsize=fontsize)
            ax.tick_params(labelsize=fontsize)

        cax = fig.add_axes([0.92, 0.35, 0.015, 0.3])
        cbar = fig.colorbar(contour, cax=cax)
        if pos_or_vel == "vel":
            cbar.set_label(fr"$\log{{10}} ( f(\mathbf{{v}}) / f_{{\mathrm{{center}}}} )$", fontsize=fontsize)
        else:
            cbar.set_label(fr"$\log{{10}} ( n(\mathbf{{r}}) / n_{{\mathrm{{center}}}} )$", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

        fig.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.07, wspace=0.3, hspace=0.4)
        save_file = None
        if pos_or_vel == "vel":
            save_file = save_path + f"velocity_DF_contour_compare_"+labels[idx]+".pdf"
        else:
            save_file = save_path + f"spacial_density_contour_compare_"+labels[idx]+".pdf"
        fig_tmp = plt.gcf()
        fig_tmp.savefig(save_file, format="pdf", dpi=200, bbox_inches='tight')
        if is_show:
            plt.show()
        plt.close()
    print("Plot velocity_DF_contour_compare, done.")

def plot_sample_3D_pos_or_vel(
    vel, path_save, Dim_frac=None, is_set_lim=True, 
    suffix="suffix", pos_or_vel="vel"
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(vel[:, 0], vel[:, 1], vel[:, 2], s=0.1, alpha=1.0, rasterized=True)
    if pos_or_vel == "vel":
        ax.set_xlabel(r"$v_x$, $\mathrm{km/s}$")
        ax.set_xlabel(r"$v_y$, $\mathrm{km/s}$")
        ax.set_xlabel(r"$v_z$, $\mathrm{km/s}$")
    else:
        ax.set_xlabel(r"$x$, $\mathrm{kpc}$")
        ax.set_xlabel(r"$x$, $\mathrm{kpc}$")
        ax.set_xlabel(r"$x$, $\mathrm{kpc}$")
    if is_set_lim:
        all_data = vel
        percent_clip = 98.0
        bounds = np.array([
            [-np.percentile(np.abs(all_data[:, i]), percent_clip), np.percentile(np.abs(all_data[:, i]), percent_clip)]
            for i in range(3)
        ])
        bounds[0][0] = np.min([bounds[0][0], bounds[1][0]]) #use the max bound of x and y
        bounds[1][0] = np.min([bounds[0][0], bounds[1][0]])
        bounds[0][1] = np.max([bounds[0][1], bounds[1][1]])
        bounds[1][1] = np.max([bounds[0][1], bounds[1][1]])
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])
        ax.set_zlim(bounds[2][0], bounds[2][1])
    if Dim_frac is not None:
        ax.set_title(fr"fractal dim $D_\mathrm{{frac}}$ = {Dim_frac:.4f}")
    if pos_or_vel == "vel":
        save_file = path_save+"samplepoints_vel_{}.pdf".format(suffix)
    else:
        save_file = path_save+"samplepoints_pos_{}.pdf".format(suffix)
    fig_tmp = plt.gcf()
    fig_tmp.savefig(save_file, format="pdf", dpi=200, bbox_inches='tight')
    plt.close()

    print("Plot samplepoints_{}, done.".format(suffix))
    return 0

def plot_normalized_pdf_components(Diffu_data_components, label_list, Diffu_0, suffix="suffix", pos_or_vel="vel", k_neighbors=32):
    fontsize = 40.
    pointsize = 3.2
    figsize = 20, 15 #for 3, 3
    dpi = None
    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True)
    colors = ["k", "r", "orange", "yellow", "g", "b", "purple"]

    for i in range(7):
        Diffu_data = np.abs(Diffu_data_components[:, i+1])
        min_val = np.min(Diffu_data)
        max_val = np.max(Diffu_data)
        mean_val = np.mean(Diffu_data)
        median_val = np.median(Diffu_data)
        count_total = len(Diffu_data)
        # ads.DEBUG_PRINT_V(1, min_val, max_val, mean_val, median_val, count_total, "Diffu_data")
        
        v_min, v_max = np.percentile(Diffu_data, [0., 100.])
        # v_min, v_max = np.percentile(Diffu_data, [0., 98.])
        v_grid = np.geomspace(v_min, v_max, 100)
        # ads.DEBUG_PRINT_V(0, v_grid, "v_grid")
        # bandwidth = (v_max-v_min)/50.
        # normalized_distribution = kde_bandwidth_1d(Diffu_data, v_grid, bandwidth)
        normalized_distribution = knn_kde_1d(Diffu_data, v_grid, k_neighbors)
        plt.plot(v_grid, normalized_distribution, 'o-', label='abs of component {}'.format(label_list[i]), color=colors[i], lw=pointsize)
        # plt.plot([mean_val, mean_val], [0., np.max(normalized_distribution)], '--', label='Diffu_mean = {:.4f}'.format(mean_val), color=colors[i], lw=pointsize)
        # plt.plot([median_val, median_val], [0., np.max(normalized_distribution)], '-.', label='Diffu_median = {:.4f}'.format(median_val), color=colors[i], lw=pointsize)
    
    if Diffu_0 is not None:
        plt.plot([Diffu_0, Diffu_0], [0., np.max(normalized_distribution)], label='Diffu_0 = {:.4f}'.format(Diffu_0), color='k', lw=pointsize)
    
    plt.xscale("log")
    plt.yscale("log")
    # plt.title(r"histogram of main diffusion coefficient of all particles", fontsize=fontsize)
    if pos_or_vel == "vel":
        plt.xlabel(r"diffusion $D_\mathrm{vel}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
    else:
        plt.xlabel(r"diffusion $D_\mathrm{pos}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
    plt.ylabel(r"distribution $f$, $\mathrm{kpc/(km/s)^3}$", fontsize=fontsize)
    
    plt.legend(fontsize=fontsize*0.6, loc=0)
    plt.tick_params(which='major', length=0, labelsize=fontsize * 1.)
    plt.tight_layout()
    if pos_or_vel == "vel":
        save_file = "../data/examples_vel/diffu_v_components_DF_{}.pdf".format(suffix)
    else:
        save_file = "../data/examples_pos/diffu_r_components_DF_{}.pdf".format(suffix)
    fig_tmp = plt.gcf()
    fig_tmp.savefig(save_file, format="pdf", bbox_inches='tight')
    plt.close()
    
    print("Plot diffu_components_{}, done.".format(suffix))
    return 0

def plot_normalized_pdf_from_eachpoints_histogram(Diffu_data, Diffu_0, suffix="suffix", pos_or_vel="vel"):
    # data = np.loadtxt(filename)
    # Diffu_data = data[:, 1]
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
    plt.grid(True)

    N_bins = 100
    bins = np.geomspace(min_val, max_val, N_bins)
    counts, bin_edges, _ = plt.hist(Diffu_data, bins=bins, density=False, alpha=0.5, edgecolor='black')
    bin_widths = np.diff(bin_edges)
    normalized_distribution = counts / (bin_widths * count_total)
    plt.clf()  # Clear the previous histogram
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    plt.plot(bin_centers, normalized_distribution, 'o-', label='Normalized Distribution', color='b', lw=pointsize)

    plt.grid(True)
    plt.plot([mean_val, mean_val], [0., np.max(normalized_distribution)], '--', label='Diffu_mean = {:.4f}'.format(mean_val), color='k', lw=pointsize)
    plt.plot([median_val, median_val], [0., np.max(normalized_distribution)], '-.', label='Diffu_median = {:.4f}'.format(median_val), color='k', lw=pointsize)
    if Diffu_0 is not None:
        plt.plot([Diffu_0, Diffu_0], [0., np.max(normalized_distribution)], label='Diffu_0 = {:.4f}'.format(Diffu_0), color='k', lw=pointsize)
    
    plt.xscale("log")
    plt.yscale("log")
    # plt.title(r"histogram of main diffusion coefficient of all particles", fontsize=fontsize)
    if pos_or_vel == "vel":
        plt.xlabel(r"diffusion $D_\mathrm{vel}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
    else:
        plt.xlabel(r"diffusion $D_\mathrm{pos}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
    plt.ylabel(r"distribution $f$, $\mathrm{kpc/(km/s)^3}$", fontsize=fontsize)
    
    plt.legend(fontsize=fontsize*0.6, loc=0)
    plt.tick_params(which='major', length=0, labelsize=fontsize * 1.)
    plt.tight_layout()
    if pos_or_vel == "vel":
        save_file = "../data/examples_vel/diffu_eff_DF_{}_histogram.eps".format(suffix)
    else:
        save_file = "../data/examples_pos/diffu_r_DF_{}_histogram.eps".format(suffix)
    fig_tmp = plt.gcf()
    fig_tmp.savefig(save_file, format="eps", bbox_inches='tight')
    plt.close()
    
    print("Plot {}, done.".format(suffix))
    return 0

def plot_normalized_pdf_from_eachpoints_histogram_vel_data(
    Diffu_data, Diffu_0, suffix="suffix", pos_or_vel="vel", vel_data=None, D_0=None
):
    """
    Upper subplot: scatter of diffusion coefficient vs particle speed.
    Lower subplot: normalized histogram (PDF) of diffusion coefficient.
    
    Labels, scaling, and saving follow the original layout.
    """
    if vel_data is None:
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
        plt.grid(True)

        # N_bins = 100
        # bins = np.geomspace(min_val, max_val, N_bins)
        # counts, bin_edges, _ = plt.hist(Diffu_data, bins=bins, density=False, alpha=0.5, edgecolor='black')
        # bin_widths = np.diff(bin_edges)
        # normalized_distribution = counts / (bin_widths * count_total)
        # plt.clf()  # Clear the previous histogram
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        # plt.plot(bin_centers, normalized_distribution, 'o-', label='Normalized Distribution', color='b', lw=pointsize)

        N_bins = 100
        bins = np.geomspace(min_val, max_val, N_bins)
        counts, bin_edges = np.histogram(Diffu_data, bins=bins)
        widths = np.diff(bin_edges)
        pdf_vals = counts / (widths * count_total)
        centers = bin_edges[:-1] * np.sqrt(bin_edges[1:] / bin_edges[:-1])
        plt.plot(centers, pdf_vals, 'o-', lw=pointsize, label='Normalized Distribution')
        
        # plt.grid(True)
        # plt.plot([mean_val, mean_val], [0., np.max(normalized_distribution)], '--', label='Diffu_mean = {:.4f}'.format(mean_val), color='k', lw=pointsize)
        # plt.plot([median_val, median_val], [0., np.max(normalized_distribution)], '-.', label='Diffu_median = {:.4f}'.format(median_val), color='k', lw=pointsize)
        # if Diffu_0 is not None:
        #     plt.plot([Diffu_0, Diffu_0], [0., np.max(normalized_distribution)], label='Diffu_0 = {:.4f}'.format(Diffu_0), color='k', lw=pointsize)
        
        plt.axvline(mean_val,   color='k', ls='--', lw=pointsize, label=f'Diffu_mean = {mean_val:.4f}')
        plt.axvline(median_val, color='k', ls='-.', lw=pointsize, label=f'Diffu_median = {median_val:.4f}')
        if Diffu_0 is not None:
            plt.axvline(Diffu_0, color='k', lw=pointsize, label=f'Diffu_ref = {Diffu_0:.4f}')
        
        plt.xscale("log")
        plt.yscale("log")
        # plt.title(r"histogram of main diffusion coefficient of all particles", fontsize=fontsize)
        if pos_or_vel == "vel":
            plt.xlabel(r"diffusion $D_\mathrm{vel}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
        else:
            plt.xlabel(r"diffusion $D_\mathrm{pos}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
        plt.ylabel(r"distribution $f$, $\mathrm{kpc/(km/s)^3}$", fontsize=fontsize)
        
        plt.legend(fontsize=fontsize*0.6, loc=0)
        plt.tick_params(which='major', length=0, labelsize=fontsize * 1.)
        plt.tight_layout()
        if pos_or_vel == "vel":
            save_file = "../data/examples_vel/diffu_eff_DF_{}_histogram.eps".format(suffix)
        else:
            save_file = "../data/examples_pos/diffu_r_DF_{}_histogram.eps".format(suffix)
        fig_tmp = plt.gcf()
        fig_tmp.savefig(save_file, format="eps", bbox_inches='tight')
        plt.close()
    
    else:
        # compute speeds
        speeds = np.linalg.norm(vel_data, axis=1)
        
        # basic stats
        min_val = Diffu_data.min()
        max_val = Diffu_data.max()
        mean_val = Diffu_data.mean()
        median_val = np.median(Diffu_data)
        count_total = len(Diffu_data)
        
        fontsize = 40.
        pointsize = 3.2
        figsize = 20, 20
        dpi = None
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Upper panel: scatter(Diffu_data, speeds)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.scatter(Diffu_data, speeds, s=5, alpha=1.0, rasterized=True)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.grid(True)
        if pos_or_vel == "vel":
            ax1.set_xlabel(r"diffusion $D_\mathrm{vel}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
            ax1.set_ylabel(r"speed $v$, $\mathrm{km/s}$", fontsize=fontsize)
        else:
            ax1.set_xlabel(r"diffusion $D_\mathrm{pos}$, $\mathrm{(kpc)^3/kpc}$", fontsize=fontsize)
            ax1.set_ylabel(r"position scale, $\mathrm{kpc}$", fontsize=fontsize)
        # ax1.set_title("Particle speed vs. diffusion coef", fontsize=fontsize)
        
        # Lower panel: normalized histogram
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        N_bins = 100
        bins = np.geomspace(min_val, max_val, N_bins)
        counts, bin_edges = np.histogram(Diffu_data, bins=bins)
        widths = np.diff(bin_edges)
        pdf_vals = counts / (widths * count_total)
        centers = bin_edges[:-1] * np.sqrt(bin_edges[1:] / bin_edges[:-1])
        ax2.plot(centers, pdf_vals, 'o-', lw=pointsize, label='Normalized Distribution')
        
        ax2.axvline(mean_val,   color='k', ls='--', lw=pointsize, label=f'Diffu_mean = {mean_val:.4f}')
        ax2.axvline(median_val, color='k', ls='-.', lw=pointsize, label=f'Diffu_median = {median_val:.4f}')
        if Diffu_0 is not None:
            ax2.axvline(Diffu_0, color='k', lw=pointsize, label=f'Diffu_ref = {Diffu_0:.4f}')
        
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True)
        if pos_or_vel == "vel":
            ax2.set_xlabel(r"diffusion $D_\mathrm{vel}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
            ax2.set_ylabel(r"distribution $f$, $\mathrm{kpc/(km/s)^3}$", fontsize=fontsize)
        else:
            ax2.set_xlabel(r"diffusion $D_\mathrm{pos}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
            ax2.set_ylabel(r"distribution $f$, $\mathrm{kpc/(km/s)^3}$", fontsize=fontsize)
        ax2.legend(fontsize=fontsize*0.6, loc=0)
        ax2.tick_params(which='major', length=0, labelsize=fontsize)
        
        plt.tight_layout()
        if pos_or_vel == "vel":
            save_file = f"../data/examples_vel/diffu_eff_DF_{suffix}_histogram.pdf"
        else:
            save_file = f"../data/examples_pos/diffu_r_DF_{suffix}_histogram.pdf"
        fig.savefig(save_file, format="pdf", bbox_inches='tight')
        plt.close(fig)
        
        print(f"Plot {suffix}, done.")
    return 0

def plot_normalized_pdf_from_eachpoints_KDE(Diffu_data, Diffu_0, suffix="suffix", pos_or_vel="vel", k_neighbors=32):
    # data = np.loadtxt(filename)
    # Diffu_data = data[:, 1]
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
    plt.grid(True)

    v_min, v_max = np.percentile(Diffu_data, [0., 100.])
    # v_min, v_max = np.percentile(Diffu_data, [0., 98.])
    v_grid = np.geomspace(v_min, v_max, 100)
    # ads.DEBUG_PRINT_V(0, v_grid, "v_grid")
    # bandwidth = (v_max-v_min)/50.
    # normalized_distribution = kde_bandwidth_1d(Diffu_data, v_grid, bandwidth)
    normalized_distribution = knn_kde_1d(Diffu_data, v_grid, k_neighbors)
    plt.plot(v_grid, normalized_distribution, 'o-', label='Normalized Distribution', color='b', lw=pointsize)

    plt.plot([mean_val, mean_val], [0., np.max(normalized_distribution)], '--', label='Diffu_mean = {:.4f}'.format(mean_val), color='k', lw=pointsize)
    plt.plot([median_val, median_val], [0., np.max(normalized_distribution)], '-.', label='Diffu_median = {:.4f}'.format(median_val), color='k', lw=pointsize)
    if Diffu_0 is not None:
        plt.plot([Diffu_0, Diffu_0], [0., np.max(normalized_distribution)], label='Diffu_0 = {:.4f}'.format(Diffu_0), color='k', lw=pointsize)
    
    plt.xscale("log")
    plt.yscale("log")
    # plt.title(r"histogram of main diffusion coefficient of all particles", fontsize=fontsize)
    if pos_or_vel == "vel":
        plt.xlabel(r"diffusion $D_\mathrm{vel}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
    else:
        plt.xlabel(r"diffusion $D_\mathrm{pos}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
    plt.ylabel(r"distribution $f$, $\mathrm{kpc/(km/s)^3}$", fontsize=fontsize)
    
    plt.legend(fontsize=fontsize*0.6, loc=0)
    plt.tick_params(which='major', length=0, labelsize=fontsize * 1.)
    plt.tight_layout()
    if pos_or_vel == "vel":
        save_file = "../data/examples_vel/diffu_eff_DF_{}_KDE.eps".format(suffix)
    else:
        save_file = "../data/examples_pos/diffu_r_DF_{}_KDE.eps".format(suffix)
    fig_tmp = plt.gcf()
    fig_tmp.savefig(save_file, format="eps", bbox_inches='tight')
    plt.close()
    
    print("Plot {}, done.".format(suffix))
    return 0

def plot_normalized_pdf_from_percentile(data, mean_val, median_val, Diffu_0, suffix, pos_or_vel="vel"):
    # data = np.loadtxt(filename)
    q = data[:, 0]  # Percentiles
    values = data[:, 1]  # Corresponding values
    # Diffu_0 = Diffu_represent[1]
    # mean_val = Diffu_represent[2]
    # median_val = Diffu_represent[3]
    ads.DEBUG_PRINT_V(1, q, values, "perentile")

    # bin_centers = np.diff(values)
    bin_centers = (values[:-1] + values[1:]) / 2.0
    normalized_distribution = np.diff(q)/np.diff(values)

    # Plotting settings
    fontsize = 40
    pointsize = 3.2
    figsize = (20, 15)  # For 3x3 layout
    dpi = 400
    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True)

    plt.plot(bin_centers, normalized_distribution, 'o-', label="Normalized DF", markersize=pointsize)
    plt.plot([mean_val, mean_val], [0., np.max(normalized_distribution)], '--', label='Diffu_mean = {:.4f}'.format(mean_val), color='k', lw=pointsize)
    plt.plot([median_val, median_val], [0., np.max(normalized_distribution)], '-.', label='Diffu_median = {:.4f}'.format(median_val), color='k', lw=pointsize)
    if Diffu_0 is not None:
        plt.plot([Diffu_0, Diffu_0], [0., np.max(normalized_distribution)], label='Diffu_0 = {:.4f}'.format(Diffu_0), color='k', lw=pointsize)
    
    # Set logarithmic scale for x and y axes
    plt.xscale("log")
    plt.yscale("log")

    # Title and labels
    # plt.title(r"Histogram of main diffusion coefficient of all particles", fontsize=fontsize)
    if pos_or_vel == "vel":
        plt.xlabel(r"Diffusion $D_\mathrm{vel}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
    else:
        plt.xlabel(r"Diffusion $D_\mathrm{pos}$, $\mathrm{(km/s)^3/kpc}$", fontsize=fontsize)
    plt.ylabel(r"Distribution $f$, $\mathrm{kpc/(km/s)^3}$", fontsize=fontsize)

    plt.legend(fontsize=fontsize * 0.6, loc=0)
    plt.tick_params(which='major', length=0, labelsize=fontsize * 1.)
    plt.tight_layout()
    if pos_or_vel == "vel":
        save_file = "../data/examples_vel/diffu_eff_DF_{}_percentile.eps".format(suffix)
    else:
        save_file = "../data/examples_pos/diffu_r_DF_{}_percentile.eps".format(suffix)
    fig_tmp = plt.gcf()
    fig_tmp.savefig(save_file, format="eps", bbox_inches='tight')
    plt.close()
    print("Saved fig of diffueff_percentile.")
    return 0



def eta_ralaxaiton_time_ratio_frac(N_particles, M_total, R0, v0, Dim_frac):
    '''
    Note the mass in calculating Diffu should be m = M_total/N_particles.
    '''
    Rm = R0/lambda2
    coef_eta_frac_inv = 2.*lambda1*lambda3 * (G**2*M_total**2)/(Rm**2*v0**4) * np.sqrt(6./np.pi)*GXX
    eta_relax = None
    if Dim_frac<=0. or Dim_frac>3.:
        print("Wrong value of Dim_frac. Please check.")
        eta_relax = np.nan
    elif Dim_frac==2.: #be 2
        eta_relax = N_particles / (Dim_frac * np.log(lambda4*N_particles))
    elif Dim_frac>2.: #in (2,3]
        eta_relax = (Dim_frac-2.) * N_particles / Dim_frac
    else: #in (0,2)
        eta_relax = (2.-Dim_frac) * N_particles**(Dim_frac-1.) / (lambda4**(2.-Dim_frac) * Dim_frac)
    return eta_relax/coef_eta_frac_inv

def eta_ralaxaiton_time_ratio(Diffu_parallel_statistics, R0, v0):
    '''
    Note the mass in calculating Diffu should be m = M_total/N_particles.
    '''
    eta_relax = v0**3 / (R0*Diffu_parallel_statistics)
    return eta_relax

def plot_relaxation_time_with_N_and_dim_pos(
    R0, v0, M_total, Dim_frac_lb, 
    diffueff_uniform_median_list, diffueff_uniform_meanvalue_list, 
    diffueff_noised_median_list, diffueff_noised_meanvalue_list, 
    diffueff_observed_list=None, diffueff_simulated_list=None, 
    diffueff_referencevalue_list=None, suffix="suffix"
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
        diffu_N_arr_referencevalue[i] = diffusion_reference_value_Diffu_ref(N_particles=N_particles_arr[i], M_total=M_total, R0=R0, v0=v0)
        eta_N_arr_referencevalue[i] = eta_ralaxaiton_time_ratio(diffu_N_arr_referencevalue[i], R0, v0)
    
    for j in np.arange(N_dimlb_plot):
        for i in np.arange(N_points_plot):
            eta_N_arr_count[j, i] = eta_ralaxaiton_time_ratio_frac(N_particles_arr[i], M_total, R0, v0, Dim_frac=Dim_frac_lb[j])

    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True)
    color = {
        "pos_uniform": "b", "pos_noised": "r", "pos_referencevalue": "k",
        "pos_simu": "g", "pos_obs": "orange", "pos_extra": "purple", 
    }
    # colorj = ["red", "orange", "green", "blue", "purple"]
    colorj = ["r", "b", "k", "orange", "g"]

    plt.plot(N_particles_arr, eta_N_arr_referencevalue, "-", label="by referencevalue", color="k", lw=pointsize*0.5)
    for j in np.arange(N_dimlb_plot):
        # ads.DEBUG_PRINT_V(1, j, eta_N_arr_count[j])
        plt.plot(N_particles_arr, eta_N_arr_count[j], "-.", label="by single fractal model, Dim_frac={:.2f}".format(Dim_frac_lb[j]), color=colorj[j], lw=pointsize*0.5)

    dldl = [diffueff_uniform_median_list, diffueff_noised_median_list, ]
    for dl in dldl:
        N_list = dl[0]
        diffueff_list = dl[1]
        N_N_list = len(N_list)
        if dl[2] is not None:
            h = dl[2]*N_list/1e4
        d = dl[3]
        pos_type_name = dl[4]
        eta_IR2_list = np.zeros(N_N_list)
        eta_diffu_list = np.zeros(N_N_list)
        for i in np.arange(N_N_list):
            # eta_IR2_list[i] = dci.rate_t_relax_to_t_cross_IR2(N_list[i], h_frac=h[i], Dim_frac=d, R_p=rmean)
            eta_diffu_list[i] = eta_ralaxaiton_time_ratio(diffueff_list[i], R0, v0)
        # ads.DEBUG_PRINT_V(1, diffueff_list, eta_diffu_list, "diffur")
        # plt.plot(N_list, eta_IR2_list, "-.", label="eta_IR2, Dim_frac={:.2f}".format(d), color=color[pos_type_name], lw=pointsize*0.5)
        plt.scatter(N_list, eta_diffu_list, label="by median value of {}".format(pos_type_name), color=color[pos_type_name], s=pointsize*60., marker="*")
    
    # dldl = [diffueff_uniform_meanvalue_list, diffueff_noised_meanvalue_list, ]
    # for dl in dldl:
    #     N_list = dl[0]
    #     diffueff_list = dl[1]
    #     N_N_list = len(N_list)
    #     if dl[2] is not None:
    #         h = dl[2]*N_list/1e4
    #     d = dl[3]
    #     pos_type_name = dl[4]
    #     eta_IR2_list = np.zeros(N_N_list)
    #     eta_diffu_list = np.zeros(N_N_list)
    #     for i in np.arange(N_N_list):
    #         # eta_IR2_list[i] = dci.rate_t_relax_to_t_cross_IR2(N_list[i], h_frac=h[i], Dim_frac=d, R_p=rmean)
    #         eta_diffu_list[i] = eta_ralaxaiton_time_ratio(diffueff_list[i], R0, v0)
    #     # ads.DEBUG_PRINT_V(1, diffueff_list, eta_diffu_list, "diffur")
    #     # plt.plot(N_list, eta_IR2_list, "-.", label="eta_IR2, Dim_frac={:.2f}".format(d), color=color[pos_type_name], lw=pointsize*0.5)
    #     plt.scatter(N_list, eta_diffu_list, label="by mean value of {}".format(pos_type_name), color=color[pos_type_name], s=pointsize*60., marker="+")
    
    plt.xscale("log")
    plt.yscale("log")
    # plt.title(r"eta_relax versus N", fontsize=fontsize)
    plt.xlabel(r"particle count $N$", fontsize=fontsize)
    plt.ylabel(r"relaxation time ratio $\eta$", fontsize=fontsize)
    plt.legend(fontsize=fontsize*0.36, loc=0)
    plt.tick_params(which='major', length=0, labelsize=fontsize*0.8) #size of the number characters
    plt.tight_layout()

    plt.savefig("../data/examples_pos/relaxation_time_with_N_and_dim_{}.eps".format(suffix), format="eps", bbox_inches='tight')
    plt.close()
    print("Saved fig of relaxation_time_with_N_and_dim_pos.")
    return 0

def plot_relaxation_time_with_N_and_dim_vel(
    R0, v0, M_total, Dim_frac_lb, 
    diffueff_iso_median_list, diffueff_iso_mean_list, 
    diffueff_aniso_median_list, diffueff_aniso_mean_list, 
    diffueff_tail_median_list, diffueff_tail_mean_list, 
    diffueff_noised_median_list, diffueff_noised_mean_list, 
    diffueff_composite_median_list, diffueff_composite_mean_list, 
    diffueff_observed_list=None, diffueff_simulated_list=None, 
    diffueff_referencevalue_list=None, suffix="suffix"
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

    N_particles_arr = np.geomspace(1e1, 1e6, N_points_plot)
    N_dimlb_plot = len(Dim_frac_lb)
    eta_N_arr_count = np.zeros((N_dimlb_plot, N_points_plot))
    diffu_N_arr_referencevalue = np.zeros(N_points_plot)
    eta_N_arr_referencevalue = np.zeros(N_points_plot)
    for i in np.arange(N_points_plot):
        diffu_N_arr_referencevalue[i] = diffusion_reference_value_Diffu_ref(N_particles=N_particles_arr[i], M_total=M_total, R0=R0, v0=v0)
        eta_N_arr_referencevalue[i] = eta_ralaxaiton_time_ratio(diffu_N_arr_referencevalue[i], R0, v0)
    # for j in np.arange(N_dimlb_plot):
    #     for i in np.arange(N_points_plot):
    #         eta_N_arr_count[j, i] = dci.eta_ralaxaiton_time_ratio_frac(N_particles_arr[i], M_total, R0, v0, Dim_frac=Dim_frac_lb[j]) #not updated

    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True)
    color = {
        "vel_iso": "b", "vel_aniso": "g", "vel_noised": "r", "vel_referencevalue": "k",
        "vel_tail": "cyan", "vel_composite": "orange", "vel_extra": "purple", 
    }
    # colorj = ["red", "orange", "green", "blue", "purple"]
    colorj = ["r", "b", "k", "orange", "g"]

    # for j in np.arange(N_dimlb_plot):
    #     plt.plot(N_particles_arr, eta_N_arr_count[j], "-.", label="eta_count, Dim_frac={:.2f}".format(Dim_frac_lb[j]), color=colorj[j], lw=pointsize*0.5)

    plt.plot(N_particles_arr, eta_N_arr_referencevalue, "-", label="eta_count, referencevalue", color="k", lw=pointsize*0.5)
    
    dldl = [
        diffueff_iso_median_list, diffueff_aniso_median_list, diffueff_tail_median_list, 
        diffueff_noised_median_list, diffueff_composite_median_list, 
    ]
    for dl in dldl:
        N_list = dl[0]
        diffueff_list = dl[1]
        N_N_list = len(N_list)
        h = None
        if dl[2] is not None:
            h = dl[2]*N_list/1e4
        d = dl[3]
        pos_type_name = dl[4]
        eta_diffu_list = np.zeros(N_N_list)
        for i in np.arange(N_N_list):
            eta_diffu_list[i] = eta_ralaxaiton_time_ratio(diffueff_list[i], R0, v0)
        # ads.DEBUG_PRINT_V(1, diffueff_list, eta_diffu_list, "diffur")
        plt.scatter(N_list, eta_diffu_list, label="eta_diffu by median of {}".format(pos_type_name), color=color[pos_type_name], s=pointsize*60., marker="*")
    
    # dldl = [
    #     diffueff_iso_mean_list, diffueff_aniso_mean_list, diffueff_tail_mean_list, 
    #     diffueff_noised_mean_list, diffueff_composite_mean_list, 
    # ]
    # for dl in dldl:
    #     N_list = dl[0]
    #     diffueff_list = dl[1]
    #     N_N_list = len(N_list)
    #     h = None
    #     if dl[2] is not None:
    #         h = dl[2]*N_list/1e4
    #     d = dl[3]
    #     pos_type_name = dl[4]
    #     eta_diffu_list = np.zeros(N_N_list)
    #     for i in np.arange(N_N_list):
    #         eta_diffu_list[i] = eta_ralaxaiton_time_ratio(diffueff_list[i], R0, v0)
    #     # ads.DEBUG_PRINT_V(1, diffueff_list, eta_diffu_list, "diffur")
    #     plt.scatter(N_list, eta_diffu_list, label="eta_diffu by meanvalue of {}".format(pos_type_name), color=color[pos_type_name], s=pointsize*60., marker="+")
    
    plt.xscale("log")
    plt.yscale("log")
    # plt.title(r"eta_relax versus N", fontsize=fontsize)
    plt.xlabel(r"particle count $N$", fontsize=fontsize)
    plt.ylabel(r"relaxation time ratio $\eta$", fontsize=fontsize)
    plt.legend(fontsize=fontsize*0.36, loc=0)
    plt.tick_params(which='major', length=0, labelsize=fontsize*0.8) #size of the number characters
    plt.tight_layout()
    save_file = "../data/examples_vel/relaxation_time_with_N_and_dim_{}.eps".format(suffix)
    fig_tmp = plt.gcf()
    fig_tmp.savefig(save_file, format="eps", bbox_inches='tight')
    plt.close()
    print("Saved fig of relaxation_time_with_N_and_dim_vel.")
    return 0



def zeta_linear_fitting_outer(counts, zeta, counts_targets, save_path, suffix="suffix"):
    """
    Extrapolate zeta(N) to new N via a power-law fit ζ = A * N^alpha.
    
    Parameters
    ----------
    counts : array-like, shape (M,)
        Original particle counts N_i.
    zeta : array-like, shape (M,)
        Measured zeta_i at each N_i.
    counts_targets : array-like, shape (K,)
        New N values where we want zeta predictions.
    
    Returns
    -------
    zeta_targets : ndarray, shape (K,)
        Extrapolated ζ at counts_targets.
    """
    # 1) Fit zeta = a * log(N) + b
    counts = np.asarray(counts)
    zeta   = np.asarray(zeta)
    logN = np.log(counts)
    a, b = np.polyfit(logN, zeta, 1)

    # (2) Extrapolate to new counts
    counts_targets = np.asarray(counts_targets)
    zeta_targets = a * np.log(counts_targets) + b

    # (3) Create a smooth curve for the fit
    # spanning from min to max of the union of counts and targets
    N_grid = np.logspace(
        np.log10(min(counts.min(), counts_targets.min())),
        np.log10(max(counts.max(), counts_targets.max())),
        200
    )
    zeta_grid = a * np.log(N_grid) + b

    # (4) Plot
    plt.figure(figsize=(6,5))
    plt.scatter(counts, zeta, label="data", marker='o', s=40, alpha=0.8)
    plt.scatter(counts_targets, zeta_targets, label="extrapolated", marker='x', s=60, c='C1')
    plt.plot(N_grid, zeta_grid,  label=f"fit: ζ={a:.3f} ln N+{b:.3f}", lw=2, c='C2')

    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r"Particle count $N$")
    plt.ylabel(r"Amplification ratio $\zeta$")
    plt.legend()

    plt.tight_layout()
    save_file = save_path+"amplification_ratio_{}.eps".format(suffix)
    fig_tmp = plt.gcf()
    fig_tmp.savefig(save_file, format="eps", bbox_inches='tight')
    # plt.show()

    return zeta_targets



if __name__ == "__main__":

    print(R0_scale)
