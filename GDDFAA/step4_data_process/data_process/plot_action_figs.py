#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import re
import json
# import pdb
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import scipy.optimize as spopt
from scipy.interpolate import RBFInterpolator

import analysis_data_distribution as ads
import galaxy_models as gm
import change_params_galaxy_init as cpgi
import action_state_samples as asa
import transformation_some as ts
import KDTree_python as kdtp
import fit_galaxy_wrapper as fgw
import plot_galaxy_wrapper as pgw
import RW_data_CMGD as rdc
import action_error_by_near_time as aent
import triaxialize_galaxy as tg
import change_params_galaxy_init as cpgi



#### plot funcs
Dim = 3
col_actions = 78
col_frequencies = col_actions+7
mask_select_type = [1, 2, 0, 3] #[1]

def _load_fit_meta_and_vector(fit_txt_path):
    fit_json_path = fit_txt_path.replace(".txt", ".json")
    if os.path.exists(fit_json_path):
        with open(fit_json_path, "r") as f:
            meta = json.load(f)
        names = meta.get("fit_params_names", [])
        vals = meta.get("fit_params_values", [])
        if names and vals and len(names) == len(vals):
            return dict(zip(names, vals)), meta
    # fallback to legacy TXT by names provided by gm
    with open(fit_txt_path, "r") as fh:
        lines = fh.readlines()
    names = list(getattr(gm, "params_name"))
    vec = [cpgi.read_by_first_lineword_from_text(n, lines) for n in names]
    return dict(zip(names, vec)), None

def plot_actions_with_time_one_particle(
    data_path, snapshot_list, particle_ID_list, 
    savename="./savefig/savename.png", is_show=False, 
    other1=None
):
    #data
    if other1 is None:
        print("(python prog) @param other1 is needed. Exit.")
        exit(0)
    time_list = other1

    n_s = len(snapshot_list)
    n_p = len(particle_ID_list)
    aa_particles = list(range(n_s))
    for i in np.arange(n_s):
        data_path_current = data_path%(snapshot_list[i])
        data = np.loadtxt(data_path_current, dtype=float)
        # x = data
        aa_particles[i] = np.hstack( (
            data[particle_ID_list, col_actions:col_actions+Dim], 
            data[particle_ID_list, col_frequencies:col_frequencies+Dim]
        ) )
    
    aa_particles = np.array(aa_particles)
    print("np.shape(aa_particles): ", np.shape(aa_particles))

    #plot
    pointsize = 0.2
    fontsize = 6.0
    figsize = None
    dpi = 400
    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = None
    # projection = "3d"

    Dim_info = [r"\lambda", r"\mu", r"\nu"]
    for j in np.arange(Dim):
        ax = fig.add_subplot(3, 1, j+1, projection=projection)
        # ax.grid(True)
        for i in np.arange(n_p):
            ax.scatter(time_list, aa_particles[:,i,j], s=pointsize, label="particle_ID %d"%(particle_ID_list[i])) #??
            ax.plot(time_list, aa_particles[:,i,j], lw=pointsize)
        if j==2: #??
            ax.set_xlabel(r"time (Gyr)", fontsize=fontsize)
        ax.set_ylabel(r"$J_{%s}$ (kpc km/s)"%(Dim_info[j]), fontsize=fontsize)
        # ax.set_zlabel(zinfo[-1], fontsize=fontsize)
        # ax.set_xlim(xinfo[0])
        # ax.set_ylim(xinfo[0])
        # # ax.set_zlim(xinfo[0])
        # plt.suptitle(titlename, fontsize=fontsize)
        # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
        ax.legend(fontsize=fontsize, loc=0)
        ax.tick_params(labelsize=fontsize) #size of the number characters
    
    # tikzplotlib.save(savename+".tex")
    fig_tmp = plt.gcf()
    fig_tmp.savefig(savename+".png", format="png", dpi=dpi, bbox_inches='tight')
    if is_show==True:
        plt.show()
    plt.close("all")
    return 0

def plot_triaxialize_display(data_path):
    return 0

def plot_mass_contour(xv_beforepreprocess_path, snapshot_ID, x_input=None, m_input=None, 
    savename="./savefig/savename.png", is_show=False, particle_type_select=None
):
    #data
    x = None
    m = None
    if xv_beforepreprocess_path is not None: #input data path
        pot_data_path = xv_beforepreprocess_path%(snapshot_ID)
        data = np.loadtxt(pot_data_path, dtype=float)
        #optional per-type mask, column 6 typically stores gadget type 0 to 5
        if particle_type_select is not None and data.shape[1] > 7:
            cand = data[:, 6]
            try:
                cand_i = cand.astype(int)
                if np.all((cand_i >= -1) & (cand_i <= 5)) and np.allclose(cand, cand_i, atol=1e-8):
                    mask = (cand_i == int(particle_type_select))
                    data = data[mask]
            except Exception:
                pass
        x = data[:, 0:3]
        m = data[:, 8]
        pers = [50., 90., 95., 97., 98., 99., 99.5]
        meds = ads.percentiles_by_xv_data(data[:,0:6], pers=pers)
        # ads.DEBUG_PRINT_V(0, pers, meds[3], meds[7], "xvmeds")
    else: #input data
        x = x_input
        m = m_input

    N_grid_x = 100
    N_grid_y = 80
    N_grid_z = 9
    bounds = np.zeros((3,2))
    for i in np.arange(3):
        bd = 100. #kpc
        # bd = np.percentile( np.abs(x[:,i]), 98. ) #only halo
        # bd = np.percentile( np.abs(x[:,i]), 96. ) #multi component
        bounds[i] = np.array([-bd, bd])
    grid_x = np.linspace(bounds[0][0], bounds[0][1], N_grid_x)
    grid_y = np.linspace(bounds[1][0], bounds[1][1], N_grid_y)
    grid_z = np.linspace(bounds[2][0], bounds[2][1], N_grid_z)
    # grid_z = grid_z**3/np.max(grid_z)**2
    mg1, mg2, mg3 = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    ads.DEBUG_PRINT_V(1, bounds, np.shape(mg1))
    rho = np.zeros_like(mg1)
    rho_activate = np.zeros_like(mg1)
    # rho_activate = np.random.random((N_grid_x, N_grid_y, N_grid_z))*-10.
    # rho_activate = np.log10 ( 1. / (mg1**2+mg2**2) )
    KD = kdtp.KDTree_galaxy_particles(x, weight_extern_instinct=m)
    for i in np.arange(N_grid_x):
        for j in np.arange(N_grid_y):
            for k in np.arange(N_grid_z):
                targets = [[grid_x[i], grid_y[j], grid_z[k]]]
                rho[i,j,k] = KD.density_SPH(targets)
    rho0 = KD.density_SPH([[0., 0., 0.]])
    # rho_activate = np.log10(rho/rho0)
    rho_activate = np.log10(rho)
    ads.DEBUG_PRINT_V(1, np.shape(rho_activate))

    #plot
    fontsize = 20.
    pointsize = 0.2
    # figsize = 46, 16 #for 4, 2
    figsize = 20, 15 #for 3, 3
    dpi = 400
    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = None
    # projection = "3d"
    levels = np.linspace(-8., 0., 16) #same range
    contour1 = None
    for k in np.arange(N_grid_z):
        ax = fig.add_subplot(3, 3, k+1, projection=projection)
        # import matplotlib.colors
        # normc = matplotlib.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        contour1 = plt.contourf(mg1[:,:,k], mg2[:,:,k], rho_activate[:,:,k], cmap="viridis", 
            levels=levels
            # norm=normc
        )
        cbar = plt.colorbar(contour1)
        cbar.set_label(r"$\log_{10}(\rho)$", fontsize=fontsize)
        ax.set_aspect("equal")
        ax.set_title("z = %f $\mathrm{kpc}$"%(grid_z[k]), fontsize=fontsize)
        if 1: #k>=4 and k<=7:
            ax.set_xlabel(r"$x$ ($\mathrm{kpc}$)", fontsize=fontsize)
        if 1: #k==0 or k==4:
            ax.set_ylabel(r"$y$ ($\mathrm{kpc}$)", fontsize=fontsize)
        # ax.legend(fontsize=fontsize, loc=0)
        ax.tick_params(labelsize=fontsize)

    cax = fig.add_axes([0.95, 0.4, 0.015, 0.3]) #same colorbar
    cbar1 = fig.colorbar(contour1, cax=cax) #, orientation='horizontal'
    cbar1.set_label(r"$\log_{10}(\rho)$ ($\mathrm{1e10\,M_\mathrm{Sun}\, kpc^{-3}}$)", fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    # plt.tight_layout() #to let them not cover each other, auto
    fig_tmp = plt.gcf()
    # # fig_tmp.savefig(savename+".png", format="png", dpi=dpi, bbox_inches='tight')
    # fig_tmp.savefig(savename+".eps", format="eps", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    #append type suffix to avoid overwriting all-particle figure
    save_suf = "" if particle_type_select is None else f".type_{int(particle_type_select)}"
    # fig_tmp.savefig(savename+save_suf+".png", format="png", dpi=dpi, bbox_inches='tight')
    # fig_tmp.savefig(savename+save_suf+".eps", format="eps", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    fig_tmp.savefig(savename+save_suf+".pdf", format="pdf", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    ads.DEBUG_PRINT_V(1, savename+save_suf+".eps", "savename")
    if is_show==True:
        plt.show()
    plt.close("all")
    return 0

import inspect
def CDFA_from_data_and_fitpars(J_grid, pars_kv, meta):
    """
    Evaluate log10 f on a J-grid.
    - Picks the DF function by meta['fit_function'].
    - If the DF expects 'h' as its first arg, build h(J) from meta['combination'].
      Otherwise pass the J-grid (JO) directly (e.g. for *_lmn_* functions).
    - Uses meta['fit_params_values'] (ordered) to build theta.
    """
    # --- sanitize meta
    meta = meta or {}
    if not isinstance(meta, dict):
        meta = {}

    # --- DF function
    fn_name = meta.get("fit_function")
    fh_func = getattr(gm, fn_name, None) if fn_name else None
    if fh_func is None:
        fh_func = gm.fh_MPLTF_Exp_log10  # safe default

    # --- parameter vector (use JSON order, not gm.params_name)
    theta = meta.get("fit_params_values")
    if theta is None:
        # fallback: build from names if values array missing
        names = meta.get("fit_params_names", [])
        theta = [pars_kv[n] for n in names if n in pars_kv]
    # cast and clean
    theta = [float(x) for x in theta if x is not None]

    # --- trim/pad to match DF signature (excluding the first arg)
    sig = inspect.signature(fh_func)
    params = list(sig.parameters.values())
    if not params:
        raise RuntimeError(f"Unexpected signature for {fh_func}")
    n_required = len(params) - 1
    if len(theta) > n_required:
        theta = theta[:n_required]
    elif len(theta) < n_required:
        theta = theta + [0.0] * (n_required - len(theta))

    # --- choose whether to pass h or JO based on first parameter name
    first_name = params[0].name.lower()  # e.g., 'h' vs 'jo'
    if first_name in ("h", "x", "h_grid"):
        # Build h(J) based on combination
        comb = meta.get("combination") or {}
        cname = str(comb.get("name", "")).strip().lower()
        is_free = cname in {
            "aa_combination_freecoef", "aa_combination_free_coefficient",
            "freecoef", "free-coef", "free"
        }
        if is_free:
            args = comb.get("args", [])
            if len(args) >= 2 and args[0] is not None and args[1] is not None:
                km, kn = float(args[0]), float(args[1])
                h = gm.AA_combination_freeCoef(J_grid, km, kn)
            else:
                h = gm.AA_combination_sumWeightFrequency_rateF1(J_grid)
        else:
            # legacy J·Ω surrogate on a J-grid
            h = gm.AA_combination_sumWeightFrequency_rateF1(J_grid)

        f_grid = fh_func(h, *theta)
    else:
        # Function expects JO (e.g., fh_MPLTF_lmn_log10(JO, J0,J1,J2,J3,p1,p2,p3,b,km,kn))
        f_grid = fh_func(J_grid, *theta)

    return f_grid

def read_params_vectors_firstfile(paramfiles_name, params_name, suffix_name, x_name=None, plot_index_of_params=None):
    '''
    To read values of certain @params_name of certain file.
    @paramfiles_name: must be a list, here we only read the file of the first element.
    '''
    #: read parameters value from files
    # N_paramfiles_name = len(paramfiles_name) #models
    N_paramfiles_name = 1 #models
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
    return value_vectors[:,j]

def plot_actions_contour(data_path, fitpars_data_path, snapshot_ID, 
    savename=None, is_compare_fit=False, is_show=False, particle_type_select=None
):
    #data
    bdDP_NDFA_path = data_path
    #: if a particle type is requested, load the per-type density file produced by 
    #\ read_action_data_and_plot_actions_2d(..., particle_type_select=...)
    # bdDP_NDFA_path = data_path
    # if particle_type_select is not None:
    #     root, ext = os.path.splitext(data_path)
    #     if ext == "":
    #         ext = ".txt"
    #     candidate = f"{root}.type_{int(particle_type_select)}{ext}"
    #     if os.path.exists(candidate):
    #         bdDP_NDFA_path = candidate
    #     else:
    #         print(f"Warning: per-type file not found: {candidate}; falling back to {data_path}") #??
    data = np.loadtxt(bdDP_NDFA_path, dtype=float)
    JO_screened = data[:, 0:6] #screened AA6
    f_points = data[:, 6]
    mass = data[:, 7]

    N_grid_lambda = 8
    N_grid_mu = 100
    N_grid_nu = 80
    # N_grid_mu = 200
    # N_grid_nu = 160
    bounds = np.zeros((3,2))
    for i in np.arange(3):
        bd1 = 1e-2 #bounds for large range
        bd2 = 5e4
        # bd1 = 1e1 #bounds for accuarate small range
        # bd2 = 1e3
        bounds[i] = np.array([bd1, bd2])
    grid_lambda = np.geomspace(bounds[0][0], bounds[0][1], N_grid_lambda)
    grid_mu = np.geomspace(bounds[1][0], bounds[1][1], N_grid_mu)
    grid_nu = np.geomspace(bounds[2][0], bounds[2][1], N_grid_nu)
    # mg1, mg2, mg3 = np.meshgrid(grid_lambda, grid_mu, grid_nu)
    mg1, mg2, mg3 = np.meshgrid(grid_mu, grid_nu, grid_lambda, indexing="ij")
    ads.DEBUG_PRINT_V(1, bounds, np.shape(mg1))
    rho = np.zeros_like(mg1)

    # rho_activate = np.log10 ( 1. / (mg1**2+mg2**2) ) #[optional] debug
    rho_activate = np.zeros_like(mg1) #[optional] calculate
    KD = kdtp.KDTree_galaxy_particles(JO_screened[:,0:3], weight_extern_instinct=mass)
    for i in np.arange(N_grid_mu):
        for j in np.arange(N_grid_nu):
            for k in np.arange(N_grid_lambda): #to plot iso-J_lambda space
                targets = [[grid_lambda[k], grid_mu[i], grid_nu[j]]]
                rho[i, j, k] = KD.density_SPH(targets)
    # rho0 = KD.density_SPH([[0., 0., 0.]])
    rho_activate = np.log10(rho/1.)
    ads.DEBUG_PRINT_V(1, np.shape(rho_activate))

    #plot
    if not is_compare_fit:
        fontsize = 20.
        pointsize = 0.2
        figsize = 24, 12 #for 2, 4
        # figsize = 20, 15 #for 3, 3
        dpi = 400
        fig = plt.figure(figsize=figsize, dpi=dpi)
        projection = None
        # projection = "3d"
        levels = np.linspace(-16., -5., 16)
        contour1 = None
        for k in np.arange(N_grid_lambda):
            ax = fig.add_subplot(2, 4, k+1, projection=projection)
            contour1 = plt.contourf(mg1[:,:,k], mg2[:,:,k], rho_activate[:,:,k], cmap="viridis", 
                levels = levels
            )
            # cbar = plt.colorbar(contour1)
            # cbar.set_label(r"$\log_{10}(f)$ ($(\mathrm{kpc\, km/s})^3$)$", fontsize=fontsize) #?? M/1e4
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_aspect("equal")
            ax.set_title(r"$J_\lambda$ = %.2f ($\mathrm{kpc\, km/s}$)"%(grid_lambda[k]), fontsize=fontsize)
            if 1: #k>=4 and k<=7:
                ax.set_xlabel(r"$J_\mu$ ($\mathrm{kpc\, km/s}$)", fontsize=fontsize)
            if 1: #k==0 or k==4:
                ax.set_ylabel(r"$J_\nu$ ($\mathrm{kpc\, km/s}$)", fontsize=fontsize)
            # ax.legend(fontsize=fontsize, loc=0)
            ax.tick_params(labelsize=fontsize)
        
        cax = fig.add_axes([0.95, 0.4, 0.015, 0.3])
        cbar1 = fig.colorbar(contour1, cax=cax) #, orientation='horizontal'
        cbar1.set_label(r"$\log_{10}(f)$ ($\mathrm{kpc\, km/s})^3$)", fontsize=fontsize)
        cbar1.ax.tick_params(labelsize=fontsize)
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
        # plt.tight_layout() #to let them not cover each other, auto
        fig_tmp = plt.gcf()
        # # fig_tmp.savefig(savename+".png", format="png", dpi=dpi, bbox_inches='tight')
        # fig_tmp.savefig(savename+".eps", format="eps", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        save_suf = "" if particle_type_select is None else f".type_{int(particle_type_select)}"
        # fig_tmp.savefig(savename+save_suf+".png", format="png", dpi=dpi, bbox_inches='tight')
        # fig_tmp.savefig(savename+save_suf+".eps", format="eps", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        fig_tmp.savefig(savename+save_suf+".pdf", format="pdf", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        if is_show==True:
            plt.show()
        plt.close("all")
    
    else: #if is_compare_fit
        #data
        # fitpars_data_path_to_read = fitpars_data_path%(snapshot_ID, int(particle_type_select))
        # paramfiles_name = [fitpars_data_path_to_read]
        # params_name = [
        #     "J1_scale_fit", "J2_scale_fit", 
        #     "J3_scale_fit", "J4_scale_fit", 
        #     "powerAA_P1_fit", "powerAA_P2_fit", 
        #     "powerAA_P3_fit", "poly_coeff_k1_fit"
        # ]

        # rbf_func = kdtp.RBF_interp_wrap(JO_screened[:,0:3], JO_screened[:,3:6])[0]
        # # yte = rbf_func(JO_screened[0:2, 0:3])
        # # ads.DEBUG_PRINT_V(0, yte, JO_screened[0:2, 3:6], "yte")
        # # pars = read_params_vectors_firstfile(paramfiles_name, params_name, 
        # #     suffix_name="grid", x_name=None, plot_index_of_params=None) #??
        
        #Build a robust Ω(J) interpolant that tolerates duplicate/collinear samples.
        J_train = JO_screened[:, 0:3]
        O_train = JO_screened[:, 3:6]
        # keep only finite rows
        m = np.all(np.isfinite(J_train), axis=1) & np.all(np.isfinite(O_train), axis=1)
        J_train, O_train = J_train[m], O_train[m]
        # drop exact duplicate J-rows (common source of singular systems)
        if J_train.size and np.unique(J_train, axis=0).shape[0] < J_train.shape[0]:
            idx = np.unique(J_train, axis=0, return_index=True)[1]
            J_train, O_train = J_train[idx], O_train[idx]
        rbf_func = kdtp.RBF_interp_wrap(J_train, O_train, neighbors=32)[0]
        
        # _rbf = RBFInterpolator(J_train, O_train, kernel="thin_plate_spline", smoothing=1e-3)
        # rbf_func = lambda X: _rbf(X)

        fitpars_data_path_to_read = fitpars_data_path % (snapshot_ID, int(particle_type_select))
        pars_kv, meta = _load_fit_meta_and_vector(fitpars_data_path_to_read)
        
        # # f_fit_grid = (np.random.random(mg1.shape))*-30. #[optional] debug
        # f_fit_grid = np.zeros_like(mg1) #[optional] calculate
        # for k in np.arange(N_grid_lambda):
        #     for i in np.arange(N_grid_mu):
        #         for j in np.arange(N_grid_nu):
        #             J_grid1 = np.array([[grid_lambda[k], grid_mu[i], grid_nu[j]]])
        #             O_grid1 = rbf_func(J_grid1)
        #             JO_grid1 = np.hstack((J_grid1, O_grid1))
        #             f_fit_grid[i, k, j] = CDFA_from_data_and_fitpars(JO_grid1, pars_kv, meta) #log10

        #vectorized grid build: J-grid (N,3) -> Ω(J) via RBF -> JO (N,6)
        J_grid = np.stack([mg1.ravel(), mg2.ravel(), mg3.ravel()], axis=1)   # (N,3)
        O_grid = rbf_func(J_grid)                                            # (N,3)
        JO_grid = np.hstack((J_grid, O_grid))                                # (N,6)
        # Evaluate fitted DF once over full JO grid (returns log10 f)
        f_fit_flat = CDFA_from_data_and_fitpars(JO_grid, pars_kv, meta)      # (N,)
        f_fit_grid = f_fit_flat.reshape(mg1.shape)

        #plot
        fontsize = 20.
        pointsize = 0.2
        # figsize = 46, 16 #for 4, 2
        # figsize = 20, 15 #for 3, 3
        # figsize = 30, 15 #for 3, 6
        # figsize = 28, 21 #for 4, 4
        figsize = 24, 24 #for 4, 4
        dpi = 400
        fig = plt.figure(figsize=figsize, dpi=dpi)
        projection = None
        # projection = "3d"
        # levels = np.linspace(-27.5, -5., 10) #range of data bd
        #\ only display normal range, mask the extrem value (because of the value of fit function at some location)
        levels = np.linspace(-16., -5., 16)
        contour1 = None
        for k in np.arange(N_grid_lambda):
            for iplt in np.arange(1, 3):
                ax = fig.add_subplot(4, 4, k*2+iplt, projection=projection)
                if (iplt)%2 != 0:
                    contour1 = plt.contourf(mg1[:,:,k], mg2[:,:,k], rho_activate[:,:,k], cmap="viridis", levels=levels)
                    # cbar1 = plt.colorbar(contour1)
                    # cbar1.set_label(r"$\log_{10}(f)$ ($\mathrm{kpc\, km/s})^3$)", fontsize=fontsize)
                else:
                    contour2 = plt.contourf(mg1[:,:,k], mg2[:,:,k], f_fit_grid[:,:,k], cmap="viridis", levels=levels)
                    # cbar2 = plt.colorbar(contour2)
                    # cbar2.set_label(r"$\log_{10}(f)$ ($\mathrm{kpc\, km/s})^3$)", fontsize=fontsize)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_aspect("equal")
                ax.set_title(r"$J_\lambda$ = %.2f ($\mathrm{kpc\, km/s}$)"%(grid_lambda[k]), fontsize=fontsize)
                if 1: #k>=4 and k<=7:
                    ax.set_xlabel(r"$J_\mu$ ($\mathrm{kpc\, km/s}$)", fontsize=fontsize)
                if 1: #k==0 or k==4:
                    ax.set_ylabel(r"$J_\nu$ ($\mathrm{kpc\, km/s}$)", fontsize=fontsize)
                # ax.legend(fontsize=fontsize, loc=0)
                ax.tick_params(labelsize=fontsize)
        
        cax = fig.add_axes([0.95, 0.4, 0.015, 0.3])
        cbar1 = fig.colorbar(contour1, cax=cax) #, orientation='horizontal'
        cbar1.set_label(r"$\log_{10}(f)$ ($\mathrm{kpc\, km/s})^3$)", fontsize=fontsize)
        cbar1.ax.tick_params(labelsize=fontsize)
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
        # plt.tight_layout() #to let them not cover each other, auto
        fig_tmp = plt.gcf()
        # # fig_tmp.savefig(savename+".png", format="png", dpi=dpi, bbox_inches='tight')
        # fig_tmp.savefig(savename+".eps", format="eps", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        save_suf = "" if particle_type_select is None else f".type_{int(particle_type_select)}"
        # fig_tmp.savefig(savename+save_suf+".png", format="png", dpi=dpi, bbox_inches='tight')
        # fig_tmp.savefig(savename+save_suf+".eps", format="eps", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        fig_tmp.savefig(savename+save_suf+".pdf", format="pdf", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        if is_show==True:
            plt.show()
        plt.close("all")
    return 0

def plot_potential_compare(data_path, snapshot_ID, 
    savename=None, is_show=False
):
    #plot
    Dim_info = [r"x", r"y", r"z"]
    for k in np.arange(Dim):
        pointsize = 4.
        fontsize = 12.
        # figsize = None
        figsize = 15, 10
        dpi = 400
        fig = plt.figure(figsize=figsize, dpi=dpi)
        projection = None
        # projection = "3d"

        pot_data_path = data_path%(snapshot_ID, k)
        data = np.loadtxt(pot_data_path, dtype=float)
        N_total = len(data)
        x = data[:,0:3]
        r = x[:,k]
        # r = ads.norm_l(x, axis=1)
        # ads.DEBUG_PRINT_V(0, r)

        P_SCF_rot = data[:, 6]
        F_SCF_rot = data[:, 7:10]
        P_DS_rot = data[:, 10]
        F_DS_rot = data[:, 11:14]
        P_SCF_inl = data[:, 14]
        F_SCF_inl = data[:, 15:18]
        P_DS_inl = data[:, 18]
        F_DS_inl = data[:, 19:22]
        label = [
            "SCF R", "DS R", "SCF I", "DS I"
        ]
        err_sd = np.abs(P_SCF_inl-P_DS_inl)/np.abs(P_DS_inl)
        err_ri = np.abs(P_SCF_rot-P_SCF_inl)/np.abs(P_SCF_inl)
        # ads.DEBUG_PRINT_V(1, np.mean(err_sd), np.mean(err_ri), "P_err_mean")

        ax = fig.add_subplot(2, 3, 1, projection=projection)
        ax.plot(r, np.abs(P_DS_inl),    label=label[3], color="r")
        ax.plot(r, np.abs(P_SCF_inl),   label=label[2], color="b") #??
        ax.plot(r, np.abs(P_DS_rot),    label=label[1], color="r", marker="+")
        ax.plot(r, np.abs(P_SCF_rot),   label=label[0], color="b", marker="+")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"line along $%s$-axis (kpc)"%(Dim_info[k]), fontsize=fontsize)
        ax.set_ylabel(r"potential abs ($\mathrm{(km/s)^2})$", fontsize=fontsize)
        ax.legend(fontsize=fontsize, loc=0)
        ax.tick_params(labelsize=fontsize) #size of the number characters

        ax = fig.add_subplot(2, 3, 2, projection=projection)
        ax.plot(r, ads.norm_l(F_DS_inl, axis=1),    label=label[3], color="r")
        ax.plot(r, ads.norm_l(F_SCF_inl, axis=1),   label=label[2], color="b")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"line along $%s$-axis (kpc)"%(Dim_info[k]), fontsize=fontsize)
        ax.set_ylabel(r"force magnitude ($\mathrm{(km/s)^2})$", fontsize=fontsize)
        ax.legend(fontsize=fontsize, loc=0)
        ax.tick_params(labelsize=fontsize) #size of the number characters

        for j in np.arange(3):
            ax = fig.add_subplot(2, 3, j+4, projection=projection)
            ax.plot(r, (F_DS_inl[:,j]),     label=label[3], color="r")
            ax.plot(r, (F_SCF_inl[:,j]),    label=label[2], color="b")
            ax.set_xscale("log")
            # ax.set_yscale("log")
            ax.set_xlabel(r"line along $%s$-axis (kpc)"%(Dim_info[k]), fontsize=fontsize)
            ax.set_ylabel(r"force of $%s$-component ($\mathrm{(km/s)^2/kpc})$"%(Dim_info[j]), fontsize=fontsize) #??
            # ax.set_zlabel(zinfo[-1], fontsize=fontsize)
            # ax.set_xlim(xinfo[0])
            # ax.set_ylim(xinfo[0])
            # # ax.set_zlim(xinfo[0])
            # plt.suptitle(titlename, fontsize=fontsize)
            # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
            ax.legend(fontsize=fontsize, loc=0)
            ax.tick_params(labelsize=fontsize) #size of the number characters
        
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
        # plt.tight_layout() #to let them not cover each other, auto
        # tikzplotlib.save(savename+".tex")
        # plt.savefig(savename+".%s"%(Dim_info[k])+".png", format="png", dpi=dpi, bbox_inches='tight')
        fig_tmp = plt.gcf()
        fig_tmp.savefig(savename+".%s"%(Dim_info[k])+".png", format="png", dpi=dpi, bbox_inches='tight')
        if is_show==True:
            plt.show()
        plt.close("all")
    return 0

def plot_orbit_each_2d(x, savename, is_show=False):
    fig = plt.figure()
    pointsize = 0.2
    fontsize = 2.
    xyz_name = ["x (kpc)", "y (kpc)", "z (kpc)"]
    for i in np.arange(3):
        ax = fig.add_subplot(2,2,i+1)
        ax.grid(True)
        ax.scatter(x[:,i%3], x[:,(i+1)%3], s=pointsize, marker="+")
        ax.set_xlabel(xyz_name[i%3])
        ax.set_ylabel(xyz_name[(i+1)%3])
    fig_tmp = plt.gcf()
    fig_tmp.savefig(savename+".png", format="png", dpi=300, bbox_inches='tight')
    if is_show:
        plt.show()
    plt.close("all")

def plot_foci_table(foci_data_path, elliporbit_data_path, snapshot_ID, 
    savename=None, is_show=False
):
    '''
    After running path/to/data_process/recalculate_foci_table.py.
    '''
    #data
    elliporbit_data_path_plot = elliporbit_data_path%(snapshot_ID)
    orbit_snapshot = elliporbit_data_path_plot
    foci_data_path_plot = foci_data_path%(snapshot_ID)
    foci_table_to_use = np.loadtxt(foci_data_path_plot, dtype=float)

    ## (1) plot foci table
    pointsize = 0.2
    fontsize = 6.0
    figsize = None
    dpi = 400
    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = None
    # projection = "3d"

    ax = fig.add_subplot(2, 1, 1, projection=projection)
    plt.plot(foci_table_to_use[:,4], foci_table_to_use[:,1], marker=".", label=r"$-b^2$")
    plt.plot(foci_table_to_use[:,4], foci_table_to_use[:,2], marker=".", label=r"$-a^2$")
    plt.xlabel(r"y cut ($\mathrm{kpc}$)")
    plt.ylabel(r"minus semi-axis square ($\mathrm{kpc}^2$)")
    plt.legend()

    ax = fig.add_subplot(2, 1, 2, projection=projection)
    plt.plot(foci_table_to_use[:,0], foci_table_to_use[:,1], marker=".", label=r"$-b^2$")
    plt.plot(foci_table_to_use[:,0], foci_table_to_use[:,2], marker=".", label=r"$-a^2$")
    plt.xlabel(r"energy ($(\mathrm{km/s})^2$)")
    plt.ylabel(r"minus semi-axis square ($\mathrm{kpc}^2$)")
    plt.legend()

    plt.tight_layout() #to let them not cover each other, auto
    fig_tmp = plt.gcf()
    fig_tmp.savefig(savename+".png", format="png", dpi=dpi, bbox_inches='tight')
    if is_show==True:
        plt.show()
    plt.close("all")

    ## (2) plot some orbits
    idx_grid_plot = np.array([1, 11, 31]) #selected orbit number
    for i in idx_grid_plot:
        print("plot orbit_%d ..."%(i))
        #|| swit 0
        swit = 0 #orbit rotates along x-axis (swit 0) and is at yz plane
        filename = orbit_snapshot+"orbit_%d_b2.dat"%(i)
        savepath = savename+"_orbit_%d_b2"%(i)
        data = np.loadtxt(filename) #cols: time, x, y, z, vx, vy, vz, pot
        x = data[:, 1:4]
        v = data[:, 4:7]
        # plot_orbit_3d(x, savepath, is_show=is_show)
        plot_orbit_each_2d(x, savepath)

        #|| swit 2
        swit = 2 #orbit rotates along z-axis (swit 2) and is at xy plane
        filename = orbit_snapshot+"orbit_%d_a2.dat"%(i)
        savepath = savename+"_orbit_%d_a2"%(i)
        data = np.loadtxt(filename) #cols: time, x, y, z, vx, vy, vz, pot
        x = data[:, 1:4]
        v = data[:, 4:7]
        # plot_orbit_3d(x, savepath, is_show=is_show)
        plot_orbit_each_2d(x, savepath)
    return 0

def read_action_data_and_plot_actions_2d(data_path, snapshot_ID, 
    savename=None, is_show=False, particle_type_select=None
):
    #data
    data_path_plot = data_path%(snapshot_ID)
    data = np.loadtxt(data_path_plot, dtype=float)
    #Optional per-type mask: type column is data[:,-6] (int)
    if particle_type_select is not None:
        try:
            ptype = data[:, -6].astype(int)
            sel = (ptype == int(particle_type_select))
            if np.any(sel):
                data = data[sel]
            else:
                print(f"Info: No rows for particle_type={particle_type_select} in {data_path_plot}.")
        except Exception as e:
            print("Waning: Failed to apply per-type mask:", e)
    N_data = len(data)
    mass = data[:, 7]
    x = data[:, 0:0+Dim] #xv
    v = data[:, Dim:Dim+Dim]
    X = ads.norm_l(x, axis=1)
    V = ads.norm_l(v, axis=1)
    xmed = np.median(X)
    vmed = np.median(V)
    xmean = np.mean(X)
    vmean = np.mean(V)
    P_F = data[:, 10] #potential
    P_D = data[:, 11]

    ## actions
    iast = 28 #actions cols
    adur = 10
    AA_TF_FP = data[:, iast+adur*0:iast+adur*0+Dim]
    AA_OD_FP = data[:, iast+adur*1:iast+adur*1+Dim] #none
    AA_GF_FP = data[:, iast+adur*2:iast+adur*2+Dim] #none
    iast += adur*5 # = 78
    AA_TF_DP = data[:, iast+adur*0:iast+adur*0+Dim]
    AA_OD_DP = data[:, iast+adur*1:iast+adur*1+Dim]
    AA_GF_DP = data[:, iast+adur*2:iast+adur*2+Dim] #none

    AA_TF_DP_all = data[:, iast+adur*0:iast+adur*0+adur]
    Act = AA_TF_DP_all[:, 0:3]
    Ang = AA_TF_DP_all[:, 3+1:7]
    Fre = AA_TF_DP_all[:, 7:10]
    AA = np.hstack((Act, Fre))
    
    bd = 1e6
    # bd = 1e8-1.
    cols = [0,1,2]
    AA_TF_FP, cl_TF_FP, cln = ads.screen_boundary_some_cols(AA_TF_FP, cols, 0., bd, value_discard=bd*1e4)
    AA, cl_TF_DP, cln = ads.screen_boundary_some_cols(AA, cols, 0., bd, value_discard=bd*1e4)
    # AA_TF_DP, cl_TF_DP, cln = ads.screen_boundary_some_cols(AA_TF_DP, cols, 0., bd, value_discard=bd*1e4)
    AA_OD_DP, cl_OD_DP, cln = ads.screen_boundary_some_cols(AA_OD_DP, cols, 0., bd, value_discard=bd*1e4)
    AA_GF_DP, cl_GF_DP, cln = ads.screen_boundary_some_cols(AA_GF_DP, cols, 0., bd, value_discard=bd*1e4)
    # note: the AA_* has been changed
    AA_TF_DP = AA[:,0:3]
    print("bd fraction: %f, %f"%(len(AA_TF_FP)/N_data, len(AA_TF_DP)/N_data))

    AA_TF_FP = ads.merge_array_by_hstack([AA_TF_FP, np.sum(AA_TF_FP, axis=1)])
    AA_OD_FP = ads.merge_array_by_hstack([AA_OD_FP, np.sum(AA_OD_FP, axis=1)])
    AA_GF_FP = ads.merge_array_by_hstack([AA_GF_FP, np.sum(AA_GF_FP, axis=1)])
    AA_TF_DP = ads.merge_array_by_hstack([AA_TF_DP, np.sum(AA_TF_DP, axis=1)])
    AA_OD_DP = ads.merge_array_by_hstack([AA_OD_DP, np.sum(AA_OD_DP, axis=1)])
    AA_GF_DP = ads.merge_array_by_hstack([AA_GF_DP, np.sum(AA_GF_DP, axis=1)])
    
    m = np.ones(N_data) #unit mass
    LL = tg.angularMoment(x, v, m) #each angular moment
    LS = ads.merge_array_by_hstack([x*v, X*V])
    LL = ads.merge_array_by_hstack([LL, ads.norm_l(LL, axis=1)])
    # LL = ads.merge_array_by_hstack([LL, np.sum(LL, axis=1)])
    LS = np.abs(LS)
    LL = np.abs(LL)
    Lmean = np.mean(LL[-1,:])
    Lmed = np.median(LL[-1,:])
    # ads.DEBUG_PRINT_V(0, LL_xyz, LL, LS)
    # ads.DEBUG_PRINT_V(0, xmean, xmed, vmean, vmed, Lmean, Lmed)

    As_TF_FP = np.median(AA_TF_FP, axis=0)
    As_OD_FP = np.median(AA_OD_FP, axis=0)
    As_GF_FP = np.median(AA_GF_FP, axis=0)
    As_TF_DP = np.median(AA_TF_DP, axis=0)
    As_OD_DP = np.median(AA_OD_DP, axis=0)
    As_GF_DP = np.median(AA_GF_DP, axis=0)
    As = [[As_TF_FP, As_OD_FP, As_GF_FP], 
        [As_TF_DP, As_OD_DP, As_GF_DP]]

    ## NDFA of DP
    mass_DP = mass[cl_TF_DP]
    KD = kdtp.KDTree_galaxy_particles(AA_TF_DP[:,cols], weight_extern_instinct=mass_DP)
    NDFADP = KD.density_SPH(AA_TF_DP[:,cols]) #some are None #?? debug
    AA_TF_DP_DF = ads.merge_array_by_hstack([AA, NDFADP, mass_DP])
    bdDP_NDFA_path = data_path_plot+".bdDP_NDFA.txt"
    if particle_type_select is not None:
        root, ext = os.path.splitext(bdDP_NDFA_path)
        if ext == "":
            ext = ".txt"
        bdDP_NDFA_path = f"{root}.type_{int(particle_type_select)}{ext}"
    np.savetxt(bdDP_NDFA_path, AA_TF_DP_DF)

    #plot
    ## (1) plot 2d
    pointsize = 0.2
    fontsize = 8.0
    figsize = None
    dpi = 400
    bd = bd

    xplot_list = [X, V]
    yplot_list = [0, 1, 2, 3]
    # xplot = X #disjunctor
    # # xplot = V
    # i_yplot = 0 #J_lambda or L_x #disjunctor
    # # i_yplot = 1
    # # i_yplot = 2
    # # i_yplot = -1 #J_total or L_norm
    i_xplot_name = [["radius", "kpc"], ["total velocity magnitude", "km/s"]]
    i_yplot_name = [["\lambda", "x"], ["\mu", "y"], [r"\nu", "z"], ["\mathrm{sum}", "\mathrm{total}"]]
    
    for i in range(2):
        xplot = xplot_list[i]
        plt.figure(dpi=300)
        for j in range(4):
            i_yplot = yplot_list[j]
            plt.subplot(2,2,i_yplot+1)
            plt.scatter(xplot, LL[:,i_yplot], s=pointsize, label=
                "angular moment, mean=%e"%(np.mean(LL[:,i_yplot]) ))
            # plt.scatter(xplot[cl_TF_FP], AA_TF_FP[:,i_yplot], s=pointsize, label=
            #     "TSFF (1e-6, 1e6), mean=%e"%(np.mean(AA_TF_FP[:,i_yplot]
            #     [(AA_TF_FP[:,i_yplot]>1./bd)&(AA_TF_FP[:,i_yplot]<bd)]) ))
            plt.scatter(xplot[cl_TF_DP], AA_TF_DP[:,i_yplot], s=pointsize, color="g", label=
                r"TSFD$\in$(1e-6, 1e6), fraction=%.2f, mean=%.2f"%(1.*len(AA_TF_DP)/N_data, np.mean(AA_TF_DP[:,i_yplot]
                [(AA_TF_DP[:,i_yplot]>1./bd)&(AA_TF_DP[:,i_yplot]<bd)]) ))
            # plt.scatter(xplot, AA_OD_DP[:,i_yplot], s=pointsize, label="each "
            #     "action by AA_OD_DP, \t\tmean=%e"%(np.mean(AA_OD_DP[:,i_yplot]
            #     [(AA_OD_DP[:,i_yplot]>1./bd)&(AA_OD_DP[:,i_yplot]<bd)])))
            
            # plt.xlim(19., 20.)
            # plt.ylim(0., 13000.)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(r"%s ($\mathrm{%s}$)"
                %(i_xplot_name[i][0], i_xplot_name[i][1]), 
                fontsize=fontsize)
            plt.ylabel(r"action $J_{%s}$ or angular moment $L_{%s}$, ($\mathrm{kpc\, km/s}$)"
                %(i_yplot_name[i_yplot][0], i_yplot_name[i_yplot][1]), 
                fontsize=fontsize/1.5)
            plt.legend(fontsize=fontsize*0.6, loc=0)
            plt.tick_params(which='major', length=0, labelsize=fontsize*0.8) #size of the number characters
        
        plt.tight_layout() #to let them not cover each other, auto
        # tikzplotlib.save(savename+".tex")
        fig_tmp = plt.gcf()
        # ads.DEBUG_PRINT_V(0, savename)
        # fig_tmp.savefig(savename+".%d.png"%(i), format="png", dpi=dpi, bbox_inches='tight')
        save_suf = "" if particle_type_select is None else f".type_{int(particle_type_select)}"
        fig_tmp.savefig(savename+save_suf+".%d.png"%(i), format="png", dpi=dpi, bbox_inches='tight')
        if is_show==True:
            plt.show()
        plt.close("all")

    ## (2) plot E~L~g~h of AA_TF_DP
    pointsize = 0.2
    fontsize = 8.0
    figsize = None
    dpi = 400

    E = mass[:]*(0.5*V**2+P_D)
    g = gm.AA_combination_sum(AA)
    h = gm.AA_combination_sumWeightFrequency_rateF1(AA)
    plt.figure(dpi=300)
    plt.scatter(E[cl_TF_DP], g, s=pointsize, label=r"$g$")
    plt.scatter(E[cl_TF_DP], h, s=pointsize, label=r"$h$")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"energy ($\mathrm{(km/s)^2}$)", fontsize=fontsize)
    plt.ylabel(r"action combination, ($\mathrm{kpc\, km/s}$)", fontsize=fontsize/1.5)
    plt.legend(fontsize=fontsize*0.6, loc=0)
    plt.tick_params(which='major', length=0, labelsize=fontsize*0.8) #size of the number characters

    plt.tight_layout() #to let them not cover each other, auto
    # tikzplotlib.save(savename+".tex")
    fig_tmp = plt.gcf()
    # ads.DEBUG_PRINT_V(0, savename)
    # fig_tmp.savefig(savename+"_ELgh.png", format="png", dpi=dpi, bbox_inches='tight')
    save_suf = "" if particle_type_select is None else f".type_{int(particle_type_select)}"
    fig_tmp.savefig(savename+save_suf+"_ELgh.png", format="png", dpi=dpi, bbox_inches='tight')
    if is_show==True:
        plt.show()
    plt.close("all")

    return bdDP_NDFA_path

def plot_actions_3d(
    data_path, fitpars_data_path, snapshot_ID, 
    savename=None, is_autosave=True, is_show=True, particle_type_select=None, use_fit_slope_for_omega=True
):
    ## data
    bdDP_NDFA_path = data_path
    data = np.loadtxt(bdDP_NDFA_path, dtype=float)
    cols = [0,1,2]

    # fitpars_data_path_to_read = fitpars_data_path%(snapshot_ID, int(particle_type_select))
    # paramfiles_name = [fitpars_data_path_to_read]
    # params_name = None
    fitpars_data_path_to_read = fitpars_data_path % (snapshot_ID, int(particle_type_select))
    paramfiles_name = [fitpars_data_path_to_read]
    pars_kv, meta = _load_fit_meta_and_vector(fitpars_data_path_to_read)
    # params_name = list(getattr(gm, "params_name"))
    # pars = np.array([pars_kv[n] for n in params_name], dtype=float)

    pointsize = 0.2
    fontsize = 8.
    figsize = None
    dpi = 400
    cm = plt.cm.get_cmap("gist_rainbow") #rainbow
    # projection = None
    projection = "3d"

    ## (1) plot NDFA
    #for log10
    bd = 5e4
    data_plot = data*1.
    data_plot, cl_data_plot, cln = ads.screen_boundary_some_cols(data, cols, 0., bd, value_discard=bd*1e4)
    J = data_plot[:, 0:3]
    log10J = np.log10(J)
    O = data_plot[:, 3:6]
    f = data_plot[:, 6]
    log10f = np.log10(f)

    figsize = 16, 8
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(2, 1, 1, projection=projection)
    axsc = ax.scatter(J[:,0], J[:,1], J[:,2], s=pointsize, label=None, c=-log10f, cmap=cm)
    cbar = plt.colorbar(axsc, shrink=0.6)
    cbar.set_label(r"$\log_{10}(f_N)$", fontsize=fontsize)
    # ax.set_xlim([0., 6e4])
    # ax.set_ylim([0., 3e4])
    # ax.set_zlim([0., 3e4])
    ax.set_xlabel(r"${J_\lambda}$ ($\mathrm{kpc\, km/s}$)",  fontsize=fontsize)
    ax.set_ylabel(r"${J_\mu}$ ($\mathrm{kpc\, km/s}$)",      fontsize=fontsize)
    ax.set_zlabel(r"${J_\nu}$ ($\mathrm{kpc\, km/s}$)",      fontsize=fontsize)
    # ax.legend(fontsize=fontsize, loc=0)
    ax.tick_params(labelsize=fontsize/2.) #size of the number characters

    ## plot NDFA
    #for log10
    bd = 5e4
    data_plot = data*1.
    data_plot, cl_data_plot, cln = ads.screen_boundary_some_cols(data, cols, 0., bd, value_discard=bd*1e4)
    J = data_plot[:, 0:3]
    log10J = np.log10(J)
    O = data_plot[:, 3:6]
    f = data_plot[:, 6]
    log10f = np.log10(f)

    ax = fig.add_subplot(2, 1, 2, projection=projection)
    axsc = ax.scatter(log10J[:,0], log10J[:,1], log10J[:,2], s=pointsize, label=None, c=-log10f, cmap=cm)
    cbar = plt.colorbar(axsc, shrink=0.6)
    cbar.set_label(r"$\log_{10}(f_N)$", fontsize=fontsize)
    # ax.set_xlim([0., 6e4])
    # ax.set_ylim([0., 3e4])
    # ax.set_zlim([0., 3e4])
    ax.set_xlabel(r"$\log_{10}({J_\lambda})$ ($\log_{10}(\mathrm{kpc\, km/s})$)",  fontsize=fontsize)
    ax.set_ylabel(r"$\log_{10}({J_\mu})$ ($\log_{10}(\mathrm{kpc\, km/s})$)",      fontsize=fontsize)
    ax.set_zlabel(r"$\log_{10}({J_\nu})$ ($\log_{10}(\mathrm{kpc\, km/s})$)",      fontsize=fontsize)
    # ax.legend(fontsize=fontsize, loc=0)
    ax.tick_params(labelsize=fontsize/2.) #size of the number characters

    # plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    # tikzplotlib.save(savename+".tex")
    fig_tmp = plt.gcf()
    save_suf = "" if particle_type_select is None else f".type_{int(particle_type_select)}"
    if is_show:
        plt.show()
    # fig_tmp.savefig(savename+".%s.png"%("bounds"), format="png", dpi=dpi, bbox_inches='tight')
    fig_tmp.savefig(savename+save_suf+".%s.png"%("bounds"), format="png", dpi=dpi, bbox_inches='tight')
    plt.close("all")
    
    ## (2) plot slices, iso-DF in actions space
    # bd = 5e4
    bd = 1e6
    data_plot = data*1.
    data_plot, cl_data_plot, cln = ads.screen_boundary_some_cols(data, cols, 0., bd, value_discard=bd*1e4)
    JO_screened = data_plot[:, 0:6]
    J = data_plot[:, 0:3]
    log10J = np.log10(J)
    O = data_plot[:, 3:6]
    JO = data_plot[:, 0:6]
    f = data_plot[:, 6]
    log10f = np.log10(f)
    ads.DEBUG_PRINT_V(1, np.min(log10f), np.median(log10f), np.max(log10f))

    # #default: old behavior
    # omg_with_args = O
    # #if JSON sidecar exists and tells us the combination used
    # a_mu_over_l, a_nu_over_l = None, None
    # if meta is not None and meta.get("combination", {}).get("name") == "AA_combination_freeCoef":
    #     args = meta["combination"].get("args", [])
    #     if len(args) >= 2:
    #         a_mu_over_l, a_nu_over_l = float(args[0]), float(args[1])
    #         Ol = O[:, 0]
    #         # Construct an Ω* whose μ,ν components have the (constant) fitted ratios to Ωλ
    #         omg_with_args = np.vstack([Ol, a_mu_over_l*Ol, a_nu_over_l*Ol]).T
    # JJO = [J, J*O, O, omg_with_args] #note: all kinds of slices
    # if meta is not None and use_fit_slope_for_omega:
    #     omg = JJO[3] #use the guided omega, not default omega
    # else:
    #     omg = JJO[2] #default omega
    # omg = JJO[3]

    # build guided omega (Ω*) if requested
    use_fit_slope_for_omega = True  # or pass this in as a function arg
    omg_with_args = None
    comb = (meta or {}).get("combination", {})
    cname = str(comb.get("name", "")).strip().lower()
    if use_fit_slope_for_omega and cname in {"aa_combination_freecoef","freecoef","free-coef","free"}:
        args = comb.get("args", [])
        if len(args) >= 2 and args[0] is not None and args[1] is not None:
            km, kn = float(args[0]), float(args[1])
            Ol = O[:, 0]
            omg_with_args = np.vstack([Ol, km*Ol, kn*Ol]).T  # Ω* with fitted ratios

    JJO = [J, J*O, O, omg_with_args]  # note last entry might be None
    label_JJO = ["J", "JO"]
    # choose which Ω to use for slice summaries:
    omg = JJO[3] if JJO[3] is not None else JJO[2]  # guided if available, else raw

    disjunctor = 0 #keep slicing in J-space as before
    # disjunctor = 1
    x = JJO[disjunctor]
    xlog = np.log(x)
    x_dy = x
    xlog_dy = xlog
    y = log10f
    yT = np.array([y]).T
    y0, dy0 = -1.e0, 1.e2 #-18. ~ -8. #for all PODD
    # y0, dy0 = 10., 0.1
    # y0, dy0 = 10.2, 0.1 #bad
    # y0, dy0 = 10.5, 0.1 #bad
    # y0, dy0 = 11., 0.1 #bad
    # y0, dy0 = 11.5, 0.1
    # y0, dy0 = 12., 0.03 #bad
    # y0, dy0 = 12., 0.1 #bad
    # y0, dy0 = 13., 0.1
    # y0, dy0 = 13.5, 0.1
    # y0, dy0 = 14., 0.1
    # y0, dy0 = 15., 0.1
    # y_dy, cl, cnl = ads.screen_boundary(yT, y0, y0+dy0)
    # ads.DEBUG_PRINT_V(1, cl, len(cl))

    # ymin = -16.
    ymin = np.floor(np.min(y))
    # ymax = -10.
    ymax = np.ceil(np.max(y))
    dy = 0.1
    # dy = 0.2
    N_slices = int((ymax-ymin)/dy)+1
    # ads.DEBUG_PRINT_V(0, N_slices, ymin, ymax, "N_slices, ymin, ymax")
    DY = np.linspace(ymin, ymax, N_slices)
    DY_useful = []
    N_sample_min = 12
    # N_sample_min = 4
    N_plot_slices = 4
    points_plot = list(range(N_plot_slices))
    # i_points_plot = 0

    params = np.zeros((N_slices-1, 5)) #params
    params_useful = []

    cl_enough_list = []
    cl_plot_list = list(range(N_plot_slices))

    for i in np.arange(N_slices-1):
        y0 = DY[i]
        dy0 = DY[i+1]-DY[i]
        absmin = DY[i]
        absmax = DY[i+1]
        ymiddle = (absmin+absmax)/2.
        # ads.DEBUG_PRINT_V(1, yT, absmin, absmax)
        yT_dy, cl, cnl = ads.screen_boundary_PM(yT, absmin, absmax)
        n_cl = len(cl)

        x_dy = x[cl]
        omg_dy = omg[cl]
        y_dy = y[cl]
        # # if (i)%(N_slices/N_plot_slices)==0 and i_points_plot<N_plot_slices:
        # if (N_slices-1.)/N_plot_slices*(i_points_plot+0)<=i and \
        #     i<(N_slices-1.)/N_plot_slices*(i_points_plot+1) and \
        #     len(cl)>=N_sample_min and i_points_plot<N_plot_slices\
        # :
        #     points_plot[i_points_plot] = cl
        #     ads.DEBUG_PRINT_V(1, i_points_plot, i, len(cl), "i_points_plot, i, length of a points_plot")
        #     i_points_plot += 1
        # ads.DEBUG_PRINT_V(1, points_plot, i_points_plot)
        # ads.DEBUG_PRINT_V(1, (N_slices-1.)/N_plot_slices*(i_points_plot+0), i, len(cl))
        # ads.DEBUG_PRINT_V(1, N_slices, absmin, absmax, len(yT), len(cl))

        if n_cl<N_sample_min: #too less point
            params[i] = np.zeros(5) #bad value
        else:
            cl_enough_list.append([i, cl, absmin, absmax])
            # slope_ref = 10.
            # scale_ref = -np.mean(y_dy/np.sum(x_dy, axis=1))
            # funcfit = gm.surface_plane1
            # p0 = [scale_ref*1., scale_ref*1., scale_ref*1.]
            # boundsD = [scale_ref/slope_ref, scale_ref/slope_ref, scale_ref/slope_ref]
            # boundsU = [scale_ref*slope_ref, scale_ref*slope_ref, scale_ref*slope_ref]
            # funcfit = gm.surface_plane2
            # slope_direct_ref = 100.
            # cut_xaxis = np.max(x_dy[:,0])
            # p0 = [1., 1., cut_xaxis]
            # boundsD = [1./slope_direct_ref, 1./slope_direct_ref, cut_xaxis/10.]
            # boundsU = [slope_direct_ref, slope_direct_ref, cut_xaxis*10.]
            # optimization, covariance = spopt.curve_fit(funcfit, x_dy, y_dy, 
            #     p0 = p0, bounds = (boundsD, boundsU), maxfev = 5000)
            optimization = ads.plane_2d_fit_by_leastsq(x_dy[:,1], x_dy[:,2], x_dy[:,0])
            optimization[0] = -optimization[0]
            optimization[1] = -optimization[1]
            OO_frac_mean = [np.mean(omg_dy[:,1]/omg_dy[:,0]), np.mean(omg_dy[:,2]/omg_dy[:,0])]
            params[i] = np.append( np.array(optimization), np.array(OO_frac_mean) )
            DY_useful.append(ymiddle)
            params_useful.append(params[i])

    DY_useful = np.array(DY_useful)
    DYL = DY_useful
    params_useful = np.array(params_useful)
    # slope_YX = params_useful[:,1]/params_useful[:,0] #slopes of Y-X and Z-X
    # slope_ZX = params_useful[:,2]/params_useful[:,0]
    slope_ml = params_useful[:,0]
    slope_nl = params_useful[:,1]
    cut_l = params_useful[:,2]
    OmOl = params_useful[:,3]
    OnOl = params_useful[:,4]
    # ads.DEBUG_PRINT_V(1, slope_YX, slope_ZX, "slices slopes")

    n_clel = len(cl_enough_list)
    if n_clel<N_plot_slices:
        print("Too less of slices with enough points. Wrong actions. Exit.")
        exit(0)
    for j in np.arange(N_plot_slices):
        idx = int( float(j)/N_plot_slices*n_clel )
        cl_plot_list[j] = cl_enough_list[idx]

    figsize = 16, 16
    pointsize = 10.
    fontsize = 16.
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # projection = None
    projection = "3d"
    for k in range(N_plot_slices):
        cl = cl_plot_list[k][1]
        fdmin = cl_plot_list[k][2]
        fdmax = cl_plot_list[k][3]
        JO_plot = x[cl]
        ax = fig.add_subplot(2, 2, k+1, projection=projection)
        axsc = ax.scatter(JO_plot[:,0], JO_plot[:,1], JO_plot[:,2], s=pointsize, label=None)
        # axsc = ax.scatter(JO_plot[:,0], JO_plot[:,1], JO_plot[:,2], s=pointsize, label=None, c=-log10f[cl], cmap=cm)
        # plt.colorbar(axsc)
        if disjunctor==0:
            ax.set_xlabel(r"${J_\lambda}$ ($\mathrm{kpc\, km/s}$)",  fontsize=fontsize)
            ax.set_ylabel(r"${J_\mu}$ ($\mathrm{kpc\, km/s}$)",      fontsize=fontsize)
            ax.set_zlabel(r"${J_\nu}$ ($\mathrm{kpc\, km/s}$)",      fontsize=fontsize)
        else:
            ax.set_xlabel(r"${J_\lambda \Omega_\lambda}$ ($\mathrm{kpc\, km/s\, Gyr^{-1}}$)",  fontsize=fontsize)
            ax.set_ylabel(r"${J_\mu \Omega_\lambda}$ ($\mathrm{kpc\, km/s\, Gyr^{-1}}$)",      fontsize=fontsize)
            ax.set_zlabel(r"${J_\nu \Omega_\lambda}$ ($\mathrm{kpc\, km/s\, Gyr^{-1}}$)",      fontsize=fontsize)
        ax.set_title(r"$\log_{10}(f)\, \in$ [%.2f, %.2f]"%(fdmin, fdmax),      fontsize=fontsize)
        # ax.legend(fontsize=fontsize, loc=0)
        ax.tick_params(labelsize=fontsize/2.) #size of the number characters

    # plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    # tikzplotlib.save(savename+".tex")
    fig_tmp = plt.gcf()
    save_suf = "" if particle_type_select is None else f".type_{int(particle_type_select)}"
    if is_show:
        plt.show()
    # fig_tmp.savefig(savename+".%s_%s.png"%("slices", label_JJO[disjunctor]), format="png", dpi=dpi, bbox_inches='tight')
    fig_tmp.savefig(savename+save_suf+".%s_%s.png"%("slices", label_JJO[disjunctor]), format="png", dpi=dpi, bbox_inches='tight')
    plt.close("all")

    ## (3) plot slopes, combination function of actions with slopes e.g. frequencies or free fixed coefficients
    figsize = None
    fontsize = 10.
    pointsize = 4.
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # plt.scatter(DYL, slope_ml, s=pointsize, color="red",  label=r"minus slope of $J_\mu$ to $J_\lambda$")
    # plt.scatter(DYL, slope_nl, s=pointsize, color="blue", label=r"minus slope of $J_\nu$ to $J_\lambda$")
    # plt.scatter(DYL, OmOl, s=pointsize, color="orange",   label=r"mean freq frac of $J_\mu$ to $J_\lambda$")
    # plt.scatter(DYL, OnOl, s=pointsize, color="green",    label=r"mean fraq frac of $J_\nu$ to $J_\lambda$")

    label_mu_ratio = r"mean $\Omega_\mu/\Omega_\lambda$"
    label_nu_ratio = r"mean $\Omega_\nu/\Omega_\lambda$"
    if use_fit_slope_for_omega and omg_with_args is not None:
        label_mu_ratio = r"fitted ratio (fixed) $k_\mu=\Omega_\mu/\Omega_\lambda$"
        label_nu_ratio = r"fitted ratio (fixed) $k_\nu=\Omega_\nu/\Omega_\lambda$"
    plt.scatter(DYL, slope_ml, s=pointsize, color="red",  label=r"slope of $J_\mu$ vs $J_\lambda$")
    plt.scatter(DYL, slope_nl, s=pointsize, color="blue", label=r"slope of $J_\nu$ vs $J_\lambda$")
    plt.scatter(DYL, OmOl,     s=pointsize, color="orange", label=label_mu_ratio)
    plt.scatter(DYL, OnOl,     s=pointsize, color="green",  label=label_nu_ratio)

    plt.plot([ymin, ymax], [0., 0.], color="k", lw=pointsize/6.)
    plt.plot([ymin, ymax], [1., 1.], color="k", lw=pointsize/6.)
    plt.xlabel(r"$\log_{10}(f_N)$ ($\log_{10}(\mathrm{(1/(kpc\, km/s)^3)})$)", fontsize=fontsize)
    if disjunctor==0:
        plt.ylabel(r"slopes of actions", fontsize=fontsize)
    else:
        plt.ylabel(r"slopes of actions with frequencies", fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    # tikzplotlib.save(savename+".tex")
    fig_tmp = plt.gcf()
    # fig_tmp.savefig(savename+".%s_%s.png"%("slopes", label_JJO[disjunctor]), format="png", dpi=dpi, bbox_inches='tight')
    save_suf = "" if particle_type_select is None else f".type_{int(particle_type_select)}"
    fig_tmp.savefig(savename+save_suf+".%s_%s.png"%("slopes", label_JJO[disjunctor]), format="png", dpi=dpi, bbox_inches='tight')
    if is_show:
        plt.show()
    plt.close("all")
    return 0

def plot_NDFA_bin_and_its_fit(data_path, particle_type_select):
    '''
    Run path/to/data_process/fit_galaxy_distribution_function.py, with tag 1 and 2.
    '''
    print("\n\n\n\nNot here. Please run path/to/data_process/fit_galaxy_distribution_function.py, with tag 1 and 2.")
    return 0

def plot_params_relation_xv_OJ(data_path):
    '''
    Run path/to/data_process/fit_galaxy_distribution_function.py, with tag 3.
    '''
    return 0





#### path
galaxy_name = sys.argv[1]
# galaxy_name = "galaxy_general"
# galaxy_name = "galaxy_general_NFW_spinH_axisLH1"
# galaxy_name = "galaxy_general_Ein_spinH_axisLH0"
# galaxy_name = "galaxy_general_Ein_spinL_axisLH1"
# galaxy_name = "galaxy_general_Ein_spinH_axisLH0_rotvelpot"
# galaxy_name = "galaxy_general_Ein_spinH_axisLH1_rotvelpot"
# galaxy_name = "galaxy_general_Ein_spinH_axisLH2_spininter1_rotvelpot"
# galaxy_name = "galaxy_general_DPLNFW_axisratioz_unmodify0"
snapshot_ID = int(sys.argv[2])
# snapshot_ID = 10 #fixed
snapshot_list = [8, 9, 10, 11, 12]
time_list = np.array(snapshot_list)*0.1 + 0.0
# is_show = True
is_show = False
if galaxy_name=="":
    print("(python prog) Vacant galaxy_name input. Exit.")
    exit(0)
triaxialize_data_path = cpgi.galaxy_general_location_path+galaxy_name+"/aa/snapshot_%d_triaxialize.txt"
potential_compare_path = cpgi.galaxy_general_location_path+galaxy_name+"/intermediate/potential_compare_%d_%d.txt"
elliporbit_data_path = cpgi.galaxy_general_location_path+galaxy_name+"/intermediate/orbit_%d/"
foci_data_path = cpgi.galaxy_general_location_path+galaxy_name+"/intermediate/snapshot_%d_lmn_foci_Pot.txt"
xv_beforepreprocess_path = cpgi.galaxy_general_location_path+galaxy_name+"/txt/snapshot_%03d.txt"
aa_data_path = cpgi.galaxy_general_location_path+galaxy_name+"/aa/snapshot_%d.action.method_all.txt"
save_total_path = cpgi.galaxy_general_location_path+"/params_statistics/"
save_single_path = cpgi.galaxy_general_location_path+galaxy_name+"/fit/"
#### run
# $ python3 plot_action_figs.py galaxy_general 10



#### main
if __name__ == '__main__':

    ## choose which particle types to fit
    try:
        type_list = list(mask_select_type)
    except NameError:
        type_list = [1] #default: halo only
    if not type_list:
        type_list = [1]

    ## not loop
    # if 0: #debug, only run loop
    if 1: #run all
        ## (0) debug
        plot_info = "debug"
        savename = save_single_path+plot_info
        # plot_mass_contour()
        print("Plot (%d) %s. Done."%(0, plot_info))

        ## (1) lmn actions in the 5 snapshot to 
        #\ aprroximate error
        plot_info = "actions_with_time_one_particle"
        particle_ID_list = [17, 18, 180, 1800]
        savename = save_single_path+plot_info
        # plot_actions_with_time_one_particle(
        #     aa_data_path, snapshot_list, particle_ID_list=particle_ID_list, 
        #     savename=savename, is_show=is_show, other1=time_list
        # ) #* #plot when after enabling action error
        print("Plot (%d) %s. Done."%(1, plot_info))

        ## (2) triaxialize display on xv+-, v DF, 
        #\ main axis direction and v direction, *
        plot_info = "triaxialize_display"
        # savename = save_single_path+plot_info
        print("Plot (%d) %s. Done."%(2, plot_info))

        ## (3) tau-ptau variance at some foci to display, *
        plot_info = "tau_ptau_variance"
        # savename = save_single_path+plot_info
        print("Plot (%d) %s. Done."%(3, plot_info))

        ## (4) potential compare of FF_pot, DS_pot and SCF_pot
        plot_info = "potential_compare"
        savename = save_single_path+plot_info
        plot_potential_compare(
            potential_compare_path, snapshot_ID, 
            savename=savename, is_show=is_show
        ) #plot
        print("Plot (%d) %s. Done."%(4, plot_info))

        ## (5) foci-ycut table and elliptical closed orbit fit
        plot_info = "foci_table"
        savename = save_single_path+plot_info
        plot_foci_table(
            foci_data_path, elliporbit_data_path, snapshot_ID, savename
        ) #plot
        print("Plot (%d) %s. Done."%(5, plot_info))

    ## loop about each type or component
    for gadget_type in type_list:
        ## (6) density profile contour
        plot_info = "mass_contour"
        savename = save_single_path+plot_info
        plot_mass_contour(
            xv_beforepreprocess_path, snapshot_ID, savename=savename, 
            particle_type_select=gadget_type
        ) #plot
        print("Plot (%d) %s. Done."%(6, plot_info))
        # exit(0)

        ## (7) actions all in 2d by various method
        plot_info = "actions_2d"
        savename = save_single_path+plot_info
        bdDP_NDFA_path = read_action_data_and_plot_actions_2d(
            aa_data_path, snapshot_ID, 
            savename, is_show=is_show, particle_type_select=gadget_type
        ) #plot
        #override BD file path (explicit per-type)
        bdDP_NDFA_path = cpgi.galaxy_general_location_path+galaxy_name\
            +f"/aa/snapshot_{snapshot_ID}.action.method_all.txt.bdDP_NDFA.type_{int(gadget_type)}.txt"
        print("Plot (%d) %s. Done."%(7, plot_info))

        ## (8) actions all in 3d, h-surface and Omega-cut
        #\ data from bdDP_NDFA_path
        plot_info = "actions_3d"
        savename = save_single_path+plot_info
        fitpars_data_path = save_single_path+"snapshot_%d.type_%d.fit.txt"
        plot_actions_3d(
            bdDP_NDFA_path, fitpars_data_path, snapshot_ID, 
            savename, is_autosave=True, particle_type_select=gadget_type,
            # is_show=True
            is_show=False
        ) #plot
        print("Plot (%d) %s. Done."%(8, plot_info))

        ## (9) NDFA and its fit contour
        plot_info = "action_contour"
        savename = save_single_path+plot_info
        # plot_actions_contour(
        #     bdDP_NDFA_path, fitpars_data_path, snapshot_ID, savename, 
        #     particle_type_select=gadget_type
        # ) #plot
        plot_info = "action_contour_compare"
        savename = save_single_path+plot_info
        plot_actions_contour(
            bdDP_NDFA_path, fitpars_data_path, snapshot_ID, savename, is_compare_fit=True, 
            particle_type_select=gadget_type
        ) #plot
        print("Plot (%d) %s. Done."%(9, plot_info))

        ## (10) aa DF fit of 2*2 models
        #\ data from bdDP_NDFA_path
        plot_info = "NDFA_bin_and_its_fit"
        savename = save_single_path+plot_info
        plot_NDFA_bin_and_its_fit(
            save_total_path, particle_type_select=gadget_type
        ) #plot
        print("Plot (%d) %s. Done."%(10, plot_info))

        ## (11) params relation
        plot_info = "params_relation_xv_OJ"
        savename = save_total_path+plot_info
        # plot_params_relation_xv_OJ() #*
        print("Plot (%d) %s. Done."%(11, plot_info))
