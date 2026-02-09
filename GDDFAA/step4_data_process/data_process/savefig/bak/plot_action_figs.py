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
from sklearn.neighbors import KDTree
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
import actions_error_with_time as aent
import triaxialize_galaxy as tg
import change_params_galaxy_init as cpgi



# [] column index of actions data file
Dim = 3
mask_select_type = [1, 2, 0, 3] #[1]
col_x = 0
col_v = 3
col_particle_IDs=6
col_particle_mass=7
col_actions = 78 #triaxial Staeckel Fudge (TSF) method
col_frequencies = col_actions+7
col_particle_type=-6
col_potential = -4

# [] path
galaxy_name = sys.argv[1]
# galaxy_name = "galaxy_general"
# galaxy_name = "galaxy_general_NFW_spinH_axisLH1"
# galaxy_name = "galaxy_general_Ein_spinH_axisLH0"
# galaxy_name = "galaxy_general_Ein_spinL_axisLH1"
# galaxy_name = "galaxy_general_Ein_spinH_axisLH0_rotvelpot"
# galaxy_name = "galaxy_general_Ein_spinH_axisLH1_rotvelpot"
# galaxy_name = "galaxy_general_Ein_spinH_axisLH2_spininter1_rotvelpot"
# galaxy_name = "galaxy_general_DPLNFW_axisratioz_unmodify0"
# snapshot_ID = 10
snapshot_ID = int(sys.argv[2])
# snapshot_list = [snapshot_ID-1, snapshot_ID]
snapshot_list = [snapshot_ID-2, snapshot_ID-1, snapshot_ID, snapshot_ID+1, snapshot_ID+2]
TimeBetSnapshot = 0.1
time_list = np.array(snapshot_list).astype(float)*TimeBetSnapshot + 0.0
# is_show = True
is_show = False

galaxy_general_location_path = "../../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"
triaxialize_data_path = galaxy_general_location_path+galaxy_name+"/aa/snapshot_%d_triaxialize.txt"
potential_compare_path = galaxy_general_location_path+galaxy_name+"/intermediate/potential_compare_%d_%d.txt"
elliporbit_data_path = galaxy_general_location_path+galaxy_name+"/intermediate/orbit_%d/"
foci_data_path = galaxy_general_location_path+galaxy_name+"/intermediate/snapshot_%d_lmn_foci_Pot.txt"
xv_beforepreprocess_path = galaxy_general_location_path+galaxy_name+"/txt/snapshot_%03d.txt"
aa_data_path = galaxy_general_location_path+galaxy_name+"/aa/snapshot_%d.action.method_all.txt"
aa_data_path_variation = galaxy_general_location_path+galaxy_name+"/aa/snapshot_%d.action.method_all.variation.txt"
aa_data_path_bdDP = galaxy_general_location_path+galaxy_name+"/aa/snapshot_%d.action.method_all.bdDP_NDFA.type_%d.txt"
save_total_path = galaxy_general_location_path+"/params_statistics/"
save_single_path = galaxy_general_location_path+galaxy_name+"/fit/"

# [] running example
# $ python3 plot_action_figs.py galaxy_general 10



# [] plot funcs
def _load_potential_grid_from_file(potential_contour_file, snapshot_ID):
    """
    Load a precomputed potential grid written by pot_grid.cpp.

    Parameters
    ----------
    potential_contour_file : str
        Either a path template with a single %d placeholder for snapshot_ID
        (e.g. "potential_contour_data_%d.txt") or a concrete file path.
    snapshot_ID : int
        Snapshot index used to resolve the template, if present.

    Returns
    -------
    grid_x, grid_y, grid_z : 1D ndarray
        Unique coordinate axes along x, y, z.
    mg1, mg2, mg3 : 3D ndarray
        Meshgrids built from (grid_x, grid_y, grid_z) with indexing="ij".
    phi_grid : 3D ndarray
        Potential values Φ(x, y, z) reshaped on the same grid.
    """
    if potential_contour_file is None:
        raise ValueError("potential_contour_file must not be None in grid loader.")

    # Resolve template like "..._%d.txt" if present.
    if ("%d" in potential_contour_file) or ("%i" in potential_contour_file):
        path = potential_contour_file % int(snapshot_ID)
    else:
        path = potential_contour_file

    data_grid = np.loadtxt(path, dtype=float)
    if data_grid.ndim == 1:
        data_grid = data_grid[None, :]

    if data_grid.shape[1] < 4:
        raise ValueError(
            f"Potential grid file '{path}' must have at least 4 columns: x y z Phi."
        )

    xg = data_grid[:, 0]
    yg = data_grid[:, 1]
    zg = data_grid[:, 2]
    phig = data_grid[:, 3]
    # print("count of wrong potential: ", len(phig[phig>0]))

    grid_x = np.unique(xg)
    grid_y = np.unique(yg)
    grid_z = np.unique(zg)

    Nx = grid_x.size
    Ny = grid_y.size
    Nz = grid_z.size

    if Nx * Ny * Nz != phig.size:
        raise RuntimeError(
            f"Grid dimensions inferred from '{path}' do not match number of rows."
        )

    phi_grid = phig.reshape((Nx, Ny, Nz))
    mg1, mg2, mg3 = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")

    return grid_x, grid_y, grid_z, mg1, mg2, mg3, phi_grid

def plot_potential_contour_KDE(
    xv_potential_path, snapshot_ID, x_input=None, pot_input=None, 
    potential_contour_file=None, 
    particle_type_select=1, 
    savename="./savefig/potential_contour", is_show=False, 
    n_grid_x=100, n_grid_y=100, n_grid_z=9,
    n_neighbors=32, bd_kpc=100.0,
):
    """
    Potential contour on (x, y) slices using KNN-smoothed particle potentials.
    The plotted quantity is log10(-Phi).

    Parameters
    ----------
    xv_potential_path : str or None
        Format string with '%d' for snapshot_ID.
        If None, x_input / pot_input are used directly.
    snapshot_ID : int
        Snapshot index to plug into xv_potential_path.
    x_input : (N,3) array or None
        Direct positions if xv_potential_path is None.
    pot_input : (N,) array or None
        Direct potential values if xv_potential_path is None.
    particle_type_select : int
        Gadget particle type to use (1 = halo, etc.).
    savename : str
        Output file prefix (no extension).
    is_show : bool
        Whether to call plt.show().
    n_grid_x, n_grid_y, n_grid_z : int
        Grid sizes along x, y, z.
    n_neighbors : int
        Number of nearest neighbours in the KNN smoothing.
    bd_kpc : float
        Half-size of the cube to sample ([-bd, bd] in each axis).
    """
    # --- load data ---------------------------------------------------------
    if xv_potential_path is not None:
        pot_data_path = xv_potential_path % (snapshot_ID)
        data = np.loadtxt(pot_data_path, dtype=float)

        # Optional per-type mask (Gadget type is in col_particle_type == -6)
        if particle_type_select is not None:
            ptype = data[:, col_particle_type].astype(int)
            sel_type = (ptype == int(particle_type_select))
            if np.any(sel_type):
                data = data[sel_type]
            else:
                print(
                    f"No rows with particle_type={particle_type_select}; using all types."
                )
        else:
            raise ValueError("No particle_type_select provided. Exit.")

        x = data[:, col_x:col_x+Dim]               # positions
        phi = data[:, col_potential].astype(float) # potential Φ(x)
    else:
        # Direct input
        if x_input is None or pot_input is None:
            raise ValueError(
                "Either xv_potential_path or (x_input, pot_input) must be provided."
            )
        x = np.asarray(x_input, float)
        phi = np.asarray(pot_input, float)

    # --- sanitize & switch to log10(-Phi) ----------------------------------
    # Only accept finite, *negative* potentials so that -Phi > 0
    mask = np.isfinite(phi) & np.all(np.isfinite(x), axis=1) & (phi < 0.0)
    x = x[mask]
    phi = phi[mask]

    if len(x) == 0:
        raise RuntimeError(
            "No valid (finite, negative) potential values found; "
            "cannot plot log10(-Phi)."
        )

    # Work with log10(-Phi) as the field we smooth
    phi_log = np.log10(-phi)  # dimensionless

    # --- build 3D grid in configuration space ------------------------------
    if potential_contour_file is not None:
        # Use precomputed SCF potential grid (x, y, z, Phi) from file.
        grid_x, grid_y, grid_z, mg1, mg2, mg3, phi_grid = _load_potential_grid_from_file(
            potential_contour_file, snapshot_ID
        )

        N_grid_x = grid_x.size
        N_grid_y = grid_y.size
        N_grid_z = grid_z.size

        # Convert to log10(-Phi) on the grid, masking non-finite or non-negative values.
        phi_log_smoothed = np.full_like(phi_grid, np.nan, dtype=float)
        mask_grid = np.isfinite(phi_grid) & (phi_grid < 0.0)
        if not np.any(mask_grid):
            raise RuntimeError(
                f"No negative potential values found in potential grid file "
                f"for snapshot_ID={snapshot_ID}; cannot plot log10(-Phi)."
            )
        phi_log_smoothed[mask_grid] = np.log10(-phi_grid[mask_grid])
    else:
        # Original KNN-smoothed particle-based field.
        N_grid_x = n_grid_x
        N_grid_y = n_grid_y
        N_grid_z = n_grid_z

        bounds = np.zeros((3, 2))
        for i in range(3):
            bounds[i] = np.array([-bd_kpc, bd_kpc])

        grid_x = np.linspace(bounds[0, 0], bounds[0, 1], N_grid_x)
        grid_y = np.linspace(bounds[1, 0], bounds[1, 1], N_grid_y)
        grid_z = np.linspace(bounds[2, 0], bounds[2, 1], N_grid_z)

        mg1, mg2, mg3 = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        pts = np.vstack([mg1.ravel(), mg2.ravel(), mg3.ravel()]).T  # (N_grid, 3)

        # --- KNN smoothing of log10(-Phi) --------------------------------------
        k_use = min(n_neighbors, len(x))
        tree = KDTree(x)  # Euclidean metric by default
        dist, ind = tree.query(pts, k=k_use)  # shapes: (N_grid, k_use)

        # Avoid division by zero: add tiny floor to distances
        d2 = dist**2 + 1e-12
        w = 1.0 / d2
        w /= np.sum(w, axis=1, keepdims=True)  # normalized weights

        phi_log_neighbors = phi_log[ind]                   # (N_grid, k_use)
        phi_log_smoothed_flat = np.sum(w * phi_log_neighbors, axis=1)
        phi_log_smoothed = phi_log_smoothed_flat.reshape(mg1.shape)
        # print(phi_log_smoothed[:,:,0]); exit(0)

    # finite_log = phi_log_smoothed[np.isfinite(phi_log_smoothed)]
    # vmin, vmax = np.percentile(finite_log, [5.0, 95.0]) #clip display range robustly
    # levels = np.linspace(vmin, vmax, 16)
    levels = None

    # --- plot slices --------------------------------------------------------
    fontsize = 20.0
    figsize = (20, 15)
    dpi = 400
    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = None

    contour_last = None
    for k in range(N_grid_z):
        ax = fig.add_subplot(3, 3, k + 1, projection=projection)
        contour_last = ax.contourf(
            mg1[:, :, k], mg2[:, :, k], phi_log_smoothed[:, :, k],
            levels=levels, cmap="viridis"
        )
        ax.set_aspect("equal")
        ax.set_title(r"$z = %.2f\ \mathrm{kpc}$" % (grid_z[k]), fontsize=fontsize)
        ax.set_xlabel(r"$x\ (\mathrm{kpc})$", fontsize=fontsize)
        ax.set_ylabel(r"$y\ (\mathrm{kpc})$", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

    # Shared colorbar
    cax = fig.add_axes([0.95, 0.4, 0.015, 0.3])
    cbar = fig.colorbar(contour_last, cax=cax)
    cbar.set_label(
        r"$\log_{10}(-\Phi)\ \left[(\mathrm{km/s})^2\right]$",
        fontsize=fontsize
    )
    cbar.ax.tick_params(labelsize=fontsize)

    fig.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9,
        wspace=0.4, hspace=0.4
    )

    fig_tmp = plt.gcf()
    savename_fig = savename + "_%d" % (snapshot_ID) + ".pdf"
    fig_tmp.savefig(
        savename_fig,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=dpi,
    )

    if is_show:
        plt.show()
    plt.close("all")
    print("Plot %s, done." % (savename_fig))
    return 0

def plot_potential_contour_SPH(
    xv_potential_path, snapshot_ID, x_input=None, pot_input=None,
    particle_type_select=1,
    savename="./savefig/potential_contour", is_show=False,
    n_grid_x=100, n_grid_y=100, n_grid_z=9,
    n_neighbors=32, bd_kpc=100.0
):
    """
    Plot log10(-Phi) on (x, y) slices, using the same SPH-style KDE.
    """
    # --- 1. load positions and potential -----------------------------------
    if xv_potential_path is not None:
        pot_data_path = xv_potential_path % (snapshot_ID)
        data = np.loadtxt(pot_data_path, dtype=float)

        # select Gadget type (same logic as in mass contour)
        if particle_type_select is not None:
            ptype = data[:, col_particle_type].astype(int)
            sel = (ptype == int(particle_type_select))
            if np.any(sel):
                data = data[sel]
            else:
                print(
                    f"No rows with particle_type={particle_type_select}; using all types."
                )

        x = data[:, col_x:col_x + Dim]
        phi = data[:, col_potential].astype(float)
    else:
        if x_input is None or pot_input is None:
            raise ValueError(
                "Either xv_potential_path or (x_input, pot_input) must be provided."
            )
        x = np.asarray(x_input, float)
        phi = np.asarray(pot_input, float)

    # --- 2. basic cleaning --------------------------------------------------
    # we *only* keep finite and negative Φ, since we need log10(-Φ)
    mask = np.all(np.isfinite(x), axis=1) & np.isfinite(phi) & (phi < 0.0)
    x = x[mask]
    phi = phi[mask]
    if x.size == 0:
        raise RuntimeError("No finite, negative potentials after cleaning.")

    # --- 3. build regular grid -----------
    N_grid_x = int(n_grid_x)
    N_grid_y = int(n_grid_y)
    N_grid_z = int(n_grid_z)

    bounds = np.zeros((3, 2))
    for i in range(3):
        bounds[i] = np.array([-bd_kpc, bd_kpc])

    grid_x = np.linspace(bounds[0, 0], bounds[0, 1], N_grid_x)
    grid_y = np.linspace(bounds[1, 0], bounds[1, 1], N_grid_y)
    grid_z = np.linspace(bounds[2, 0], bounds[2, 1], N_grid_z)

    mg1, mg2, mg3 = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")

    # --- 4. SPH "KDE" for -Phi, using KDTree_galaxy_particles ---------------
    # We treat (-Phi) as a positive "weight" field, analogous to mass.
    # The SPH kernel is the same as used for rho in plot_mass_contour().
    # print(phi); exit(0)

    KD_phi = kdtp.KDTree_galaxy_particles(x, weight_extern_instinct=phi)
    KD_1   = kdtp.KDTree_galaxy_particles(x, weight_extern_instinct=np.ones_like(phi))
    phi_mag_grid = np.zeros_like(mg1)

    for i in range(N_grid_x):
        for j in range(N_grid_y):
            for k in range(N_grid_z):
                target = [[grid_x[i], grid_y[j], grid_z[k]]]
                num  = KD_phi.density_SPH(target, k=n_neighbors)  # ∑ φ_j W_ij
                den  = KD_1.density_SPH(target, k=n_neighbors)    # ∑ 1·W_ij = “kernel mass”
                phi_mag_grid[i, j, k] = -num/den
    
    # avoid non-positive / non-finite values before log10
    phi_mag_grid[~np.isfinite(phi_mag_grid)] = np.nan
    phi_mag_grid[phi_mag_grid <= 0.0] = np.nan

    finite_mask = np.isfinite(phi_mag_grid)
    phi_log = np.full_like(phi_mag_grid, np.nan, dtype=float)
    phi_log[finite_mask] = np.log10(phi_mag_grid[finite_mask])
    # print(phi_log[:,:,0])

    # vmin, vmax = np.percentile(phi_log[finite_mask], [5.0, 95.0])
    # if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
    #     vmin = np.nanmin(phi_log[finite_mask])
    #     vmax = np.nanmax(phi_log[finite_mask])
    # levels = np.linspace(vmin, vmax, 16) # choose contour range robustly
    levels = None

    # --- 5. plot (same style as mass contour) ------------------------------
    fontsize = 20.0
    figsize = (20, 15)
    dpi = 400

    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = None

    contour_last = None
    for k in range(N_grid_z):
        ax = fig.add_subplot(3, 3, k + 1, projection=projection)
        contour_last = ax.contourf(
            mg1[:, :, k],
            mg2[:, :, k],
            phi_log[:, :, k],
            levels=levels,
            cmap="viridis",
        )
        ax.set_aspect("equal")
        ax.set_title(r"$z = %.2f\ \mathrm{kpc}$" % (grid_z[k]), fontsize=fontsize)
        ax.set_xlabel(r"$x\ (\mathrm{kpc})$", fontsize=fontsize)
        ax.set_ylabel(r"$y\ (\mathrm{kpc})$", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

    # shared colorbar on the right
    cax = fig.add_axes([0.95, 0.4, 0.015, 0.3])
    cbar = fig.colorbar(contour_last, cax=cax)
    cbar.set_label(
        r"$\log_{10}\!\left(-\Phi\right)\ \left[(\mathrm{km/s})^2\right]$",
        fontsize=fontsize,
    )
    cbar.ax.tick_params(labelsize=fontsize)

    fig.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.4,
        hspace=0.4,
    )

    fig_tmp = plt.gcf()
    savename_fig = f"{savename}_{snapshot_ID:d}.pdf"
    fig_tmp.savefig(
        savename_fig,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=dpi,
    )

    if is_show:
        plt.show()
    plt.close("all")
    print(f"Plot {savename_fig}, done.")
    return 0

def plot_potential_contour_notlog_diff(
    xv_potential_path, snapshot_ID, snapshot_ID_prev=None, 
    x_input=None, pot_input=None, x_input_prev=None, pot_input_prev=None, 
    potential_contour_file=None, 
    particle_type_select=1, is_relative=True, 
    savename="./savefig/potential_contour_diff", is_show=False, 
    n_grid_x=100, n_grid_y=100, n_grid_z=9,
    n_neighbors=32, bd_kpc=100.0,
):
    """
    Plot the potential difference ΔΦ = Φ(snapshot_ID) − Φ(snapshot_ID_prev)
    on (x, y) slices, using KNN-smoothed particle potentials.

    Parameters
    ----------
    xv_potential_path : str or None
        Format string with '%d' for snapshot_ID.
        If None, x_input/x_input_prev and pot_input/pot_input_prev must be given.
    snapshot_ID : int
        "New" snapshot index.
    snapshot_ID_prev : int or None
        "Old" snapshot index. If None, uses snapshot_ID - 1.
    x_input, pot_input : arrays or None
        Positions (N,3) and potentials (N,) for the "new" snapshot,
        used only if xv_potential_path is None.
    x_input_prev, pot_input_prev : arrays or None
        Positions (N,3) and potentials (N,) for the "old" snapshot,
        used only if xv_potential_path is None.
    particle_type_select : int
        Gadget particle type to use (1 = halo, etc.) when reading from file.
    savename : str
        Output filename prefix (without extension).
    is_show : bool
        Whether to call plt.show().
    n_grid_x, n_grid_y, n_grid_z : int
        Grid sizes in x, y, z.
    n_neighbors : int
        Number of nearest neighbours for the KNN smoothing.
    bd_kpc : float
        Half-size of the cube; grid spans [-bd_kpc, +bd_kpc] in each axis.
    """
    # --------------------------
    # helper: load one snapshot
    # --------------------------
    def _load_x_phi_from_file(snap_id):
        pot_data_path = xv_potential_path % (snap_id)
        data = np.loadtxt(pot_data_path, dtype=float)

        if particle_type_select is not None:
            ptype = data[:, col_particle_type].astype(int)
            sel = (ptype == int(particle_type_select))
            if np.any(sel):
                data = data[sel]
            else:
                print(
                    f"snapshot {snap_id}: no rows with particle_type="
                    f"{particle_type_select}; using all types."
                )

        x = data[:, col_x:col_x + Dim]
        phi = data[:, col_potential].astype(float)

        mask = np.isfinite(phi) & np.all(np.isfinite(x), axis=1)
        x = x[mask]
        phi = phi[mask]
        return x, phi

    # --------------------------
    # 1. load the two snapshots
    # --------------------------
    if snapshot_ID_prev is None:
        snapshot_ID_prev = int(snapshot_ID) - 1

    snap_new = int(snapshot_ID)
    snap_old = int(snapshot_ID_prev)

    if xv_potential_path is not None:
        # from files
        x_new, phi_new = _load_x_phi_from_file(snap_new)
        x_old, phi_old = _load_x_phi_from_file(snap_old)
    else:
        # from direct input
        if (x_input is None) or (pot_input is None) \
           or (x_input_prev is None) or (pot_input_prev is None):
            raise ValueError(
                "If xv_potential_path is None, you must provide "
                "(x_input, pot_input) and (x_input_prev, pot_input_prev)."
            )
        x_new = np.asarray(x_input, float)
        phi_new = np.asarray(pot_input, float)
        x_old = np.asarray(x_input_prev, float)
        phi_old = np.asarray(pot_input_prev, float)

        mask_new = np.isfinite(phi_new) & np.all(np.isfinite(x_new), axis=1)
        mask_old = np.isfinite(phi_old) & np.all(np.isfinite(x_old), axis=1)
        x_new, phi_new = x_new[mask_new], phi_new[mask_new]
        x_old, phi_old = x_old[mask_old], phi_old[mask_old]

    if (len(x_new) == 0) or (len(x_old) == 0):
        raise RuntimeError(
            "One of the snapshots has zero "
            "valid particles after cleaning."
        )

    # --------------------------------
    # 2. build regular configuration grid
    # --------------------------------
    if potential_contour_file is not None:
        # Use precomputed SCF potential grids for both snapshots.
        grid_x_new, grid_y_new, grid_z_new, mg1_new, mg2_new, mg3_new, phi_new_grid = _load_potential_grid_from_file(
            potential_contour_file, snap_new
        )
        grid_x_old, grid_y_old, grid_z_old, mg1_old, mg2_old, mg3_old, phi_old_grid = _load_potential_grid_from_file(
            potential_contour_file, snap_old
        )

        # Sanity check: both grids must share the same coordinates.
        if not (
            np.array_equal(grid_x_new, grid_x_old)
            and np.array_equal(grid_y_new, grid_y_old)
            and np.array_equal(grid_z_new, grid_z_old)
        ):
            raise RuntimeError(
                "Potential grid files for snapshots %d and %d do not share the same (x, y, z) grid."
                % (snap_new, snap_old)
            )

        grid_x = grid_x_new
        grid_y = grid_y_new
        grid_z = grid_z_new
        mg1, mg2, mg3 = mg1_new, mg2_new, mg3_new

        N_grid_x = grid_x.size
        N_grid_y = grid_y.size
        N_grid_z = grid_z.size

        phi_new_grid_flat = phi_new_grid.reshape(-1)
        phi_old_grid_flat = phi_old_grid.reshape(-1)
    else:
        N_grid_x = int(n_grid_x)
        N_grid_y = int(n_grid_y)
        N_grid_z = int(n_grid_z)

        bounds = np.zeros((3, 2))
        for i in range(3):
            bounds[i] = np.array([-bd_kpc, bd_kpc])

        grid_x = np.linspace(bounds[0, 0], bounds[0, 1], N_grid_x)
        grid_y = np.linspace(bounds[1, 0], bounds[1, 1], N_grid_y)
        grid_z = np.linspace(bounds[2, 0], bounds[2, 1], N_grid_z)

        mg1, mg2, mg3 = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        pts = np.vstack([mg1.ravel(), mg2.ravel(), mg3.ravel()]).T  # (N_grid, 3)

        # --------------------------------
        # 3. KNN smoothing for each snapshot
        # --------------------------------
        def _smooth_phi_at_grid(x_part, phi_part):
            k_use = min(n_neighbors, len(x_part))
            if k_use < 1:
                raise RuntimeError("Not enough particles for KNN smoothing.")
            tree = KDTree(x_part)
            dist, ind = tree.query(pts, k=k_use)  # (N_grid, k_use)

            d2 = dist**2 + 1e-12   # avoid division by zero
            w = 1.0 / d2
            w /= np.sum(w, axis=1, keepdims=True)

            phi_neighbors = phi_part[ind]         # (N_grid, k_use)
            phi_smoothed_flat = np.sum(w * phi_neighbors, axis=1)
            return phi_smoothed_flat

        phi_new_grid_flat = _smooth_phi_at_grid(x_new, phi_new)
        phi_old_grid_flat = _smooth_phi_at_grid(x_old, phi_old)

    delta_phi_flat = phi_new_grid_flat - phi_old_grid_flat #absolute
    if is_relative:
        delta_phi_flat /= phi_old_grid_flat #relative
    delta_phi = delta_phi_flat.reshape(mg1.shape)
    delta_phi = delta_phi_flat.reshape(mg1.shape)

    # --------------------------------
    # 4. choose plotting range (symmetric)
    # --------------------------------
    finite_delta = delta_phi[np.isfinite(delta_phi)]
    if finite_delta.size == 0:
        raise RuntimeError("All delta Phi values are non-finite.")

    max_abs = np.percentile(np.abs(finite_delta), 95.0)
    if (not np.isfinite(max_abs)) or (max_abs <= 0.0):
        max_abs = np.nanmax(np.abs(finite_delta))
    if max_abs <= 0.0:
        max_abs = 1.0  # fallback

    vmin, vmax = -max_abs, max_abs
    # levels = np.linspace(vmin, vmax, 16)
    levels = None

    # --------------------------------
    # 5. plot
    # --------------------------------
    fontsize = 20.0
    figsize = (20, 15)
    dpi = 400

    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = None

    contour_last = None
    for k in range(N_grid_z):
        ax = fig.add_subplot(3, 3, k + 1, projection=projection)
        contour_last = ax.contourf(
            mg1[:, :, k],
            mg2[:, :, k],
            delta_phi[:, :, k],
            levels=levels,
            cmap="coolwarm",
        )
        ax.set_aspect("equal")
        ax.set_title(
            r"$z = %.2f\ \mathrm{kpc}$" % (grid_z[k]),
            fontsize=fontsize,
        )
        ax.set_xlabel(r"$x\ (\mathrm{kpc})$", fontsize=fontsize)
        ax.set_ylabel(r"$y\ (\mathrm{kpc})$", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

    # shared colorbar on the right
    cax = fig.add_axes([0.95, 0.4, 0.015, 0.3])
    cbar = fig.colorbar(contour_last, cax=cax)
    cbar.set_label(
        r"$\Delta\Phi\ \left[(\mathrm{km/s})^2\right]$",
        fontsize=fontsize,
    )
    cbar.ax.tick_params(labelsize=fontsize)

    fig.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.4,
        hspace=0.4,
    )

    # optional global title showing which snapshots are used
    title = None
    if is_relative:
        title = r"$\eta_\Phi = \Phi(%d) / \Phi(%d) - 1$" % (snap_new, snap_old)
    else:
        title = r"$\Delta\Phi = \Phi(%d) - \Phi(%d)$" % (snap_new, snap_old)
    fig.suptitle(title, fontsize=fontsize, y=0.98)

    fig_tmp = plt.gcf()
    savename_fig = None
    if is_relative:
        savename_fig = f"{savename}_diff_{snap_new:d}_{snap_old:d}.pdf"
    else:
        savename_fig = f"{savename}_diff_{snap_new:d}_{snap_old:d}_absolute.pdf"
    fig_tmp.savefig(
        savename_fig,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=dpi,
    )

    if is_show:
        plt.show()
    plt.close("all")
    print(f"Plot {savename_fig}, done.")
    return 0

def plot_actions_variation_with_time(
    data_path_pattern, snapshot_list, time_list=None, 
    particle_type_select=1, n_particles=10, is_plot_log_variation=False, 
    is_is_mask_relative_variation=True, 
    savename="./savefig/actions_variation_with_time", is_show=False,
):
    """
    Plot TSF actions as a function of time for a small set of halo particles.

    Parameters
    ----------
    data_path_pattern : str
        Format string with '%d' where snapshot_ID is inserted.
    snapshot_list : sequence of int
        Snapshot indices to read in time order.
    time_list : sequence of float or None
        Physical time (e.g. Gyr) for each snapshot. If None, uses snapshot
        indices; if a global `TimeBetSnapshot` exists, uses snapshot_list * TimeBetSnapshot.
    particle_type_select : int or None
        Gadget particle type to use (1 = halo). If None, all types are used.
    n_particles : int
        Target number of tracer particles (will be reduced if not enough).
    savename : str
        Output file prefix (no extension).
    is_show : bool
        Whether to call plt.show().

    Returns
    -------
    snapshot_arr : (N_snap,) int
    t_arr        : (N_snap,) float  (time axis used for the plot)
    particle_ID_list : (N_tracer,) int
    radii_sel       : (N_tracer,) float, radius in middle snapshot (kpc)
    aa_particles    : (N_snap, N_tracer, 3) actions J_lambda, J_mu, J_nu
    """
    snapshot_arr = np.asarray(snapshot_list, dtype=int)
    n_snap = len(snapshot_arr)
    if n_snap < 2:
        raise ValueError("Need at least two snapshots to plot actions vs time.")

    # Parameters for TSF [J, Omega] screening
    bd_down = 1e-2
    # bd_up = 5e4
    bd_up = 1e6
    cols_judge = np.arange(6)  # 3 actions + 3 frequencies

    # --- 1. Select representative halo particles from a middle snapshot ----
    middle_snapshot = int(snapshot_arr[n_snap // 2])  # robust to any n_snap >= 2
    data0 = np.loadtxt(data_path_pattern % (middle_snapshot), dtype=float)

    # Per-type selection
    if particle_type_select is not None:
        ptype0 = data0[:, col_particle_type].astype(int)
        mask_type = (ptype0 == int(particle_type_select))
        if np.any(mask_type):
            data0 = data0[mask_type]
        else:
            print(
                f"No rows with particle_type={particle_type_select} "
                f"in snapshot {middle_snapshot}; using all types."
            )
    else:
        raise ValueError("No particle_type_select provided. Exit.")

    # --- TSF [J, Omega] screening on the middle snapshot -----------------------
    # Act = J_lambda, J_mu, J_nu ; Fre = Omega_lambda, Omega_mu, Omega_nu
    Act0 = data0[:, col_actions:col_actions + 3]
    Fre0 = data0[:, col_frequencies:col_frequencies + 3]
    AA0 = np.hstack((Act0, Fre0))  # shape (N0, 6)

    AA0_screened, idx_good0, idx_bad0 = ads.screen_boundary_some_cols(
        AA0, cols_judge, bd_down, bd_up, value_discard=None
    )
    if AA0_screened.shape[0] == 0:
        raise RuntimeError(
            "No particles in the middle snapshot pass the TSF [J,Omega] screening."
        )

    # positions, radii, IDs restricted to TSF-good rows
    x0 = data0[idx_good0, col_x:col_x + Dim]
    r0 = ads.norm_l(x0, axis=1)
    ids0 = data0[idx_good0, col_particle_IDs]
    r_median = np.median(r0)
    print("r_median of the middle snapshot: ", r_median)
    # energy (kinetic + potential) on the same TSF-good rows
    v0 = data0[idx_good0, col_v:col_v + Dim]
    phi0 = data0[idx_good0, col_potential].astype(float)
    v0n = ads.norm_l(v0, axis=1)
    E0 = 0.5 * v0n**2 + phi0  # (km/s)^2

    # clean basic NaNs
    good0 = np.isfinite(E0) & np.isfinite(r0) & np.isfinite(ids0)
    x0 = x0[good0]
    r0 = r0[good0]
    ids0 = ids0[good0]
    E0 = E0[good0]

    if len(ids0) == 0:
        raise RuntimeError("No finite energy, radii, IDs in middle snapshot after screening.")
    
    # # sort by radius or energy (kinetic + potential)
    # order = np.argsort(r0)
    order = np.argsort(E0)
    E_sorted = E0[order]
    r_sorted = r0[order]
    ids_sorted = ids0[order].astype(int)

    n_available = len(ids_sorted)
    if n_available < n_particles:
        print(
            f"Only {n_available} particles "
            f"available after TSF screening; reducing n_particles from {n_particles}."
        )
        n_particles = n_available

    # Choose tracer particles distributed across the energy-ordered list.
    #\ Energy can be negative, so we use linear targets in E rather than geomspace.
    E_median = np.median(E_sorted)
    # E_min, E_max = E_sorted[0], E_sorted[-1]
    # E_min, E_max = -1e5, -1e3 #fixed
    # E_min, E_max = E_median*0.1, E_median*2. #small range
    # E_min, E_max = E_median*2e-2, E_median*1e1 #default
    E_min, E_max = E_median*1e-2, E_median*1e2 #default

    if E_max == E_min:
        target_E = None
    else:
        target_E = np.geomspace(E_min, E_max, n_particles)

    if target_E is None:
        idx_sel = np.linspace(0, n_available - 1, n_particles).astype(int)
    else:
        idx_sel = np.zeros(n_particles, dtype=int)
        used = np.zeros(n_available, dtype=bool)
        for i, Et in enumerate(target_E):
            j0 = int(np.argmin(np.abs(E_sorted - Et)))
            if not used[j0]:
                j_sel = j0
            else:
                j_sel = j0
                delta = 1
                while True:
                    candidates = []
                    j_left = j0 - delta
                    j_right = j0 + delta
                    if j_left >= 0:
                        candidates.append(j_left)
                    if j_right < n_available:
                        candidates.append(j_right)
                    if not candidates:
                        break
                    found = False
                    for jj in candidates:
                        if not used[jj]:
                            j_sel = jj
                            found = True
                            break
                    if found:
                        break
                    delta += 1
                    if delta > n_available:
                        break
            used[j_sel] = True
            idx_sel[i] = j_sel
        idx_sel = np.sort(idx_sel)

    particle_ID_list = ids_sorted[idx_sel]
    radii_sel = r_sorted[idx_sel]
    energies_sel = E_sorted[idx_sel]

    print("Selected tracer IDs sorted by energy (E = 0.5|v|^2 + Phi) in middle snapshot:")
    for pid, Ee, rr in zip(particle_ID_list, energies_sel, radii_sel):
        print(f"ID = {pid:10d} (E_midsnap = {Ee: .6e} (km/s)^2, r_midsnap = {rr:10.3f} kpc)")

    # --- 2. Read actions for those IDs in each snapshot --------------------
    # aa_particles[t_snap, i_particle, j_action], j_action = 0..2 => Jλ, Jμ, Jν
    aa_particles = np.full((n_snap, n_particles, Dim), np.nan, dtype=float)

    for isnap, snap in enumerate(snapshot_arr):
        path = data_path_pattern % (snap)
        data = np.loadtxt(path, dtype=float)

        ids_here = data[:, col_particle_IDs].astype(int)
        # Map from ID -> row index
        id_to_index = {int(i): idx for idx, i in enumerate(ids_here)}

        for ip, pid in enumerate(particle_ID_list):
            idx = id_to_index.get(int(pid), None)
            if idx is None:
                # particle may have been lost / out of this snapshot
                continue

            # actions and frequencies for this particle at this snapshot
            J_vec = data[idx, col_actions:col_actions + Dim]
            Fre_vec = data[idx, col_frequencies:col_frequencies + Dim]
            AA_local = np.hstack((J_vec, Fre_vec))

            # TSF [J, Omega] mask on this particular row
            # equivalent to screen_boundary_some_cols() with one row
            if (
                np.all(np.isfinite(AA_local))
                and np.all(AA_local > bd_down)
                and np.all(AA_local < bd_up)
            ):
                aa_particles[isnap, ip, :] = J_vec
            else:
                # leave as NaN -> line broken at this snapshot in the plot
                continue

    # the relative variation with time
    mid_idx = n_snap // 2
    aa_frac_particles = np.full_like(aa_particles, np.nan)
    denom = aa_particles[mid_idx, :, :]   # shape (N_tracer, 3)
    # avoid division by zero / NaN
    valid = np.isfinite(denom) & (np.abs(denom) > 0.0)
    for isnap in range(n_snap):
        aa_frac_particles[isnap, valid] = (
            aa_particles[isnap, valid] / denom[valid]
        )
    # aa_particles = aa_frac_particles

    # --- 3. Time axis ------------------------------------------------------
    if time_list is not None:
        t_arr = np.asarray(time_list, dtype=float)
        if len(t_arr) != n_snap:
            raise ValueError("time_list must have the same length as snapshot_list.")
        x_label = r"time (Gyr)"
    else:
        # Try to use a global TimeBetSnapshot if available
        try:
            dt = float(TimeBetSnapshot)  # noqa: F821  (may not exist)
            t_arr = snapshot_arr.astype(float) * dt
            x_label = r"time (Gyr)"
        except NameError:
            t_arr = snapshot_arr.astype(float)
            x_label = r"snapshot index"

    # --- 4. Plot -----------------------------------------------------------
    pointsize = 2.0
    fontsize = 12.0
    dpi = 400
    figsize = (16, 9)  # 3 panels stacked

    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = None
    Dim_info = [r"\lambda", r"\mu", r"\nu"]

    for j in range(Dim):
        # ax = fig.add_subplot(3, 1, j + 1, projection=projection)
        axL = fig.add_subplot(3, 2, 2 * j + 1, projection=projection)  # absolute
        axR = fig.add_subplot(3, 2, 2 * j + 2, projection=projection)  # fraction

        # One line per particle
        for i in range(n_particles):
            label = (
                f"ID {particle_ID_list[i]} (E_midsnap={energies_sel[i]:.2e} (km/s)^2)"
                # f"ID {particle_ID_list[i]} (E_midsnap={energies_sel[i]:.1f} (km/s)^2, r_midsnap={radii_sel[i]:.1f} kpc)"
                if j == 0 else None
            )

            # --- left: absolute actions
            y_abs = aa_particles[:, i, j]
            finite_abs = np.isfinite(y_abs)
            if finite_abs.any():
                axL.plot(
                    t_arr[finite_abs], y_abs[finite_abs],
                    marker="o", markersize=pointsize,
                    linewidth=0.7, label=label,
                )

            # --- right: fraction to midsnap
            y_frac = aa_frac_particles[:, i, j]
            finite_frac = np.isfinite(y_frac)
            if finite_frac.any():
                # reference lines on fraction panel
                axR.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6,)
                axR.axhline(0.1, color="k", linestyle="--", linewidth=0.8, alpha=0.6,)
                axR.axhline(10., color="k", linestyle="--", linewidth=0.8, alpha=0.6,)
                axR.plot(
                    t_arr[finite_frac], y_frac[finite_frac],
                    marker="o", markersize=pointsize,
                    linewidth=0.7, label=None,
                )

        # left panel styling: NO logscale
        axL.tick_params(labelsize=fontsize)
        axL.set_ylabel(
            r"$J_{%s}\,(\mathrm{kpc\, km/s})$" % Dim_info[j],
            fontsize=fontsize
        )

        # right panel styling: keep existing behaviour
        axR.tick_params(labelsize=fontsize)
        if is_plot_log_variation:
            axR.set_yscale("log")
        axR.set_ylabel(
            r"$J_{%s}/J_{%s, \mathrm{midsnap}}$" % (Dim_info[j], Dim_info[j]),
            fontsize=fontsize
        )

        if j == Dim - 1:
            axL.set_xlabel(x_label, fontsize=fontsize)
            axR.set_xlabel(x_label, fontsize=fontsize)

    # only first panel gets legend
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.axes[0].legend(
            handles, labels, loc="lower right", fontsize=fontsize/2, frameon=False
        )

    savename_fig = savename + ".variation_actions" + ".pdf"
    fig.tight_layout()
    fig.savefig(
        savename_fig, 
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=dpi,
    )
    if is_show:
        plt.show()
    plt.close(fig)
    print(f"Plot {savename}.pdf, done.")

    # return data so orbits etc. can reuse the same tracer set
    return snapshot_arr, t_arr, particle_ID_list, radii_sel, aa_particles

def plot_3d_orbit_particles(
    data_path_pattern, snapshot_list,
    particle_ID_list, is_plot_log_variation=False, 
    savename="./savefig/orbits_some_particles",
    is_show=False,
):
    """
    Plot 3D configuration-space orbits (x,y,z) of a small set of particles,
    using positions from several snapshots, and also plot variation of
    radius, |v| and potential versus time.

    Parameters
    ----------
    data_path_pattern : str
        Format string with '%d' where snapshot_ID is inserted.
    snapshot_list : sequence of int
        Snapshot indices to read in time order (e.g. 5 snapshots).
    particle_ID_list : sequence of int
        IDs of tracer particles.
    savename : str
        Output file prefix (no extension).
    is_show : bool
        Whether to call plt.show().

    Returns
    -------
    info_particles_select : array, shape (N_snap, N_tracer, Dim*2+3)
        Per snapshot and tracer:
            0:Dim      = x         (kpc)
            Dim:2*Dim  = v         (km/s)
            2*Dim      = |x|       (kpc)
            2*Dim + 1  = |v|       (km/s)
            2*Dim + 2  = potential ((km/s)^2)
    """
    snapshot_arr = np.asarray(snapshot_list, dtype=int)
    n_snap = len(snapshot_arr)
    pids = np.asarray(particle_ID_list, dtype=int)
    n_tracer = len(pids)

    if n_snap < 2:
        raise ValueError("Need at least two snapshots to plot orbits.")
    if n_tracer == 0:
        raise ValueError("particle_ID_list is empty.")

    # info_particles_select[t_snap, i_tracer, :]
    # 0:Dim      -> x
    # Dim:2*Dim  -> v
    # 2*Dim      -> |x|
    # 2*Dim + 1  -> |v|
    # 2*Dim + 2  -> potential
    info_particles_select = np.full(
        (n_snap, n_tracer, Dim * 2 + 3), np.nan, dtype=float
    )

    # --- read positions/velocities for each snapshot -----------------------
    for isnap, snap in enumerate(snapshot_arr):
        path = data_path_pattern % (snap)
        data = np.loadtxt(path, dtype=float)

        ids_here = data[:, col_particle_IDs].astype(int)
        id_to_index = {int(i): idx for idx, i in enumerate(ids_here)}

        for ip, pid in enumerate(pids):
            idx = id_to_index.get(int(pid), None)
            if idx is None:
                # this tracer may be absent in this snapshot
                continue
            # x, v
            info_particles_select[isnap, ip, 0:Dim] = data[idx, col_x:col_x + Dim]
            info_particles_select[isnap, ip, Dim:2*Dim] = data[
                idx, col_x + Dim : col_x + Dim * 2
            ]
            # |x|, |v|
            info_particles_select[isnap, ip, 2*Dim] = ads.norm_l(
                data[idx, col_x:col_x + Dim]
            )
            info_particles_select[isnap, ip, 2*Dim + 1] = ads.norm_l(
                data[idx, col_x + Dim : col_x + Dim * 2]
            )
            # potential (snapshot convention: last-4 column)
            info_particles_select[isnap, ip, 2*Dim + 2] = data[idx, -4]

    # small diagnostic print, as you already had
    for (ip, ID) in enumerate(particle_ID_list):
        print("particle ID = %d:" % (ID), info_particles_select[:, ip, :])

    # ----------------------------------------------------------------------
    # A) Plot variation of r, |v|, Phi versus time (style ~ actions plot)
    # ----------------------------------------------------------------------
    # time axis
    if "TimeBetSnapshot" in globals():
        try:
            dt = float(TimeBetSnapshot)
            t_arr = snapshot_arr.astype(float) * dt
            x_label = r"time (Gyr)"
        except Exception:
            t_arr = snapshot_arr.astype(float)
            x_label = r"snapshot index"
    else:
        t_arr = snapshot_arr.astype(float)
        x_label = r"snapshot index"

    pointsize = 2.0
    fontsize = 12.0
    dpi_var = 400
    figsize_var = (16, 12)  # 4 panels (E, r, |v|, Phi)

    fig_var = plt.figure(figsize=figsize_var, dpi=dpi_var)
    projection = None

    # indices for r, |v|, Phi inside info_particles_select
    idx_r = 2 * Dim
    idx_vn = 2 * Dim + 1
    idx_phi = 2 * Dim + 2
    y_labels_left = [
        r"$E = ((\mathrm{km/s})^2)$",
        r"$r\ (\mathrm{kpc})$",
        r"$|v|^2/2\ (\mathrm{km/s})$",
        r"$-\Phi\ ((\mathrm{km/s})^2)$",
    ]
    y_labels_right = [
        r"$E/E_{\mathrm{midsnap}}$",
        r"$r/r_{\mathrm{midsnap}}$",
        r"$|v|^2/|v|^2_{\mathrm{midsnap}}$",
        r"$\Phi/\Phi_{\mathrm{midsnap}}$",
    ]

    # radius at middle snapshot for labelling
    mid_idx = n_snap // 2

    for j in range(4):
        # ax = fig_var.add_subplot(4, 1, j + 1, projection=projection)
        axL = fig_var.add_subplot(4, 2, 2 * j + 1, projection=projection)  # absolute
        axR = fig_var.add_subplot(4, 2, 2 * j + 2, projection=projection)  # fraction

        for ip in range(n_tracer):
            # --- build absolute y (used for both left and right)
            if j == 0:
                # E = 0.5 |v|^2 + Phi
                vn = info_particles_select[:, ip, idx_vn]
                phi = info_particles_select[:, ip, idx_phi]
                y_abs = 0.5 * vn**2 + phi
            elif j == 1:
                y_abs = info_particles_select[:, ip, idx_r]
            elif j == 2:
                y_abs = 0.5*(info_particles_select[:, ip, idx_vn])**2
            else:
                # plot -Phi (positive) (matches existing convention)
                y_abs = -info_particles_select[:, ip, idx_phi]
            finite_abs = np.isfinite(y_abs)

            # approximate radius at middle snapshot
            r_mid = info_particles_select[mid_idx, ip, idx_r]
            label = (
                # f"ID {pids[ip]} (E_midsnap={y[mid_idx]:.1f} (km/s)^2)"
                f"ID {pids[ip]} (r_midsnap={r_mid:.1f} kpc)"
                # f"ID {pids[ip]} (E_midsnap={y[mid_idx]:.1f} (km/s)^2, r_midsnap={r_mid:.1f} kpc)"
                if (j == 0)
                else None
            )

            # left: absolute (NO logscale)
            if finite_abs.any():
                axL.plot(
                    t_arr[finite_abs], y_abs[finite_abs],
                    marker="o", markersize=pointsize,
                    linewidth=0.7, label=label,
                )

            # right: fraction to midsnap (guard zero/NaN denom)
            denom = y_abs[mid_idx]
            if np.isfinite(denom) and (denom != 0.0):
                y_frac = y_abs / denom
            else:
                y_frac = np.full_like(y_abs, np.nan)
            finite_frac = np.isfinite(y_frac)
            if finite_frac.any():
                axR.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6,)
                axR.axhline(0.1, color="k", linestyle="--", linewidth=0.8, alpha=0.6,)
                axR.axhline(10., color="k", linestyle="--", linewidth=0.8, alpha=0.6,)
                axR.plot(
                    t_arr[finite_frac], y_frac[finite_frac],
                    marker="o", markersize=pointsize,
                    linewidth=0.7, label=None,
                )

        # style / labels
        axL.tick_params(labelsize=fontsize)
        axL.set_ylabel(y_labels_left[j], fontsize=fontsize)

        axR.tick_params(labelsize=fontsize)
        if is_plot_log_variation:
            axR.set_yscale("log")
        axR.set_ylabel(y_labels_right[j], fontsize=fontsize)

        if j == 3:
            axL.set_xlabel(x_label, fontsize=fontsize)
            axR.set_xlabel(x_label, fontsize=fontsize)

    # legend only on first panel
    handles, labels = fig_var.axes[0].get_legend_handles_labels()
    if handles:
        fig_var.axes[0].legend(
            handles, labels, loc="lower right", fontsize=fontsize/2, frameon=False,
        )

    fig_var.tight_layout()
    savename_fig = savename + ".variation_pot" + ".pdf"
    fig_var.savefig(
        savename_fig,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=dpi_var,
    )
    if is_show:
        plt.show()
    plt.close(fig_var)
    print(f"Plot {savename_fig}, done.")

    # ----------------------------------------------------------------------
    # B) 3D orbit plot (your original behaviour, unchanged in style)
    # ----------------------------------------------------------------------
    dpi = 400
    figsize = (7, 7)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # global radial range for axis limits (using x only)
    all_xyz = info_particles_select[:, :, 0:Dim].reshape(-1, Dim)
    mask_finite = np.all(np.isfinite(all_xyz), axis=1)
    if np.any(mask_finite):
        xyz_finite = all_xyz[mask_finite]
        r_max = np.max(ads.norm_l(xyz_finite, axis=1))
    else:
        r_max = 1.0

    # draw each orbit as a line with markers at the sampled snapshots
    for ip in range(n_tracer):
        xyz = info_particles_select[:, ip, 0:Dim]
        mline = np.all(np.isfinite(xyz), axis=1)
        if not np.any(mline):
            continue
        x_line = xyz[mline, 0]
        y_line = xyz[mline, 1]
        z_line = xyz[mline, 2]
        label = f"ID {pids[ip]}"

        ax.plot(
            x_line,
            y_line,
            z_line,
            linewidth=0.8,
            marker="o",
            markersize=2.0,
            label=label,
        )

    ax.set_xlabel(r"$x\ \mathrm{(kpc)}$", fontsize=12)
    ax.set_ylabel(r"$y\ \mathrm{(kpc)}$", fontsize=12)
    ax.set_zlabel(r"$z\ \mathrm{(kpc)}$", fontsize=12)

    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(-r_max, r_max)

    ax.tick_params(axis="both", which="major", labelsize=10)

    ax.legend(fontsize=8, loc="upper right", ncol=2, frameon=False)
    ax.view_init(elev=25.0, azim=45.0)

    fig.tight_layout()
    savename_fig = savename + ".orbit_3d" + ".pdf"
    fig_tmp = plt.gcf()
    fig_tmp.savefig(
        savename_fig,
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.1,
    )

    if is_show:
        plt.show()
    plt.close(fig)
    print(f"Plot {savename_fig}, done.")
    return info_particles_select





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
        if len(data)==0:
            raise ValueError("Wrong length of data.")
        x = data[:, 0:3]
        m = data[:, 8]
        pers = [50., 90., 95., 97., 98., 99., 99.5]
        meds = ads.percentiles_by_xv_data(data[:,0:6], pers=pers)
        # ads.DEBUG_PRINT_V(0, pers, meds[3], meds[7], "xvmeds")
    else: #input data
        x = x_input
        m = m_input

    N_grid_x = 100
    N_grid_y = 100
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
            # legacy J·Omega surrogate on a J-grid
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
        
        #Build a robust Omega(J) interpolant that tolerates duplicate/collinear samples.
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

        #vectorized grid build: J-grid (N,3) -> Omega(J) via RBF -> JO (N,6)
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

def load_orbit_stream(filename, ncol=8):
    vals = []
    idx = 0
    with open(filename, "r") as f:
        for line in f:
            # line = (line.strip()).split()
            line = (line).split()
            if not line or len(line) != ncol:
                continue
            try:
                for tok in line:
                    vals.append(float(tok))
            except ValueError:
                continue
            idx += 1
    arr = np.array(vals, dtype=float).reshape(idx, 8)
    return arr

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
        # data = np.loadtxt(filename) #cols: time, x, y, z, vx, vy, vz, pot
        data = load_orbit_stream(filename) #cols: time, x, y, z, vx, vy, vz, pot
        x = data[:, 1:4]
        v = data[:, 4:7]
        plot_orbit_each_2d(x, savepath)

        #|| swit 2
        swit = 2 #orbit rotates along z-axis (swit 2) and is at xy plane
        filename = orbit_snapshot+"orbit_%d_a2.dat"%(i)
        savepath = savename+"_orbit_%d_a2"%(i)
        # data = np.loadtxt(filename) #cols: time, x, y, z, vx, vy, vz, pot
        data = load_orbit_stream(filename) #cols: time, x, y, z, vx, vy, vz, pot
        x = data[:, 1:4]
        v = data[:, 4:7]
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
    
    bd_down = 1e-2
    bd_up = 1e6
    cols = [0,1,2]
    AA_TF_FP, cl_TF_FP, cln = ads.screen_boundary_some_cols(AA_TF_FP, cols, bd_down, bd_up, value_discard=bd_up*1e4)
    AA, cl_TF_DP, cln = ads.screen_boundary_some_cols(AA, cols, bd_down, bd_up, value_discard=bd_up*1e4)
    # AA_TF_DP, cl_TF_DP, cln = ads.screen_boundary_some_cols(AA_TF_DP, cols, bd_down, bd_up, value_discard=bd_up*1e4)
    AA_OD_DP, cl_OD_DP, cln = ads.screen_boundary_some_cols(AA_OD_DP, cols, bd_down, bd_up, value_discard=bd_up*1e4)
    AA_GF_DP, cl_GF_DP, cln = ads.screen_boundary_some_cols(AA_GF_DP, cols, bd_down, bd_up, value_discard=bd_up*1e4)
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
    
    # bdDP_NDFA_path = data_path_plot+".bdDP_NDFA.txt"
    # if particle_type_select is not None:
    #     root, ext = os.path.splitext(bdDP_NDFA_path)
    #     if ext == "":
    #         ext = ".txt"
    #     bdDP_NDFA_path = f"{root}.type_{int(particle_type_select)}{ext}"
    bdDP_NDFA_path = aa_data_path_bdDP%(snapshot_ID, int(particle_type_select))
    print("bdDP_NDFA_path: ", bdDP_NDFA_path)
    np.savetxt(bdDP_NDFA_path, AA_TF_DP_DF)

    #plot
    ## (1) plot 2d
    pointsize = 0.2
    fontsize = 8.0
    figsize = None
    dpi = 400

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
            #     [(AA_TF_FP[:,i_yplot]>1./bd_up)&(AA_TF_FP[:,i_yplot]<bd_up)]) ))
            plt.scatter(xplot[cl_TF_DP], AA_TF_DP[:,i_yplot], s=pointsize, color="g", label=
                r"TSFD$\in$(1e-6, 1e6), fraction=%.2f, mean=%.2f"%(1.*len(AA_TF_DP)/N_data, np.mean(AA_TF_DP[:,i_yplot]
                [(AA_TF_DP[:,i_yplot]>1./bd_up)&(AA_TF_DP[:,i_yplot]<bd_up)]) ))
            # plt.scatter(xplot, AA_OD_DP[:,i_yplot], s=pointsize, label="each "
            #     "action by AA_OD_DP, \t\tmean=%e"%(np.mean(AA_OD_DP[:,i_yplot]
            #     [(AA_OD_DP[:,i_yplot]>1./bd_up)&(AA_OD_DP[:,i_yplot]<bd_up)])))
            
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
    bd_down = 1e-2
    bd_up = 5e4
    data_plot = data*1.
    data_plot, cl_data_plot, cln = ads.screen_boundary_some_cols(data, cols, bd_down, bd_up, value_discard=bd_up*1e4)
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
    bd_down = 1e-2
    bd_up = 5e4
    data_plot = data*1.
    data_plot, cl_data_plot, cln = ads.screen_boundary_some_cols(data, cols, bd_down, bd_up, value_discard=bd_up*1e4)
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
    bd_down = 1e-2
    # bd_up = 5e4
    bd_up = 1e6
    data_plot = data*1.
    data_plot, cl_data_plot, cln = ads.screen_boundary_some_cols(data, cols, bd_down, bd_up, value_discard=bd_up*1e4)
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
    #         # Construct an Omega* whose μ,ν components have the (constant) fitted ratios to Omegaλ
    #         omg_with_args = np.vstack([Ol, a_mu_over_l*Ol, a_nu_over_l*Ol]).T
    # JJO = [J, J*O, O, omg_with_args] #note: all kinds of slices
    # if meta is not None and use_fit_slope_for_omega:
    #     omg = JJO[3] #use the guided omega, not default omega
    # else:
    #     omg = JJO[2] #default omega
    # omg = JJO[3]

    # build guided omega (Omega*) if requested
    use_fit_slope_for_omega = True  # or pass this in as a function arg
    omg_with_args = None
    comb = (meta or {}).get("combination", {})
    cname = str(comb.get("name", "")).strip().lower()
    if use_fit_slope_for_omega and cname in {"aa_combination_freecoef","freecoef","free-coef","free"}:
        args = comb.get("args", [])
        if len(args) >= 2 and args[0] is not None and args[1] is not None:
            km, kn = float(args[0]), float(args[1])
            Ol = O[:, 0]
            omg_with_args = np.vstack([Ol, km*Ol, kn*Ol]).T  # Omega* with fitted ratios

    JJO = [J, J*O, O, omg_with_args]  # note last entry might be None
    label_JJO = ["J", "JO"]
    # choose which Omega to use for slice summaries:
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



# [] main
if __name__ == '__main__':

    ## [] choose which particle types to fit
    try:
        type_list = list(mask_select_type)
    except NameError:
        type_list = [1] #default: halo only
    if not type_list:
        type_list = [1]

    ## [] not loop
    # if 0: #debug, only run loop
    if 1: #run all
        ## (0) debug
        plot_info = "debug"
        savename = save_single_path+plot_info
        # plot_mass_contour()
        print("Plot (%d) %s. Done."%(0, plot_info))

        ## (1) triaxialized potential contour with time in the 5 snapshot
        plot_info = "potential_triaxialized_snapshot_with_time"
        xv_potential_path = aa_data_path
        savename = save_single_path+plot_info #+snapshot_ID #5 figure
        bd_kpc = 30.
        # bd_kpc = 100.
        # potential_contour_file = None
        potential_contour_file = "../../step3_actions/step2_Nbody_TACT/DataInterface/SCF/orbitIntegSCF_adjust_a2b2/src/"\
            +"potential_contour_data_%d.txt"
        for i_snapshot in snapshot_list:
            # plot_potential_contour_SPH(
            plot_potential_contour_KDE(
                xv_potential_path, i_snapshot, potential_contour_file=potential_contour_file, 
                savename=savename, bd_kpc=bd_kpc
            )
        plot_potential_contour_notlog_diff(
            xv_potential_path, snapshot_ID, snapshot_ID-1, potential_contour_file=potential_contour_file, 
            savename=savename, bd_kpc=bd_kpc
        )
        print("Plot (%d) %s. Done."%(1, plot_info))

        ## (1) lmn actions in the 5 snapshot
        plot_info = "info_of_some_particles_with_time"
        savename = save_single_path+plot_info
        is_plot_log_variation = True
        # is_plot_log_variation = False
        is_mask_relative_variation = True
        # is_mask_relative_variation = False
        snap_arr, t_arr, particle_ID_list, tracer_r, aa_tracers = \
            plot_actions_variation_with_time(
                aa_data_path, snapshot_list, time_list, 
                particle_type_select=1, n_particles=20, is_plot_log_variation=is_plot_log_variation, 
                is_mask_relative_variation=is_mask_relative_variation, 
                savename=savename, is_show=is_show
            )
        savename = save_single_path+plot_info
        info_particles_select = plot_3d_orbit_particles(
            aa_data_path, snapshot_list, particle_ID_list, is_plot_log_variation=is_plot_log_variation, 
            savename=savename, is_show=is_show
        )
        print("Plot (%d) %s. Done."%(1, plot_info))
        # sys.exit(2) #debug

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

    ## [] loop about each type or component
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
            # aa_data_path, snapshot_ID, 
            aa_data_path_variation, snapshot_ID, 
            savename, is_show=is_show, particle_type_select=gadget_type
        ) #plot
        # #override BD file path (explicit per-type)
        # bdDP_NDFA_path = galaxy_general_location_path+galaxy_name\
        #     +f"/aa/snapshot_{snapshot_ID}.action.method_all.txt.bdDP_NDFA.type_{int(gadget_type)}.txt"
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
