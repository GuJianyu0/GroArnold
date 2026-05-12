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

# sys.path.insert(0, os.path.abspath("../path/to"))
import analysis_data_distribution as ads
import galaxy_models as gm
import change_params_galaxy_init as cpgi

# import paraview



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
aa_data_path_DF_csv = galaxy_general_location_path+galaxy_name+"/aa/csv/snapshot_%d.action.type_%d.DF_%s.csv"
save_total_path = galaxy_general_location_path+"/params_statistics/"
save_single_path = galaxy_general_location_path+galaxy_name+"/fit/"



def export_paraview_action_files(
    snapshot_ID,
    particle_type_select=None,
    savename=None,
):
    """
    Export ParaView-friendly CSV files from the same bdDP_NDFA txt used by plot_actions_3d().

    Input:
        snapshot_ID
        particle_type_select
        savename:
            path template like
            aa_data_path_DF_csv = galaxy_general_location_path + galaxy_name + \
                "/aa/csv/snapshot_%d.action.type_%d.DF_%s.csv"

    Output files:
        - one all-points CSV
        - one summary CSV
        - one CSV for every log10(f) slice

    Return:
        csv_all, csv_summary, csv_slice_list
    """
    import os
    import numpy as np

    if particle_type_select is None:
        particle_type_select = 1
    if savename is None:
        raise ValueError("savename must be a path template like aa_data_path_DF_csv")

    # ------------------------------------------------------------------
    # local helpers
    # ------------------------------------------------------------------
    def _ensure_parent_dir(path):
        parent = os.path.dirname(path)
        if parent != "":
            os.makedirs(parent, exist_ok=True)

    def _save_csv(path, arr, header):
        _ensure_parent_dir(path)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        np.savetxt(
            path,
            arr,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.8e",
        )

    def _name_num(x):
        # safe string for filename
        s = "%.2f" % (float(x))
        s = s.replace("-", "m").replace(".", "p")
        return s

    def _slice_name(i_slice, fdmin, fdmax):
        return "slice_%03d_log10f_%s_%s" % (
            int(i_slice),
            _name_num(fdmin),
            _name_num(fdmax),
        )

    # ------------------------------------------------------------------
    # same source logic as your existing project
    # requires global aa_data_path_bdDP to be already defined
    # ------------------------------------------------------------------
    bdDP_NDFA_path = aa_data_path_bdDP % (snapshot_ID, int(particle_type_select))
    if not os.path.exists(bdDP_NDFA_path):
        raise FileNotFoundError("Input txt not found:\n%s" % (bdDP_NDFA_path))

    data = np.loadtxt(bdDP_NDFA_path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 7:
        raise ValueError(
            "Input txt must have at least 7 columns: "
            "[J_lambda, J_mu, J_nu, Omega_lambda, Omega_mu, Omega_nu, f, ...]"
        )

    # ------------------------------------------------------------------
    # keep the same screening idea as plot_actions_3d()
    # ------------------------------------------------------------------
    cols = [0, 1, 2]
    bd_down = 1e-2
    bd_up = 1e6

    mask = np.ones(len(data), dtype=bool)
    mask &= np.all(np.isfinite(data[:, :7]), axis=1)
    for c in cols:
        mask &= (data[:, c] >= bd_down) & (data[:, c] <= bd_up)
    mask &= (data[:, 6] > 0.0)

    data_plot = data[mask]
    if len(data_plot) == 0:
        raise RuntimeError("No valid rows remain after boundary screening.")

    J = data_plot[:, 0:3]
    O = data_plot[:, 3:6]
    f = data_plot[:, 6]

    log10J = np.log10(J)
    log10f = np.log10(f)

    # disjunctor == 0 in your current plot_actions_3d() usage
    # so x = J
    x = J
    y = log10f

    # ------------------------------------------------------------------
    # same slice setup as plot_actions_3d()
    # ------------------------------------------------------------------
    ymin = np.floor(np.min(y))
    ymax = np.ceil(np.max(y))
    dy = 0.1
    N_slices = int((ymax - ymin) / dy) + 1

    if N_slices < 2:
        DY = np.array([ymin, ymin + dy], dtype=float)
        N_slices = 2
    else:
        DY = np.linspace(ymin, ymax, N_slices)

    # ------------------------------------------------------------------
    # also keep the representative-slice selection logic compatible
    # with plot_actions_3d(), but here we export ALL slices anyway
    # ------------------------------------------------------------------
    N_sample_min = 12
    N_plot_slices = 4

    slice_id = -np.ones(len(y), dtype=int)
    slice_lo = np.full(len(y), np.nan, dtype=float)
    slice_hi = np.full(len(y), np.nan, dtype=float)

    slice_count = np.zeros(N_slices - 1, dtype=int)
    enough_mask_per_slice = np.zeros(N_slices - 1, dtype=int)

    # assign each point to one slice
    for i in range(N_slices - 1):
        fdmin = DY[i]
        fdmax = DY[i + 1]

        if i < (N_slices - 2):
            cl = (y >= fdmin) & (y < fdmax)
        else:
            # include the right edge in the last slice
            cl = (y >= fdmin) & (y <= fdmax)

        n_cl = int(np.sum(cl))
        slice_count[i] = n_cl
        if n_cl >= N_sample_min:
            enough_mask_per_slice[i] = 1

        slice_id[cl] = i
        slice_lo[cl] = fdmin
        slice_hi[cl] = fdmax

    # choose the same representative slices as plot_actions_3d(), if possible
    plot_selected_slice = np.zeros(N_slices - 1, dtype=int)
    cl_enough_list = [i for i in range(N_slices - 1) if slice_count[i] >= N_sample_min]
    n_clel = len(cl_enough_list)
    if n_clel >= N_plot_slices:
        for j in np.arange(N_plot_slices):
            idx = int(float(j) / N_plot_slices * n_clel)
            i_sel = cl_enough_list[idx]
            plot_selected_slice[i_sel] = 1
    elif n_clel > 0:
        for i_sel in cl_enough_list:
            plot_selected_slice[i_sel] = 1

    is_plot_selected_row = np.zeros(len(y), dtype=int)
    good_row = (slice_id >= 0)
    is_plot_selected_row[good_row] = plot_selected_slice[slice_id[good_row]]

    row_id = np.arange(len(x), dtype=float)
    weight = np.ones(len(x), dtype=float)

    all_table = np.column_stack([
        J[:, 0], J[:, 1], J[:, 2], 
        # log10J[:, 0], log10J[:, 1], log10J[:, 2], 
        # O[:, 0], O[:, 1], O[:, 2], 
        log10f, f, 
        slice_id.astype(float), slice_lo, slice_hi, 
        row_id, 
        # is_plot_selected_row.astype(float), 
        weight,
    ])

    header_all = ",".join([
        "J_lambda", "J_mu", "J_nu", 
        # "log10J_lambda", "log10J_mu", "log10J_nu", 
        # "Omega_lambda", "Omega_mu", "Omega_nu", 
        "log10f", "f", 
        "slice_id", "slice_log10f_min", "slice_log10f_max", 
        "row_id", 
        # "is_plot_actions3d_selected", 
        "weight",
    ])

    # ------------------------------------------------------------------
    # save all-points CSV
    # ------------------------------------------------------------------
    csv_all = savename % (snapshot_ID, int(particle_type_select), "all")
    _save_csv(csv_all, all_table, header_all)

    # ------------------------------------------------------------------
    # save one CSV for every slice
    # ------------------------------------------------------------------
    csv_slice_list = []

    for i in range(N_slices - 1):
        fdmin = DY[i]
        fdmax = DY[i + 1]

        if i < (N_slices - 2):
            cl = (y >= fdmin) & (y < fdmax)
        else:
            cl = (y >= fdmin) & (y <= fdmax)

        slice_name = _slice_name(i, fdmin, fdmax)
        csv_slice = savename % (snapshot_ID, int(particle_type_select), slice_name)
        csv_slice_list.append(csv_slice)

        if np.any(cl):
            _save_csv(csv_slice, all_table[cl], header_all)
        else:
            # header-only empty csv with correct number of columns
            empty_arr = np.zeros((0, all_table.shape[1]), dtype=float)
            _save_csv(csv_slice, empty_arr, header_all)

    # ------------------------------------------------------------------
    # save summary CSV
    # ------------------------------------------------------------------
    summary_rows = []
    for i in range(N_slices - 1):
        fdmin = DY[i]
        fdmax = DY[i + 1]
        fdmid = 0.5 * (fdmin + fdmax)
        summary_rows.append([
            float(i),                    # slice_id
            float(fdmin),                # log10f_min
            float(fdmax),                # log10f_max
            float(fdmid),                # log10f_mid
            float(slice_count[i]),       # n_points
            float(enough_mask_per_slice[i]),
            # float(plot_selected_slice[i]),
        ])

    summary_arr = np.array(summary_rows, dtype=float)
    header_summary = ",".join([
        "slice_id",
        "log10f_min",
        "log10f_max",
        "log10f_mid",
        "n_points",
        "enough_for_fit",
        # "is_plot_actions3d_selected",
    ])

    csv_summary = savename % (snapshot_ID, int(particle_type_select), "summary")
    _save_csv(csv_summary, summary_arr, header_summary)

    print("nslice: %d" % (N_slices - 1))
    print("nrows : %d" % (len(all_table)))
    return csv_all, csv_summary, csv_slice_list

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
    # pointsize = 0.1
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

    ### a. 3d slices
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
        pointsize_s = 10.
        alpha_s = 0.5
        if k <= 2:
            pointsize_s *= 1.5
            alpha_s *= 1.5
        if k >= 2:
            pointsize_s /= 10.
            alpha_s /= 10.
        axsc = ax.scatter(JO_plot[:,0], JO_plot[:,1], JO_plot[:,2], s=pointsize_s, alpha=alpha_s, label=None)
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
    fig_tmp.savefig(
        savename+save_suf+".%s_%s.pdf"%("slices", label_JJO[disjunctor]), 
        format="pdf", dpi=dpi, bbox_inches="tight",
    )
    # plt.close("all")
    plt.close(fig)

    if is_show:
        plt.show()
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

    ## [] loop about each type or component
    for gadget_type in type_list:
        ## (8) actions slices paraview
        #\ data from bdDP_NDFA_path
        bdDP_NDFA_path = aa_data_path_bdDP%(snapshot_ID, int(gadget_type))
        plot_info = "actions_paraview"
        savename = save_single_path+plot_info
        df_csv_path, summary_csv_path, slice_csv_path_list = export_paraview_action_files(
            snapshot_ID,
            particle_type_select=gadget_type,
            savename=aa_data_path_DF_csv
        )
        print("Plot (%d) %s. Done."%(8, plot_info))
        sys.exit(2)

        ## (8) actions all in 3d, h-surface and Omega-cut
        #\ data from bdDP_NDFA_path
        bdDP_NDFA_path = aa_data_path_bdDP%(snapshot_ID, int(gadget_type))
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
