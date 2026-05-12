#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

import analysis_data_distribution as ads
import triaxialize_galaxy as tg



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

bd_down = 1e-2
# bd_up = 5e4
bd_up = 1e6

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
aa_data_path_error = galaxy_general_location_path+galaxy_name+"/aa/snapshot_%d.action.method_all.error.txt"
aa_data_path_relative_variation = galaxy_general_location_path+galaxy_name+"/aa/snapshot_%d.action.type_%d.relative.txt"
save_total_path = galaxy_general_location_path+"/params_statistics/"
save_single_path = galaxy_general_location_path+galaxy_name+"/fit/"
statistics_path_types = save_single_path + "snapshot_%d.type_%d.statistics.json"



def compute_actions_error(aa_data_path, snapshot_ID, snapshot_list):
    snapshot_arr = np.asarray(snapshot_list, dtype=int)
    n_snap = len(snapshot_arr)
    if n_snap < 5:
        raise ValueError("Need at least 5 snapshots to plot actions vs time.")

    # AA_save = None
    # AA_save_error = None

    # for i in snapshot_arr:
    #     data = aa_data_path%(snapshot_arr)
    #     # --- TSF [J, Omega] screening on the middle snapshot -----------------------
    #     # Act = J_lambda, J_mu, J_nu ; Fre = Omega_lambda, Omega_mu, Omega_nu
    #     Act = data[:, col_actions:col_actions + 3]
    #     Fre = data[:, col_frequencies:col_frequencies + 3]
    #     AA = np.hstack((Act, Fre))
    #     AA_error = np.zeros_like(AA)
    
    # #() compute actions error by standard deviation efficiently

    # save_path_aa = aa_data_path_variation%(snapshot_ID)
    # save_path_aa_snapshot_ID = aa_data_path%(snapshot_arr)
    # #() open and save as same format of save_path_aa_snapshot_ID

    # save_path_error = aa_data_path_variation%(snapshot_ID)
    # AA_error_data = np.hstack((AA_save, AA_save_error))
    # np.savetxt(save_path_error, AA_error_data)

    # We compute per-particle mean and std of:
    #   [J_lambda, J_mu, J_nu, Omega_lambda, Omega_mu, Omega_nu]
    # across snapshot_list, while preserving row count (no deletions).
    #
    # Valid samples: finite and != 0.0.
    # Invalid samples simply do not contribute to mean/std; output remains NaN if no valid samples.

    idxA = [col_actions + 0, col_actions + 1, col_actions + 2]
    idxF = [col_frequencies + 0, col_frequencies + 1, col_frequencies + 2]
    idx_all = idxA + idxF
    max_idx = max(idx_all)

    save_path_aa = aa_data_path_variation % int(snapshot_ID)
    # save_path_error = save_path_aa.replace(".variation.txt", ".error.txt")
    # if save_path_error == save_path_aa:
    #     save_path_error = save_path_aa + ".error.txt"
    save_path_error = aa_data_path_error % int(snapshot_ID)

    def _read_next_data_line(fh):
        """Return next non-empty, non-comment line, or None at EOF."""
        while True:
            line = fh.readline()
            if line == "":
                return None
            s = line.strip()
            if (not s) or line.lstrip().startswith("#"):
                continue
            return line

    def _parse_6vals_from_parts(parts):
        """Extract 6 AA floats from token list; return np.array(6,) with NaN if missing/bad."""
        out = np.full(6, np.nan, dtype=float)
        if len(parts) <= max_idx:
            return out
        for kk, col in enumerate(idx_all):
            try:
                out[kk] = float(parts[col])
            except Exception:
                out[kk] = np.nan
        return out

    # Open all snapshot files (AA full format)
    fh_by_sid = {}
    for sid in snapshot_arr:
        fh_by_sid[int(sid)] = open(aa_data_path % int(sid), "r")

    # Base file (whose non-AA columns we preserve) is snapshot_ID
    base_sid = int(snapshot_ID)
    if base_sid not in fh_by_sid:
        # snapshot_ID not in snapshot_list; still write output based on snapshot_ID file
        fh_base = open(aa_data_path % base_sid, "r")
    else:
        fh_base = fh_by_sid[base_sid]

    try:
        with open(save_path_aa, "w") as fout_aa, open(save_path_error, "w") as fout_err:
            # Copy header/comments from base file into variation output,
            # and advance all files to their first data line.
            # Base: preserve header lines verbatim in save_path_aa
            first_line_by_sid = {}

            # Base header copy
            while True:
                pos = fh_base.tell()
                line = fh_base.readline()
                if line == "":
                    first_line_by_sid[base_sid] = None
                    break
                s = line.strip()
                if (not s) or line.lstrip().startswith("#"):
                    fout_aa.write(line)
                    continue
                # first data line
                first_line_by_sid[base_sid] = line
                break

            # Other files: skip headers (do not write)
            for sid, fh in fh_by_sid.items():
                if sid == base_sid:
                    continue
                first_line_by_sid[sid] = _read_next_data_line(fh)

            # Optional header for error file
            fout_err.write(
                "# cols: Jl_mean Jm_mean Jn_mean Ol_mean Om_mean On_mean "
                "Jl_std Jm_std Jn_std Ol_std Om_std On_std\n"
            )

            row = 0
            base_line = first_line_by_sid.get(base_sid, None)
            while base_line is not None:
                # Ensure all snapshot lines exist (row alignment)
                lines_row = {}
                for sid in snapshot_arr:
                    sid = int(sid)
                    line_sid = first_line_by_sid.get(sid, None)
                    if line_sid is None:
                        raise RuntimeError(
                            f"EOF mismatch: snapshot {sid} ended early at row {row}."
                        )
                    lines_row[sid] = line_sid

                # Parse base tokens for output formatting
                base_parts = base_line.split()
                # Compute per-row mean/std over snapshots for 6 AA values
                mean6 = np.full(6, np.nan, dtype=float)
                std6  = np.full(6, np.nan, dtype=float)

                # Gather 6-vectors for each snapshot line
                vals_by_sid = {}
                for sid in snapshot_arr:
                    sid = int(sid)
                    parts = lines_row[sid].split()
                    vals_by_sid[sid] = _parse_6vals_from_parts(parts)

                # Welford per component (6 scalars) across snapshots
                for k in range(6):
                    mu = 0.0
                    M2 = 0.0
                    cnt = 0
                    for sid in snapshot_arr:
                        sid = int(sid)
                        v = vals_by_sid[sid][k]
                        if (not np.isfinite(v)) or (v == 0.0):
                            continue
                        cnt += 1
                        delta = v - mu
                        mu += delta / cnt
                        delta2 = v - mu
                        M2 += delta * delta2
                    if cnt > 0:
                        mean6[k] = mu
                    if cnt > 1:
                        std6[k] = np.sqrt(M2 / (cnt - 1.0))

                # Write variation-format AA file: replace AA cols with MEAN values
                if len(base_parts) > max_idx:
                    base_parts[idxA[0]:idxA[0] + 3] = [
                        f"{mean6[0]:.8e}", f"{mean6[1]:.8e}", f"{mean6[2]:.8e}"
                    ]
                    base_parts[idxF[0]:idxF[0] + 3] = [
                        f"{mean6[3]:.8e}", f"{mean6[4]:.8e}", f"{mean6[5]:.8e}"
                    ]
                    fout_aa.write(" ".join(base_parts) + "\n")
                else:
                    # If base line is unexpectedly short, keep it unchanged (row alignment)
                    fout_aa.write(base_line)

                # Write error file: 12 cols = mean(6) + std(6)
                out12 = np.hstack((mean6, std6))
                fout_err.write(" ".join(f"{x:.8e}" if np.isfinite(x) else "nan" for x in out12) + "\n")

                # advance all snapshot data lines
                row += 1
                # next base
                base_line = _read_next_data_line(fh_base)
                first_line_by_sid[base_sid] = base_line
                # next others
                for sid, fh in fh_by_sid.items():
                    if sid == base_sid:
                        continue
                    first_line_by_sid[sid] = _read_next_data_line(fh)

    finally:
        # Close file handles
        for fh in fh_by_sid.values():
            try:
                fh.close()
            except Exception:
                pass
        if base_sid not in fh_by_sid:
            try:
                fh_base.close()
            except Exception:
                pass

    print("Save actions mean and std. Done.")
    return 0

def roughly_yerror_and_mask(xe, tgts, DF_log10):
    xe_rate = ads.norm_l(xe[:,0:3]/tgts[:,0:3], axis=1)
    xe_rate_freq = ads.norm_l(xe[:,3:6]/tgts[:,3:6], axis=1)
    xe_rate_OJsum = xe_rate+xe_rate_freq
    finite_count_xerror = np.sum(np.isfinite( xe_rate_OJsum ))
    count_xerror = len(xe)
    print("finite_count_xerror and count_xerror: ", finite_count_xerror, " ", count_xerror)
    
    mask_xe_rate_notfinite = (np.isfinite(xe_rate_OJsum)^True)
    xe_rate[mask_xe_rate_notfinite] = 10.
    mask_xe_rate_toolarge = (np.abs(xe_rate)<1e-2)
    xe_rate[mask_xe_rate_toolarge] = 10.
    mask_xe_rate_tooless = (np.abs(xe_rate)<1e-2)
    xe_rate[mask_xe_rate_tooless] = 0.1
    print(np.sum(mask_xe_rate_notfinite), np.sum(mask_xe_rate_tooless), np.min(xe_rate), np.max(xe_rate))
    xe_rate_log = np.log10(xe_rate)
    ye = xe_rate_log*DF_log10 #the ye has been log10
    # ye = np.log10(xe_rate)*DF_log10*3 #?? approximation from error propagation
    # pers = [0., 0.05, 0.2, 0.5, 0.8, 0.95, 1.]
    # ads.DEBUG_PRINT_V(1, np.percentile(xe_rate[np.argsort(xe_rate)], pers)) #note: not need to argsort when percentile
    # ads.DEBUG_PRINT_V(1, np.percentile(xe_rate_log[np.argsort(xe_rate_log)], pers))
    # ads.DEBUG_PRINT_V(1, np.percentile(DF_log10[np.argsort(DF_log10)], pers))
    # ads.DEBUG_PRINT_V(0, np.percentile(ye[np.argsort(ye)], pers))
    return ye



def _action_uncertainty_errorfile_path(galaxymodel_name, snapshot_ID):
    """
    Multi-snapshot AA mean+std file.
    Expected 12 columns per particle:
      mean(Jλ,Jμ,Jν,Ωλ,Ωμ,Ων), std(Jλ,Jμ,Jν,Ωλ,Ωμ,Ων)
    """
    return os.path.join(
        galaxymodel_name, "aa", f"snapshot_{int(snapshot_ID)}.action.method_all.error.txt"
    )

def _read_rows_by_sorted_index_text(path, row_idx_sorted, ncol=12):
    """
    Stream-read only selected *data* rows (0-based, counting only non-empty non-comment rows).
    This avoids loading the full file when it is huge.
    """
    row_idx_sorted = np.asarray(row_idx_sorted, dtype=int)
    out = np.full((len(row_idx_sorted), ncol), np.nan, dtype=float)
    if len(row_idx_sorted) == 0:
        return out

    k = 0
    target = int(row_idx_sorted[k])
    row = 0
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or line.lstrip().startswith("#"):
                continue
            if row == target:
                parts = line.split()
                if len(parts) >= ncol:
                    try:
                        out[k, :] = [float(parts[i]) for i in range(ncol)]
                    except Exception:
                        # keep NaN row if parse fails
                        pass
                k += 1
                if k >= len(row_idx_sorted):
                    break
                target = int(row_idx_sorted[k])
            row += 1
    return out

def _relative_uncertainty(mean_arr, std_arr, eps=1e-40):
    """
    rel = std / max(|mean|, eps), elementwise; keeps NaN if inputs are NaN.
    """
    mean_arr = np.asarray(mean_arr, dtype=float)
    std_arr = np.asarray(std_arr, dtype=float)
    denom = np.maximum(np.abs(mean_arr), eps)
    rel = std_arr / denom
    rel[~np.isfinite(rel)] = np.nan
    return rel

def _pct_summary(x, p=(16.0, 50.0, 84.0)):
    x = np.asarray(x, dtype=float).reshape(-1)
    m = np.isfinite(x)
    if not np.any(m):
        return {"n": 0, "p16": np.nan, "p50": np.nan, "p84": np.nan}
    q = np.percentile(x[m], list(p))
    return {"n": int(m.sum()), "p16": float(q[0]), "p50": float(q[1]), "p84": float(q[2])}

def report_action_uncertainty_for_type(
    snapshot_ID,
    gadget_type,
    cl,
    ptype_full,
    tgts_cl,          # tgts returned by main_step2_DF for tag==2 (already |AA| and already cl-applied)
    mass_full,
    auxiliary_function_plot2d_x,
    auxiliary_function_plot2d_args,
    save_prefix,
    galaxymodel_name,
):
    """
    Build an uncertainty report for the particles that:
      - pass AA screening (cl), and
      - belong to gadget_type
    using multi-snapshot AA mean+std file.

    Outputs:
      - save_prefix + ".action_uncertainty.png"
      - save_prefix + ".action_uncertainty.summary.txt"
      - returns a JSON-safe dict for meta.
    """
    # ferr = _action_uncertainty_errorfile_path(galaxymodel_name, snapshot_ID)
    ferr = aa_data_path_error
    if not os.path.exists(ferr):
        print(f"Warning: action-uncertainty file not found: {ferr}")
        return None

    cl = np.asarray(cl, dtype=int)
    ptype_full = np.asarray(ptype_full).reshape(-1)
    if cl.size == 0:
        return None

    # selection within cl for this type (this mirrors the type filtering used later in fitting)
    ptype_cl = ptype_full[cl]
    sel = (ptype_cl.astype(int) == int(gadget_type))
    if np.sum(sel) == 0:
        return None

    # indices in the original AA file rows that correspond to selected particles
    idx_sel = cl[sel]

    # stream read only required rows from error file (12 cols)
    idx_sorted = np.sort(idx_sel)
    err_sorted = _read_rows_by_sorted_index_text(ferr, idx_sorted, ncol=12)
    # map back to the original selection order (order in idx_sel)
    pos = np.searchsorted(idx_sorted, idx_sel)
    err = err_sorted[pos, :]

    mean6 = err[:, 0:6]
    std6  = err[:, 6:12]
    rel6 = _relative_uncertainty(mean6, std6)
    relJ = rel6[:, 0:3]
    relO = rel6[:, 3:6]

    # scalar summaries (median across the 3 components)
    relJ_med = np.nanmedian(relJ, axis=1)
    relO_med = np.nanmedian(relO, axis=1)

    # compute h (same helper as plotting)
    _aux_args = auxiliary_function_plot2d_args
    if _aux_args is None:
        _aux_args = ()
    else:
        _aux_args = tuple(a for a in _aux_args if a is not None)
    tgts_type = np.asarray(tgts_cl, dtype=float)[sel]
    h = auxiliary_function_plot2d_x(tgts_type, *_aux_args)
    h = np.asarray(h, dtype=float).reshape(-1)

    # meta summary (store log10 rel for paper readability)
    log_relJ_med = np.log10(relJ_med)
    log_relO_med = np.log10(relO_med)
    meta_unc = {
        "snapshot_ID": int(snapshot_ID),
        "gadget_type": int(gadget_type),
        "log10_relJ_median": _pct_summary(log_relJ_med),
        "log10_relO_median": _pct_summary(log_relO_med),
        "log10_relJ_lambda": _pct_summary(np.log10(relJ[:, 0])),
        "log10_relJ_mu":     _pct_summary(np.log10(relJ[:, 1])),
        "log10_relJ_nu":     _pct_summary(np.log10(relJ[:, 2])),
        "log10_relO_lambda": _pct_summary(np.log10(relO[:, 0])),
        "log10_relO_mu":     _pct_summary(np.log10(relO[:, 1])),
        "log10_relO_nu":     _pct_summary(np.log10(relO[:, 2])),
    }

    # write a tiny text summary for paper drafts
    summary_txt = save_prefix + ".action_uncertainty.summary.txt"
    with open(summary_txt, "w") as f:
        f.write("# action uncertainty summary (log10 relative std)\n")
        f.write(json.dumps(meta_unc, indent=2))
        f.write("\n")

    # plot
    try:
        from matplotlib.colors import LogNorm
        fig = plt.figure(figsize=(12, 8), dpi=300)

        ax1 = fig.add_subplot(2, 2, 1)
        m1 = np.isfinite(log_relJ_med)
        if np.any(m1):
            ax1.hist(log_relJ_med[m1], bins=60)
        ax1.set_xlabel(r"$\log_{10}(\sigma_J/|\bar{J}|)$")
        ax1.set_ylabel("count")
        ax1.set_title("relative action uncertainty (median over 3)")

        ax2 = fig.add_subplot(2, 2, 2)
        m2 = np.isfinite(log_relO_med)
        if np.any(m2):
            ax2.hist(log_relO_med[m2], bins=60)
        ax2.set_xlabel(r"$\log_{10}(\sigma_\Omega/|\bar{\Omega}|)$")
        ax2.set_ylabel("count")
        ax2.set_title("relative frequency uncertainty (median over 3)")

        ax3 = fig.add_subplot(2, 2, 3)
        mh = np.isfinite(h) & (h > 0.0) & np.isfinite(relJ_med) & (relJ_med > 0.0)
        if np.any(mh):
            x = np.log10(h[mh])
            y = np.log10(relJ_med[mh])
            H, xe, ye = np.histogram2d(x, y, bins=[80, 60])
            Hm = np.ma.masked_less(H, 1.0)
            pcm = ax3.pcolormesh(xe, ye, Hm.T, norm=LogNorm(vmin=1.0, vmax=max(1.0, np.nanmax(H))), shading="auto")
            fig.colorbar(pcm, ax=ax3, label="count per bin")
        ax3.set_xlabel(r"$\log_{10} h$")
        ax3.set_ylabel(r"$\log_{10}(\sigma_J/|\bar{J}|)$")

        ax4 = fig.add_subplot(2, 2, 4)
        mh2 = np.isfinite(h) & (h > 0.0) & np.isfinite(relO_med) & (relO_med > 0.0)
        if np.any(mh2):
            x = np.log10(h[mh2])
            y = np.log10(relO_med[mh2])
            H, xe, ye = np.histogram2d(x, y, bins=[80, 60])
            Hm = np.ma.masked_less(H, 1.0)
            pcm = ax4.pcolormesh(xe, ye, Hm.T, norm=LogNorm(vmin=1.0, vmax=max(1.0, np.nanmax(H))), shading="auto")
            fig.colorbar(pcm, ax=ax4, label="count per bin")
        ax4.set_xlabel(r"$\log_{10} h$")
        ax4.set_ylabel(r"$\log_{10}(\sigma_\Omega/|\bar{\Omega}|)$")

        fig.tight_layout()
        outpng = save_prefix + ".action_uncertainty.png"
        fig.savefig(outpng, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print("Save action uncertainty plot:", outpng)
    except Exception as e:
        print("Warning: plotting action uncertainty failed:", e)

    return meta_unc

def load_mean_and_std_of_actions(aa_data_path_variation, aa_data_path_error, snapshot_ID):
    data = np.loadtxt(aa_data_path_error%(snapshot_ID)) #it has been masked by actions bounds
    Jmean_lj_all = data[:, 0:3] #where l donates the particle index and j donates the lambda, mu and nu
    Jstd_lj_all = data[:, 6:9]
    data2 = np.loadtxt(aa_data_path_variation%(snapshot_ID))
    particles_type = (data2[:,col_particle_type]).astype(int)
    N_particles_masked = len(data)
    ads.DEBUG_PRINT_V(1, Jmean_lj_all[0], N_particles_masked, "N_particles_masked")
    return Jmean_lj_all, Jstd_lj_all, particles_type

def relative_variation_lmn_each_type(bd_down, bd_up, types_input):
    Jmean_lj_all, Jstd_lj_all, particles_type = load_mean_and_std_of_actions(
        aa_data_path_variation, aa_data_path_error, snapshot_ID
    )
    eps_floor = 0.05  # denominator floor: max(|Jbar_lj|, eps_floor * Jtot_l)
    
    for each_type in types_input:
        # particles of the current gadget type
        cl_each_type = np.where(particles_type.astype(int) == int(each_type))[0]
        if len(cl_each_type) == 0:
            savename = aa_data_path_relative_variation % (snapshot_ID, each_type)
            np.savetxt(savename, np.array([np.nan, np.nan, np.nan]))
            print("Warning: No particles of gadget type %d. Skip it."%(each_type))
            continue

        Jmean_lj = Jmean_lj_all[cl_each_type]
        Jstd_lj  = Jstd_lj_all[cl_each_type]
        # ads.DEBUG_PRINT_V(0, np.shape(Jstd_lj), "rel_lmn")
        Jmean_lj, cl, cln = ads.screen_boundary_some_cols(
            Jmean_lj, [0,1,2], bd_down, bd_up, value_discard=None
        )
        Jstd_lj = Jstd_lj[cl]

        # after masking, nothing remains
        if len(Jmean_lj) == 0:
            rel_lambda, rel_mu, rel_nu = np.nan, np.nan, np.nan
        else:
            # Jtot_l from the 3 action means of each particle
            Jtot_l = np.sqrt(np.sum(Jmean_lj**2, axis=1))

            # denominator: max(|Jbar_lj|, eps_floor * Jtot_l)
            denom_lj = np.maximum(np.abs(Jmean_lj), eps_floor * Jtot_l[:, None])

            with np.errstate(divide='ignore', invalid='ignore'):
                rel_lj = Jstd_lj / denom_lj

            rel_lj[~np.isfinite(rel_lj)] = np.nan

            # collapse particle index l -> one scalar for each action component
            # saved as percent
            rel_lambda = 100.0 * np.nanmedian(rel_lj[:, 0])
            rel_mu     = 100.0 * np.nanmedian(rel_lj[:, 1])
            rel_nu     = 100.0 * np.nanmedian(rel_lj[:, 2])

        rel_lmn = np.array([rel_lambda, rel_mu, rel_nu])
        ads.DEBUG_PRINT_V(1, each_type, rel_lmn, "each_type, rel_lmn")
        savename = aa_data_path_relative_variation % (snapshot_ID, each_type)
        np.savetxt(savename, rel_lmn)
    return 0

def compute_statistics(bd_down, bd_up, types_input):
    data = np.loadtxt(aa_data_path_variation%(snapshot_ID))
    x_all = data[:,col_x:col_x+3]
    v_all = data[:,col_v:col_v+3]
    mass_all = data[:,col_particle_mass]
    pot_all = data[:,col_potential]
    AA_all = data[:,col_actions:col_actions+3]
    particles_type = (data[:,col_particle_type]).astype(int)
    data = None #to decrease the occupication of running memory, thie does not influence later variables
    # ads.DEBUG_PRINT_V(1, x_all[0])

    for each_type in types_input:
        # particles of the current gadget type
        cl_each_type = np.where(particles_type.astype(int) == int(each_type))[0]
        if len(cl_each_type) == 0:
            savename = aa_data_path_relative_variation % (snapshot_ID, each_type)
            np.savetxt(savename, np.array([np.nan, np.nan, np.nan]))
            print("Warning: No particles of gadget type %d. Skip it."%(each_type))
            continue

        x = x_all[cl_each_type]
        v = v_all[cl_each_type]
        mass = mass_all[cl_each_type]
        pot = pot_all[cl_each_type]
        AA = AA_all[cl_each_type]

        M = np.sum(mass)
        r = ads.norm_l(x, axis=1)
        r_median = np.median(r) #a reference value of scale length
        q_axisratio = ads.axis_ratio_by_configuration_data(x)
        spin_L_processed_direct = tg.spin_lambda_Nbody(x, v, mass=mass, pot=pot)

        AA_cl, cl, cln = ads.screen_boundary_some_cols(AA, [0,1,2], bd_down, bd_up, value_discard=None)
        action_axisratio_direct = ads.axis_ratio_by_configuration_data(AA_cl)
        
        savename = statistics_path_types%(snapshot_ID, each_type)
        ads.DEBUG_PRINT_V(1, each_type, spin_L_processed_direct, len(AA_cl), "each_type, spin_L_processed_direct, len(AA_cl)")
        
        # save to json with variable name: M--(1,), r_median--(1,), spin_L_processed_direct--(1,), q_axisratio--(3,), action_axisratio_direct--(3,), 
        stats_dict = {
            "M": float(M),
            "r_median": float(r_median),
            "spin_L_processed_direct": float(spin_L_processed_direct),
            "q_axisratio": np.asarray(q_axisratio).tolist(),
            "action_axisratio_direct": np.asarray(action_axisratio_direct).tolist()
        }
        with open(savename, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print("Save %s, done."%(savename))
    return 0



# [] main
if __name__ == '__main__':

    # ## [] actions variation with time
    # compute_actions_error(aa_data_path, snapshot_ID, snapshot_list)

    # ## [] actions relative variation with time
    # relative_variation_lmn_each_type(bd_down=bd_down, bd_up=bd_up, types_input=mask_select_type)

    ## [] compute statistics
    print(mask_select_type)
    compute_statistics(bd_down, bd_up, types_input=mask_select_type)
