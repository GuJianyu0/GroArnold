#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import sys
import analysis_data_distribution as ads



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
save_total_path = galaxy_general_location_path+"/params_statistics/"
save_single_path = galaxy_general_location_path+galaxy_name+"/fit/"



def compute_actions_error(aa_data_path, aa_data_path_variation, snapshot_ID, snapshot_list):
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
    # Valid samples: finite and != 0.0 (0 used as sentinel in your pipeline).
    # Invalid samples simply do not contribute to mean/std; output remains NaN if no valid samples.

    idxA = [col_actions + 0, col_actions + 1, col_actions + 2]
    idxF = [col_frequencies + 0, col_frequencies + 1, col_frequencies + 2]
    idx_all = idxA + idxF
    max_idx = max(idx_all)

    save_path_aa = aa_data_path_variation % int(snapshot_ID)
    # error file path: same prefix, but ".error.txt"
    save_path_error = save_path_aa.replace(".variation.txt", ".error.txt")
    if save_path_error == save_path_aa:
        save_path_error = save_path_aa + ".error.txt"

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



# [] main
if __name__ == '__main__':

    ## [] actions variation with time
    bd_down = 1e-2
    # bd_up = 5e4
    bd_up = 1e6
    compute_actions_error(aa_data_path, aa_data_path_variation, snapshot_ID, snapshot_list)
