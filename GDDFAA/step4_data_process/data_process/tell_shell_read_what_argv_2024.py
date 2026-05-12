#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import re
import sys

if __name__ == "__main__":

    ## DF target snapshots
    is_actionerror = int(sys.argv[1]) #note: this file should be copied and run in path/to/step1_prepprocess/
    print("python code, is_actionerror: ", is_actionerror)
    # exit(0)
    path_file = "../../../install_and_run/user_settings_multi.txt"
    file_handle = open(path_file, mode="r")
    char_read = file_handle.readlines()
    file_handle.close()

    snapshots_settings = char_read[8]
    print("argv: ", snapshots_settings)
    atss = snapshots_settings.split(" ")
    time_snapshot_init = float(atss[0])
    time_snapshot_final = float(atss[1])
    # print(atss)
    # exit(0)
    time_snapshot_target_interval = float(atss[2])
    time_snapshot_each = float(atss[3])
    n_cut_snapshot = float(atss[4])
    # snapshot_ID_each_compared = float(atss[5])
    n_mpi_gadget = int(atss[6])

    snapshot_init = n_cut_snapshot+np.round(time_snapshot_init/time_snapshot_each) #[learn code] np.round() return float
    snapshot_final = np.round(time_snapshot_final/time_snapshot_each)
    snapshot_target_interval = np.round(time_snapshot_target_interval/time_snapshot_each)
    
    ts = np.array([])
    s = snapshot_init
    while s<=snapshot_final:
        ts = np.append(ts, s)
        s += snapshot_target_interval
    ts = ts.astype(int)
    N_ts = len(ts)
    print("DF target snapshots: ", ts)
    path_file_w = "argv_target_snapshots_DF.txt" #"10 "
    file_handle = open(path_file_w, mode="w")
    for a in ts:
        file_handle.write("%s "%(a))
    file_handle.close()

    ## AA target snapshots
    if is_actionerror:
        print("Use actionerror, add target snapshots.")
        snapshot_init -= snapshot_target_interval*2
        snapshot_final += snapshot_target_interval*2
        time_snapshot_init -= time_snapshot_each*2
        time_snapshot_final += time_snapshot_each*2
        if snapshot_init<0:
            print("snapshot_init: ", snapshot_init)
            print("The snapshot_init should not less than zero when there is the actionerror.")
            exit(0)
    print("snapshot_init: ", snapshot_init)
    print("snapshot_final: ", snapshot_final)

    path_file_w = "snapshot_interval_stde.txt" #"1 "
    file_handle = open(path_file_w, mode="w")
    file_handle.write("%s "%(int(snapshot_target_interval)))
    file_handle.close()

    ts = np.array([])
    s = snapshot_init
    while s<=snapshot_final:
        ts = np.append(ts, s)
        s += snapshot_target_interval
    ts = ts.astype(int)
    N_ts = len(ts)
    print("AA target snapshots: ", ts)
    path_file_w = "argv_target_snapshots.txt"
    file_handle = open(path_file_w, mode="w")
    for a in ts:
        file_handle.write("%s "%(a))
    file_handle.close()
    print(N_ts, " ", time_snapshot_final)
    # exit(0)

    for i in np.arange(N_ts):
        argv_run_AA = char_read[10]
        aras = re.split(r"[ ]+", argv_run_AA)
        # aras = argv_run_AA.split(" ") #[learn code] too much splitors
        a4 = float(ts[i])*time_snapshot_each #t_init
        # print(ts[i], float(ts[i]), time_snapshot_each, a4)
        # print(str(ts[i]), str(float(ts[i])), str(time_snapshot_each), str(a4))
        a5 = np.min([float(ts[i]+0.1)*time_snapshot_each, time_snapshot_final]) #+1 #t_final
        a6 = np.max([0.01, time_snapshot_each]) #d_t/d_DataTACT_load_snapshot
        a7 = time_snapshot_each #d_t/d_snapshot
        a16 = float(ts[i])*time_snapshot_each
        a17 = float(ts[i]+0.1)*time_snapshot_each

        aras[2] = n_mpi_gadget #??
        aras[4] = "%.3f"%(a4)
        aras[5] = "%.3f"%(a5)
        aras[6] = "%.3f"%(a6)
        aras[7] = "%.3f"%(a7)
        aras[16] = "%.3f"%(a16)
        aras[17] = "%.3f"%(a17)
        print("argv: ", aras, len(aras))
        path_file_w = "argv_run_AA_%s.txt"%(ts[i]) #"... data.exe ..."
        file_handle = open(path_file_w, mode="w")
        for a in aras:
            file_handle.write("%s "%(a))
        file_handle.close()
