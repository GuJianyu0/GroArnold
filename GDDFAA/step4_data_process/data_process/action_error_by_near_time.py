#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import sys
import analysis_data_distribution as ads

def get_action_data_path(s, gm_model_name=None, suffix=None):  
    folder = "../../step2_Nbody_simulation/gadget/Gadget-2.0.7/"
    if gm_model_name is None:
        gm_model_name = "galaxy_general"
    base = "snapshot_%d"%(s)
    if suffix is None:
        suffix = ".action.method_all"
    return folder+gm_model_name+"/"+"aa/"+base+suffix+".txt"

def get_certain_actions_and_frequencies(data, col_actions, col_frequencies):
    dim = 3
    d1 = np.hstack( (
        data[:,col_actions:col_actions+dim], 
        data[:,col_frequencies:col_frequencies+dim]
    ) )
    return d1

# def roughly_yerror_and_mask(xe, tgts, DF_log10):
#     xe_rate = np.sign( np.sum(xe[:,0:3], axis=1) )*ads.norm_l(xe[:,0:3]/tgts[:,0:3], axis=1)
#     mask_xe_rate = (np.isfinite(xe_rate)^True)
#     print("rate of finate value: ", len(mask_xe_rate)/len(xe_rate))
#     xe_rate[mask_xe_rate] = 1.
#     mask_xe_rate = (xe_rate<-1.)
#     xe_rate[mask_xe_rate] = -1.
#     mask_xe_rate = (xe_rate>1.)
#     xe_rate[mask_xe_rate] = 1.
#     xe_rate = np.abs(xe_rate)
#     xe_rate[xe_rate<0.1] = 0.1
#     ye = np.log10(xe_rate)*DF_log10 #the ye has been log10
#     # ye = np.log10(xe_rate)*DF_log10*3 #approximation from error propagation
#     ads.DEBUG_PRINT_V(0, xe_rate[0:10], ye[0:10])
#     return ye

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



if __name__ == '__main__':

    # #debug
    # f1 = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/snapshot_10.action.method_all.txt"
    # data = np.loadtxt(f1, dtype=float)
    # data[:, 78] *= 1.5**-1
    # np.savetxt(f1+".v2", data)
    # ads.DEBUG_PRINT_V(0, data[17, 78])



    #to calculate standard error from the near 5 snpshots
    print(sys.argv[1], " ", sys.argv[2])
    dealwith = int(sys.argv[1]) #like 10
    interval = int(sys.argv[2]) #like 1
    start = dealwith-interval*2
    end = dealwith+interval*2
    snapshot_list = np.linspace(start, end, 5)
    snapshot_list = snapshot_list.astype(int)
    col_actions = 78 #denpendent to file fit_galaxy....py
    col_frequencies = col_actions+7

    # for i=2
    n, m = None, None
    data2 = None
    for i in np.arange(2,3):
        print("near time i: ", i)
        filename = get_action_data_path(snapshot_list[i])
        data1 = np.loadtxt(filename)
        data2 = get_certain_actions_and_frequencies(data1, col_actions, col_frequencies)
        n, m = np.shape(data2) #get shape
    data = np.zeros((5,n,m))
    data[2] = data2
    tgts = data2[:, 0:3]
    DF_log10 = np.log10( np.ones(len(data2))*1e-10 )
    
    #for i=0, 1
    for i in np.arange(0,2):
        print("near time i: ", i)
        filename = get_action_data_path(snapshot_list[i])
        data1 = np.loadtxt(filename)
        data2 = get_certain_actions_and_frequencies(data1, col_actions, col_frequencies)
        data[i] = data2

    #for i=3, 4
    for i in np.arange(3,5):
        print("near time i: ", i)
        filename = get_action_data_path(snapshot_list[i])
        data1 = np.loadtxt(filename)
        data2 = get_certain_actions_and_frequencies(data1, col_actions, col_frequencies)
        data[i] = data2

    data_stde = np.std(data, axis=0)
    print("data_stde snape: ", np.shape(data_stde))
    filename_stde = get_action_data_path(dealwith, suffix=".action.method_all.variance") #the filename of target
    np.savetxt(filename_stde, data_stde)



    # #debug
    # f2 = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/snapshot_10.action.method_all.variance.txt"
    # xe = np.loadtxt(f2, dtype=float)
    # ye = roughly_yerror_and_mask_not_log(xe, tgts, DF_log10)
    # ads.DEBUG_PRINT_V(0, ye)
