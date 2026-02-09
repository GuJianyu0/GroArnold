#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import inspect
import traceback
import scipy.optimize
import scipy.integrate
from scipy.interpolate import Rbf
# import random
# from distutils.debug import DEBUG
from sklearn.neighbors import KDTree



#### useful functions
def activate_range(x, xm=1.):
    return np.tanh(x/xm)
    # return np.log10(np.abs(x))

def count_function_args(func):
    """
    Return the number of explicit positional or keyword parameters
    (excluding *args/**kwargs) that a function defines.
    """
    sig = inspect.signature(func)
    n = sum(
        1 for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    return n

def check_count_of_particle_type(particle_type, mask_select_type):
    N_types = len(mask_select_type)
    counts = list(range(N_types))
    for i in range(N_types):
        counts[i] = len(particle_type[particle_type==mask_select_type[i]])
    return counts

def random_choice_without_prior(pos, N_targets):
    '''
    Random choice without prior and deep copy.
    '''
    N = len(pos)
    sample_indices = np.random.choice(N, size=N_targets, replace=False)
    pos_targets = copy.deepcopy(pos[sample_indices]) #copy
    return pos_targets

def rescale_dispersion_keep_ratio(vmean_three_fixed, sigma_three_old, v2m_sqrt_target):
    '''
    Rescale the diag-ccomponent of velocity dispersion: 
    fix the 3 mean velocity to vmean_three_fixed (usually zeros), 
    keep the ratio of the diag-ccomponent of velocity dispersion to sigma_three_old, 
    and keep the rms of speed to v2m_sqrt_target.
    '''
    sigma_total_old = np.linalg.norm(sigma_three_old)
    ratio = sigma_three_old/sigma_total_old
    sigma_total_target = np.sqrt( v2m_sqrt_target**2 - np.linalg.norm(vmean_three_fixed)**2 )
    sigma_three_target = sigma_total_target*ratio
    return sigma_three_target

def get_mean_center(sample):
    return np.mean(sample, axis=0)

def get_std_dispersion(sample):
    return np.std(sample, axis=0)

def get_mean_radius(sample, is_center=False):
    pos = sample*1.
    if is_center:
        pos -= np.mean(pos, axis=0)
    r = norm_l(pos, axis=1)
    return np.mean(r)

def readjust_positions(positions, r_mean):
    sample = positions*1.
    old_mean = np.mean(sample, axis=0)
    sample -= old_mean

    old_r_mean = get_mean_radius(sample)
    scaling_factor = r_mean / old_r_mean
    sample *= scaling_factor
    return sample

def get_quadratic_mean(sample, is_center=False):
    smp = sample*1.
    if is_center:
        smp -= np.mean(smp, axis=0)
    qm = np.sqrt(np.mean(np.sum(smp**2, axis=1)))
    return qm

def get_percentile_radius(positions, is_center=False, pers=[50., 99., 100.]):
    pos = positions*1.
    if is_center:
        pos -= np.mean(pos, axis=0)
    r = norm_l(pos, axis=1)
    return np.percentile(r, pers)

def get_mean_unit_kinetic_energy(velocities, is_center=False):
    vel = velocities*1.
    if is_center:
        vel -= np.mean(vel, axis=0)
    v_size = norm_l(vel, axis=1)
    return 0.5*np.mean(v_size**2.)

def readjust_velocities(velocities, mean_3, dispersion_3):
    sample = velocities*1.
    old_mean = np.mean(sample, axis=0)
    sample -= old_mean
    sample += mean_3

    old_quadratic_mean = np.sqrt(np.mean(np.sum(sample**2, axis=1)))
    quadratic_mean = np.sqrt(np.sum(mean_3**2+dispersion_3**2))
    scaling_factor = quadratic_mean / old_quadratic_mean
    sample *= scaling_factor
    return sample

def print_sample_info(sample, sample_name=None):
    if len(np.shape(sample)) != 2:
        DEBUG_PRINT_V(
            1, "Wrong shape of sample, please check.", 
            sample_name
        )
        return 1
    else:
        DEBUG_PRINT_V(
            1, np.shape(sample), 
            get_mean_center(sample), get_std_dispersion(sample), 
            get_mean_radius(sample), get_quadratic_mean(sample), 
            sample_name
        )
        return 0

def screen_radius_samples(positions, radius_bound_down=-1e-20, radius_bound_up=np.inf):
    r = norm_l(positions, axis=1)
    mask = (r>=radius_bound_down) & (r<=radius_bound_up)
    indices = np.where(mask)[0]
    indices_not = np.where(~mask)[0]
    return positions[indices], indices, indices_not

def screen_sample_by_min_radius_count(positions, N_select, mean_radius_setting=None):
    if len(positions) < N_select:
        return positions, None
    else:
        pos = positions*1.
        pos -= np.mean(pos, axis=0)
        radius = norm_l(pos, axis=1)
        indices = np.argsort(radius)
        pos = (pos[indices])[0:N_select]
        # ads.DEBUG_PRINT_V(0, (radius[indices])[N_select], "original screen radius")

        r_mean_old = get_mean_radius(pos)
        if mean_radius_setting is not None:
            scaling_factor = mean_radius_setting/r_mean_old
            pos *= scaling_factor
        DEBUG_PRINT_V(1, np.shape(pos), scaling_factor, "scaling_factor")
        return pos, r_mean_old

def screen_sample_by_random_select(positions, N_select, mean_radius_setting=None):
    if len(positions) < N_select:
        return positions
    else:
        pos = positions*1.
        np.random.shuffle(pos) #the outeast axis
        pos = pos[:N_select]
        if mean_radius_setting is not None:
            scaling_factor = mean_radius_setting/get_mean_radius(pos)
            pos *= scaling_factor
            DEBUG_PRINT_V(1, np.shape(pos), scaling_factor, "scaling_factor")
        return pos

def remain_only_finite_index(x):
    x_sum = np.sum(x, axis=1)
    mask = (np.isfinite(x_sum))
    index_mask = np.where(mask)[0]
    # ads.DEBUG_PRINT_V(1, np.shape(x), np.shape(index_mask), "np.shape(index_mask)")
    return x[index_mask]

def normalization_vector_designated(vec_init, norm):
    '''
    The vector is with shape (n,).
    '''
    # return vec_init/norm_l(vec_init)*norm
    vec = np.array(vec_init, dtype=float)
    n0 = norm_l(vec)
    # Robust normalization: avoid NaN/Inf when vec_init is zero or non-finite.
    # If n0 is zero/non-finite, direction is undefined.
    # - If requested norm is ~0, return zeros.
    # - Otherwise, return vec unchanged (do not rescale) to avoid not-finite.
    if (not np.isfinite(n0)) or (n0 <= 0.0):
        if np.isfinite(norm) and (abs(norm) <= 0.0):
            return np.zeros_like(vec)
        return vec
    if not np.isfinite(norm):
        return vec
    return vec / n0 * norm

def calculate_spin_L_before_preprocess(xv, mass, pot):
    x = xv[:, 0:3]
    v = xv[:, 3:6]
    vnorm = norm_l(v, axis=1)
    L = np.zeros(( len(mass), 3 ))
    L[:, 0] = mass[:]*(x[:, 1]*v[:, 2] + x[:, 2]*v[:, 1])
    L[:, 1] = mass[:]*(x[:, 2]*v[:, 0] + x[:, 0]*v[:, 2])
    L[:, 2] = mass[:]*(x[:, 0]*v[:, 1] + x[:, 1]*v[:, 0])
    Lt = norm_l( np.sum(L, axis=0) )
    # DEBUG_PRINT_V(0, Lt, np.sum( norm_l(L, axis=0) ), np.sum( norm_l(L, axis=1) ))
    E = mass[:]*(0.5*vnorm**2+pot)
    Et = np.sum(E)
    G = 43007.1
    Mt = np.sum(mass)
    return (Lt*np.abs(Et)**0.5)/(G*Mt**2.5)

def calculate_beta_sigma_total_from_xv(xv):
    x = xv[:,0:3]
    v = xv[:,3:6]
    r = norm_l(x, axis=1)+1e-40
    rxy = np.sqrt(x[:,0]**2+x[:,1]**2)
    vr = (x[:,0]*v[:,0]+x[:,1]*v[:,1]+x[:,2]*v[:,2])/r
    sigr_total = np.std(vr)
    vtheta = (x[:,2]*v[:,0]-x[:,0]*v[:,2])/(r*np.sqrt(1-(x[:,2]/r)**2))
    sigtheta_total = np.std(vtheta)
    vphi = ( (x[:,0]*v[:,1]-x[:,1]*v[:,0])*x[:,0]**2/(x[:,0]**2+x[:,1]**2+1e-40) - (x[:,0]*v[:,0]+x[:,1]*v[:,1])*x[:,2]/r )/rxy
    sigphi_total = np.std(vphi)
    return 1.-(sigtheta_total**2+sigphi_total**2)/sigr_total**2

def split_various_bin_1d_percentile(xdata, ydata, yerror_data=None, percentile_split=[0., 0.2, 0.98, 1.], 
    N_bins_arr=[None,1000,None], is_geomspace=False
): #used to fit
    s = np.argsort(xdata)
    xs = xdata[s]
    ys = ydata[s]
    yedata = None
    if yerror_data is None: #if None, it will not be used
        yedata = np.ones_like(ydata)
    else:
        yedata = yerror_data
    ye_s = yedata[s]
    nd = len(xs)
    npct = len(percentile_split)
    N_each_percent = np.zeros(npct-1).astype(int)
    cut_each_percent = np.zeros(npct-1)
    xd1 = np.array([])
    yd1 = np.array([])
    yed1 = np.array([])
    for i in np.arange(npct-1):
        indexd = floor_int(percentile_split[i]*nd)
        indexu = floor_int(percentile_split[i+1]*nd)
        # DEBUG_PRINT_V(0, npct, nd, indexd, indexu, "npct")
        xsp, ysp, yesp= None, None, None
        if i==npct-1:
            xsp = xs[indexd:]
            ysp = ys[indexd:]
            yesp = ye_s[indexd:]
        else:
            xsp = xs[indexd:indexu]
            ysp = ys[indexd:indexu]
            yesp = ye_s[indexd:indexu]
        cut_each_percent[i] = xs[indexu-1]
        xdp, ydp, yedp = None, None, None
        if N_bins_arr[i] is None:
            xdp, ydp, yedp = xsp, ysp, yesp
        else:
            xspb = None
            if is_geomspace:
                xspb = np.geomspace(np.min(xsp), np.max(xsp), N_bins_arr[i])
            else:
                xspb = np.linspace(np.min(xsp), np.max(xsp), N_bins_arr[i])
            xdp, ydp, yedp = get_mean_ydata_by_xdata_1d_bin(xsp, ysp, xspb, yerror_data=yesp)
        N_each_percent[i] = len(xdp)
        # DEBUG_PRINT_V(1, "xdp", indexd, indexu, N_bins_arr[i])
        xd1 = np.append(xd1, xdp)
        yd1 = np.append(yd1, ydp)
        yed1 = np.append(yed1, yedp)
    return np.array(xd1), np.array(yd1), np.array(yed1), cut_each_percent, N_each_percent

def get_mean_ydata_by_xdata_1d_bin(xdata, ydata, xdata_1d_bin_le, yerror_data=None): #used to fit
    '''
    To get the mean xdata and weighted ydata in given bins from original xdata and ydata. 
    Then remove the data points that is nan.
    '''
    s = np.argsort(xdata)
    xs = xdata[s]
    ys = ydata[s]
    yedata = None
    if yerror_data is None: #if None, it will not be used
        yedata = np.ones_like(ydata)
    else:
        yedata = yerror_data
    ye_s = yedata[s]
    n = len(xdata_1d_bin_le)
    xd1 = np.zeros(n-1)
    yd1 = np.zeros(n-1)
    yed1 = np.zeros(n-1)
    for i in np.arange(n-1):
        xsbd = xdata_1d_bin_le[i]
        xsbu = xdata_1d_bin_le[i+1]
        mask = None
        if i==0:
            mask = (xs>=xsbd)
        else:
            mask = (xs>xsbd)
        mask = mask & (xs<=xsbu)
        xsm = xs[mask]
        ysm = ys[mask]
        yesm = ye_s[mask]
        if len(xsm)==0:
            xd1[i] = np.nan
            yd1[i] = np.nan
            yed1[i] = np.nan
        else:
            xd1[i] = np.mean(xsm)
            yd1[i] = np.sum(xsm*ysm)/np.sum(xsm)
            yed1[i] = np.sum(xsm*yesm)/np.sum(xsm)
        # if i==n-2:
        #     # DEBUG_PRINT_V(0, xsbd, xsbu, mask, np.shape(mask))
        #     DEBUG_PRINT_V(1, xsm, np.shape(xsm), ysm, np.shape(ysm))
        #     DEBUG_PRINT_V(0, xd1[i], yd1[i])
    mask1 = np.isfinite(xd1)
    # DEBUG_PRINT_V(0, len(xd1), len(mask1), len(xd1[mask1]))
    return xd1[mask1], yd1[mask1], yed1[mask1]

def floor_int(a):
    return int(np.floor(a))

def split_mesh_by_percentile_2d(particle_data, n_block_1d):
    dim = 2
    spls = np.linspace(0., 1., n_block_1d+1)
    # pers_alldim = np.zeros((dim, n_block_1d+1))
    # asort = list(range(dim))
    # N = np.zeros(dim)
    blocks = list(range(n_block_1d)) #blocks index dim 0
    for j in np.arange(n_block_1d):
        blocks[j] = list(range(n_block_1d)) #blocks index dim 1
    
    i = 0
    for j in np.arange(n_block_1d):
        i = 0
        pers_alldim = np.percentile(particle_data[:,i], spls) #each j in dim 0
        asort = np.argsort(particle_data[:,i])
        N = len(particle_data[:,i])
        # blocks0 = particle_data[asort[i]][floor_int(pers_alldim[i][j]*N[i]):floor_int(pers_alldim[i][j+1]*N[i])]
        blocks0 = particle_data[asort][floor_int(N/n_block_1d)*j:floor_int(N/n_block_1d)*(j+1)]
        # DEBUG_PRINT_V(0, np.shape(particle_data), np.shape(asort), np.shape(particle_data[asort]), np.shape(blocks0))

        for j1 in np.arange(n_block_1d):
            i = 1
            pers_alldim = np.percentile(blocks0[:,i], spls) #each j in dim 0
            asort = np.argsort(blocks0[:,i])
            N = len(blocks0[:,i])
            blocks1 = blocks0[asort][floor_int(N/n_block_1d)*j1:floor_int(N/n_block_1d)*(j1+1)]
            blocks[j][j1] = blocks1
            # DEBUG_PRINT_V(1, np.shape(blocks0), np.shape(asort), np.shape(blocks0[asort]), np.shape(blocks1))
    
    # DEBUG_PRINT_V(0, np.shape(blocks), "shape of blocks")
    return blocks

def split_mesh_by_percentile(particle_data, n_block_1d, dim=2): #?? dim>2
    spls = np.linspace(0., 1., n_block_1d+1)
    pers_alldim = np.zeros(dim, n_block_1d+1)
    blocks = list(range(n_block_1d))
    for i in np.arange(dim):
        blocks[j] = list(range(n_block_1d))
        pers_alldim[i] = np.percentile(particle_data[:,i], spls)
        Ni = len(particle_data[:,i])
        for j in np.arange(n_block_1d):
            asort = np.argsort(particle_data[:,i])
            blocks[j] = particle_data[:,i][asort][floor_int(pers_alldim[i][j]*Ni), floor_int(pers_alldim[i][j+1]*Ni)]
    return 0

def split_mesh_by_tree(particle_data, n_block_1d, dim):
    #??
    return 0

def axis_ratio_by_configuration_data(xyz):
    '''
    @argument xyz are triaxialized xyz-configuration space data.
    '''
    sx = np.sum(xyz[:,0]**2)
    sy = np.sum(xyz[:,1]**2)
    sz = np.sum(xyz[:,2]**2)
    # DEBUG_PRINT_V(1, np.max(xyz[:,0]))
    # DEBUG_PRINT_V(1, np.max(xyz[:,1]))
    # DEBUG_PRINT_V(1, np.mean(xyz[:,0]))
    # DEBUG_PRINT_V(1, np.mean(xyz[:,1]))
    # DEBUG_PRINT_V(0, sx, sy, sz)
    return [1., (sy/sx)**0.5, (sz/sx)**0.5]

def percentiles_by_xv_data(xv, DF=[0.], pers=[0.5, 20., 50., 80., 99.5]):
    '''
    Note whether the input xv is the xv or abs of xv.
    '''
    meds = [
        np.percentile(xv[:,0], pers), 
        np.percentile(xv[:,1], pers), 
        np.percentile(xv[:,2], pers), 
        np.percentile(norm_l(xv[:,0:3], axis=1), pers), 
        np.percentile(xv[:,3], pers), 
        np.percentile(xv[:,4], pers), 
        np.percentile(xv[:,5], pers), 
        np.percentile(norm_l(xv[:,3:6], axis=1), pers), 
        np.percentile(DF, pers)
    ]
    return meds

def percentiles_by_angle_action_data(aa, DF=[0.], pers=[0.5, 20., 50., 80., 99.5]):
    '''
    @argument aa is array of multi-dimentional actions and frequencies (or angles).
    '''
    cl = [0,1,2]
    meds = [
        np.percentile(aa[cl,0], pers), 
        np.percentile(aa[cl,1], pers), 
        np.percentile(aa[cl,2], pers), 
        np.percentile(norm_l(aa[cl,0:3], axis=1), pers), 
        np.percentile(aa[cl,3], pers), 
        np.percentile(aa[cl,4], pers), 
        np.percentile(aa[cl,5], pers), 
        np.percentile(norm_l(aa[cl,3:6], axis=1), pers), 
        np.percentile(DF, pers)
    ]
    return meds

def averages_by_angle_action_data(aa, DF=[0.]):
    cl = [0,1,2]
    ms = [
        np.mean(aa[cl,0]), 
        np.mean(aa[cl,1]), 
        np.mean(aa[cl,2]), 
        np.mean(norm_l(aa[cl,0:3], axis=1)), 
        np.mean(aa[cl,3]), 
        np.mean(aa[cl,4]), 
        np.mean(aa[cl,5]), 
        np.mean(norm_l(aa[cl,3:6], axis=1)), 
        np.mean(DF)
    ]
    return ms

def select_min(x, y):
    if x<y:
        return x
    else:
        return y

def select_max(x, y):
    if x>y:
        return x
    else:
        return y

def frac_to_median(arr, axis=None, is_logP1=False):
    a = arr/np.median(arr, axis=axis)
    if is_logP1==True:
        a = log_abs_P1(a)
    return a

def integer_dividing_ceil_int(a, b):
    return int(np.ceil(float(a)/b))

def read_single_comment_of_each_line(file_name, comment_symbol):
    file_handle = open(file_name, mode="r")
    content_string = file_handle.readlines()
    file_handle.close()
    comments = []
    for i in content_string:
        i_comment = i.find(comment_symbol)
        comments.append(i[i_comment:-1])
    # DEBUG_PRINT_V(0, content_string, comments)
    return comments

def generate_actions_grid_3d_3d(bd2, N_grid, tag_func=2):
    bd = bd2
    p_AC = None
    func_AC = None
    if tag_func==0: #linspace
        bin_grid = np.linspace(1./bd, bd, N_grid)
        ps = np.hstack((
            np.array([bin_grid]).T, 
            np.array([bin_grid]).T, 
            np.array([bin_grid]).T, 
            np.ones((len(bin_grid), 3))
        ))
        return ps
    if tag_func==1: #logspace
        bin_grid = np.logspace(np.log10(1./bd), np.log10(bd), N_grid)
        ps = np.hstack((
            np.array([bin_grid]).T, 
            np.array([bin_grid]).T, 
            np.array([bin_grid]).T, 
            np.ones((len(bin_grid), 3))
        ))
        return ps
    elif tag_func==2:
        k, b, h = -5e-4/5e3, 5e-4, 1e1
        bd_most = -b/k
        p_AC = [k/2, b, h]
        #: h is not effective; b_interval[-1] is determinded by 1./F'(xmin) because it is invert-function of F(x)
        func_AC = poly2_1d_accumulate_integrand
        bd = bd_most
        # bd = bd2
    elif tag_func==3:
        p_AC = [1., 1.]
        # func_AC = exp_poly1_1d_accumulate_integrand
        func_AC = exp_poly1_1d_accumulate_integrand_log
        bd = bd2
    else:
        print("No such tag_func provided. Exit.")
        exit(0)

    bin_grid = generate_space_1d_by_accumulte(1./bd, bd, N_grid, func_AC=func_AC, p=p_AC)
    bin_grid_2 = np.linspace(bd_most, bd2, int(N_grid/10+1))
    bin_grid = np.hstack(( bin_grid[:-1], bin_grid_2 ))
    # bin_grid = np.hstack((bin_grid, bin_grid_2[1:]))

    ps = np.hstack((
        np.array([bin_grid]).T, 
        np.array([bin_grid]).T, 
        np.array([bin_grid]).T, 
        np.ones((len(bin_grid), 3))
    ))

    # print(np.shape(ps))
    # bin_grid_interval = bin_grid[1:]-bin_grid[:-1]
    # add.DEBUG_PRINT_V(1, bin_grid, bin_grid_interval)
    # plt.scatter(np.arange(len(bin_grid)), bin_grid)
    # plt.scatter(np.arange(len(bin_grid_interval)), bin_grid_interval)
    # # plt.yscale("log")
    # plt.show()
    return ps

def exp_poly1_1d(x, p):
    k, b = p[0], p[1]
    return np.exp(-k*x)*b

def exp_poly1_1d_accumulate_integrand(x, p):
    k, b = p[0], p[1]
    return np.exp(-k*x-1.)*(-b/k)

def exp_poly1_1d_accumulate_integrand_log(x, p):
    k, b = p[0], p[1]
    return np.log( (np.exp(-k*x)-1.)*(-b/k) )

def poly2_1d_accumulate_integrand(x, p):
    a2, a1, a0 = p[0], p[1], p[2]
    # add.DEBUG_PRINT_V(0, a2, a1)
    return a2*x*x + a1*x +a0

def generate_space_1d_by_DF(
    xmin, xmax, N, func_DF=None, p=None
): #bad result, long tail??
    """
    @func_dscrp: name of selected function.
    @p: parameters input.
    """
    b = np.linspace(xmin, xmax, N)
    if func_DF is None: #linspace
        b = b
    else:
        xs = np.linspace(0., 1., N)
        fi = func_DF(xmin, p)
        ff = func_DF(xmax, p)
        C = scipy.integrate.quad(func_DF, xmin, xmax, args=p)[0]
        # DEBUG_PRINT_V(0, xs, fi, ff, C)
        if not (np.isfinite(C) and fi>=0. and ff>=0.):
            print("The distribution function cannot be normalized. Exit.")
            exit(0)
        for n in np.arange(N):
            gridvalue = xs[n]
            def func_accumulate_root(x, p):
                return scipy.integrate.quad(func_DF, xmin, x, args=p)[0]/C - gridvalue
            #: [learn code]: python can do def function in def function
            sol = scipy.optimize.fsolve(func_accumulate_root, x0=xmin, args=p, maxfev=5000)
            #: [learn code]: x0 is the initial guess, return len(x0), each is root near any of x0
            b[n] = sol
            f_accu = scipy.integrate.quad(func_DF, 0., sol, args=p)[0]/C
            DEBUG_PRINT_V(1, n, gridvalue, sol, f_accu)
    return b

def generate_space_1d_by_accumulte(
    xmin, xmax, N, func_AC=None, p=None
):
    """
    @func_dscrp: name of selected function.
    @p: parameters input.
    """
    b = np.linspace(xmin, xmax, N)
    if func_AC is None: #linspace
        b = b
    else:
        xs = np.linspace(0., 1., N)
        fi = func_AC(xmin, p)
        ff = func_AC(xmax, p)
        C = ff-fi
        DEBUG_PRINT_V(1, xmin, xmax, fi, ff, C)
        if not (np.isfinite(C) and C>0.): #finite positive(??) slowly increaing function #and fi>0.
            print("The distribution function cannot be normalized. Exit.")
            exit(0)
        sol_ref = xmin
        for n in np.arange(N):
            gridvalue = xs[n]
            def func_accumulate_root(x, p):
                return (func_AC(x, p)-fi)/C - gridvalue
            #: [learn code]: python can do def function in def function
            sol = scipy.optimize.root(func_accumulate_root, x0=sol_ref, args=p).x
            # sol = scipy.optimize.fsolve(func_accumulate_root, x0=sol_ref, args=p, maxfev=5000)
            #: [learn code]: x0 is the initial guess, return len(x0), each is root near any of x0
            sol_ref = sol+1e-20
            b[n] = sol
            f_accu = (func_AC(sol, p)-fi)/C
            # add.DEBUG_PRINT_V(1, [n, gridvalue], [sol, f_accu])
    return b

def change_weight_of_fitting_knn(x, pers=[5., 95.], N_neighbour=32):
    # '''
    # Note: It has changed input.
    # '''
    n = len(x)
    if not (len(pers)==2 and n*(pers[-1]-pers[0])>N_neighbour):
        print("change_weight_of_fitting_knn(): Bad boundary. Exit.")
        exit(0)
    cl_sort = np.argsort(x)
    # xresort = x[cl_sort]
    ii = (n*np.array(pers)/100.).astype(int)
    cl_1 = np.arange(0,ii[0])
    cl_bd = np.arange(ii[0], ii[1], N_neighbour) #linear
    cl_2 = np.arange(ii[1],n)
    cl_newweight = np.hstack(( cl_1, cl_bd, cl_2 ))
    return cl_sort, cl_newweight

def logspace_1d_by_boundaries(xmin, xmax, count):
    x1min = np.log10(xmin)
    x1max = np.log10(xmax)
    return np.logspace(x1min, x1max, count)

def check_x_un_finite(x, symbol=""):
    for j in np.arange(len(x)):
        if ~np.isfinite(sum(x[j])):
            print("There are not finite value. Exit.")
            DEBUG_PRINT_V(0, x[j], j, symbol)

def screen_remove_far_back(h, log10f, h_far=2e5, is_mask_log10f=True):
    # log10h = np.log10(h)
    # hmed = np.median(log10h)
    n = len(h)
    mask = (h>h_far)
    # mask = (h<1e2)
    if is_mask_log10f:
        mask = mask & (log10f>-10.)
    mask = ~np.array(mask)
    h_screen, log10f_screen = h[mask], log10f[mask]
    return h_screen, log10f_screen

def screen_boundary_some_cols(AA, cols_judge, bd_low, bd_up, value_discard=None):
    '''
    Warning: It is not deep copy and it may change the input.
    '''
    n1 = len(AA)
    n2 = len(cols_judge)
    mask = (np.ones(n1)>0.) #Trues of shape (n1,)
    for i in np.arange(n2):
        mask = mask & ( (AA[:, cols_judge[i]]>bd_low) & (AA[:, cols_judge[i]]<bd_up) ) #here use "&" instead of "and"
    condition_list = np.where(mask)[0]
    condition_list_not = np.where(~mask)[0]
    if value_discard is not None:
        AA[condition_list_not] = value_discard #change the input value
    return AA[condition_list], condition_list, condition_list_not

def judge_read_line(linestr):
    if linestr==False: #EOF
        return 0
    if linestr=="\n": # or linestr==None #empty line
        return -1
    if len(linestr)<1:
        return -2
    elif linestr[0]=="#" or linestr[0]=="/" or linestr[0]=="%": #notes
        return 1
    elif (48<=ord(linestr[0]) and ord(linestr[0])<=57) \
        or (linestr[0] in [".", "+", "-"]): #might usual number
        return 2
    else:
        return 3 #other string

def swap_return(a,b):
    a_ = b
    b_ = a
    return a_, b_

def sort_by_standard(Target_array, Standard_array, is_from_small_to_large=True):
    orders = 1.
    if not is_from_small_to_large:
        orders = -1.
    S = Standard_array*orders
    sortlist = np.argsort(S)
    T = Target_array[sortlist] #anyaxis??
    S = Standard_array[sortlist]
    return T, S

def generate_sample_square(n1=10,n2=10,ls=10.):
    x1 = np.arange(n1)*ls*6/n1
    x1 -= np.max(x1)/2
    x2 = np.arange(n2)*ls*6/n2
    x2 -= np.max(x2)/2
    X, Y = np.meshgrid(x1, x2)
    # Z = ls-X-Y
    Z = np.zeros(X.shape)
    return X,Y,Z
    
def merge_array_by_hstack(array_list):
    '''
    @param array_list: arrays list with same length of rank, like [a[:,0:N],b[:,0:N]]
    '''
    # l = len(array_list)
    A = array_list[0]
    if len(np.shape(A))==1:
        A = np.array([ A ]).T
    for a in array_list[1:]:
        b = a
        if len(np.shape(b))==1:
            b = np.array([ b ]).T
        A = np.hstack(( A, b ))
    return A

def neighbourAverage_bin(x, which_column=-1, N_neighbour=10, is_all=True): #x,y: 1d-arrays with same length
    '''
    To return the local average of nearest points by deviding bins for 2-d data by a certain column. 
    Apart for this function, k-neareast neighbour method is alternative.
    @param x: (N,M) array, each x[:,i] (i:0~M) column is a parameter.
    @param which_column: int, which column is the standard of sorting.
    @param N_neighbour: int, how many points in a bin, bias behind.
    '''
    sh = x.shape
    if not( len(sh)==2 ):
        print("x: should be 2d-array data.")
        return x
    if not( N_neighbour>0 ):
        print("N_neighbour: should >0.")
        return x
    N = sh[0]
    M = sh[1]
    # if not N==len(y):
    #     print("Unequal lengthes of x and y variables.")
    #     return 0

    ind = np.argsort(x[:,which_column])
    x_ = x[ind]
    
    xla = 0
    xls = 0
    if is_all==0:
        N_blocks = int(np.ceil(float(N)/N_neighbour))
        xla = np.zeros((N_blocks,M))
        xls = np.zeros((N_blocks,M))
        i = 0
        for i in np.arange(N_blocks):
            down = N_blocks*i
            up = N_blocks*(i+1)
            if down<0:
                down = 0
            if up>=N_blocks:
                up = N_blocks-1
            xla[i] = np.mean(x_[down:up],axis=0)
            xls[i] = np.std(x_[down:up],axis=0)
            i += 1
    else:
        xla = np.zeros((N,M))
        xls = np.zeros((N,M))
        i = 0
        for i in np.arange(N):
            down = i-int(np.floor(N_neighbour/2))
            up = i+int(np.ceil(N_neighbour/2))
            if down<0:
                down = 0
            if up>=N:
                up = N-1
            xla[i] = np.mean(x_[down:up],axis=0)
            xls[i] = np.std(x_[down:up],axis=0)
            i += 1
    return xla, xls, ind

def log_abs_P1(x, logbasis=np.exp(1.), is_keep_sign=True, added=1.): #move 1
    # xn = np.array(x) #change
    xn = np.array(x)*1. #not change
    xn = np.log(np.abs(xn)+added)
    if is_keep_sign:
        xn *= np.sign(x)
        # DEBUG_PRINT_V(0, xn.shape, [xn==0.], xn[xn==0.])
        # xn[xn==0.] = (xn*1.)[xn==0.]
        # xn[xn<0.] = np.log(abs(xn-added))[xn<0.]
        # xn[xn>0.] = np.log(abs(xn+added))[xn>0.]
    return xn/np.log(logbasis)

def logabs(x, logbasis=np.exp(1.), is_keep_sign=False): #cannot distinguish +-
    xn = np.array(x)
    sg = 1.
    if is_keep_sign:
        sg = xn/np.abs(xn)
    return sg*np.log(abs(xn))/np.log(logbasis)

def rate_abs_log(A, B, logbasis=np.exp(1.)):
    return np.log(abs(A/B))/np.log(logbasis)

def relative_error(a, a0):
    return (a-a0)/a0

def norm_l(a, b=0, axis=0, l=2, is_abs=False):
    #without input shape judgement
    #b is no use, should a-b; or a**l+b**l
    if is_abs:
        return ( np.sum(abs(a-b)**l, axis=axis) )**(1./l)
    else:
        return ( np.sum((a-b)**l, axis=axis) )**(1./l)
    # if len(a.shape)<axis+1:
    #     exit(0)
    # x=0
    # if b.shape==(1,): #norm_l of (a: array, b: number)
    #     for i in range(len(a)):
    #         x += (a[i]-b)**l
    #     return x**(1./l)
    # else: #norm_l of (a: array, b; array) #without input shape judgement
    #     for i in range(len(a)):
    #         x += (a[i]-b[i])**l
    #     return x**(1./l)

def DEBUG_PRINT_V(is_continue=0, *V):
    print("DEBUG_PRINT_V(): ")
    for v in V:
        print(v)
    if is_continue==0:
        print("Then exit the prog compulsively.")
        # exit(0)
        # sys.exit(0)
        sys.exit(2)
    else:
        print("go on ...")




#### data process briefly
def plane_2d_fit_by_leastsq(x, y, z):
    '''
    To fit a plane 
    z = a0*x + a1*y +a2 (where a0 = -A/C, a1 = B/C, a2 = -D/C),
    by leastsq by solving matrix.
    @param x, y, z: Data points coordinate. These should have the same size.
    Return: value of np.array([a0, a1, a2])
    '''
    if not ( len(x)==len(z) and len(x)==len(z) ):
        print("Wrong size of input x, y, z points. Please check. Exit.")
        exit(0)
    N = len(x)
    A = np.array([
        [np.sum(x*x), np.sum(x*y), np.sum(x)], 
        [np.sum(x*y), np.sum(y*y), np.sum(y)], 
        [np.sum(x), np.sum(y), N]
    ])
    B = np.array([
        [np.sum(x*z), np.sum(y*z), np.sum(z)], 
    ])
    X = np.linalg.solve(A, B.T)
    return X

def put_certain_3ddatapoints_inaline(x1_points,x2_points,x3_points,fxxx_points, i_x1i,i_x1f,i_x2i,i_x2f,i_x3i,i_x3f):
    if not( len(x1_points.shape)==1 and len(x2_points.shape)==1 and len(x2_points.shape)==1 ):
        print(r"Bad shape of independent variables of data points! Please 1d only.")
        sys.exit(0)
    dim = 3
    if len(fxxx_points.shape)!=dim:
        print(r"Bad shape of dependent variables of data points! Please 3d only.")
        sys.exit(0)
    lx1 = len(x1_points)
    lx2 = len(x2_points)
    lx3 = len(x3_points)
    ly1,ly2,ly3 = fxxx_points.shape
    if not( lx1==ly1 and lx2==ly2 and lx3==ly3 ):
        print(r"Inconsistent lengthes of independent variables (x1,x2,x3) and dependent variables (f(x1,x2,x3)) of data points! Please equally only.")
        sys.exit(0)
    if not( type(i_x1i)==int and type(i_x1f)==int and type(i_x2i)==int and type(i_x2f)==int and type(i_x3i)==int and type(i_x3f)==int ):
        print(r"Bad type of appointment indexes of data points! Please int only.")
        sys.exit(0)
    if not( 1<=i_x1f-i_x1i<=lx1 and 1<=i_x2f-i_x2i<=lx2 and 1<=i_x3f-i_x3i<=lx3 ):
        print(r"Too large length of appointment indexes of data points! Please 1<=xf-xi<=x_points.shape only.")
        sys.exit(0)

    nx1 = i_x1f-i_x1i+1
    nx2 = i_x2f-i_x2i+1
    nx3 = i_x3f-i_x3i+1
    n_data_inaline = nx1*nx2*nx3
    xxx_data_inaline = np.zeros((n_data_inaline,3))
    fxxx_data_inaline = np.zeros(n_data_inaline)

    n = 0
    for n1 in range(nx1):
        for n2 in range(nx2):
            for n3 in range(nx3):
                xxx_data_inaline[n] = np.array([ x1_points[n1], x2_points[n2], x3_points[n3] ])
                fxxx_data_inaline[n] = fxxx_points[ n1, n2, n3 ]
                n+=1

    if not( n==n_data_inaline ):
        print(r"Wrong count caused by this function! Please check.")
        sys.exit(0)
    return xxx_data_inaline, fxxx_data_inaline, n_data_inaline



def divided_bins_123(dataread,colD=0,colD2=3,datarange="x", nmesh=100,nmesh2=10,nmesh3=6,whatbin=1, datamin=1.e-8,datamax=1.e8, \
    param1=0,param2=19.6,param3="purple",param4=1.): #np.histogram2d()

    ## judge input params
    shape1_0 = 40 #80 #the least data points
    shape2_0 = 3 #space dim
    if type(colD)!=int or type(nmesh)!=int or type(nmesh2)!=int or type(nmesh3)!=int:
        print(r"False type of some input arguments!")
        sys.exit(0)
    if len(dataread.shape)!=2:
        print(r"Input data should be a 2d-vector!")
        sys.exit(0)
    shape1,shape2 = dataread.shape
    print(r"The shape of 2d-data = %d %d, colD = %d." % (shape1, shape2, colD) )
    if shape1<shape1_0:
        print(r"Too less data points!")
        sys.exit(0)
    if colD<0 or colD+3>shape2:
        print(r"Bad rank appointment!")
        sys.exit(0)

    ## set inf and nan to 0.
    D_num = np.where(dataread==np.inf, 0., dataread)
    D = np.array(D_num)
    D = abs(D)
    D0 = D[:, colD:colD+3] #data in step 1, none inf or nan

    ## conditions
    err = 1e-10 #should: datamin<=err
    spr = 1e10 #should: datamax>=err
    print("min: ", min(D0[0]))
    LD = len(D0) #the outest dim
    A_judge = np.zeros(LD)
    for i in range(LD):
        # t = norm_l(J0[i])
        # if t>1.e-2 and t<1.e6: #no +-
        t = D[i, colD:colD+3]
        # if 1:
        # if t[0]>0 and t[1]>0 and t[2]>0:
        if t[0]>datamin and t[1]>datamin and t[2]>datamin and t[0]<datamax and t[1]<datamax and t[2]<datamax:
            A_judge[i] = 1
    N_eff = int(sum(A_judge))
    N_record = 0

    ## screen data
    A_nonezero = np.zeros((N_eff,3)) #data in step 2, none zero
    D1_datapoints = np.zeros((N_eff,3))
    D2_datapoints = np.zeros((N_eff,3))
    idx = 0
    for i in range(LD):
        if A_judge[i]:
            A_nonezero[idx] = D[i, colD:colD+3]
            D1_datapoints[idx] = D[i, colD:colD+3]
            D2_datapoints[idx] = D[i, colD2:colD2+3]
            idx += 1
    rate = float(N_eff)/float(LD)
    print(r"The length of data points of group 1 and 2 = %d %d." % (len(D1_datapoints), len(D2_datapoints)))
    print(r"The rate of reliable actions = %d %f." % (N_eff, rate))

    ## devided bins
    if nmesh==0:
        nmesh = int((LD*1./shape1_0)**(1./1))
    x_points = np.zeros((nmesh, 3)) #the ,3 represents 3 indepandent x values in C_3^2=3 combination of 3 coordinates
    fx_points = np.zeros((nmesh, 3)) #corresponded y values
    x_output = np.zeros((nmesh, 3)) #put each data points in a line
    fx_output = np.zeros((nmesh, 3)) #put each data points in a line
    if nmesh2==0:
        nmesh2 = int((LD*1./shape1_0)**(1./2))
    xx_points = np.zeros((nmesh2,nmesh2, 3)) #the ,3 represents 3 x-x values in C_3^2=3 combination of 3 coordinates
    fxx_points = np.zeros((nmesh2,2, 3)) #corresponded y values
    xx_output = np.zeros((nmesh2**2,2, 3)) #put each data points in a line
    fxx_output = np.zeros((nmesh2**2, 3)) #put each data points in a line
    if nmesh3==0:
        nmesh3 = int((LD*1./shape1_0)**(1./3)) #to let the count of data points in a cell not too less
    xxx_points = np.zeros((nmesh3,3)) #the x-x-x values in C_3^3=1 combination of 3 coordinates
    fxxx_points = np.zeros((nmesh3,nmesh3,nmesh3)) #corresponded y values
    xxx_output = np.zeros((nmesh3**3,3)) #put each data points in a line
    fxxx_output = np.zeros(nmesh3**3) #put each data points in a line
    print(r"cell count of 1d (nmesh**1) = %d, of 2d (nmesh2**2) = %d, of 3d (nmesh3**3) = %d." % (nmesh**1, nmesh2**2, nmesh3**3))



    ## 1 dim
    ## y = f1(x1), f1(x2), f1(x3), where $f1(x_i) = \int f3(x_1,x_2,x_3) \mathrm{d}J_\mathrm{i\,other\,1} \mathrm{d}J_\mathrm{i\,other\,2}$, counts 3
    for k in range(3):

        A = A_nonezero[:,k] #data in step 2, each
        ## no use
        if datarange=="x":
            A = A_nonezero[:,k]
        elif datarange=="r":
            A = np.sqrt(A_nonezero[:, 0]**2+A_nonezero[:, 1]**2+A_nonezero[:, 2]**2)
        else:
            print(r"Unknown data range.")
            sys.exit(0)

        A_mesh = np.zeros(nmesh+1) #defined as mesh, counts (nmesh+1)
        DA = np.zeros(nmesh) #difined as bin
        A_count = np.zeros(nmesh) #defined
        A_frequency = np.zeros(nmesh) #defined
        A_distribution = np.zeros(nmesh) #defined

        A_min = min(A)
        A_max = max(A)
        A_median = np.median(A)
        if whatbin==1:
            A_mesh = np.linspace(A_min-err, A_max+err, nmesh+1) #linear, from 0 to max
        elif whatbin==2:
            A_mesh = np.logspace(np.log10(A_min-err), np.log10(A_max+err), nmesh+1) #logarithmic
        elif whatbin==3:
            A_mesh[0:-1] = np.linspace(A_min-err, 3*A_median+err, nmesh) #line, from 0 to median and to max, more data points will not be recorded
            A_mesh[-1] = A_max+err
        elif whatbin==4:
            A_mesh[0:-1] = np.logspace(np.log10(A_min-err), np.log10(3*A_median+err), nmesh) #logarithmic
            A_mesh[-1] = A_max+err
        elif whatbin==5:
            A_mesh = np.linspace(A_min-err, (param4+err), nmesh+1) #line,  setting bounds
        elif whatbin==6:
            A_mesh = np.logspace(np.log10(A_min-err), np.log10((param4+err)), nmesh+1) #logarithmic
        else:
            print(r"No such bin provided! Exit.")
            sys.exit(0)
        DA = A_mesh[1:]-A_mesh[:-1] #the next difference
        print(r"data A: min max median = %f %f %f" % (A_min, A_max, A_median))
        # print(r"data A: DJ1 = ", DA)

        N_record = 0
        for j in A:
            for n in range(nmesh):
                if j>=A_mesh[n] and j<A_mesh[n+1]: #remove zeros before
                    A_count[n]+=1
                    N_record+=1

        A_count*=1.0 #to let int -> float
        A_frequency = A_count/N_record
        A_distribution = A_frequency/DA
        # print(r"count in each bin: ", A_count)
        print(r"the N_record = %d" % N_record)
        print(r"the sum of A_count = %f" % sum(A_count))
        print(r"the sum of A_frequency = %f", sum(A_frequency))

        x_points[:,k] = A_mesh[:-1]+DA/2 #we make a dislocation to set the output datapoint are in the center of the cell
        fx_points[:,k] = A_distribution #it has devided the volumne of the cell so as be a probability density distribution
    x_output = x_points
    fx_output = fx_points


    ## 2 dim
    ## y = f2(x2,x3), f2(x1,x3), f2(x1,x2), where $f2(x_i,x_j) = \int f3(x_1,x_2,x_3) \mathrm{d}x_\mathrm{i,j\,other}$, counts 3
    # #...



    ## 3 dim
    ## y = f3(x1,x2,x3), where $f3(x_1,x_2,x_3) = f3(x_1,x_2,x_3)$, counts 1
    AAA = A_nonezero #data in step 2, each dim -> 3 dim
    A3_mesh = np.zeros((nmesh3+1,3)) #defined as mesh, each dim -> 3 dim
    DA3 = np.zeros((nmesh3,3)) #difined as bin, each dim -> 3 dim
    DADADA = np.zeros((nmesh3,nmesh3,nmesh3)) #defined
    # AAA_mesh = np.zeros((nmesh3+1,nmesh3+1,nmesh3+1))
    AAA_count = np.zeros((nmesh3,nmesh3,nmesh3)) #defined
    AAA_frequency = np.zeros((nmesh3,nmesh3,nmesh3)) #defined
    AAA_distribution = np.zeros((nmesh3,nmesh3,nmesh3)) #defined

    AAA_min = np.array([min(AAA[:,0]), min(AAA[:,1]), min(AAA[:,2])])
    AAA_max = np.array([max(AAA[:,0]), max(AAA[:,1]), max(AAA[:,2])])
    AAA_median = np.array([np.median(AAA[0]), np.median(AAA[1]), np.median(AAA[2])])
    for k in range(3):
        if whatbin==1:
            A3_mesh[:,k] = np.linspace(AAA_min[k]-err, AAA_max[k]+err, nmesh3+1) #linear, from 0 to max
        elif whatbin==2:
            A3_mesh[:,k] = np.logspace(np.log10(AAA_min[k]-err), np.log10(AAA_max[k]+err), nmesh3+1) #logarithmic
        elif whatbin==3:
            A3_mesh[0:-1,k] = np.linspace(AAA_min[k]-err, 3*AAA_median[k]+err, nmesh3) #line, from 0 to median to max, more data points will not be recorded
            A3_mesh[-1,k] = AAA_max[k]+err
        elif whatbin==4:
            A3_mesh[0:-1,k] = np.logspace(np.log10(AAA_min[k]-err), np.log10(3*AAA_median[k]+err), nmesh3) #logarithmic
            A3_mesh[-1,k] = AAA_max[k]+err
        elif whatbin==5:
            A3_mesh[:,k] = np.linspace(AAA_min[k]-err, (param4+err), nmesh3+1) #line, setting bounds
        elif whatbin==6:
            A3_mesh[:,k] = np.logspace(np.log10(AAA_min[k]-err), np.log10((param4+err)), nmesh3+1) #logarithmic
        else:
            print(r"No such bin provided! Exit.")
            sys.exit(0)
        DA3[:,k] = A3_mesh[1:,k]-A3_mesh[:-1,k] #the next difference
        print(r"data AAA's one dim: min max median = %f %f %f" % (AAA_min[k], AAA_max[k], AAA_median[k]))
        # print(r"data A: DJ1 = ", DA3[k])

    N_record = 0
    for aaa in AAA: #fxxx to traverse each data point with reliable actions #remove zeros before #the order is AAA[x1,x2,x3]
        for n1 in range(nmesh3): #x1 ~ k=0
            if aaa[0]>=A3_mesh[n1,0] and aaa[0]<A3_mesh[n1+1,0]:
                for n2 in range(nmesh3): #x2 ~ k=1
                    if aaa[1]>=A3_mesh[n2,1] and aaa[1]<A3_mesh[n2+1,1]:
                        for n3 in range(nmesh3): #x3 ~ k=2
                            if aaa[2]>=A3_mesh[n3,2] and aaa[2]<A3_mesh[n3+1,2]:
                                AAA_count[n1,n2,n3] += 1
                                N_record+=1
            # else ... #else continue is no use

    for n1 in range(nmesh3): #dxdxdx
        for n2 in range(nmesh3):
            for n3 in range(nmesh3):
                DADADA[n1,n2,n3] = DA3[n1,0]*DA3[n2,1]*DA3[n3,2]

    AAA_count*=1.0 #to let int -> float
    AAA_frequency = AAA_count/N_record
    AAA_distribution = AAA_frequency/DADADA
    print(r"the N_record = %d" % N_record)
    print(r"the sum of AAA_count = %f" % sum(sum(sum(AAA_count))))
    print(r"the sum of AAA_frequency = %f" % sum(sum(sum(AAA_frequency))))

    xxx_points = A3_mesh[:-1,:]+DA3/2 #we make a dislocation to set the output datapoint are in the center of the cell
    fxxx_points = AAA_distribution #it has devided the volumne of the cell so as be a probability density distribution
    xxx_output, fxxx_output, N3 = put_certain_3ddatapoints_inaline(xxx_points[:,0],xxx_points[:,1],xxx_points[:,2],fxxx_points, 0,nmesh3-1,0,nmesh3-1,0,nmesh3-1)

    return x_output, fx_output, xx_output, fxx_output, xxx_output, fxxx_output, D1_datapoints, D2_datapoints



def interpolation_Rbf_xxx_yyy_xxx0_k(xxx_knn, fxxx_knn, xxx0, k=10, funcname="gaussian"):

    # ## judge input params
    # we do not do it for efficiency.

    return 0




def fromAAAtoBBB_byinterpolation_atsomepoints(xxx_points, fxxx_points, xxx0, k=10, funcname="gaussian"): #for example, J1J2J3_datapoints~Omg1Omg2Omg3_datapoints to J1J2J3_mesh~Omg1Omg2Omg3_mesh

    # ## judge input params
    if not( len(xxx_points.shape)==2 and len(fxxx_points.shape)==2 ):
        print(r"All the two input datas should be a 2d-vector!")
        sys.exit(0)
    shape11,shape12 = xxx_points.shape
    shape21,shape22 = fxxx_points.shape
    shape31,shape32 = xxx0.shape
    if not( shape12==3 and shape22==3 and shape32==3 ):
        print(r"All the input datas should be a (n,3) vector!")
        sys.exit(0)
    if not( shape11==shape21 ):
        print(r"The two input datas are not the same size!")
        sys.exit(0)

    ## finding K neighbors
    tree = KDTree(xxx_points, leaf_size=40)
    distances, indices = tree.query(xxx0, k=k)
    # print("indices: ", indices)

    ##interpolation
    yyy0 = np.zeros(xxx0.shape)
    L = len(xxx0)
    for l in range(L):
        func1l = Rbf(xxx_points[indices[l],0],xxx_points[indices[l],1],xxx_points[indices[l],2], fxxx_points[indices[l],0], function=funcname)
        func2l = Rbf(xxx_points[indices[l],0],xxx_points[indices[l],1],xxx_points[indices[l],2], fxxx_points[indices[l],1], function=funcname)
        func3l = Rbf(xxx_points[indices[l],0],xxx_points[indices[l],1],xxx_points[indices[l],2], fxxx_points[indices[l],2], function=funcname)
        f1xxx0l = func1l(xxx0[l,0],xxx0[l,1],xxx0[l,2])
        f2xxx0l = func2l(xxx0[l,0],xxx0[l,1],xxx0[l,2])
        f3xxx0l = func3l(xxx0[l,0],xxx0[l,1],xxx0[l,2])
        yyy0[l] = np.array([f1xxx0l,f2xxx0l,f3xxx0l])
    print("yyy0.shape: ", yyy0.shape)

    return yyy0



def divided_bins_1(dataread,colD=0, N_dim=3,datarange="x", nmesh=100,whatbin=1, datamin=1.e-10,datamax=1.e10, param1=0,param2=19.6,param3="purple",param4=1.):

    #### data
    ## set inf and nan to 0.
    D_finite = np.array(np.where(np.isfinite(dataread), dataread,0.)) #remove not finite value
    D_finite = abs(D_finite) #not minus

    D = D_finite #change D
    D0 = D[:,colD:colD+N_dim]

    ## screen data by some conditions
    LD = len(D) #particle count
    D_judge = np.zeros(LD) #where D is on condition below
    err = 1e-10 #should: datamin<=err
    spr = 1e10 #should: datamax>=err
    for i in range(LD):
        t = D0[i]
        if t[0]>datamin and t[1]>datamin and t[2]>datamin and t[0]<datamax and t[1]<datamax and t[2]<datamax:
            D_judge[i] = 1
    N_eff = int(sum(D_judge))

    ## efficent data
    D_eff = np.zeros((N_eff,N_dim)) #efficent data
    A = np.zeros(N_eff)

    N_record = 0
    idx = 0
    for i in range(LD):
        if D_judge[i]:
            D_eff[idx] = D0[i]
            idx += 1
    rate = float(N_eff)/float(LD)
    if idx!=N_eff:
        print(r"Wrong count of efficient particle datas. Please check!")
        sys.exit(0)
    print(r"The rate of reliable actions = %d %f." % (N_eff, rate))

    if datarange=="x": #x-coordiante
        A = D_eff[:, 0]
    elif datarange=="r": #radius
        A = np.sqrt(D_eff[:, 0]**2+D_eff[:, 1]**2+D_eff[:, 2]**2)
    else:
        print(r"Unknown data range.")
        sys.exit(0)
    # print("data: min max: ", min(A), max(A))

    #### bin
    ## devided bins
    if nmesh==0:
        print(r"Invalid value of nmesh, set 100.")
        nmesh = 100
    x_points = np.zeros((nmesh, 3)) #the ,3 represents 3 indepandent x values in C_3^2=3 combination of 3 coordinates
    fx_points = np.zeros((nmesh, 3)) #corresponded y values
    x_output = np.zeros((nmesh, 3)) #put each data points in a line
    fx_output = np.zeros((nmesh, 3)) #put each data points in a line

    ## 1 dim
    for k in range(1):
        A_mesh = np.zeros(nmesh+1) #defined as mesh, counts (nmesh+1)
        DA = np.zeros(nmesh) #difined as bin
        A_count = np.zeros(nmesh) #defined
        A_frequency = np.zeros(nmesh) #defined
        A_distribution = np.zeros(nmesh) #defined

        A_min = min(A)
        A_max = max(A)
        A_median = np.median(A)
        kmedian = 10
        if whatbin==1:
            A_mesh = np.linspace(A_min-err, A_max+err, nmesh+1) #linear, from 0 to max
        elif whatbin==2:
            A_mesh = np.logspace(np.log10(A_min-err), np.log10(A_max+err), nmesh+1) #logarithmic
        elif whatbin==3:
            A_mesh[0:-1] = np.linspace(A_min-err, kmedian*A_median+err, nmesh) #line, from 0 to median and to max, more data points will not be recorded
            A_mesh[-1] = A_max+err
        elif whatbin==4:
            A_mesh[0:-1] = np.logspace(np.log10(A_min-err), np.log10(kmedian*A_median+err), nmesh) #logarithmic
            A_mesh[-1] = A_max+err
        elif whatbin==5:
            A_mesh = np.linspace(A_min-err, (param4+err), nmesh+1) #line,  setting bounds
        elif whatbin==6:
            A_mesh = np.logspace(np.log10(A_min-err), np.log10((param4+err)), nmesh+1) #logarithmic
        else:
            print(r"No such bin provided! Exit.")
            sys.exit(0)

        if datarange=="x":
            DA = A_mesh[1:]-A_mesh[:-1] #the next difference
            # DA = A_mesh[:-1] #max
        if datarange=="r":
            DA = 4.*np.pi/3*(A_mesh[1:]**3-A_mesh[:-1]**3)
        print(r"data A: min max median = %f %f %f" % (A_min, A_max, A_median))
        # print(r"data A: DJ1 = ", DA)

        N_record = 0
        for j in A:
            for n in range(nmesh):
                if j>=A_mesh[n] and j<A_mesh[n+1]: #remove zeros before
                    A_count[n]+=1
                    N_record+=1

        A_count *= 1.0 #to let int -> float
        # A_frequency = A_count/N_record
        A_frequency = A_count/LD
        A_distribution = A_frequency/DA
        print(r"grid: ", A_mesh)
        print(r"cellvolumn in each bin: ", DA)
        print(r"count in each bin: ", A_count)
        print(r"the N_record = %d" % N_record)
        print(r"the sum of A_count = %f" % sum(A_count))
        print(r"the sum of A_frequency = %f", sum(A_frequency))

        x_points[:,k] = A_mesh[:-1] #+DA/2 #we make a dislocation to set the output datapoint are in the center of the cell
        fx_points[:,k] = A_distribution #it has devided the volumne of the cell so as be a probability density distribution
    x_output = x_points
    fx_output = fx_points
    print(x_output[:,0])
    print(fx_output[:,0])
    print(nmesh,len(fx_output))

    return x_output[:,0],fx_output[:,0], D0, D0



def isCondition_boundary(x, datamin,datamax, is_abs=False):
    a = 1
    if is_abs:
        for xi in x:
            a *= (datamin<abs(xi)<datamax)
    else:
        for xi in x:
            a *= (datamin<xi<datamax)
    return a

def screen_boundary_PM(data, datamin,datamax, is_abs=False):
    '''
    To screen data by given boundary, without abs().
    '''
    #### data
    if len(data.shape)!=2:
        print(r"Input data should be a 2d-vector!")
        sys.exit(0)
    shape1,shape2 = data.shape #particle counts and dimension
    # print(r"The shape of 2d-data = %d, %d" % (shape1, shape2) )

    ## set inf and nan to 0.
    # D_finite = np.array(np.where(np.isfinite(data), data,0.)) #remove not finite value
    # D = abs(D_finite) #not minus
    D = data

    ## screen data by some conditions
    D_judge = np.zeros(shape1) #where D is on condition below
    for i in range(shape1):
        x = D[i]
        if isCondition_boundary(x, datamin, datamax, is_abs=is_abs):
            D_judge[i] = 1

    # ## efficent data
    # N_eff = int(sum(D_judge))
    # D_eff = np.zeros((N_eff,shape2)) #efficent data
    # idx = 0
    # for i in range(shape1):
    #     if D_judge[i]:
    #         D_eff[idx] = D[i]
    #         idx += 1
    # rate = float(N_eff)/float(shape1)
    # if idx!=N_eff:
    #     print(r"Wrong count of efficient particle datas. Please check!")
    #     sys.exit(0)
    # print(r"The rate of reliable actions = %d %f." % (N_eff, rate))

    D_listCondition = np.where(D_judge==1)[0]
    D_listConditionNot = np.where(D_judge!=1)[0]
    D_eff = D[D_listCondition]
    # print(D_listConditionNot)
    print(r"screen_boundary(): particle counts of condition and notcondition = %d, %d" %( len(D_listCondition), len(D_listConditionNot) ))

    return D_eff, D_listCondition, D_listConditionNot

def screen_boundary(data, datamin,datamax):
    '''
    To screen data by abs boundary.
    '''
    #### data
    if len(data.shape)!=2:
        print(r"Input data should be a 2d-vector!")
        sys.exit(0)
    shape1,shape2 = data.shape #particle counts and dimension
    print(r"The shape of 2d-data = %d, %d" % (shape1, shape2) )

    ## set inf and nan to 0.
    # D_finite = np.array(np.where(np.isfinite(data), data,0.)) #remove not finite value
    # D = abs(D_finite) #not minus
    D = data

    ## screen data by some conditions
    D_judge = np.zeros(shape1) #where D is on condition below
    for i in range(shape1):
        x = D[i]
        if isCondition_boundary(x, datamin,datamax, is_abs=True):
            D_judge[i] = 1

    # ## efficent data
    # N_eff = int(sum(D_judge))
    # D_eff = np.zeros((N_eff,shape2)) #efficent data
    # idx = 0
    # for i in range(shape1):
    #     if D_judge[i]:
    #         D_eff[idx] = D[i]
    #         idx += 1
    # rate = float(N_eff)/float(shape1)
    # if idx!=N_eff:
    #     print(r"Wrong count of efficient particle datas. Please check!")
    #     sys.exit(0)
    # print(r"The rate of reliable actions = %d %f." % (N_eff, rate))

    D_listCondition = np.where(D_judge==1)[0]
    D_listConditionNot = np.where(D_judge!=1)[0]
    D_eff = D[D_listCondition]
    # print(D_listConditionNot)
    print(r"screen_boundary(): particle counts of condition and notcondition = %d, %d" %( len(D_listCondition), len(D_listConditionNot) ))

    return D_eff, D_listCondition, D_listConditionNot



##:: local main()
if __name__ == '__main__':

    p = None

    # data = np.loadtxt("../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/txt/snapshot_000.txt")
    # l = spin_L(data[:,0:6], data[:,8], data[:,14])
    # print(l) #?? spin_L_old of snapshot_000 is 0.4, the expected value is 0.1

    # N = 100
    # L = 4
    # k = 2
    # # xxx = np.random.random((N,3))
    # # xxx0 = np.random.random((L,3))
    # xxx = np.arange(N*3)
    # xxx = xxx.reshape(N,3)
    # xxx0 = np.array([[1.,2.,3.]])
    # tree = KDTree(xxx, leaf_size=40)
    # distances, indices = tree.query(xxx0, k=k)
    # print("indices: ", indices)