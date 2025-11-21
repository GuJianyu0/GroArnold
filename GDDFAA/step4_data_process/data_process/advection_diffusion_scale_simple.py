#!/usr/bin/env python
# -*- coding:utf-8 -*-

# ============================================================================================
# Description: A wrapper to simply estimate coarse-grained advection-diffusion scale.
# Author: Jianyu Gu
# ============================================================================================

##[] modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as spopt
from scipy.fftpack import fft, ifft

import RW_data_CMGD as rdc
import analysis_data_distribution as add
import galaxy_models as gm
import triaxialize_galaxy as tg
import KDTree_python as kdtp
import fit_galaxy_wrapper as fgw
import scipy.integrate as spitg



##[] functions
## estimation
N_mesh_eachdim = 51

def get_actions_and_Kernel_debug():

    gm_name = ""
    galaxymodel_name = "galaxy_general"+gm_name+"/"
    snapshot_Id = 160

    # bd = 1e5/2
    # bd = 2e5
    # bd = 1e4
    bd = 1e6
    # bd = 2e6
    bd_min = 1e-2
    filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/"\
        +"snapshot_160.action.method_all.txt"
    
    xv, AA_cl, mass = rdc.read_actions(filename, bd=bd, bd_min=bd_min, is_angles=False, actionmethod="AA_TF_DP")
    add.DEBUG_PRINT_V(1, len(AA_cl), "len(AA_cl)")

    cols = [0,1,2]
    KD = kdtp.KDTree_galaxy_particles(AA_cl[:,cols], weight_extern_instinct=mass)
    
    # DF = KD.density_SPH(AA_cl[:,cols]) #some are None
    # DF_log10 = np.log10(DF)
    # add.DEBUG_PRINT_V(1, len(DF_log10), "len(DF_log10)")
    
    # fJ_m = 1.
    # Jl = AA_cl[:,0]
    # Jm = AA_cl[:,1]
    # Jn = AA_cl[:,2]
    # fJ = DF_log10
    # Jl_m = np.median(AA_cl[:,0])
    # Jm_m = np.median(AA_cl[:,1])
    # Jn_m = np.median(AA_cl[:,2])
    # Jl_frac_m = Jl/Jl_m
    # Jm_frac_m = Jm/Jm_m
    # Jn_frac_m = Jn/Jn_m
    # fJ_frac_m = fJ/fJ_m
    # add.DEBUG_PRINT_V(1, Jl_m, Jm_m, Jn_m, fJ_m, "medians")
    return AA_cl, KD

def DF_KDE_some(AA_ponits, KD):
    cols = [0,1,2]
    return KD.density_SPH(AA_ponits[:,cols])

def integrate_mesh_1d_by_data_trapezoid(
    xm, ym, bound_1d_down=np.inf, bound_1d_up=np.inf, N_grid_1d=20
):
    n = len(xm)
    I = 0.
    for i in np.arange(n-1):
        I += (ym[i+1]+ym[i])*(xm[i+1]-xm[i])/2
    return I

def integrate_mesh_3d_by_data_trapezoid(
    xm, ym, bound_1d_down=np.inf, bound_1d_up=np.inf, N_grid_1d=20
):
    # spitg.nquad() #3d func one by one
    # spitg.trapezoid()
    xm1 = xm[0]
    xm2 = xm[1]
    xm3 = xm[2]
    n1, n2, n3 = np.shape(ym)
    # add.DEBUG_PRINT_V(1, xm1[:,0,0], xm1[0,:,0], xm1[0,0,:])
    # add.DEBUG_PRINT_V(1, xm2[:,0,0], xm2[0,:,0], xm2[0,0,:])
    # add.DEBUG_PRINT_V(0, xm3[:,0,0], xm3[0,:,0], xm3[0,0,:])
    # add.DEBUG_PRINT_V(0, ym)
    N_allintervals = (n1-1)*(n2-1)*(n3-1)
    I = 0.
    for i in np.arange(n1-1):
        for j in np.arange(n2-1):
            for k in np.arange(n3-1):
                hx = xm1[i,j+1,k]-xm1[i,j,k] #?? ix, iy, iz: order [1,0,2]
                hy = xm2[i+1,j,k]-xm2[i,j,k] #\ ix, iy, iz
                hz = xm3[i,j,k+1]-xm3[i,j,k]
                I_cubic = ( ym[i+1,j+1,k+1] + ym[i+1,j+1,k] 
                    + ym[i+1,j,k+1] + ym[i+1,j,k]
                    + ym[i,j+1,k+1] + ym[i,j+1,k]
                    + ym[i,j,k+1] + ym[i,j,k]
                    ) * hx*hy*hz/8
                I += I_cubic
                # add.DEBUG_PRINT_V(1, ym[i+1,j+1,k+1], ym[i,j,k], hx, hy ,hz)
                # print("grid_interval {i=%d, j=%d, k=%d} of %d"%(i, j, k, N_allintervals))
    return I

## integrate
def grids_integration_single_peak_by_func(
    x, N_grid_1d=20, dim=3, #\param N_grid_1d should be 4 times of a int
    x_bound_d=None, x_bound_u=None, 
    x_scale_d=None, x_scale_u=None, 
    x_peak_d=None, x_peak_u=None, 
):
    N1 = np.ceil(N_grid_1d/4.)
    logspace_1d_d1 = np.logspace(np.log10(x_bound_d[0]), np.log10(x_scale_d[0]), N1) #?? manual mesh
    logspace_1d_d2 = np.logspace(np.log10(x_scale_d[0]), np.log10(x_peak_d[0]), N1)
    logspace_1d_u1 = np.logspace(np.log10(x_peak_u[0]), np.log10(x_scale_u[0]), N1)
    logspace_1d_u2 = np.logspace(np.log10(x_scale_u[0]), np.log10(x_bound_u[0]), N1)
    grid_0 = np.hstack((logspace_1d_d1, logspace_1d_d2[1:], logspace_1d_u1[1:], logspace_1d_u2[1:]))
    grid_1 = grid_0
    grid_2 = grid_0
    mg1, mg2, mg3 = np.meshgrid(grid_0, grid_1, grid_2) #each mgi is a N1*N2*N3 array of coor i 
    return mg1, mg2, mg3

def get_value_at_grids(
    mgs, func, args=None, dim=3
):
    mg1 = mgs[0]
    mg2 = mgs[1]
    mg3 = mgs[2]
    y = mg1
    ng = len(mg1[0][0])
    for i in np.arange(ng):
        for j in np.arange(ng):
            for k in np.arange(ng):
                y[i][j][k] = func([mg1[i][j][k], mg2[i][j][k], mg3[i][j][k]], args) #func(points, args) at [ix][jy][kz]
    return y

def odd_grids_1d_int(n):
    '''
    The moving is consist with fft.
    '''
    arr = np.arange(n)
    # for i in np.arange(n/2,n):
    #     arr[i] = i-n/2
    # for i in np.arange(0,n-n/2):
    #     arr[i] = i-n/2
    # return arr
    return arr-int(np.floor(n/2))

def vector_1d_to_vector_nd_h(vecs_tuple):
    N_vecs_list = len(vecs_tuple)
    vecs_list_merge = list(range(N_vecs_list))
    for i in np.arange(N_vecs_list):
        vecs_list_merge[i] = np.array([vecs_tuple[i]]).T
    return np.hstack(tuple(vecs_list_merge))

def mesh_cubic_3dN_to_1dN(m3d):
    '''
    Convert from a 3d cubic mesh to a 1d vector.
    '''
    m1d = np.reshape(m3d, (np.size(m3d)))
    return m1d

def mesh_cubic_1dN_to_3dN(m1d, n3d_tuple):
    m3d = np.reshape(m1d, n3d_tuple)
    return m3d

# def translation_cubic_grids_half_3d(g): #file add
#     g1 = g
#     center = np.mean(g[0][0][:])
#     g1 = g1[0][0][:]-center
#     # ... for i,j,k
#     return g1

# def convolution_numerical(DF, h):
#     # from astropy.convolution import convolve, Gaussian2DKernel
#     # gauss_kernel = convolve(3.)
#     # gauss_kernel = Gaussian2DKernel(3.)
#     # DF1 = convolve(DF, gauss_kernel)

#     # from scipy import signal
#     # signal.convolve2d()
#     return 0.

def tophat_kernel_3d(xcg, d_xm_1d):
    n = int(np.ceil(xcg/d_xm_1d))
    TK = np.ones((n,n))/n**2
    return TK

def Gauss_kernel_2d(sigma, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size//2
    if sigma<=0:
        sigma = ((kernel_size-1)*0.5-1)*0.3+0.8
    s = sigma**2
    sum_val =  0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i-center, j-center
            kernel[i, j] = np.exp(-(x**2+y**2)/2*s)
            sum_val += kernel[i, j]
    kernel = kernel/sum_val
    return kernel

def convolve_3d(img, kernel, mode="same"):
    for i in np.arange(img.shape[0]):
        for j in np.arange(img.shape[1]):
            oneline = img[i,j,:]
            img[i,j,:] = np.convolve(oneline, kernel[0], mode=mode)
    for i in np.arange(img.shape[1]):
        for j in np.arange(img.shape[2]):
            oneline = img[:,i,j]
            img[:,i,j] = np.convolve(oneline, kernel[1], mode=mode)
    for i in np.arange(img.shape[0]):
        for j in np.arange(img.shape[2]):
            oneline = img[i,:,j]
            img[i,:,j]=np.convolve(oneline, kernel[2],mode=mode)
    return  img

def integrand_on_grids_3d_with_discrete_Fourior(yg, p, d_xm_1d):
    convolution = Gauss_kernel_2d(p, 3)
    # convolution = tophat_kernel_3d(p, d_xm_1d) #??
    yg_conv = convolve_3d(yg, convolution, "same") #?? conserve
    n1, n2, n3 = np.shape(yg)
    mg1, mg2, mg3 = np.meshgrid(
        odd_grids_1d_int(n2), odd_grids_1d_int(n1), odd_grids_1d_int(n3)
    ) #real number #the order should be [b,a,c] ~ [a,b,c] #?? translation grids
    # fr_yg = fft(yg_conv)/(n1*n2*n3) #Fourior transformation #?? 1./N**3
    fr_yg = fft(yg_conv)
    integrand = np.abs(fr_yg)**2*(mg1**2+mg2**2+mg3**2) #?? discrete to continus in Fourior
    # add.DEBUG_PRINT_V(1, yg, yg_conv)
    # add.DEBUG_PRINT_V(1, mg1, fr_yg, integrand)
    return integrand, mg1, mg2, mg3

def integrand_on_grids_1d_with_diffusion(x_i, x_f, n=N_mesh_eachdim, lam=None):
    x = np.linspace(x_i, x_f, n)
    integrand = x**0.5/(1.+lam*np.exp(x)) #?? overflow
    # add.DEBUG_PRINT_V(1, x, lam, integrand, "v2_diffusion") #?? integrand is too small
    return integrand, x

## DF
def coarse_grained_scale_init(scale_galaxy, tag):
    return 0.

def diffusion_tenser_1():
    return 0.

def rootfunc_coarse_grained_scale(xcg, vcg, xs, vs, eta, beta, lam, xm=None, DFm=None, d_xm_1d=None):
    # rootfunc = xcg*f_v + vcg*f_x - pdD/pdv
    if not xcg/xs<1:
        print("Too small xcg: {not xcg/xs<1}. Exit.")
        print("xcg = %f, xs=%f"%(xcg, xs))
        exit(0)
    lnA = np.log(2*xs/xcg) #??> hypothesis
    #: integrate mesh of Fourior is in the after line
    Fr_DFm, im1, im2, im3 = integrand_on_grids_3d_with_discrete_Fourior(DFm, xcg, d_xm_1d) #?? DF is too various
    im = [im1, im2, im3]
    Chi = integrate_mesh_3d_by_data_trapezoid(im, Fr_DFm)
    im_f = 0.5*beta*eta*vcg**2
    I_diffu, im1 = integrand_on_grids_1d_with_diffusion(0., im_f, N_mesh_eachdim, lam)
    Itg = integrate_mesh_1d_by_data_trapezoid(im1, I_diffu)
    Itg = (im_f-0.)**(3./2) #?? debug
    coef_D = 16.*2.**0.5*np.pi**2*gm.G**2*xcg**3*vcg**3*lnA/(eta**0.5*beta**2.5*vs)
    # add.DEBUG_PRINT_V(1, xs, vs, eta, beta, lam)
    # add.DEBUG_PRINT_V(1, Chi, im_f, np.shape(I_diffu), lnA, coef_D, Itg)
    D = coef_D*Itg
    rootfuncvalue = vcg - Chi*D #?? the last two are too small
    add.DEBUG_PRINT_V(1, coef_D, Itg) #\ 1e-100
    add.DEBUG_PRINT_V(1, vcg, Chi, D, rootfuncvalue, "rootfuncvalue")
    return rootfuncvalue

def display_rootfunc_1d1d(xlist, rootfunc, *args):
    N_xlist = len(xlist)
    ylist = np.zeros(N_xlist)
    for i in np.arange(N_xlist):
        ylist[i] = rootfunc(xlist[i], args)
    return 0.

def DF_LB1_versus_energy(e, beta, eta0, lam, epslm): #?? em=v2/2+phi>0
    # return eta0/(1. + lam*np.exp(-beta*eta0*epslm))
    return (np.exp(-beta*eta0*e) - np.exp(-beta*eta0*epslm))/(lam + np.exp(-beta*eta0*epslm))

def fit_DF_leastsq(DF_func, xdata, ydata, p0=None, bounds=None):
    popt, pcov = spopt.curve_fit(DF_func, xdata, ydata, p0=p0, bounds=bounds)
    beta, eta0, lam, epslm = popt
    if not beta>0.:
        print("Bad fit value: {not beta>0.}. Exit.")
        print("beta = %f"%(beta))
        exit(0)
    if not eta0>0.:
        print("Bad fit value: {not eta0>0.}. Exit.")
        print("eta0 = %f"%(eta0))
        exit(0)
    return popt, pcov



##[] simulation
def reidentity_num(data):
    return 

def get_num_wave(data):
    return 

def plot_num_wave(data):
    return 


## compare
def plot_by_coarse_grained_scale_paramter():
    return 0.

def transpose_1d_to_column(x):
    return np.array([x]).T

def set_color_by_init(n, nb=4):
    '''
    The data_after[argsort_IDs_after] is the array sorted by IDs_after from 0 to N-1, 
    such that color_after[[10,11]] is the color info of particle with IDs=[10,11] 
    (other info are the same index); 
    so the init color (unchanged) color_before should be set before sorted by IDs 
    (color_after and color_before is the same because they all sorted by the unchanged IDs), 
    and color_before can be put in a (N,m) shape array;
    do not forget to sort the data_after or its subset by IDs when compared with color info.
    '''
    colorinfo = np.zeros(n)
    b = int(np.floor(n/nb))
    colorinfo[0:b] = 0
    colorinfo[b:b*2] = 1
    colorinfo[b*2:b*3] = 2
    colorinfo[b*3:] = 3
    # add.DEBUG_PRINT_V(1, b, colorinfo)
    return colorinfo

def get_mean_interval_in_new_convection():
    return 0.

def calculation_to_motion_for_background():
    #a #b
    return 0.

def neural_network_on_lumps_DF_in_CDE():
    return 0.

def neural_network_on_evolution_effect_in_CDE():
    return 0.

def parameter_space_DF_action(): #as w1
    return 0.

def recognize_lumps_size_from_data(): #as w2
    return 0.



##[] main
if __name__ == "__main__":

    filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/"\
        +"snapshot_160.action.method_all.txt"
    filename1 = filename+".small"
    # data = np.loadtxt(filename)
    # np.savetxt(filename1, data[::1000, :])
    data = np.loadtxt(filename1)
    xv = data[:, 0:6]
    x = data[:,0:3]
    rx = add.norm_l(x, axis=1)
    v = data[:,3:6]
    rv = add.norm_l(v, axis=1)
    IDs = data[6]
    mass = np.ones(len(data))
    potd = data[:,11]
    energy = 0.5*add.norm_l(v,axis=1)**2 + potd
    energy_T = np.array([energy]).T
    percentiles = add.percentiles_by_xv_data(xv)
    add.DEBUG_PRINT_V(1, percentiles, "percentiles")
    
    '''
    KD_x = kdtp.KDTree_galaxy_particles(x, weight_extern_instinct=mass)
    xm_1d = np.linspace(-150., 150., N_mesh_eachdim)
    xm = (xm_1d, xm_1d, xm_1d)
    xm_stroke_1 = mesh_cubic_3dN_to_1dN(xm[0])
    xm_stroke_2 = mesh_cubic_3dN_to_1dN(xm[1])
    xm_stroke_3 = mesh_cubic_3dN_to_1dN(xm[2])
    xm_stroke = vector_1d_to_vector_nd_h((xm_stroke_1, xm_stroke_2, xm_stroke_3))
    DF_xm_stroke = KD_x.density_SPH(xm_stroke)
    DF_xm = mesh_cubic_1dN_to_3dN(DF_xm_stroke, (N_mesh_eachdim, N_mesh_eachdim, N_mesh_eachdim))
    # KD_v = kdtp.KDTree_galaxy_particles(v, weight_extern_instinct=mass)
    # vm = xm
    # DF2m = KD_v.density_SPH(vm)
    KD_e = kdtp.KDTree_galaxy_particles(energy_T, weight_extern_instinct=mass)
    em = energy
    DF_em = KD_e.density_1d(energy_T)
    add.DEBUG_PRINT_V(1, np.shape(DF_xm), "np.shape(DF_xm)")
    add.DEBUG_PRINT_V(1, np.shape(em), "np.shape(em)")
    add.DEBUG_PRINT_V(1, np.shape(DF_em), "np.shape(DF_em)")
    # exit(0)

    p0_em = [1., 1., 1., 1.]
    bounds_em = ([1e-10, 1e-10, 1e-10, 1e-10], [1e10, 1e10, 1e10, 1e10])
    par_LB, cov_LB = fit_DF_leastsq(DF_LB1_versus_energy, em, DF_em, p0=p0_em, bounds=bounds_em)
    beta, eta0, lam, epslm = par_LB #??
    DF_em_fit = DF_LB1_versus_energy(em, beta, eta0, lam, epslm)
    add.DEBUG_PRINT_V(1, par_LB, "par_LB")
    plt.scatter(em, DF_em, marker=".", label="em~DF_em_data")
    plt.scatter(em, DF_em_fit, marker="x", label="em~DF_em_fit")
    plt.ylim(-0.01, 0.07)
    plt.legend()
    # plt.show()
    plt.close()
    eta = eta0 #??
    xs = np.mean(rx)
    vs = np.mean(rv)
    vcg = vs #??> hypothesis
    # exit(0)

    N_xcg = 10
    xcg_trys = np.logspace(-2., 1.5, N_xcg)
    deltas = np.ones_like(xcg_trys)*1e10
    d_xm_1d = xm_1d[1]-xm_1d[0]
    for i in np.arange(N_xcg):
        deltas[i] = rootfunc_coarse_grained_scale(xcg_trys[i], vcg, xs, vs, eta, beta, lam, xm, DF_xm, d_xm_1d)
        add.DEBUG_PRINT_V(1, i, xcg_trys[i], deltas[i], "xcg_trys~deltas")
    print("xcg_trys : ", xcg_trys)
    print("deltas   : ", deltas)
    # xcg_trys = [
    #     1.00000000e-02, 2.44843675e-02, 5.99484250e-02, 1.46779927e-01, 
    #     3.59381366e-01, 8.79922544e-01, 2.15443469e+00, 5.27499706e+00, 
    #     1.29154967e+01, 3.16227766e+01
    # ]
    # deltas = [
    #     -3.72964373e+17, -4.91632207e+15, -7.10641939e+13, -1.06377700e+12, 
    #     -1.61011581e+10, -2.22294174e+08, -8.21070493e+04,  2.20975369e+02, 
    #     2.20975369e+02,  2.20975369e+02
    # ]
    xplot = add.log_abs_P1(xcg_trys)
    yplot = add.log_abs_P1(deltas)
    plt.plot(xplot, yplot, marker=".", label="xcg_trys~deltas")
    # plt.xscale("log") #??
    # plt.yscale("log")
    plt.legend()
    plt.show()
    plt.close()
    exit(0)
    # '''
    
    filename_after = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/"\
        +"snapshot_80.action.method_all.txt"
    filename_after1 = filename_after+".small"
    # data_after1 = np.loadtxt(filename_after)
    # np.savetxt(filename_after1, data_after1[::1000, :])
    data_after = np.loadtxt(filename_after1)
    IDs_after = data_after[:,6]
    x_after = data_after[:,0:3]

    x_a = x_after #(3,N)
    IDs_x_a = data_after[:,6]
    x_a_by_x_a_0 = x_a[np.argsort(x_a[:,0])] #(3,N)
    color_arr = set_color_by_init(len(x_a), 4)
    x_a_supper_by_IDs = np.hstack((x_a_by_x_a_0, transpose_1d_to_column(IDs_x_a), transpose_1d_to_column(color_arr)))
    x_a_supper_by_IDs = x_a_supper_by_IDs[np.argsort(IDs_x_a)] #(5,N)
    
    x_b = x #(3,N)
    IDs_x_b = data[:,6]
    IDs_x_b_by_IDs = IDs_x_b[np.argsort(IDs_x_b)]
    x_b_by_IDs = x_b[np.argsort(IDs_x_b)] #(3,N)
    color_arr_by_IDs = color_arr[np.argsort(IDs_x_b)]
    x_b_supper_by_IDs = np.hstack((x_b_by_IDs, transpose_1d_to_column(IDs_x_b_by_IDs), transpose_1d_to_column(color_arr_by_IDs)))

    cm = plt.cm.get_cmap("gist_rainbow") #rainbow
    plt.subplot(1,2,1)
    f_color = x_a_supper_by_IDs[:,-1]
    plt.scatter(x_a_supper_by_IDs[:,0], x_a_supper_by_IDs[:,1], s=0.2, c=f_color, cmap=cm, label="t = 0.8 Gyr")
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    plt.xlim(-600., 600.)
    plt.ylim(-600., 600.)
    plt.legend()
    plt.subplot(1,2,2)
    f_color = x_b_supper_by_IDs[:,-1]
    plt.scatter(x_b_supper_by_IDs[:,0], x_b_supper_by_IDs[:,1], s=0.2, c=f_color, cmap=cm, label="t = 1.6 Gyr")
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    plt.xlim(-600., 600.)
    plt.ylim(-600., 600.)
    plt.legend()
    plt.show()
    plt.close()
    exit(0)





    # ##examples
    # N = 100
    # x = np.linspace(0., 2., N)
    # y = np.linspace(0., 2., N)
    # xm, ym = np.meshgrid(x, y)
    # freq_x = np.arange(len(x))
    # freq_y = np.arange(len(y))
    # freq_xm, freq_ym = np.meshgrid(freq_x, freq_y)
    # add.DEBUG_PRINT_V(1, np.shape(xm), "a")

    # def func1(x1, y1=0., args=None):
    #     return 1.5*np.sin(( x1+y1 )*20.*np.pi) + 0.5 #period 20., total 100.
    # um = func1(xm, ym)
    # print(um)
    # add.DEBUG_PRINT_V(1, np.shape(um), "b")
    # F_um = fft(um)/N
    # add.DEBUG_PRINT_V(1, np.shape(F_um), "c")
    # abs_um=np.abs(F_um)
    # ang_um=np.angle(F_um)
    # add.DEBUG_PRINT_V(1, np.shape(abs_um), "d")
    # add.DEBUG_PRINT_V(1, np.shape(ang_um), "e")
    
    # vw = func1(x)
    # F_vw = fft(vw)/N
    # abs_vw = np.abs(F_vw)
    # ang_vw = np.angle(F_vw)
    # # add.DEBUG_PRINT_V(0, np.shape(x), np.shape(F_vw), np.shape(abs_vw))

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection=None)
    # # ax.scatter(x, F_vw, label="Original")
    # ax.scatter(freq_x, abs_vw, label="Fourior: abs")
    # ax.scatter(freq_x, ang_vw, label="Fourior: ang")
    # plt.legend()
    # plt.show()
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection="3d")
    # # ax.scatter(xm, ym, um, label="Original: u=u(x,y)")
    # # ax.scatter(freq_xm, freq_ym, F_um, label="Fourior: F_u=abs_u(k_x,k_y)", color="red")
    # ax.scatter(freq_xm, freq_ym, abs_um, label="Fourior: abs_u=abs_u(k_x,k_y)")
    # ax.scatter(freq_xm, freq_ym, ang_um, label="Fourior: ang_u=abs_u(k_x,k_y)")
    # plt.legend()
    # plt.show()
    # plt.close()
    # exit(0)



    # ##examples
    # def func1(x, p):
    #     return p[0]*x*x + p[1]*x + p[2]
    # def func1r(p, x, y):
    #     return func1(x, p) - y
    # def func1_log(x, p):
    #     return np.log10(func1(x, p))
    # def func1r_log(p, x, y):
    #     return np.log10(func1(x, p)) - y
    
    # x = np.linspace(0.1, 10.0, 100)
    # y = 0.5*x*x +10. + 0.2*(np.random.random(100)-0.5)
    # y_log = np.log10(y)
    # print(x, y, y_log)

    # res = spopt.leastsq(func1r, x0=(0.4, 0.1, 0.), args=(x, y))
    # print(res[0])

    # res = spopt.leastsq(func1r_log, x0=(0.4, 0.1, 0.), args=(x, y_log))
    # print(res[0])

    # CFMM = fgw.Minimize_fit()
    # res = CFMM.leastsq_residual_log(func1_log, x0=(0.4, 0.1, 0.), args=(x, y_log))
    # print(res[0])

    # ##other
    # #采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
    # x=np.linspace(0,1,1400) 
    
    # #设置需要采样的信号，频率分量有180，390和600
    # y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)
    
    # yy=fft(y)      #快速傅里叶变换
    # yreal = yy.real    # 获取实数部分
    # yimag = yy.imag    # 获取虚数部分
    
    # yf=abs(fft(y))    # 取绝对值
    # yf1=abs(fft(y))/len(x)   #归一化处理
    # yf2 = yf1[range(int(len(x)/2))] #由于对称性，只取一半区间
    
    # xf = np.arange(len(y))  # 频率
    # xf1 = xf
    # xf2 = xf[range(int(len(x)/2))] #取一半区间
    
    # plt.subplot(221)
    # plt.plot(x[0:50],y[0:50])
    # plt.title('Original wave')
    
    # plt.subplot(222)
    # plt.plot(xf,yf,'r')
    # plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B') #注意这里的颜色可以查询颜色代码表
    
    # plt.subplot(223)
    # plt.plot(xf1,yf1,'g')
    # plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')
    
    # plt.subplot(224)
    # plt.plot(xf2,yf2,'b')
    # plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')
    # plt.show()

    # Fs = 150.0;     # sampling rate采样率
    # Ts = 1.0/Fs;    # sampling interval 采样区间
    # t = np.arange(0,1,Ts)  # time vector,这里Ts也是步长
    
    # ff = 25;     # frequency of the signal
    # y = np.sin(2*np.pi*ff*t)
    
    # n = len(y)     # length of the signal
    # k = np.arange(n)
    # T = n/Fs
    # frq = k/T     # two sides frequency range
    # frq1 = frq[range(int(n/2))] # one side frequency range
    
    # YY = np.fft.fft(y)   # 未归一化
    # Y = np.fft.fft(y)/n   # fft computing and normalization 归一化
    # Y1 = Y[range(int(n/2))]
    
    # fig, ax = plt.subplots(4, 1)
    
    # ax[0].plot(t,y)
    # ax[0].set_xlabel('Time')
    # ax[0].set_ylabel('Amplitude')
    
    # ax[1].plot(frq,abs(YY),'r') # plotting the spectrum
    # ax[1].set_xlabel('Freq (Hz)')
    # ax[1].set_ylabel('|Y(freq)|')
    
    # ax[2].plot(frq,abs(Y)) # plotting the spectrum
    # ax[2].set_xlabel('Freq (Hz)')
    # ax[2].set_ylabel('|Y(freq)|')
    
    # ax[3].plot(frq1,abs(Y1)) # plotting the spectrum
    # ax[3].set_xlabel('Freq (Hz)')
    # ax[3].set_ylabel('|Y(freq)|')
    
    # plt.show()
    # exit(0)
