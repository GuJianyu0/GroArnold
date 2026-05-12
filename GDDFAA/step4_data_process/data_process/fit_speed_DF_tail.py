#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Fit a combined core-tail speed DF model to 1D speed PDF data,
using mpfit for non-linear least squares. Compares the fitted
model with the PDF estimated via kNN-KDE.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import mpfit

import analysis_data_distribution as ads



def kde_bandwidth_1d(speed_data, grid, bandwidth):
    '''
    kNN-KDE for 1D speed data
    '''
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(speed_data[:, None])
    log_dens = kde.score_samples(grid[:, None])
    return np.exp(log_dens)

def f_Gaussian(v, par):
    A_G  = par[0]
    sigma  = par[1]

    # Gaussian core
    f_G = A_G * (2.0*np.pi * sigma**2)**(-1.5) * np.exp(-0.5 * (v/sigma)**2)
    
    return 4.0*np.pi * v**2 * f_G

def f_powerlaw(v, par):
    vc     = par[0]
    gamma  = par[1]
    A_pl   = par[2]
    epsilon= par[3]
    
    # Power-law tail (note: translated by epsilon)
    # f_pl = A_pl * ( 1 + (v/vc)**2 )**(-gamma)
    f_pl = A_pl / vc**3 * ( epsilon + (v/vc)**2 )**(-gamma)

    return 4.0*np.pi * v**2 * f_pl

def g_func(v, k, v1):
    """Logistic blending function"""
    return 1.0 / (1.0 + np.exp(-k * (v - v1)))

def f_comb(v, par):
    """
    Compute the composite PDF.
    
    Parameters:
       v   : array of speed values.
       par : parameter vector [sigma, vc, gamma, k, v1, A_pl, epsilon]
    
    Returns:
       Model PDF values at v.
    """
    sigma  = par[0]
    vc     = par[1]
    gamma  = par[2]
    k      = par[3]
    v1     = par[4]
    A_pl   = par[5]
    epsilon= par[6]
    
    # Gaussian core
    # f_G = 0.
    f_G = (2.0*np.pi * sigma**2)**(-1.5) * np.exp(-0.5 * (v/sigma)**2)
    
    # Power-law tail (note: translated by epsilon)
    # f_pl = 0.
    # f_pl = A_pl * ( 1 + (v/vc)**2 )**(-gamma)
    f_pl = A_pl / vc**3 * ( epsilon + (v/vc)**2 )**(-gamma)
    
    # Blending function: logistic form
    # g = 0.5
    g = g_func(v, k, v1)
    
    # Combined model
    f_combined = 4.0*np.pi * v**2 * ( f_G * (1-g) + f_pl * g )
    
    return f_combined

def model_resid(par, fjac=None, x=None, y=None):
    """Residual function for mpfit. Returns model - data."""
    model = f_comb(x, par)
    return [0, model - y]

def model_resid_Gaussian(par, fjac=None, x=None, y=None):
    """Residual function for mpfit. Returns model - data."""
    model = f_Gaussian(x, par)
    return [0, model - y]

def model_resid_powerlaw(par, fjac=None, x=None, y=None):
    """Residual function for mpfit. Returns model - data."""
    model = f_powerlaw(x, par)
    return [0, model - y]

def fit_speed_df_mpfit(v_grid, pdf_data, func_model, quiet=0):
    """
    Fit the combined speed DF model to the observed PDF data using mpfit.
    The free parameter vector is: [sigma, vc, gamma, k, v1, A_pl, epsilon].
    
    Returns:
       bestfit: best-fit parameter vector,
       mp: mpfit result object.
    """
    # Define parinfo for each parameter.
    # You can adjust the initial values and bounds as needed.
    parinfo = [
        {"value": 180.0,  "fixed": 0, "limited": [1, 1], "limits": [20.0, 500.0]},   # sigma
        {"value": 300.0,  "fixed": 0, "limited": [1, 1], "limits": [40.0, 1000.0]},   # vc
        {"value": 4.5,    "fixed": 0, "limited": [1, 1], "limits": [1.0, 20.0]},       # gamma
        {"value": 0.1,   "fixed": 0, "limited": [1, 1], "limits": [1e-20, 1e4]},      # k
        {"value": 200.0,  "fixed": 0, "limited": [1, 1], "limits": [1e2, 1e4]},      # v1
        {"value": 0.1,    "fixed": 0, "limited": [1, 1], "limits": [1e-20, 1e6]},         # A_pl
        {"value": 0.1,   "fixed": 1, "limited": [1, 1], "limits": [1e-6, 10.0]}        # epsilon: 0.1
        # {"value": 1e-6,    "fixed": 0, "limited": [1, 1], "limits": [1e-20, 1e20]},         # A_pl
        # {"value": 1,   "fixed": 1, "limited": [1, 1], "limits": [1e-6, 10.0]}        # epsilon: 0.1
    ]
    
    # Set up mpfit using parinfo.
    m = mpfit.mpfit(func_model, parinfo=parinfo, functkw={'x': v_grid, 'y': pdf_data}, maxiter=200, quiet=quiet)
    
    if m.status <= 0:
        print("Fitting failed with error:", m.errmsg)
    bestfit = m.params
    return bestfit, m

def fit_speed_df_mpfit_Gaussian(v_grid, pdf_data, func_model, quiet=0):
    """
    Fit the combined speed DF model to the observed PDF data using mpfit.
    The free parameter vector is: [sigma, vc, gamma, k, v1, A_pl, epsilon].
    
    Returns:
       bestfit: best-fit parameter vector,
       mp: mpfit result object.
    """
    # Define parinfo for each parameter.
    # You can adjust the initial values and bounds as needed.
    parinfo = [
        {"value": 1.,    "fixed": 0, "limited": [1, 1], "limits": [0.0, 1e6]},         # A_G
        {"value": 180.0,  "fixed": 0, "limited": [1, 1], "limits": [10.0, 500.0]},   # sigma
    ]
    
    # Set up mpfit using parinfo.
    m = mpfit.mpfit(func_model, parinfo=parinfo, functkw={'x': v_grid, 'y': pdf_data}, maxiter=200, quiet=quiet)
    
    if m.status <= 0:
        print("Fitting failed with error:", m.errmsg)
    bestfit = m.params
    return bestfit, m

def fit_speed_df_mpfit_powerlaw(v_grid, pdf_data, func_model, quiet=0):
    """
    Fit the combined speed DF model to the observed PDF data using mpfit.
    The free parameter vector is: [sigma, vc, gamma, k, v1, A_pl, epsilon].
    
    Returns:
       bestfit: best-fit parameter vector,
       mp: mpfit result object.
    """
    # Define parinfo for each parameter.
    # You can adjust the initial values and bounds as needed.
    parinfo = [
        {"value": 300.0,  "fixed": 0, "limited": [1, 1], "limits": [20.0, 1000.0]},   # vc
        {"value": 4.5,    "fixed": 0, "limited": [1, 1], "limits": [1.0, 10.0]},       # gamma
        {"value": 0.1,    "fixed": 0, "limited": [1, 1], "limits": [0.0, 1e6]},         # A_pl
        {"value": 1e-1,   "fixed": 0, "limited": [1, 1], "limits": [-50.0, 50.0]}        # epsilon
    ]
    
    # Set up mpfit using parinfo.
    m = mpfit.mpfit(func_model, parinfo=parinfo, functkw={'x': v_grid, 'y': pdf_data}, maxiter=200, quiet=quiet)
    
    if m.status <= 0:
        print("Fitting failed with error:", m.errmsg)
    bestfit = m.params
    return bestfit, m

def fit_and_plot_speed_df(v_grid, pdf_data, save_path):
    """
    Fit the combined speed DF model to the observed 1D PDF and plot the results.
    
    Parameters:
       v_grid   : 1D numpy array of speeds.
       pdf_data : 1D numpy array of observed PDF values at v_grid.
       save_path: filename to save the plot.
       
    Returns:
       bestfit: best-fit parameter vector,
       mp_result: mpfit result object.
    """
    quiet = 1

    model = model_resid_Gaussian
    bestfit, mp_result = fit_speed_df_mpfit_Gaussian(v_grid, pdf_data, model, quiet=quiet)
    print("Best-fit parameters of Gaussian:")
    print("A_G =", bestfit[0])
    print("sigma =", bestfit[1])
    pdf_fit_Gaussian = f_Gaussian(v_grid, bestfit)
    
    model = model_resid_powerlaw
    bestfit, mp_result = fit_speed_df_mpfit_powerlaw(v_grid, pdf_data, model, quiet=quiet)
    print("Best-fit parameters of powerlaw:")
    print("vc =", bestfit[0])
    print("gamma =", bestfit[1])
    print("A_pl =", bestfit[2])
    print("epsilon =", bestfit[3])
    pdf_fit_powerlaw = f_powerlaw(v_grid, bestfit)
    
    model = model_resid
    bestfit, mp_result = fit_speed_df_mpfit(v_grid, pdf_data, model, quiet=quiet)
    # bestfit[4] *= 0.5 #debug
    # bestfit[5] *= 2.0
    print("Best-fit parameters of combined:")
    print("sigma =", bestfit[0])
    print("vc =", bestfit[1])
    print("gamma =", bestfit[2])
    print("k =", bestfit[3])
    print("v1 =", bestfit[4])
    print("A_pl =", bestfit[5])
    print("epsilon =", bestfit[6])
    pdf_fit = f_comb(v_grid, bestfit)
    
    ## Plot DF    
    sigma, vc, gamma, k, v1, A_pl, epsilon = bestfit

    # Compute individual components on v_grid.
    A_G = 1.
    pars_component_Gaussian = A_G, sigma
    F_G = 4.0 * np.pi * v_grid**2 * f_Gaussian(v_grid, pars_component_Gaussian)
    pars_component_powerlaw = vc, gamma, A_pl, epsilon
    F_pl = 4.0 * np.pi * v_grid**2 * f_powerlaw(v_grid, pars_component_powerlaw)
    g = g_func(v_grid, k, v1)
    F_comb = F_G * (1 - g) + F_pl * g
    
    # Component fractions:
    frac_G = F_G * (1 - g) / F_comb
    frac_pl = F_pl * g / F_comb
    frac_total = frac_G+frac_pl

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Subfigure 1: Plot each curve vs v.
    ax[0].plot(v_grid, frac_G, 'g-', lw=2, label="Gaussian Core Fraction")
    ax[0].plot(v_grid, frac_pl, 'b-', lw=2, label="Power-law Tail Fraction")
    ax[0].plot(v_grid, frac_total, 'r-', lw=2, label="Total Fraction")
    ax[0].plot(v_grid, g, 'k--', lw=2, label="Blending function, g(v)")
    ax[0].set_xlabel("Speed, v")
    ax[0].set_ylabel("Component Fraction")
    ax[0].set_title("Component Fraction vs. v")
    ax[0].legend()
    ax[0].grid(True)
    
    # Subfigure 2: Plot DF vs v.
    ax[1].plot(v_grid, pdf_data, 'ko', markersize=4, label='Data (KDE)')
    ax[1].plot(v_grid, pdf_fit_Gaussian, 'g-', lw=2, label='Fitted Model of Gaussian')
    ax[1].plot(v_grid, pdf_fit_powerlaw, 'b-', lw=2, label='Fitted Model of powerlaw')
    ax[1].plot(v_grid, pdf_fit, 'r-', lw=2, label='Fitted Model of combined')
    ax[1].set_xlabel("Speed, v")
    ax[1].set_ylabel("PDF")
    ax[1].set_title("Combined Velocity DF Fit")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path+"fitted_velocity_DF.eps", format="eps", bbox_inches='tight')
    # plt.show()
    plt.close()
    print("Plot fitted DF to:", save_path)
    
    return bestfit, mp_result

def load_speed_fitted_paramters():
    return 0

def fit_speed_DF_tail_each_workflow():

    ## 1. data
    data_path = "../data/samples_simulated/snapshot_010_example_NFW.txt"
    # data_path = "../data/samples_simulated/snapshot_080_example_merge.txt"
    # data_path = "../data/snapshot_%03d_big.txt"%(snapshot_ID)
    data = np.loadtxt(data_path, dtype=float)
    N_particles = len(data)
    positions = data[:, 0:3]
    # positions = np.random.random((N, 3))
    # positions = np.random.normal(0, 1, (N, 3))
    velocities = data[:, 3:6]
    m = data[:, 8]
    pers = [50., 90., 95., 97., 98., 99., 99.5]
    meds = ads.percentiles_by_xv_data(data[:,0:6], pers=pers)
    ads.DEBUG_PRINT_V(1, pers, meds[3], meds[7], "xvmeds")
    r_max = 100.
    n_average = N_particles/(4.*np.pi/3*r_max**3) #0.238
    print("mean numbder density per kpc: {}".format(n_average))
    # beta_sph, beta_cyl = velocity_dispersion(None, xv)
    # ads.DEBUG_PRINT_V(1, beta_sph, beta_cyl, "beta_sph, beta_cyl")
    positions -= np.mean(positions, axis=0) #translate x center
    velocities -= np.mean(velocities, axis=0) #translate v center
    r_data = ads.norm_l(positions, axis=1)
    V_data = ads.norm_l(velocities, axis=1)
    r_mean = np.mean(r_data)
    V_max = np.max(V_data)
    t_cross = np.mean(r_data/V_data)

    # is_screen_radius = True
    is_screen_radius = False
    if is_screen_radius: #to screen particles whose radius within 0.5 ~ 1.5 r_mean
        idx_range = np.where((r_data>0.5*r_mean) & (r_data<1.5*r_mean))[0]
        xv_range = data[idx_range]
        positions = xv_range[:, 0:3]
        velocities = xv_range[:, 3:6]
        r_data = ads.norm_l(positions, axis=1)
        V_data = ads.norm_l(velocities, axis=1)
        ads.DEBUG_PRINT_V(1, len(idx_range), "idx_range")
    
    # For demonstration, generate a sample speed dataset from a 3D Maxwellian:
    N = 10000
    sigma_true = 180.0
    # Direct 3D sampling: each component ~ N(0, sigma_true) → speeds follow Maxwellian.
    velocities = np.random.normal(0, sigma_true, size=(N, 3))
    speeds_Gaussian = np.linalg.norm(velocities, axis=1)
    
    ## 2. fit
    # Fit the combined model to the sample's speed PDF.
    # speeds = speeds_Gaussian
    speeds = V_data
    # bandwidth = 3.
    bandwidth = 6.
    v_min, v_max = np.percentile(speeds, [0.5, 99.5])
    v_grid = np.linspace(v_min, v_max, 500)
    pdf_data = kde_bandwidth_1d(speeds, v_grid, bandwidth=bandwidth)
    
    save_path = "../data/examples_vel/"
    fit_and_plot_speed_df(v_grid, pdf_data, save_path)

    return 0



if __name__ == "__main__":

    fit_speed_DF_tail_each_workflow()
