#!/usr/bin/env python
# -*- coding:utf-8 -*-
#In[]
##:: import
# from fcntl import F_GET_SEALS
import numpy as np
import analysis_data_distribution as add
import affine_trans_group as atg



#In[]
#########################################
##:: useful variables
#########################################

## constants
G = 43007.1 #(10^-10*M_\odot, kpc, km/s)
## the sun
Sun_xv = np.array([8.178, 0.1, 0.225,    25.13, 180.01, 14.95]) #the Sun xv
Sun_R0 = 8.2 #kpc
Sun_vc = 220. #km/s
## the Milky-Way (a group of reference values)
MW_mass_total = 137. #10^-10*M_\odot
MW_halo_NFW_rho_scale = 0.000891 #10M, kpc, km/s #NFW
MW_halo_length_scale = 19.6 #kpc
MW_halo_action_scale = np.sqrt(G*MW_mass_total*MW_halo_length_scale)
## system
Dim = 3 #only Dim===3 provided


#### params
paramsresults_name = [ #paramsresults all
    "scale_length_comp1_median_direct", "flatx_comp1_direct", 
    "flaty_comp1_direct", "flatz_comp1_direct", "spin_L_old_direct", 
    "beta_sigma_total_direct", "scale_length_comp1_fit", "scale_density_comp1_fit", 
    "fitmodelfunc_name_xv_fit", "powerS1_fit", "powerA1_fit", "powerB1_fit", 
    "powerA2_fit", "powerB2_fit", 
    "Jsum_median_direct", "Jl_median_direct", "Jm_median_direct", 
    "Jn_median_direct", "Ol_median_direct", "Om_median_direct", "On_median_direct", 
    "actions_ratio_m_direct", "actions_ratio_n_direct", "spin_L_processed_direct", 
    "fitmodelfunc_name_AA_fit", "J1_scale_fit", "J2_scale_fit", "J3_scale_fit", 
    "J4_scale_fit", "poly_coeff_k1_fit", "poly_coeff_k2_fit", 
    "actions_coef_free_m_fit", "actions_coef_free_n_fit", "powerAA_E1_fit", 
    "powerAA_P1_fit", "powerAA_P2_fit", "powerAA_P3_fit", "powerAA_P4_fit", 
    "spin_L_parameter_P1_direct", "rotate_sig_parameter_P1_direct", 
    "beta_parameter_P1_direct", "beta_z_parameter_P1_direct", 
    "spin_L_parameter_P2_direct", "rotate_sig_parameter_P2_direct", 
    "beta_parameter_P2_direct", "beta_z_parameter_P2_direct"
]

params_name = [ #paramsresults to display
    "scale_length_comp1_median_direct", 
    "flaty_comp1_direct", "flatz_comp1_direct", 
    "scale_length_comp1_fit", "scale_density_comp1_fit", 
    "powerA1_fit", "powerB1_fit", 

    "Jsum_median_direct", 
    "Jl_median_direct", "Jm_median_direct", 
    "Jn_median_direct", 
    "Ol_median_direct", "Om_median_direct", 
    "On_median_direct", 
    "actions_ratio_m_direct", "actions_ratio_n_direct", 
    "J1_scale_fit", "J2_scale_fit", 
    "J3_scale_fit", "J4_scale_fit", 
    "poly_coeff_k1_fit", #"poly_coeff_k2_fit", 
    #"actions_coef_free_m_fit", "actions_coef_free_n_fit", 
    #"powerAA_E1_fit", 
    "powerAA_P1_fit", "powerAA_P2_fit", 
    "powerAA_P3_fit", #"powerAA_P4_fit", 
    
    "spin_L_parameter_P1_direct", "rotate_sig_parameter_P1_direct", 
    "beta_parameter_P1_direct", "beta_z_parameter_P1_direct", 
    "spin_L_parameter_P2_direct", "rotate_sig_parameter_P2_direct", 
    "beta_parameter_P2_direct", "beta_z_parameter_P2_direct"
]



#### some examples of fitting_model wrappers, users can add and then pick by YAML file
# #: old version
# #: default, for halo, fh_MPLTF_log10
# fitmodelfunc = None
# is_fit_1d = True

# #: a simpler fit model, to be filled, fitmodelfunc = [[0,1,2,3,4]]
# p0_mp = np.array([
#     1.e+04, 2.e+04, 6.e+04, 
#     1.e+00, 2.e+00, 1.2e+00, 
# ])
# bounds_mp = [
#     [1e-1, 2e-10, 6e1, -200., -200., -200.,  ],
#     [1e6,  2e6,   6e6,  200.,  200.,  2000., ]
# ]
# fitmodelfunc = [
#     [
#         gm.fh_MPLF_log10, "fh_MPLF_log10", 
#         [
#             "scale_free_1", "scale_free_2", #"scale_free_3", 
#             "power_free_1", "power_free_2", "power_free_3", 
#             "log_penalty"
#         ], 
#         None, None, [p0_mp, bounds_mp]
#     ]
# ]
# is_fit_1d = True

# #: for disk, fh_MPLF_lmn_log10
# p0_mp = np.array([
#     1.e+04, 2.e+04, #6.e+04, 
#     1.e+00, 2.e+00, 1.2e+00, 
#     0.2, 0.1, 
# ])
# bounds_mp = [
#     [1e-1, 2e-10, #6e1, 
#      -200., -200., -200., 0., 0., ],
#     [1e6,  2e6,   #6e6,  
#      200.,  200.,  2000., 1., 1., ]
# ]
# fitmodelfunc = [
#     [
#         gm.fh_MPLF_lmn_log10, "fh_MPLF_lmn_log10", 
#         [
#             "scale_free_1", "scale_free_2", #"scale_free_3", 
#             "power_free_1", "power_free_2", "power_free_3", 
#             "coef_free_1", "coef_free_2", 
#             "log_penalty"
#         ], 
#         None, None, [p0_mp, bounds_mp]
#     ]
# ]
# is_fit_1d = False

# #: for each, fh_MPLTF_lmn_log10
# p0_mp = np.array([
#     1.e+04, 2.e+04, 6.e+04, 8.e+05, 
#     1.e+00, 2.e+00, 1.2e+00, 
#     -0.5, 0.2, 0.1, 
# ])
# bounds_mp = [
#     [1e-1, 2e-10, 6e1, 8e-10, -200., -200., -200.,  -1000., 0.1, 0.1, ],
#     [1e6,  2e6,   6e6, 8e7,    200.,  200.,  2000.,   100., 10., 10., ]
# ]
# fitmodelfunc = [
#     [
#         gm.fh_MPLTF_lmn_log10, "fh_MPLTF_lmn_log10", 
#         [
#             "scale_free_1", "scale_free_2", "scale_free_3", "scale_free_4", 
#             "power_free_1", "power_free_2", "power_free_3", 
#             "coef_free_1", "coef_free_2", "coef_free_3", 
#             "log_penalty"
#         ], 
#         None, None, [p0_mp, bounds_mp]
#     ]
# ]
# is_fit_1d = False

#: examples
def fitting_model_MPLF_freq():
    """
    1D-in-h model: fh_MPLF_log10(h, ...)
    A simpler fit model.
    - combination: J\cdot\Omega via rateF1 (no args)
    - params_name left as None (auto-named later), it will auto-name ["pn1_fit", ..., "log_penalty"] after fit.
    """
    p0_mp = np.array([
        1.e+04, 2.e+04, 
        1.e+00, 2.e+00, 1.2e+00, 
    ])
    bounds_mp = [
        [1e-1, 2e-10, 
         -200., -200., -200., ],
        [1e6,  2e6, 
         200.,  200.,  2000., ]
    ]
    fitmodelfunc = [
        [
            fh_MPLF_log10,
            "fh_MPLF_log10",
            None,        # params_name -> None (auto)
            None,        # fixed params (unused here)
            None,        # fit value result to record
            [p0_mp, bounds_mp],        # p0/bounds come from MG/Wrap_func path; keep None
        ]
    ]
    combo = {"name": "AA_combination_sumWeightFrequency_rateF1", "args": []}
    return {"is_fit_1d": True, "combination": combo, "fitmodelfunc": fitmodelfunc}

def fitting_model_MPLF_freefixed():
    """
    Free/fixed-coefficient version using the l,m,n form:
    fh_MPLF_lmn_log10(J,lmn,...); we still collapse to h for binning/plot,
    but the combination stored to metadata will be free-coef style.
    """
    p0_mp = np.array([
        1.e+04, 2.e+04, 
        1.e+00, 2.e+00, 1.2e+00, 
        0.2, 0.1, 
    ])
    bounds_mp = [
        [1e-1, 2e-10, 
            -200., -200., -200., 0., 0., ],
        [1e6,  2e6, 
            200.,  200.,  2000., 1., 1., ]
    ]
    fitmodelfunc = [
        [
            fh_MPLF_lmn_log10,
            "fh_MPLF_lmn_log10",
            None,        # params_name -> None (auto)
            None, 
            None, 
            [p0_mp, bounds_mp],
        ]
    ]
    combo = {"name": "AA_combination_freeCoef", "args": []}  # args will be filled from fit later
    return {"is_fit_1d": False, "combination": combo, "fitmodelfunc": fitmodelfunc}

def fitting_model_MPLTF_freq():
    """
    1D-in-h model: fh_MPLTF_log10(h, ...)
    - same collapse h as above
    """
    p0_mp = np.array([
        1.e+04, 2.e+04, 6.e+04, 8.e+05, 
        1.e+00, 2.e+00, 1.2e+00, 
        -0.5, 
    ])
    bounds_mp = [
        [1e-1, 2e-10, 6e1, 8e-10, -200., -200., -200.,  -1000., ],
        [1e6,  2e6,   6e6, 8e7,    200.,  200.,  2000.,   100., ]
    ]
    fitmodelfunc = [
        [
            fh_MPLTF_log10,
            "fh_MPLTF_log10",
            None,        # params_name -> None (auto)
            None, 
            None, 
            [p0_mp, bounds_mp],
        ]
    ]
    combo = {"name": "AA_combination_sumWeightFrequency_rateF1", "args": []}
    return {"is_fit_1d": True, "combination": combo, "fitmodelfunc": fitmodelfunc}

def fitting_model_MPLTF_freefixed():
    """
    Free/fixed version with the 'TF' family:
    fh_MPLTF_lmn_log10(J,lmn,...)
    """
    p0_mp = np.array([
        1.e+04, 2.e+04, 6.e+04, 8.e+05, 
        1.e+00, 2.e+00, 1.2e+00, 
        -0.5, 0.2, 0.1, 
    ])
    bounds_mp = [
        [1e-1, 2e-10, 6e1, 8e-10, -200., -200., -200.,  -1000., 0.1, 0.1, ],
        [1e6,  2e6,   6e6, 8e7,    200.,  200.,  2000.,   100., 10., 10., ]
    ]
    fitmodelfunc = [
        [
            fh_MPLTF_lmn_log10,
            "fh_MPLTF_lmn_log10",
            None,        # params_name -> None (auto)
            None, 
            None, 
            [p0_mp, bounds_mp],
        ]
    ]
    combo = {"name": "AA_combination_freeCoef", "args": []}
    return {"is_fit_1d": False, "combination": combo, "fitmodelfunc": fitmodelfunc}



#### some $\log f(h)$ functions, Tylor/lagueree/Gaussian/Fourior/logpoly
M0 = 137.
def fh_MPLF_lmn_log10(JO, 
    # J0, Jc, Jt, p1, p2, p3, km, kn
    J0, Jc, p1, p2, p3, km, kn
):
    '''
    The influences that paramters control in curve.
    J0: total magnitude
    Jc: more horizontal in small h
    Jt: None
    p1: smaller tail cut on h
    p2: tall, degenerated when the curve is rough
    p3: tall compareness of the slope at smaller and larger h
    '''
    C = np.log(M0/J0**3)
    C1 = 1e0
    h = AA_combination_freeCoef(JO, km, kn)
    F = p2*np.log( (J0/(h+Jc))**p1+C1 ) - p3**np.log( (J0/h)+C1 )
    G = 0.
    T = 0.
    F = C+F+G+T #formula: log_10 f == log_e f * log_10 e
    return F*np.log10(np.e)

def fh_MPLF_log10( h, 
    # J0, Jc, Jt, p1, p2, p3
    J0, Jc, p1, p2, p3
):
    '''
    The influences that paramters control in curve.
    J0: total magnitude
    Jc: more horizontal in small h
    Jt: None
    p1: smaller tail cut on h
    p2: tall, degenerated when the curve is rough
    p3: tall compareness of the slope at smaller and larger h
    '''
    C = np.log(M0/J0**3)
    C1 = 1e0
    F = p2*np.log( (J0/(h+Jc))**p1+C1 ) - p3**np.log( (J0/h)+C1 )
    G = 0.
    T = 0.
    F = C+F+G+T #formula: log_10 f == log_e f * log_10 e
    return F*np.log10(np.e)

def fh_MDPL_log10( h, 
    J0, p1, p2, C1
):
    '''
    Line or L shape.
    '''
    F = (p2/(-p1))*np.log( (h/J0)**(-p1) + C1 )
    return F*np.log10(np.e)

def fh_DPL_log10( h, 
    J0, p1, p2
):
    '''
    Line shape.
    '''
    x = h/J0
    F = -p1*np.log( x ) - p2*np.log( x+1 )
    return F*np.log10(np.e)

def fh_TPL1_log10(h, 
    J0, J1, J2, J3, p1, p2, p3
    # J0, J1, J2, J3, J4, p1, p2, p3, p4
):
    '''
    The influences that paramters control in curve.
    J0: total magnitude
    p1: cuspy in small h
    p2: cuspy in tail h
    p3: tall compareness of the slope at smaller and larger h
    ym: ydata height
    '''
    # J0, J1, J2, J3, p1, p2, p3 = 8e2, 5e2, 4e3, 2e4, 0.18, 0.5, 0.18
    CD = np.log(M0/J0**3)
    x1 = h/J1
    x2 = h/J2
    x3 = h/J3
    # x4 = h/J4
    P1 = -p1*np.log( x1    + 1 )
    P2 = +p2*np.log( x2**2 + 1 )
    P3 = -p3*np.log( x3**3 + 1 ) #MTPL111
    P4 = 0.
    # P4 = -p4*np.log( x4**4 + 1 )
    
    # P1 = -0.7*( np.log( h+500. ) - np.log(500.) )
    # P2 = -(h/1e4)*0.1
    # P2 = -0.2*(np.log( h**-2+1000.**2 ) - np.log( 2000.**2 ) )
    # P2 = +0.25*(np.log( h**3+4000.**3 ) - np.log( 2000.**2 ) )
    # P2 = +0.25*(np.log( h**2+4000.**2 ) - np.log( 2000.**2 ) )
    # P3 = -4*np.log( (h/20000.)**3+500 )
    # P3 = -12*( np.log( (h+150000.) ) -np.log(20000.) )
    # P3 = -1.5*( np.log( h**3+4e4**3 ) -np.log( 2e4**3 ) )
    # P3 = -1*( np.log( h**3-2000*h**2+4e4**3 ) -np.log( 2e4**3 ) )
    # P3 = -1*( np.log( h**2.1+2.1e4**2.1 ) -np.log( 2e4**2.1 ) )
    # P3 = -0.7*( np.log( h**3+2.1e4**3 ) -np.log( 2e4**3 ) ) #MTPL111
    # P3 = -0.7*( np.log( h**3-1.*h**2*2.1e4**1+2.1e4**3 ) -np.log( 2e4**3 ) )
    # P3 = -0.9*( np.log( h**2-1.5*h*2.1e4**1+1.5*2.1e4**2 ) -np.log( 2.1e4**2 ) )
    # Y = 14.2

    # P1 = -0.8*( np.log( h+500. ) - np.log(1.) )
    # P2 = +0.4*(np.log( h+10000 ) - np.log(1.) )
    # P3 = -2.1*( np.log( h+2.1e4 ) -np.log(1.) )
    # Y = 37.

    F = CD+P1+P2+P3+P4
    # ads.DEBUG_PRINT_V(0, CD, np.log10(np.e))
    return F*np.log10(np.e)

def fh_TPL_log10(h, 
    J0, J1, J2, p1, p2, k21, k22
):
    '''
    The influences that paramters control in curve.
    J0: total magnitude
    '''
    CD = np.log(M0/J0**3)
    x1 = h/J1
    x2 = h/J2
    P1 = -p1*np.log( x1 + 1 )
    P2 = -p2*np.log( x2**2-k21*x2+k22**2 )
    # P1 = -0.8*( np.log( h+500. ) - np.log(500.) ) -3.8
    # P2 = -1.1*( np.log( h**2-1.5*h*2.1e4**1+1.5*2.1e4**2 ) -np.log( 2e4**3 ) )
    F = CD+P1+P2
    return F*np.log10(np.e)

def fh_P1PL_Exp_log10(h, 
    J0, J1, J2, J3, p1, p2, p3, b
):
    '''
    To fit cuspy (e.g. NFW), core (e.g. Einasto), n-shape (might move or disk in symmetry potential) and s-shape (might simply shrink) log curve.
    The influences that paramters control in curve. Here called this the MMPL function.
    J0: total magnitude
    '''
    CD = np.log(M0/J0**3)
    x1 = h/J1
    x2 = h/J2
    x3 = h/J3
    # print(h, J2, x2, b, x2**2 + 1., -b*x2)
    P1 = -p1*np.log( x1 + 1. )
    # P1 = 0.
    P2 = -p2*np.log( x2**2 -b*x2 + 1. ) #note: this may lead nan value because of log at some fit parameters
    # P2 = 0.
    P3 = -p3*np.log( x3**3 + 1. )
    # P3 = -x3**p3 #might uncontrolable that the J3 too less that only a exp is to fit
    # P3 = 0. #not perfect to n-shape
    F = CD+P1+P2+P3
    return F*np.log10(np.e)

def fh_MPLTF_log10(h, 
    J0, J1, J2, J3, p1, p2, p3, b
):
    '''
    Function for multiple power law with turning-points b.
    '''
    CD = np.log(M0/J0**3)
    x1 = h/J1
    x2 = h/J2
    x3 = h/J3
    P1 = -p1*np.log( x1 + 1. )
    P2 = -p2*np.log( x2**2 -b*x2 + 1. )
    P3 = -p3*np.log( x3**3 + 1. )
    # P3 = -p3*np.log( x3 + 1. ) #better 1 times form like Cole17, bigger pl index
    # P3 = p3*np.log( 1./x3 + 1. )
    # P1 = -p1*np.log( x1**2 + (1.+1./J3)*x1 + 1. )
    # P2 = -p2*np.log( x2**2 -b*x2 + 1. )
    # P3 = 0.
    F = CD+P1+P2+P3
    return F*np.log10(np.e)

def fh_MPLTF_lmn_log10(JO, 
    J0, J1, J2, J3, p1, p2, p3, b, km, kn
):
    '''
    To fit cuspy (e.g. NFW), core (e.g. Einasto), n-shape (might move or disk in symmetry potential) and s-shape (might simply shrink) log curve.
    The influences that paramters control in curve. Here called this the MMPL function.
    J0: total magnitude
    '''
    CD = np.log(M0/J0**3)
    h = AA_combination_freeCoef(JO, km, kn)
    x1 = h/J1
    x2 = h/J2
    x3 = h/J3
    # print(h, J2, x2, b, x2**2 + 1., -b*x2)
    P1 = -p1*np.log( x1 + 1. )
    # P1 = 0.
    P2 = -p2*np.log( x2**2 -b*x2 + 1. ) #note: this may lead nan value because of log at some fit parameters
    # P2 = 0.
    P3 = -p3*np.log( x3**3 + 1. )
    # P3 = -x3**p3 #might uncontrolable that the J3 too less that only a exp is to fit
    # P3 = 0. #not perfect to n-shape
    F = CD+P1+P2+P3
    return F*np.log10(np.e)

def fh_othercomp_P1PL_Exp_log10(h, 
    J0, J1, J2, J3, p1, p2, p3, b
):
    '''
    To fit cuspy (e.g. NFW), core (e.g. Einasto), n-shape (might move or disk in symmetry potential) and s-shape (might simply shrink) log curve.
    The influences that paramters control in curve. Here called this the MMPL function.
    J0: total magnitude
    '''
    # CD = 1.
    CD = np.log(M0/J0**3)
    x1 = h/J1
    x2 = h/J2
    x3 = h/J3
    # print(h, J2, x2, b, x2**2 + 1., -b*x2)
    P1 = -p1*np.log( x1 + 1. )
    # P1 = 0.
    P2 = -p2*np.log( x2**2 -b*x2 + 1. ) #note: this may lead nan value because of log at some fit parameters
    # P2 = 0.
    P3 = -p3*np.log( x3**3 + 1. )
    # P3 = -x3**p3 #might uncontrolable that the J3 too less that only a exp is to fit
    # P3 = 0. #not perfect to n-shape
    F = CD+P1+P2+P3
    return F*np.log10(np.e)

def fh_TPL_log10_log10(h, 
    J0, J1, J2, p1, p2, k21, k22
):
    '''
    The influences that paramters control in curve.
    J0: total magnitude
    '''
    CD = np.log(M0/J0**3)
    x1 = h**10/J1
    x2 = h**10/J2
    P1 = -p1*np.log( x1 + 1 )
    P2 = -p2*np.log( x2**2-k21*x2+k22**2 )
    # P1 = -0.8*( np.log( h+500. ) - np.log(500.) ) -3.8
    # P2 = -1.1*( np.log( h**2-1.5*h*2.1e4**1+1.5*2.1e4**2 ) -np.log( 2e4**3 ) )
    F = CD+P1+P2
    return F*np.log10(np.e)

def fh_SPL_log10(h, 
    J0, J2, J3, p2, p3, k21, k22, k31, k32, k33
):
    '''
    The influences that paramters control in curve.
    J0: total magnitude
    '''
    CD = np.log(M0/J0**3)
    # x1 = h/J1
    x2 = h/J2
    x3 = h/J3
    P1 = 0.
    # P1 = -p1*np.log( x1 + 1 )
    P2 = -p2*np.log( x2**2-k21*x2+k22 )
    # P2 = -p2*np.log( x2**2-k21*x2+k22**2 )
    P3 = -p3*np.log( x3**3-k31*x3**2-k32*x3+k33 )
    # P1 = -0.8*( np.log( h+500. ) - np.log(500.) ) -3.8
    # P2 = -1.1*( np.log( h**2-1.5*h*2.1e4**1+1.5*2.1e4**2 ) -np.log( 2e4**3 ) )
    F = CD+P1+P2+P3
    return F*np.log10(np.e)

# def fh_SPL_log10(h, 
#     J0, J1, J2, J3, p1, p2, p3, k21, k22, k31, k32, k33
# ):
#     '''
#     The influences that paramters control in curve.
#     J0: total magnitude
#     '''
#     CD = np.log(M0/J0**3)
#     x1 = h/J1
#     x2 = h/J2
#     x3 = h/J3
#     P1 = -p1*np.log( x1 + 1 )
#     P2 = -p2*np.log( x2**2-k21*x2+k22**2 )
#     P3 = -p3*np.log( x3**3-k31*x3**2-k32*x3+k33 )
#     # P1 = -0.8*( np.log( h+500. ) - np.log(500.) ) -3.8
#     # P2 = -1.1*( np.log( h**2-1.5*h*2.1e4**1+1.5*2.1e4**2 ) -np.log( 2e4**3 ) )
#     F = CD+P1+P2+P3
#     return F*np.log10(np.e)

def fh_Sersic_log10(h, 
    J0, p1, p2, k1, ym
): #polynomial
    C = np.log(M0/J0**3)
    C1 = 1e0
    T1 = k1*( (h/J0)**1+C1 )**p2
    # T1 = k1*( (h/J0)**p1+C1 )**p2
    F = C+T1+ym
    return F

def fh_DPL_core_log10(h):
    return 0.

def fh_cut_log10(h):
    return 0.

def fh_P18_log10( h, 
    J0, Jc, Jt, p1, p2, p3, p4, k1, ym
):
    '''
    The influences that paramters control in curve.
    '''
    C = np.log(M0/J0**3)
    C1 = 1e0
    C2 = 1e0
    F = p2*np.log( (J0/(h+C2))+C1 ) - p3**np.log( (J0/h)+C1 )
    xc = Jc/h
    G = -p4*np.log( xc**2-k1*xc+C1 )
    T = 0.
    # T = -(h/Jt)**2
    Y = ym
    F = C+F+G+T+Y #formula: log_10 f == log_e f * log_10 e
    return F*np.log10(np.e)

def fh_MGE_4p_1d(h, Is, sigs, mus):
    ng = len(Is)
    qs = np.ones(ng)
    g = list(range(ng))
    for i in range(ng):
        g[i] = Gauss_1d(h, Is[i], sigs[i], mus[i])
    return np.sum(g[i])

def fLz(Lz):
    return Lz**2*(1+np.tanh(Lz/1e0))*np.exp(-1e0*Lz**2)

def fE(E, a, b, c):
    return c*(E-a)**(-b)

def fh_sr1(X0):
    f = ( -0.42*X0 - ( -0.935/-0.083 + (-0.186+0.583)*X0 ) ) \
        / ( -0.105 - X0*-0.081 + -0.935/-0.083 + (-0.186+0.583) )
    return f

def fh_loglog_sr1(X0):
    f = -(2.*X0**2+1.3) / (X0-0.35)
    return f

def fh_sr2(X0):
    return -0.72-np.log(X0)



##:: some nbody demo
def potential_nbody_simple(x0, nbody_xarray, m_each=1., soft=5., rerr=1e-10):
    p = 0.
    # print(G, m_each, len(nbody_xarray))
    # print(nbody_xarray[0])
    for i in np.arange(len(nbody_xarray)):
        r2 = (x0[0]-nbody_xarray[i,0])**2 \
            +(x0[1]-nbody_xarray[i,1])**2 +(x0[2]-nbody_xarray[i,2])**2
        if r2>rerr**2:
            p += -1./np.sqrt(r2+soft**2)
    return G*m_each*p



##:: some simple function models
def linear_curve1d_space1d(x, k, b):
    return x*k+b

def Gauss_1d(x, A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

def exp_3d(x, C,k1,k2,k3):
    return C*np.exp(k1*x[:,0]+k2*x[:,1]+k3*x[:,2])
def Gauss_line_1d(x, A,mu,sigma,k,b):
    return A*np.exp(-(x-mu)**2/(2*sigma**2)) + k*x+b
def pl2_simple_3d(x, alpha,beta,C):
    h0 = 1.
    h = x[:,0]+x[:,1]+x[:,2]
    alpha1 = (6.-alpha)/(4.-alpha)
    beta1 = 2.*beta-3.
    # return C+h*alpha#+np.log(1.+h*beta)
    # return C+1./h*alpha#+np.log(1.+h*beta)
    # return C*(1.+h0/h)**alpha1+beta1
    return C*(1.+h0/h)**alpha1/(1.+h/h0)**beta1



#########################################
##:: mass DF models
#########################################
#() rs: scale length of radius; rhos: scale mass density; ki: a power of powerlaw
#() for scaled models, r is r_tidle = r/r_s\
##spherical
def rho_spherical_singlepowerlaw(r, rhos,rs,k1):
    return rhos / (r/abs(rs))**k1
def rho_spherical_King(r, rhos,rs):
    return rhos / (r/abs(rs))**2
def rho_spherical_isothermalzero(r, rhos,rs):
    return rhos / (r/abs(rs))**2
def rho_spherical_Plummer(r, rhos,rs):
    return rhos / (r/abs(rs)+1.)**2
def rho_spherical_Einasto(r, rhos,rs):
    return rhos*abs(rs)**3 / ( (r+abs(rs))**1 *(r**2+abs(rs)**2) )

def rho_spherical_doublepowerlaw(r, rhos,rs,k1,k2):
    return rhos / ( (r/abs(rs))**k1 *(1.+r/abs(rs))**k2 )
def rho_spherical_NFW(r, rhos,rs):
    return rhos / ( (r/abs(rs))**1 *(1.+r/abs(rs))**2 )
def rho_spherical_Hernquist(r, rhos,rs):
    return rhos / ( (r/abs(rs))**1 *(1.+r/abs(rs))**3 )
def rho_spherical_Burkert(r, rhos,rs):
    return rhos *abs(rs)**3 / ( (r+abs(rs))**1 *(r**2+abs(rs)**2) )

def rho_spherical_cLB(r, rhos,rs):
    return 0.
def rho_spherical_ExpAndPowerlaw(r, rhos,rs,k1,k2):
    return rhos *np.exp(-abs(rs)*r**k1)/r**k2
def rho_spherical_GaussAndPowerlaw(r, rhos,rs):
    return rhos *np.exp(-abs(rs)*r**2)/r**2

def rho_spherical_scaled_NFW_DICE(r):
    return 1. / ( (r)**1 *(r+1.)**2 )
def rho_spherical_scaled_Hernquist_DICE(r):
    return 1. / ( (r)**1 *(r+1.)**3 )
def rho_spherical_scaled_Burkert_DICE(r):
    return 1. / ( (r+1.)**1 *(r**2+1.**2) )
def rho_spherical_scaled_Einasto_DICE(r):
    return 1. / ( (r+1.)**1 *(r**2+1.**2) )
def rho_spherical_scaled_isothermal_DICE(r):
    return 1. / (r)**2
def rho_spherical_scaled_Plummer_DICE(r):
    return 1. / (r+1.)**2

def rho_spherical_doublepowerlaw_rhorate_log(r, alpha, beta, coef):
    #:: scaled means: r === rdim/r0, rho === rhodim/rho0
    return np.log(coef) -alpha*np.log(r) -(beta-alpha)*np.log(1+r)
def rho_spherical_doublepowerlaw_rhorate_log_shadow(r, *p):
    coef = p[0]
    alpha = p[1]
    beta = p[2]
    # return np.log(coef) -alpha*np.log(r) -(beta-alpha)*np.log(1+r)
    return lambda r,coef,alpha,beta: rho_spherical_doublepowerlaw_rhorate_log(r, alpha, beta, coef)

def rho_spherical_doublepowerlaw_scaled_length_log(r, alpha, beta, C0):
    #:: scaled means: r === rdim/r0, rho === rhodim/rho0
    return np.log(C0) -alpha*np.log(r) -(beta-alpha)*np.log(1+r)

def rho_spherical_doublepowerlaw_scalednot_log(r, alpha, beta, rho0, r0):
    #:: input: not scaled length; output: scaled mass density
    r_ = abs(r/r0)
    return np.log(rho0) -alpha*(np.log(r_)) -(beta-alpha)*np.log(1+r_)
def rho_doublepowerlaw_spherical_log(r, ds,rs, alpha,beta):
    # return np.log(ds) -alpha*(np.log(r/rs)) -(beta-alpha)*np.log(1.+r/rs)
    return np.log(ds) -alpha*(np.log(r/rs)) -(beta-alpha)*np.log(1.+r/rs)

def rho_spherical_Einasto_DICE_log(r, ds,rs, alpha):
    return np.log(ds) -1.*(r/rs)**alpha
def rho_spherical_Einasto_polygamma_plreciprocal_log(r, ds,rs, n):
    nr = 1./n
    bn = 3.*n-1./3+0.0079/n
    return np.log(ds) -bn*( (r/rs)**nr-1. )
def rho_spherical_Einasto_polygamma_log(r, ds,rs, nr):
    n = 1./nr
    bn = 3.*n-1./3+0.0079/n
    return np.log(ds) -bn*( (r/rs)**nr-1. )
def rho_spherical_Sersic_log(r, ds,rs, n):
    #in DICE, alpha_struct === n, beta_struct === bn
    nr = 1./n
    bn = 3.*n-1./3+0.0079/n
    return np.log(ds) -bn*( (r/rs)**nr-1. )

##triaxial
def rho_doublepowerlaw_triaxial(x, ds,rs, alpha,beta,qy,qz,angx=0,angy=0,angz=0):
    rx = np.zeros(x[:,0:3].shape)
    for i in range(len(x)):
        rx[i] = atg.SO3(x[i],[angx,angy,angz])
    rq = ( (rx[:,0])**2+(rx[:,1]/qy)**2+(rx[:,2]/qz)**2 )**0.5
    return ds/( (rq/rs)**alpha * (1.+rq/rs)**(beta-alpha) )
def rho_doublepowerlaw_triaxial_log(x, ds,rs, alpha,beta,qy,qz):
    rx = x[:,0:3]
    rq = ( (rx[:,0])**2+(rx[:,1]/qy)**2+(rx[:,2]/qz)**2 )**0.5
    return np.log(ds) -np.log(rq/rs)*alpha -np.log(1.+rq/rs)*(beta-alpha)
def rho_doublepowerlaw_triaxial_rotate_log(x, ds,rs, alpha,beta,qy,qz,angx=0,angy=0,angz=0):
    rx = np.zeros(x[:,0:3].shape)
    for i in range(len(x)):
        rx[i] = atg.SO3(x[i],[angx,angy,angz])
    rq = ( (rx[:,0])**2+(rx[:,1]/qy)**2+(rx[:,2]/qz)**2 )**0.5
    return np.log(ds) -np.log(rq/rs)*alpha -np.log(1.+rq/rs)*(beta-alpha)

def rho_Sersic_triaxial_log(x, ds,rs, n,qy,qz):
    #in DICE, alpha_struct === n, beta_struct === bn
    rx = x[:,0:3]
    rq = ( (rx[:,0])**2+(rx[:,1]/qy)**2+(rx[:,2]/qz)**2 )**0.5
    nr = 1./n
    bn = 3.*n-1./3+0.0079/n
    return np.log(ds) -bn*( (rq/rs)**nr-1. )
def rho_Sersic_triaxial_rotate_log(x, ds,rs, n,qy,qz,angx=0,angy=0,angz=0):
    #in DICE, alpha_struct === n, beta_struct === bn
    rx = np.zeros(x[:,0:3].shape)
    for i in range(len(x)):
        rx[i] = atg.SO3(x[i],[angx,angy,angz])
    rq = ( (rx[:,0])**2+(rx[:,1]/qy)**2+(rx[:,2]/qz)**2 )**0.5
    nr = 1./n
    bn = 3.*n-1./3+0.0079/n
    return np.log(ds) -bn*( (rq/rs)**nr-1. )

##other
def position_estimateScale(x):
    return np.median(add.norm_l(x[:,0:3], axis=1))
def Phi_doublepowerlaw_NFW_triaxial(target, ds,rs, qy,qz):
    x,y,z = target[0],target[1],target[2]
    Ms = 4.*np.pi * ds*rs**3
    rq = ( (x)**2+(y/qy)**2+(z/qz)**2 )**0.5
    return -G*Ms/rq * np.log(1.+rq/rs)
def Phi_doublepowerlaw_triaxial(target, ds,rs, alpha,beta, qy,qz):
    x,y,z = target[0],target[1],target[2]
    Ms = 4.*np.pi * ds*rs**3
    rq = ( (x)**2+(y/qy)**2+(z/qz)**2 )**0.5
    return -G*Ms/rq * np.log(1.+rq/rs)



#In[]
####[] combination and NDF of angle-actions (AA)

##[] comb
def actionScale_by_otherScale(M, rs):
    return np.sqrt(G*M*rs)

def surface_plane1(x, A,B,C): #x shape [n][3]
    # add.DEBUG_PRINT_V(0, x.shape, A.shape)
    return -(x[:,0]*A + x[:,1]*B + x[:,2]*C)

def surface_plane2(x, B, C, D): #x shape [n][3]
    return x[:,0] + x[:,1]*B + x[:,2]*C + D

def AA_combination_sum(JO):
    J = JO[:,0:3]
    return np.sum(J,axis=1)
def AA_combination_sumWeightFrequency(JO):
    J = JO[:,0:3]
    O = JO[:,3:6]
    # add.DEBUG_PRINT_V(0, J.shape, O.shape)
    return np.sum(J*O,axis=1)
def AA_combination_estimateScale(JO):
    return np.median(AA_combination_sum(JO))
def AA_combination_estimateScaleFreq(JO):
    return np.median(AA_combination_sumWeightFrequency(JO))
def AA_combination_sumWeightFrequency_rateF1(JO):
    O = JO[:,3:6]
    return AA_combination_sumWeightFrequency(JO)/O[:,0]
def AA_combination_disk_rateF1(JO):
    O = JO[:,3:6]
    return AA_combination_sumWeightFrequency(JO)/O[:,0]
def AA_combination_norml2(JO):
    J = JO[:,0:3]
    return add.norm_l(J, b=0, axis=1, l=2)
def AA_combination_norml2WeightFrequency(JO):
    J = JO[:,0:3]
    O = JO[:,3:6]
    return add.norm_l(J*O, b=0, axis=1, l=2)

def AA_combination_AxisRatio(JO, q1,q2,q3):
    J = JO[:,0:3]
    J_ar = J[:,0]*q1+J[:,1]*q2+J[:,2]*q3
    return J_ar
def AA_combination_free_AxisRatio(JO, qq1,qq2,qq3):
    J = JO[:,0:3]
    J_ar = J[:,0]*qq1+J[:,1]*qq2+J[:,2]*qq3 #other??
    return J_ar
def AA_combination_freeCoef(JO, eta1, eta2):
    Jc = JO[:,0]+JO[:,1]*eta1+JO[:,2]*eta2
    return Jc

def AA_combination_free_sum(JO, C0, C1, C2):
    Js = AA_combination_estimateScale(JO)
    return (JO[:,0]*C0+JO[:,1]*C1+JO[:,2]*C2)/Js
def AA_combination_sumCoefPowerFree(JO, C0, C1, C2, p0, p1, p2): #no C0, p0??
    Js = AA_combination_estimateScale(JO)
    return ((JO[:,0]*C0)**p0+(JO[:,1]*C1)**p1+(JO[:,2]*C2)**p1)/Js

def AA_combination_freefree_sumPower(JO, C0, C1, C2, p0, p1, p2):
    Js = AA_combination_estimateScale(JO)
    J = JO[:,0:3]
    return (J[:,0]/Js)**p0*C0 +(J[:,1]/Js)**p1*C1 +(J[:,2]/Js)**p2*C2

def AA_combination_sumModifiedWeight_expSinRate(JO,alpha,beta,L0):
    Ja = JO[:,0]
    Jb = JO[:,1]
    Jc = JO[:,2]
    L = Jb+Jc
    c = L/(L+Ja)
    jt = (1.5*Ja+L)/L0 #L0 is the scaled action
    ksi = 1. - 1./(jt**alpha+1)
    Fin = 1. #??
    Fout = 1.
    k = (1.-ksi)*Fin + ksi*Fout
    a = 0.5*(k+1.)
    b = 0.5*(k-1.)
    # Lz = Jb #Lz ??
    # Lhw = a*Lz + b*Jb*Jc/(Jb+Jc) + Jb #Lz ??
    Lhw = a*Jc + b*Jb*Jc/(Jb+Jc) + Jb
    # Lhw = L #{a=1,b=0,"in Binney and Vasiliev (2022)"}
    eschp = np.exp(beta*np.sin(c*np.pi/2.))
    gh = Ja*eschp + Lhw*0.5*(1 + c*ksi)/eschp
    return gh



##[] DFXVs
def DFXV_fCombinationFixed_DPL_log(xv, p1, p2, C0, C1,     eta1, eta2): #the last two params are fixed
    Mass = MW_mass_total
    rs = C0
    # rs = np.median(add.norm_l(np.abs(xv[:, 0:3]), axis=1))
    rhos_log = C1
    # rhos_log = np.log(Mass) -3.*np.log(rs)
    rq = ( (xv[:,0])**2+(xv[:,1]/eta1)**2+(xv[:,2]/eta2)**2 )**0.5
    rqn = rq / rs
    return rhos_log - p1*np.log(rqn) - (p2-p1)*np.log(1.+rqn)

def DFXV_fCombinationFree_DPL_log(xv, eta1, eta2, p1, p2, C0, C1):
    Mass = MW_mass_total
    rs = C0
    # rs = np.median(add.norm_l(np.abs(xv[:, 0:3]), axis=1))
    rhos_log = C1
    # rhos_log = np.log(Mass) -3.*np.log(rs)
    rq = ( (xv[:,0])**2+(xv[:,1]/eta1)**2+(xv[:,2]/eta2)**2 )**0.5
    rqn = rq / rs
    return rhos_log - p1*np.log(rqn) - (p2-p1)*np.log(1.+rqn)

def DFXV_fCombinationFree_MDPL_ExpPloy_log(
    xv, eta1, eta2, p1, p2, p3, p4, n, C1, C2, C3
):
    Mass = MW_mass_total
    return 0.

def rhorq_TPL_rhos_rs_log10(rq, rs, rs2, rhos, p1, p2, p3):
    '''
    A mass density profile of triple power-law for halo like.
    '''
    rhos_log = np.log(rhos)
    rqn = rq / rs
    rqn2 = rq / rs2
    rho_log = rhos_log - p1*np.log(rqn) - (p2-p1)*np.log(1.+rqn) + p3*np.log(rqn2)
    return rho_log*np.log10(np.e)

def rhorq_DPL_rhos_rs_log10(rq, rs, rhos, p1, p2):
    '''
    A mass density profile of double power-law for halo like.
    '''
    rhos_log = np.log(rhos)
    rqn = rq / rs
    rho_log = rhos_log - p1*np.log(rqn) - (p2-p1)*np.log(1.+rqn)
    return rho_log*np.log10(np.e)

def rhorq_Einasto_rhos_rs_log10(rq, rs, rhos, n, bn):
    '''
    A mass density profile of brief Einasto for halo like.
    '''
    rhos_log = np.log(rhos)
    rqn = rq / rs
    rho_log = rhos_log - bn*( rqn**(1./n)-1. )
    return rho_log*np.log10(np.e)

def rhorq_DPL_log10(rq, C0, C1, p1, p2):
    Mass = MW_mass_total
    rs = C0
    # rs = np.median(add.norm_l(np.abs(xv[:, 0:3]), axis=1))
    rhos_log = np.log10(C1)
    # rhos_log = np.log(Mass) -3.*np.log(rs)
    # rq = ( (xv[:,0])**2+(xv[:,1]/eta1)**2+(xv[:,2]/eta2)**2 )**0.5
    rqn = rq / rs
    return rhos_log - p1*np.log(rqn) - (p2-p1)*np.log(1.+rqn)

def rhorq_Einasto_more_log10(rq, C0, C1, n, bn):
    Mass = MW_mass_total
    # rq = ( (xv[:,0])**2+(xv[:,1]/q1)**2+(xv[:,2]/q2)**2 )**0.5
    # rs = np.median(add.norm_l(np.abs(xv[:, 0:3]), axis=1))
    # rhos_log = np.log(Mass) -3.*np.log(rs)
    rs = C0
    rqn = rq / rs
    rhos_log = np.log10(C1)
    # bn = 3.*n-1./3+0.0079/n #the n is alpha in DICE
    rho = rhos_log - bn*( rqn**(1./n)-1. )
    return rho

def rhorq_Einasto_simplify_log10(rq, C0, C1, n):
    Mass = MW_mass_total
    # rq = ( (xv[:,0])**2+(xv[:,1]/q1)**2+(xv[:,2]/q2)**2 )**0.5
    # rs = np.median(add.norm_l(np.abs(xv[:, 0:3]), axis=1))
    # rhos_log = np.log(Mass) -3.*np.log(rs)
    rs = C0
    rqn = rq / rs
    rhos_log = np.log10(C1)
    bn = 3.*n-1./3+0.0079/n #the n is alpha in DICE
    rho = rhos_log - bn*( rqn**(1./n)-1. )
    return rho

def rhorq_MDPL_core_and_cuspy_log10(rq, C0, C1, p1, p2, b):
    Mass = MW_mass_total
    rs = C0
    # rs = np.median(add.norm_l(np.abs(xv[:, 0:3]), axis=1))
    rhos_log = np.log10(C1)
    # rhos_log = np.log(Mass) -3.*np.log(rs)
    # rq = ( (xv[:,0])**2+(xv[:,1]/eta1)**2+(xv[:,2]/eta2)**2 )**0.5
    rqn = rq / rs
    return rhos_log - p1*np.log(rqn+b) - (p2-p1)*np.log(rqn**2+1.)



##[] DFAAs
#: the mainly, wrap ??
#: DPL: double power law funciton
#: EP: exponent power function
# label_code_about_adjust_fit_function

def DFAA_fCombinationFreq_DPL_Gauss_log(
    # JO, p1, p2, p3, J_0, J_cut
    # JO, p1, p2, p3, J_0, J_cut, J_core
    JO, p1, p2, J_0, J_core, J_cut
):
    Mass = MW_mass_total
    coef_log = np.log(Mass) - 3.*np.log(J_0*2.*np.pi)
    # h = AA_combination_sumModifiedWeight_expSinRate(JO, p2, p3, J_0) #might be zero
    # h = np.sum(JO[:,0:3], axis=1) #might be zero
    h = AA_combination_sumWeightFrequency_rateF1(JO) #might be zero
    # g = h #might be zero
    # g = np.sum(JO[:,0:3], axis=1) #might be zero
    # J_core = 0. #should be near 0., less than J_0
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 1e-10 )
    f_log = p1*np.log( 1./( (h+J_core)/J_0 ) + 1. ) -p2*np.log( ( h/J_0 ) + 1. )
    x = h/J_cut
    # T_log = 0.
    T_log = -( x )**2
    DF_log = coef_log + f_log + T_log
    return DF_log/np.log(10.)

def DFAA_1d_MTPL_log(
    h, p1, p2, p3, J_0, J_core
):
    Mass = MW_mass_total
    coef_log = np.log(Mass) - 3.*np.log(J_0*2.*np.pi)
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 1. )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 1. ) -p3*np.log( ( h/J_0 )**(1.) + 1. )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 0.1 ) -p3*np.log( ( h/J_0 )**(1.) + 1. )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 1. ) -p3*np.log( ( h/J_0 )**(1.) + 0.1 )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 0.1 ) -p3*np.log( ( h/J_0 )**(1.) + 0.1 )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 0.3 ) -p3*np.log( ( h/J_0 )**(1.) + 0.3 )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 0.1 ) -p3*np.log( ( h/J_0 )**(1.) + 0.1 )
    f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 1e-0 ) \
        - p3*np.log( ( h/J_0 )**(1.) + 1e-0 )
    # x = g/J_cut
    T_log = 0.
    # T_log = -( x )**p4
    DF_log = coef_log + f_log + T_log
    return DF_log/np.log(10.)

def DFAA_fCombinationFreq_MTPL_log(
    # JO, p1, p2, p3, J_0, J_cut
    # JO, p1, p2, p3, J_0, J_cut, J_core
    JO, p1, p2, p3, J_0, J_core
):
    Mass = MW_mass_total
    coef_log = np.log(Mass) - 3.*np.log(J_0*2.*np.pi)
    # h = AA_combination_sumModifiedWeight_expSinRate(JO, p2, p3, J_0) #might be zero
    # h = np.sum(JO[:,0:3], axis=1) #might be zero
    h = AA_combination_sumWeightFrequency_rateF1(JO) #might be zero
    # g = h #might be zero
    # g = np.sum(JO[:,0:3], axis=1) #might be zero
    # J_core = 0. #should be near 0., less than J_0
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 1. )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 1. ) -p3*np.log( ( h/J_0 )**(1.) + 1. )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 0.1 ) -p3*np.log( ( h/J_0 )**(1.) + 1. )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 1. ) -p3*np.log( ( h/J_0 )**(1.) + 0.1 )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 0.1 ) -p3*np.log( ( h/J_0 )**(1.) + 0.1 )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 0.3 ) -p3*np.log( ( h/J_0 )**(1.) + 0.3 )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 0.1 ) -p3*np.log( ( h/J_0 )**(1.) + 0.1 )
    f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 1e-10 ) -p3*np.log( ( h/J_0 )**(1.) + 1e-10 )
    # x = g/J_cut
    T_log = 0.
    # T_log = -( x )**p4
    DF_log = coef_log + f_log + T_log
    return DF_log/np.log(10.)

def DFAA_fCombinationFreq_MPL_Expn_log(
    # JO, p1, p2, p3, J_0, J_cut
    JO, p1, p2, p4, J_0, J_cut, J_core
):
    Mass = MW_mass_total
    coef_log = np.log(Mass) - 3.*np.log(J_0*2.*np.pi)
    # h = AA_combination_sumModifiedWeight_expSinRate(JO, p2, p3, J_0) #might be zero
    # h = np.sum(JO[:,0:3], axis=1) #might be zero
    h = AA_combination_sumWeightFrequency_rateF1(JO) #might be zero
    # g = h #might be zero
    g = np.sum(JO[:,0:3], axis=1) #might be zero
    # J_core = 0. #should be near 0., less than J_0
    f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 1. )
    # f_log = p2*np.log( ( (h+J_core)/J_0 )**(-p1) + 1. ) -p3*np.log( ( g/J_0 )**(-p1) + 1. )
    x = g/J_cut
    # T_log = 0.
    T_log = -( x )**p4 #main??
    DF_log = coef_log + f_log + T_log
    return DF_log/np.log(10.)

def DFAA_fCombinationFreq_DPL_Expn_xex_log(
    JO, p1, p2, p3, p4, J_0, J_cut#, J_core
):
    Mass = MW_mass_total
    # sumJ = np.sum(JO[:,0:3], axis=1) #might be zero
    # J_0 = np.median(sumJ)
    coef_log = np.log(Mass) - 3.*np.log(J_0*2.*np.pi) #+ np.log(C0)
    # h = AA_combination_sumModifiedWeight_expSinRate(JO, p2, p3, J_0) #might be zero
    # g = h #might be zero
    h = AA_combination_sumWeightFrequency_rateF1(JO) #might be zero
    g = np.sum(JO[:,0:3], axis=1) #might be zero
    J_core = 0.
    # J_core = J_0
    # f_log = p2*np.log(((h)/J_0)**(-p1)+1.) -p3*np.log((g/J_0)**(-p1)+1.)
    f_log = p2*np.log((h/J_0+J_core/J_0)**(-p1)+1.) -p3*np.log((g/J_0)**(-p1)+1.)
    x = g/J_cut
    # T_log = -x**p4
    T_log = -x**p4 - x*np.log(x)
    DF_log = coef_log + f_log + T_log
    return DF_log

def DFAA_fCombinationFreq_DPL_Expn_log(
    JO, p1, p2, p3, p4, J_0, J_cut, J_core
):
    Mass = MW_mass_total
    # sumJ = np.sum(JO[:,0:3], axis=1) #might be zero
    # J_0 = np.median(sumJ)
    coef_log = np.log(Mass) - 3.*np.log(J_0*2.*np.pi) #+ np.log(C0)
    # h = AA_combination_sumModifiedWeight_expSinRate(JO, p2, p3, J_0) #might be zero
    # g = h #might be zero
    h = AA_combination_sumWeightFrequency_rateF1(JO) #might be zero
    # h = np.sum(JO[:,0:3], axis=1) #might be zero
    g = np.sum(JO[:,0:3], axis=1) #might be zero
    # g = h #might be zero
    # J_core = 0.
    # J_core = J_0
    # f_log = p2*np.log(((h)/J_0)**(-p1)+1.) -p3*np.log((g/J_0)**(-p1)+1.)
    f_log = p2*np.log((h/J_0+J_core/J_0)**(-p1)+1.) -p3*np.log((g/J_0)**(-p1)+1.)
    x = g/J_cut
    T_log = -x**p4
    # T_log = -x**p4 - x*np.log(x)
    DF_log = coef_log + f_log + T_log
    return DF_log

def DFAA_fCombinationExpsin_DPL_Expn_log(
    JO, p1, p2, p3, p4, J_0, J_cut#, J_core
):
    Mass = MW_mass_total
    # sumJ = np.sum(JO[:,0:3], axis=1) #might be zero
    # J_0 = np.median(sumJ)
    coef_log = np.log(Mass) - 3.*np.log(J_0*2.*np.pi) #+ np.log(C0)
    h = AA_combination_sumModifiedWeight_expSinRate(JO, p2, p3, J_0) #might be zero
    g = h #might be zero
    J_core = 0.
    # J_core = J_0
    f_log = p2*np.log(((h+J_core)/J_0)**(-p1)+1.) -p3*np.log((g/J_0)**(-p1)+1.)
    T_log = -(g/J_cut)**p4
    DF_log = coef_log + f_log + T_log
    return DF_log

def DFAA_fCombinationFree_MDPL_Expn_1_log(
    JO, p1, p2, p3, C0, n
    # JO, p1, C0
):
    Mass = MW_mass_total
    h = AA_combination_sumWeightFrequency_rateF1(JO) #might be zero
    # return -p1*h + C0
    g = np.sum(JO[:,0:3], axis=1) #might be zero
    # h = g
    Js1 = np.median(h)
    Js2 = np.median(g)
    # add.DEBUG_PRINT_V(0, Js1, Js2, "Js12")
    # Js2 = C2
    # coef_log = np.log(C0)
    coef_log = np.log(C0) + np.log(Mass) - 3.*np.log(Js1)
    # Jc1 = AA_combination_sum(JO)
    # Jc1 = AA_combination_freeCoef(JO, eta11, eta12)
    f_log = p1*np.log((h/Js1+1)**(-p2)+1.) -p3*np.log(g/Js2+1.)
    # f_log = -p1*np.log(h/Js1+1.)
    # T_log = 0.
    # T_log = -(Jc1-Js3)**2/(2.*Js2**2)
    T_log = -(h/Js2)**(1./n)
    retu = coef_log + f_log + T_log
    # retu[~np.isfinite(retu)] = -0. #??
    return retu/np.log(10.)

def DFAA_fCombinationFree_MDPL1_ExpPloy2(
    JO, p1, p3, p4, C1, C2, C3
):
    Mass = MW_mass_total #??
    Js1, Js2, Js3 = C1, C2, C3
    coef_log = np.log(Mass) - 3.*np.log(Js1)
    Jc1 = AA_combination_sum(JO)
    # Jc1 = AA_combination_freeCoef(JO, eta11, eta12)
    # f_log = p1*np.log(Js1/Jc1+1) - p2*np.log(Jc1/Js1+1)
    f_log = -p1*np.log(Jc1/Js1+1) -p3*np.log((Jc1/Js1)**p4+1)
    T_log = -(Jc1-Js3)**2/(2.*Js2**2)
    return coef_log + f_log + T_log

def DFAA_fCombinationFree_MDPL_ExpPloy2(
    JO, p1, p2, p3, p4, C1, C2, C3
):
    Mass = MW_mass_total
    Js1, Js2, Js3 = C1, C2, C3
    coef_log = np.log(Mass) - 3.*np.log(Js1)
    Jc1 = AA_combination_sum(JO)
    # Jc1 = AA_combination_freeCoef(JO, eta11, eta12)
    # f_log = p1*np.log(Js1/Jc1+1) - p2*np.log(Jc1/Js1+1)
    f_log = p1*np.log(1./(Jc1/Js1+1)**p2+1) -p3*np.log((Jc1/Js1)**p4+1)
    T_log = -(Jc1-Js3)**2/(2.*Js2**2)
    retu = coef_log + f_log + T_log
    retu[~np.isfinite(retu)] = 0. #??
    return retu

def DFAA_MDPLE_log(JO, p1, p2, p3, p4, n, c0):
    h = np.sum(JO[:,0:3], axis=1)
    Js = np.median(h)
    Mass = 1.
    coef_log = np.log(c0) + np.log(Mass) - 3.*np.log(Js)
    f_log = -p1*np.log((Js/h)**p2+1) - p3*np.log((h/Js)**p4+1)
    t_log = -(h/Js)**(1./n)
    return coef_log + f_log + t_log

def DF_Gaussian_log(x_input, c1, mu1, sigma1):
    x = AA_combination_sum(x_input)
    retu =  c1*np.exp( -(x-mu1)**2 / (2.*sigma1**2) )
    retu[retu<=0.] = 1e-20
    return np.log(retu)

def DF_doubleGaussian_log(x_input, c1, mu1, sigma1, c2, mu2, sigma2):
    x = AA_combination_sum(x_input)
    retu =  c1*np.exp( -(x-mu1)**2 / (2.*sigma1**2) ) \
        +   c2*np.exp( -(x-mu2)**2 / (2.*sigma2**2) )
    # for j in np.arange(len(retu)):
    #     if not retu[j]>0.:
    #         add.DEBUG_PRINT_V(0, 
    #             x[j], retu[j], j, 
    #             [c1, mu1, sigma1, c2, mu2, sigma2], 
    #             [((x - mu1)**2. / (2.*sigma1**2.))[0], ((x - mu1)**2. / (2.*sigma1**2.))[0]], 
    #             [(c1*np.exp( -(x - mu1)**2 / (2.*sigma1**2) ))[0], (c1*np.exp( -(x - mu1)**2 / (2.*sigma1**2) ))[0]], 
    #         )
    retu[retu<=0.] = 1e-20
    return np.log(retu)

def DF_simple_polymal(JO, c0, c1, c2, c3, c4, c5):
    # Jc = AA_combination_freeCoef(JO, eta1, eta2)
    Jc = AA_combination_sum(JO)
    return c0 + c1*Jc + c2*Jc**2 + c3*Jc**3 + c4*Jc**4 + c5*Jc**5

def DFAA_fCombinationFree_debug_log(
    JO, eta11, eta12, p1, p2, p3, p4, n, b
):
    # Mass = MW_mass_total
    Mass = 1.
    Js = AA_combination_estimateScaleFreq(JO)
    Js1, Js2, Js3 = Js*0.1, Js, Js*2.1
    coef_log = np.log(Mass) - 3.*np.log(Js)
    Jc1 = AA_combination_freeCoef(JO, eta11, eta12)
    f_log = p1*np.log(1./(Js1**p2)+1) -p3*np.log(Js2**p4+1)
    g_log = 1.
    T_log = -b*(Jc1/Js3)**(1./n)
    return coef_log + f_log + g_log + T_log

def DFAA_fCombinationFree_debug3_log(
    JO, C0, p1, p2, p3, p4, n
):
    Js = AA_combination_estimateScaleFreq(JO)
    # Js = 1e4
    # Jc1 = AA_combination_freeCoef(JO, eta11, eta12)
    Jc1 = AA_combination_sum(JO)
    J1 = Jc1/Js
    # Jc2 = AA_combination_freeCoef(JO, eta21, eta22)
    Jc2 = AA_combination_sum(JO)
    J2 = Jc2/Js
    Jc3 = AA_combination_sum(JO)
    # Jc3 = Jc2
    J3 = Jc3/Js
    b = 1
    # return p1*np.log(1./(J1+1)**p2+1) -p3*np.log(J2**p4+1) \
    #     -b*(J3)**(1./n) -3.*np.log(Js) +np.log(C0)
    return p1*np.log(1./J1**p2+1) -p3*np.log(J2**p4+1) \
        -b*(J3)**(1./n) -3.*np.log(Js) +np.log(C0)

def DFAA_fCombinationFree_DPL_quadraticPL_Exp(
    JO, eta11, eta12, eta31, p1, p2, p3, C1, C2, C3
):
    Mass = MW_mass_total
    Js1, Js2, Js3 = C1, C2, C3
    coef_log = np.log(Mass) - 3.*np.log(Js1)
    Jc1 = AA_combination_freeCoef(JO, eta11, eta12)
    f_log = p1*np.log(Js1/Jc1+1) - p2*np.log(Jc1/Js1+1)
    g_log = -p3*np.log( (Js2/Jc1)**2 - eta31*(Js2/Jc1) + 1 ) #??
    T_log = -(Jc1/Js3)**2
    return coef_log + f_log + g_log + T_log

def DFAA_fCombinationFree_debug2_log(
    JO, C0, eta11, eta12, eta21, eta22, p1, p2, p3, p4, n
):
    Js = AA_combination_estimateScaleFreq(JO)
    # Js = 1e4
    Jc1 = AA_combination_freeCoef(JO, eta11, eta12)
    J1 = Jc1/Js
    Jc2 = AA_combination_freeCoef(JO, eta21, eta22)
    J2 = Jc2/Js
    Jc3 = AA_combination_sum(JO)
    # Jc3 = Jc2
    J3 = Jc3/Js
    b = 1
    # return p1*np.log(1./J1**p2+1) -p3*np.log(J2**p4+1) \
    #     -b*(J3)**(1./n) -3.*np.log(Js) +np.log(C0) #original
    return p1*np.log(1./(J1+1)**p2+1) -p3*np.log(J2**p4+1) \
        -b*(J3)**(1./n) -3.*np.log(Js) +np.log(C0) #original
    # return p1*np.log(1./(J1+1)**p2+1) -p3*np.log(J2**p4+1) \
    #     -b*(J3)**(1./n) -3.*np.log(Js) +np.log(C0) #bad
    # return p1*np.log(1./(J1**p2+1)) -p3*np.log(J2**p4+1) \
    #     -b*(J3)**(1./n) -3.*np.log(Js) +np.log(C0) #bad

def DFAA_fCombinationFree_debug1_log(
    JO, eta11, eta12, eta21, eta22, p1, p2, p3, p4, n, b
):
    Js = AA_combination_estimateScaleFreq(JO)
    # Js = 1e4
    Jc1 = AA_combination_freeCoef(JO, eta11, eta12)
    J1 = Jc1/Js
    Jc2 = AA_combination_freeCoef(JO, eta21, eta22)
    J2 = Jc2/Js
    Jc3 = AA_combination_sum(JO)
    # Jc3 = Jc2
    J3 = Jc3/Js
    return p1*np.log(1./J1**p2+1) -p3*np.log(J2**p4+1) \
        -b*(J3)**(1./n) -3.*np.log(Js)

def DFAA_fCombinationFree_modifiedDPLMultiplyEP_log(
    JO, eta11, eta12, eta21, eta22, p1, p2, p3, p4, n, b
):
    Js = AA_combination_estimateScaleFreq(JO)
    Jc1 = AA_combination_freeCoef(JO, eta11, eta12)
    J1 = Jc1/Js
    Jc2 = AA_combination_freeCoef(JO, eta21, eta22)
    J2 = Jc2/Js
    Jc3 = AA_combination_sum(JO)
    # Jc3 = Jc2
    J3 = Jc3/Js
    return p1*np.log(1./(J1+1)**p2+1) -p3*np.log(J2**p4+1) -b*(J3)**(1./n) -3.*np.log(Js)

def AA_fCombinationFixed_changedPowerLawDPL_log(JO, Js, p1, p2):# , Js=None):
    Jc1 = AA_combination_sumWeightFrequency_rateF1(JO)
    J1 = Jc1/Js
    Jc2 = AA_combination_sum(JO)
    J2 = Jc2/Js
    return p1*np.log(1./J1+1) -p2*np.log(J2+1) -3.*np.log(Js)
def AA_fCombinationSum_changedPowerLaw1_log(JO, p1, p2, p3):# , Js=None):
    Js = AA_combination_estimateScaleFreq(JO)
    # Jc = AA_combination_sumWeightFrequency_rateF1(JO)
    Jc = AA_combination_sum(JO)
    # Js = np.median(Jc)
    # Js = AA_combination_estimateScale(JO)
    J = Jc/Js
    # return -p1*np.log(J**p2+1) -p3*np.log(J) -3.*np.log(Js)
    return p1*np.log(1./J**p2+1) -p3*np.log(J+1) -3.*np.log(Js)
def AA_fCombinationSum_changedPowerLaw2_log(JO, p1, p2, p3, p4):
    Js = AA_combination_estimateScaleFreq(JO)
    # Jc = AA_combination_sumWeightFrequency_rateF1(JO)
    Jc = AA_combination_sum(JO)
    J1 = Jc/Js
    J2 = Jc/Js
    return -p1*np.log(J1**p2+1) -p3*np.log(J2**p4+1) -3.*np.log(Js)
def AA_fCombinationSum_changedPowerLaw3_log(JO, p1, p2, p3):
    Js = AA_combination_estimateScaleFreq(JO)
    # Jc = AA_combination_sumWeightFrequency_rateF1(JO)
    Jc = AA_combination_sum(JO)
    J = Jc/Js
    return -p3*np.log((J**p1+1)**(-p2)+1) -3.*np.log(Js)
def AA_fCombinationSum_usual1_log(JO, p1, p2, p3, n): #exppower
    Js = AA_combination_estimateScaleFreq(JO)
    # Jc = AA_combination_freeCoef(JO, eta1, eta2)
    # Jc = AA_combination_sumWeightFrequency_rateF1(JO)
    Jc = AA_combination_sum(JO)
    J = Jc/Js
    # return -p1*np.log(J**p2+1) -p3*np.log(J) -(J)**(1./n) -3.*np.log(Js)
    return p1*np.log(1./J**p2+1) -p3*np.log(J+1) -(J)**(1./n) -3.*np.log(Js)

def AA_fCombinationFree_changedPowerLaw1_log(JO, eta1, eta2, p1, p2, p3):# , Js=None):
    Jc = AA_combination_freeCoef(JO, eta1, eta2)
    Js = AA_combination_estimateScaleFreq(JO)
    # Js = np.median(Jc)
    # Js = AA_combination_estimateScale(JO)
    J = Jc/Js
    return -p1*np.log(J**p2+1) -p3*np.log(J) -3.*np.log(Js)
def AA_fCombinationFree_changedPowerLaw2_log(JO, eta11, eta12, eta21, eta22, p1, p2, p3, p4):
    Js = AA_combination_estimateScaleFreq(JO)
    Jc1 = AA_combination_freeCoef(JO, eta11, eta12)
    J1 = Jc1/Js
    Jc2 = AA_combination_freeCoef(JO, eta21, eta22)
    J2 = Jc2/Js
    # return -p1*np.log(J1**p2+1) -p3*np.log(J2**p4+1) -3.*np.log(Js)
    return p1*np.log(1./J1**p2+1) -p3*np.log(J2**p4+1) -3.*np.log(Js)
def AA_fCombinationFree_changedPowerLaw3_log(JO, eta1, eta2, p1, p2, p3):
    Jc = AA_combination_freeCoef(JO, eta1, eta2)
    Js = AA_combination_estimateScaleFreq(JO)
    J = Jc/Js
    return -p3*np.log((J**p1+1)**(-p2)+1) -3.*np.log(Js)

def AA_fCombinationFree_usual1_log(JO, eta11, eta12, eta21, eta22, p1, p2, p3, p4, n): #exppower
    Js = AA_combination_estimateScaleFreq(JO)
    Jc1 = AA_combination_freeCoef(JO, eta11, eta12)
    J1 = Jc1/Js
    Jc2 = AA_combination_freeCoef(JO, eta21, eta22)
    J2 = Jc2/Js
    # Jc3 = AA_combination_sum(JO)
    Jc3 = Jc2
    J3 = Jc3/Js
    # return -p1*np.log(J**p2+1) -p3*np.log(J) -(J)**(1./n) -3.*np.log(Js)
    return p1*np.log(1./J1**p2+1) -p3*np.log(J2**p4+1) -(J3)**(1./n) -3.*np.log(Js)



def AA_fCombination_P1Rpower_log(Jc, C, ps):
    Js = np.median(Jc)
    return np.log(1+Js/Jc)*ps +np.log(C)-3.*np.log(Js)
def AA_fCombination_exp(Jc, C):
    Js = np.median(Jc)
    # Js = 1e4 #1.4e4
    # add.DEBUG_PRINT_V(0, Jc, Js)
    return -(Jc/Js) +np.log(C)-3.*np.log(Js)
def AA_fCombination_exppower_log(Jc, C, n):
    Js = np.median(Jc)
    return -(Jc/Js)**(1./n) +np.log(C)-3.*np.log(Js)
def AA_fCombination_power1_log(Jc, C, alpha):
    Js = np.median(Jc)
    return -np.log(Jc/Js)*alpha +np.log(C)-3.*np.log(Js)
def AA_fCombination_powermult_log(Jc, C, alpha,beta,gamma,delta):
    Js = np.median(Jc)
    Jcs = Jc/Js
    Jcrs = 1./Jcs
    return np.log(Jcs)*alpha + np.log(1.+Jcs)*beta \
        + np.log(Jcrs)*gamma + np.log(1.+Jcrs)*delta \
        +np.log(C)-3.*np.log(Js)
def AA_fCombination_polysum_log(Jc, C_n3,C_n2,C_n1,C_0,C_p1,C_p2,C_p3):
    Js = np.median(Jc)
    Jcs = Jc/Js
    Jcsr = 1./Jcs
    return np.log( C_n3*(Jcsr)**(-3)+C_n2*(Jcsr)**(-2)+C_n1*(Jcsr)**(-1)+C_0\
        +C_p1*(Jcsr)**1+C_p2*(Jcsr)**2+C_p3*(Jcsr)**3 ) -3.*np.log(Js)
def AA_fCombination_powermultWithoutMC_log(Jc, alpha,beta,gamma,delta): #C not used
    Js = np.median(Jc)
    Jcs = Jc/Js
    Jcsr = 1./Jcs
    return np.log(Jcs)*alpha + np.log(1.+Jcs)*beta\
        + np.log(Jcsr)*gamma + np.log(1.+Jcsr)*delta -3.*np.log(Js)
def AA_fCombination_Posti15_log(Jc, p1,p2): #C not used
    Js = np.median(Jc)
    return Jc
# def AA_fCombination_Posti15Axis_log(Jc, p1,p2, q1,q2,q3): #C not used #q should be in comb
#     Js = np.median(Jc)
#     return Jc
def AA_fCombination_any_free_sumCoef_log(): #add params #fit locally free this #all params when fit for free fN - free comb
    return 
def AA_fCombination_any_free_powerSumCoef_log():
    return 
def AA_fCombination_any_freeCoef_log():
    return 





## other DF of actions models
# import scipy.optimize as sciopt
# sciopt.brute()
# import sympy
# xx,yy,zz = sympy.symbols("xx,yy,zz")
##2020 up
def func_poly5(x, a,b,c,d,e,f): #good3
    return a+b*x+c*x**2+d*x**3+e*x**4+f*x**5

def func_powerlaw_M1(x, a,c):
    return a*x**(-c) #donot add x+b, *+c

def func_powerlaw(x, a,b,c,d): #good1
    return a*(x+b)**(-c)+d

def func_exp(x, a,b,c): #good2
    return a*np.exp(-b*x)+c
    # if inx>=0:
    #     return 1.0/(1+exp(-inx))
    # else:
    #     return exp(inx)/(1+exp(inx))

def func_Gau(x, a,m,s):
    return a*np.exp(-((x-m)/s)**2) +0. #-, cannot from + #b??

def func_Gau2(x, a,b,c,d,e,f):
    return a*np.exp(-(x-b)**2)/(2.*c**2) +d*np.exp(-(x-e)**2)/(2.*f**2)

def func_doublepowerlaw(x, a,b,c):
    return (x-a)**b +c

def func_doublepowerlaw_Posti(x, a,b,c):
    return a*(1+b*x)**1.67/(1+c*x)**2.9

def func_doublepowerlaw_down(x, a,b,c):
    J0 = np.sqrt(G*MW_mass_total*MW_halo_length_scale)
    xx = x/J0
    return a*xx**b*(1.+xx)**c

def func_Posti15(x):
    return x
    
def func_WE15_1(x, a,b,c): #bad
    J0 = np.sqrt(G*MW_mass_total*MW_halo_length_scale)
    NJ2 = 1.*a #
    k = NJ2*MW_mass_total/(2*np.pi)**3
    TD = x/J0*b #
    LL = 2*TD*c #
    A = TD*LL**-(5./3)
    B = (J0**2+LL**2)**(5./3)
    return k*A/B

def func2_exp(x, a,k1,k2):
    return a*np.exp(-k1*x[:,0]-k2*x[:,1])

def func2_shadow_exp(x,y, a,k1,k2):
    return a*np.exp(-k1*x-k2*y)

def func3_exp(x, a,k1,k2,k3):
    return a*np.exp(-k1*x[:,0]-k2*x[:,1]-k3*x[:,2])
    # return a*np.exp(k1*x[0]+k2*x[1]+k3*x[2])



##spherical
def AA3_spherical_pl2_Posti15_frequencies(JOmg, k1,k2,C0, M0,J0): #J_r,L_z,J_z, Omega_r,Omega_\phi,Omega_z, param1,param2, const1,const2,const3
    JOmg = abs(JOmg) #here we donot consider rotation and direction??
    J = JOmg[:,0:3]
    Omg = JOmg[:,3:6]
    h = J[:,0]*1 +J[:,1]*Omg[:,1]/Omg[:,0] +J[:,2]*Omg[:,2]/Omg[:,0]
    g = J[:,0] +J[:,1] +J[:,2]
    C = C0*M0/J0**3
    A = ( 1.+J0/h )**( (6.-k1)/(4.-k1) )
    B = ( 1.+g/J0 )**( 2.*k2-3. )
    return A/B*C
def AA3_spherical_pl2_Posti15_interpolation(JOmg, k1,k2,C0):
    JOmg = abs(JOmg) #here we donot consider rotation and direction??
    J = JOmg[:,0:3]
    Omg = JOmg[:,3:6]
    h = J[:,0]*1 +J[:,1]*Omg[:,1]/Omg[:,0] +J[:,2]*Omg[:,2]/Omg[:,0]
    g = J[:,0] +J[:,1] +J[:,2]
    J0_here = np.sqrt(G*MW_halo_length_scale)
    # err = 1.e-6
    # g = np.array( np.where(g<=0., err, g) ) #remove zero
    # h = np.array( np.where(h<=0., err, h) ) #remove zero
    # k1 = np.array( np.where(k1==4., 4.+err, k1) ) #remove zero
    C = C0/J0_here**3
    A = ( 1.+J0_here/h )**( (6.-k1)/(4.-k1) )
    B = ( 1.+g/J0_here )**( 2.*k2-3. )
    rtn = C*A/B
    rtn_finite = np.array(np.where(np.isfinite(rtn), rtn,0.))
    return rtn_finite

def AA3_spherical_pl2_Posti15_returnfinite(JOmg, k1,k2,C0):
    # JOmg = abs(JOmg) #here we donot consider rotation and direction??
    J = JOmg[:,0:3]
    Omg = JOmg[:,3:6]
    h = J[:,0]*1 +J[:,1]*Omg[:,1]/Omg[:,0] +J[:,2]*Omg[:,2]/Omg[:,0]
    g = J[:,0] +J[:,1] +J[:,2]
    J0_here = np.sqrt(G*MW_mass_total*MW_halo_length_scale)
    C = C0/J0_here**3
    # C = 1./J0_here**3
    A = ( 1.+J0_here/h )**( (6.-k1)/(4.-k1) )
    B = ( 1.+g/J0_here )**( 2.*k2-3. )
    rtn = C*A/B
    rtn_finite = np.array(np.where(np.isfinite(rtn), rtn,0.))
    return rtn_finite

def AA3_spherical_pl2_Posti15_returnfinite_C1(JOmg, k1,k2):
    # JOmg = abs(JOmg) #here we donot consider rotation and direction??
    J = JOmg[:,0:3]
    Omg = JOmg[:,3:6]
    h = J[:,0]*Omg[:,0] +J[:,1]*Omg[:,1] +J[:,2]*Omg[:,2]
    h = abs(h/Omg[:,0])
    g = J[:,0] +J[:,1] +J[:,2]
    J0_here = np.sqrt(G*MW_mass_total*MW_halo_length_scale)
    C = 1./J0_here**3
    # C = 1./J0_here**3
    A = ( 1.+J0_here/h )**( (6.-k1)/(4.-k1) )
    B = ( 1.+g/J0_here )**( 2.*k2-3. )
    rtn = C*A/B
    rtn_finite = np.array(np.where(np.isfinite(rtn), rtn,0.))
    return rtn_finite

def AA3_spherical_pl2_Posti15_actionscale_log(OJ, k1,k2,J0):
    Omg = OJ[:,0:3]
    J = OJ[:,3:6]
    h = J[:,0]*Omg[:,0] +J[:,1]*Omg[:,1] +J[:,2]*Omg[:,2]
    h = abs(h/Omg[:,0]) #pm
    g = J[:,0] +J[:,1] +J[:,2]
    power1 = (6.-k1)/(4.-k1)
    power2 = (2.*k2-3.)
    base1 = 1.+J0/h
    base2 = 1.+g/J0
    rtn = -3*np.log(J0) +power1*np.log(base1) -power2*np.log(base2)
    rtn_finite = np.array(np.where(np.isfinite(rtn), rtn,0.))
    return rtn_finite

def AA3_spherical_pl2_Posti15_returnfinite_log(JOmg, k1,k2):
    J = JOmg[:,0:3]
    Omg = JOmg[:,3:6]
    h = J[:,0]*Omg[:,0] +J[:,1]*Omg[:,1] +J[:,2]*Omg[:,2]
    h = abs(h/Omg[:,0]) #pm
    g = J[:,0] +J[:,1] +J[:,2]
    J0_here = np.sqrt(G*MW_mass_total*MW_halo_length_scale)
    C = 1. #J0**3/J0**3
    power1 = (6.-k1)/(4.-k1)
    power2 = (2.*k2-3.)
    base1 = 1.+J0_here/h
    base2 = 1.+g/J0_here
    rtn = np.log(C) +power1*np.log(base1) -power2*np.log(base2)
    rtn_finite = np.array(np.where(np.isfinite(rtn), rtn,0.))
    return rtn_finite

def AA3_spherical_pl2_Posti15_simple(J, k1,k2,C0):
    JOmg = abs(J) #here we donot consider rotation and direction??
    h = J[:,0] +J[:,1] +J[:,2]
    g = J[:,0] +J[:,1] +J[:,2]
    J0_here = MW_halo_action_scale
    # err = 1.e-6
    # ovr = 1.e8
    # g = np.array( np.where(g<=0., err, g) ) #remove zero
    # h = np.array( np.where(h<=0., err, h) ) #remove zero
    # k1 = np.array( np.where(k1>=4., 4.-err*1e5, k1) ) #remove zero
    # k1 = np.array( np.where(k1<=1.5, 1.5+err*1e5, k1) ) #remove zero
    # k1 = np.array( np.where(k1<=40., 40., k1) ) #remove zero
    # k1 = np.array( np.where(k1>=15., 1.5, k1) ) #remove zero
    C = C0/J0_here**3
    # C = 1/J0_here**3
    A = ( 1.+J0_here/h )**( (6.-k1)/(4.-k1) )
    B = ( 1.+g/J0_here )**( 2.*k2-3. )
    # A = np.array( np.where(A>=ovr, ovr, A) ) #remove zero
    # B = np.array( np.where(B>=ovr, ovr, B) ) #remove zero
    return A/B*C

def AA3_spherical_pl2_WE15(J, Omg, k1,k2, C0,M0,J0): #J_r,L_z,J_z, Omega_r,Omega_\phi,Omega_z, param1,param2, const1,const2,const3
    l = (6-k1)/(4-k1)
    m = 2*k2-3
    h = J[:,0]*1 +J[:,1]*Omg[:,1]/Omg[:,0] +J[:,2]*Omg[:,2]/Omg[:,0]
    g = J[:,0] +J[:,1] +J[:,2]
    DD = 1. #??
    TT = 1.
    LL = 1.
    C = C0*M0/(2*np.pi)**3*J0**(3-m)
    A = TT*LL**(-l)
    B = ( J0**2+LL*2 )**( (m-l)/2 )
    return A/B*C

##2021 down
def AA_P1Rpower_free_log(JO, C0, C1, C2, p0, p1, p2, C, ps):
    Js = AA_combination_estimateScale(JO)
    JJr = Js/AA_combination_sumCoefPowerFree(JO, C0, C1, C2, p0, p1, p2)
    return NDF_combination_P1Rpower_free_log(JJr, Js, C, ps)
def AA_P1Rpower_free(JO, C0, C1, C2, p0, p1, p2, C, ps):
    Js = AA_combination_estimateScale(JO)
    JJr = Js/AA_combination_sumCoefPowerFree(JO, C0, C1, C2, p0, p1, p2)
    return (1+JJr)**ps *C/Js**3
def NDF_combination_P1Rpower_free_log(JJr, Js, C, ps):
    return np.log(1+JJr)*ps +np.log(C)-3.*np.log(Js)

def AA_exp_coef(JO, C0,C1,C2,C):
    Js = AA_combination_estimateScale(JO)
    JJ = AA_combination_free_sum(JO, C0,C1,C2)
    return np.exp( -JJ ) *C/Js**3
def AA_exp_coef_log(JO, C0,C1,C2,C):
    Js = AA_combination_estimateScale(JO)
    JJ = AA_combination_free_sum(JO, C0,C1,C2)
    return -JJ +np.log(C)-3.*np.log(Js)

def AA_exp_frequency(JO, C,Js):
    JJ = AA_combination_sumWeightFrequency_rateF1(JO)/Js
    return np.exp(-JJ) *C/Js**3
def AA_exp_frequency_log(JO, C,Js):
    JJ = AA_combination_sumWeightFrequency_rateF1(JO)/Js
    return -JJ +np.log(C)-3.*np.log(Js)

def AA_exppower_frequency(JO, n,C,Js):
    JJ = AA_combination_sumWeightFrequency_rateF1(JO)/Js
    nr = 1./n
    return np.exp(-JJ**nr) *C/Js**3
def AA_exppower_frequency_log(JO, n,C,Js):
    JJ = AA_combination_sumWeightFrequency_rateF1(JO)/Js
    nr = 1./n
    return -JJ**nr +np.log(C)-3.*np.log(Js)

def AA_exppower_coef(JO, n,C0,C1,C2,C):
    Js = AA_combination_estimateScale(JO)
    JJ = AA_combination_free_sum(JO, C0,C1,C2)
    nr = 1./n
    return np.exp(-JJ**nr) *C/Js**3
def AA_exppower_coef_log(JO, n,C0,C1,C2,C):
    Js = AA_combination_estimateScale(JO)
    JJ = AA_combination_free_sum(JO, C0,C1,C2)
    nr = 1./n
    return -JJ**nr +np.log(C)-3.*np.log(Js)

def AA_power1_coef(JO, alpha,C0,C1,C2,C):
    Js = AA_combination_estimateScale(JO)
    JJr = 1./AA_combination_free_sum(JO, C0,C1,C2)
    return (JJr)**alpha *C/Js**3
def AA_power1_coef_log(JO, alpha,C0,C1,C2,C):
    Js = AA_combination_estimateScale(JO)
    JJr = 1./AA_combination_free_sum(JO, C0,C1,C2)/Js
    return np.log(JJr)*alpha +np.log(C)-3.*np.log(Js)

def AA_power1_frequency(JO, alpha,Js):
    JJ = AA_combination_sumWeightFrequency_rateF1(JO)/Js
    JJr = 1./JJ
    return (JJr)**alpha /Js**3
def AA_power1_frequency_log(JO, alpha,Js):
    JJ = AA_combination_sumWeightFrequency_rateF1(JO)/Js
    JJr = 1./JJ
    return np.log(JJr)*alpha -3.*np.log(Js)

def AA_powermult_frequency(JO, alpha,beta,gamma,delta,Js):
    JJ = AA_combination_sumWeightFrequency_rateF1(JO)/Js
    JJr = 1./JJ
    return (JJ)**alpha * (1.+JJ)**beta\
        * (JJr)**gamma * (1.+JJr)**delta /Js**3
def AA_powermult_frequency_log(JO, alpha,beta,gamma,delta,Js):
    JJ = AA_combination_sumWeightFrequency_rateF1(JO)/Js
    JJr = 1./JJ
    return np.log(JJ)*alpha + np.log(1.+JJ)*beta\
        + np.log(JJr)*gamma + np.log(1.+JJr)*delta -3.*np.log(Js)

def AA_polysum_frequency(JO, C_n3,C_n2,C_n1,C_0,C_p1,C_p2,C_p3,Js):
    JJ = AA_combination_sumWeightFrequency_rateF1(JO)/Js
    JJr = 1./JJ
    return ( C_n3*(JJr)**(-3)+C_n2*(JJr)**(-2)+C_n1*(JJr)**(-1)+C_0\
        +C_p1*(JJr)**1+C_p2*(JJr)**2+C_p3*(JJr)**3 ) /Js**3
def AA_polysum_frequency_log(JO, C_n3,C_n2,C_n1,C_0,C_p1,C_p2,C_p3,Js):
    JJ = AA_combination_sumWeightFrequency_rateF1(JO)/Js
    JJr = 1./JJ
    return np.log( C_n3*(JJr)**(-3)+C_n2*(JJr)**(-2)+C_n1*(JJr)**(-1)+C_0\
        +C_p1*(JJr)**1+C_p2*(JJr)**2+C_p3*(JJr)**3 ) -3.*np.log(Js)

def AA_Gauss_disk(Lz, sig,Js): #disk Lz
    return np.exp(Lz/sig**2/Js) /Js**3
def AA_Gauss_disk_log(Lz, sig,Js): #disk Lz
    return Lz/sig**2/Js -3.*np.log(Js)

def AA_powermult_splitAction(JO, alpha,beta,gamma,delta,Js):
    JJ = AA_combination_sumWeightFrequency_rateF1(JO)/Js
    JJr = 1./JJ
    return (JJ)**alpha * (1.+JJ)**beta\
        * (JJr)**gamma * (1.+JJr)**delta /Js**3
def AA_powermult_splitAction_log(JO, alpha,beta,gamma,delta,Js):
    JJ = AA_combination_sumWeightFrequency_rateF1(JO)/Js
    JJr = 1./JJ
    return np.log(JJ)*alpha + np.log(1.+JJ)*beta\
        + np.log(JJr)*gamma + np.log(1.+JJr)*delta -3.*np.log(Js)





#In[]
## main
if __name__ == "__main__":

    M_total = MW_mass_total
    Ms_r_4pi = MW_halo_NFW_rho_scale*MW_halo_length_scale**3
    pot0 = G*M_total/MW_halo_length_scale
    # f = rho_spherical_Einasto_polygamma_plreciprocal_log
    # fx = f(MW_halo_length_scale, 1.7, MW_halo_NFW_rho_scale, MW_halo_length_scale)
    # fx_notlog = np.exp(fx)
    # print(fx, fx_notlog)

    f = Phi_doublepowerlaw_NFW_triaxial
    # qy = 0.8
    # qz = 0.3
    qy = 0.6
    qz = 0.3
    # qy = 0.5 #1.17
    # qz = 0.3
    # qy = 0.3 #1.26
    # qz = 0.3
    k_rs = 2e0
    # k_rs = 1e0
    rs = MW_halo_length_scale*k_rs
    ds = Ms_r_4pi/rs**3
    # fA = f([MW_halo_length_scale, 0., 0.], ds, rs, qy, qz)
    # fB = f([0., MW_halo_length_scale, 0.], ds, rs, qy, qz)
    fA = f([MW_halo_length_scale, 0., 0.], ds, rs, qy, qz)
    fB = f([0., MW_halo_length_scale*qy, 0.], ds, rs, qy, qz)
    # print(fA, pot0, MW_halo_length_scale)
    print(fA/pot0, fB/pot0, fB/fA)
