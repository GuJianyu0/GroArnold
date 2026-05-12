import numpy as np
import sys
import numpy.random as rnd
import analysis_data_distribution as ads
import KDTree_python as kdtp
import triaxialize_galaxy as tg



def mean_vvector_to_phidirection(x, v, N_iter=1):
    '''
    To change the velocity direction of a particle by making 
    the angle of velocity vector and the positive phi-direction 
    be half iteratively.
    Note that @param x and v is coordinate of only one count 
    of particle.
    '''
    vnorm = ads.norm_l(v)
    xv = np.hstack((x, v))
    xv_sphcoor = kdtp.CartesianToSpherical(np.array([xv]))[0]
    # vs = ads.norm_l(xv_sphcoor[3:6])
    phi = xv_sphcoor[1]
    v_phidirection = np.array([vnorm*np.cos(phi-np.pi/2), vnorm*np.sin(phi-np.pi/2), 0.0])
    #\n note: to let v vector angle less pi/2 then R vector angle
    
    # Guard against small negative values caused by numerical noise / convention.
    v_to_vphi = np.sqrt(max(vnorm - xv_sphcoor[4], 0.0))
    # ads.DEBUG_PRINT_V(0, xv_sphcoor, v_phidirection)
    
    v_ = v*1.
    if N_iter==np.inf:
        v_ = v_phidirection
    else:
        for i in np.arange(N_iter):
            vm = v_+v_phidirection
            # If vm is (near) zero, the direction is undefined; fall back to the
            # intended phi-direction to keep |v| preserved and avoid NaN.
            if (not np.all(np.isfinite(vm))) or (ads.norm_l(vm) <= 0.0):
                v_ = v_phidirection
            else:
                v_ = ads.normalization_vector_designated(vm, vnorm)
            # print("nvd: ", vm, vnorm, v_)
    
    # xv_ = np.hstack((x, v_))
    # xv_sphcoor_ = kdtp.CartesianToSpherical(np.array([xv_]))[0]
    # vs_ = ads.norm_l(xv_sphcoor_[3:6])
    return v_, v_to_vphi

def modify_IC(x, v, m=None, N_iter=1, is_centerx_and_mainaxis=False, is_center_v=True):
    '''
    See pot and spin in Jeans dim 3; 
    rotate to main axis and
    iteratively change z-angular moments of random particles; 
    run 1 Gyr and see bound particles 
    (aa file triaxialized pot, E>0, outter iter big, move v center); 
    calculate NDFA; I_new, spin_tg.
    '''
    # ads.DEBUG_PRINT_V(1, x[0], v[0])
    x_, v_ = None, None
    if is_centerx_and_mainaxis:
        x_, v_ = tg.centralize_coordinate(x, v, m)
        x_, v_ = tg.rotate_mainAxisDirection(x_, v_, m) #note: one should update potential
    else: #default
        x_, v_ = x*1., v*1. #note: a new varibale or deep copy

    # ads.DEBUG_PRINT_V(1, x_[0], v_[0])
    N = len(x)
    vv = np.zeros(N)
    for i in np.arange(N):
        v_[i], vv[i] = mean_vvector_to_phidirection(x[i], v[i], N_iter)
    
    n_not_finite = len(v_[~np.isfinite(v_)])
    ads.DEBUG_PRINT_V(1, n_not_finite, len(vv[~np.isfinite(vv)]), "length of not finite, after")
    if n_not_finite>0:
        raise ValueError("Not finite value after modifying initial condition, which would lead error for downstream gadget simulation. Exit.")

    if is_center_v: #default
        v_ -= np.mean(v_, axis=0)
    
    # ads.DEBUG_PRINT_V(1, vv, np.mean(vv), "v_to_vphi_all")
    # ads.DEBUG_PRINT_V(1, np.mean(v, axis=0), np.mean(v_, axis=0), "vmean")
    # vs = kdtp.CartesianToSpherical(np.hstack((x,v)))[:, 3:6]
    # vs_ = kdtp.CartesianToSpherical(np.hstack((x,v_)))[:, 3:6]
    # ads.DEBUG_PRINT_V(1, np.mean(vs, axis=0), np.mean(vs_, axis=0), "vsphmean")
    return x_, v_



if __name__ == '__main__':
    
    folder_IC_trans = "../../../GDDFAA/step1_galaxy_IC_preprocess/step3_preprocess_IC/step1_from_ascii_to_g1_and_run/"
    # ggtxt = "gg0.txt" #for debug
    ggtxt = sys.argv[1]
    ggg1 = ggtxt+".modified.txt"
    # target_spin = 1.0
    
    data = np.loadtxt(folder_IC_trans+ggtxt)
    x = data[:, 0:3]
    v = data[:, 3:6]
    m = data[:, 8]
    particle_type = data[:, 7]
    pot0 = data[:, -7] #14
    mask_select_type = [1, 2, 0, 3]
    counts_type_1203 = ads.check_count_of_particle_type(particle_type, mask_select_type)
    
    n_not_finite1 = len(x[~np.isfinite(x)])
    n_not_finite2 = len(v[~np.isfinite(v)])
    ads.DEBUG_PRINT_V(1, n_not_finite1, n_not_finite2, "length of not finite, begin")
    if n_not_finite1+n_not_finite2>0:
        raise ValueError("Not finite value readed, wrong initial condition. Please check. Exit.")

    ads.DEBUG_PRINT_V(1, counts_type_1203, pot0, "counts_type, pot0")
    x_, v_ = None, None
    if np.sum(np.abs(pot0))<1e-10:
        print("There is not potential data, set to zero and use the psedo energy and spin to check temperorily.")

    # spin1_direct = tg.spin_lambda_Nbody(x, v, mass=m, pot=pot0)
    spin1_direct, R_max, Menc = tg.spin_bullock_lambda_prime(x, v, m)
    fracubd1 = tg.frac_unbounded_particles(v, pot0)
    ads.DEBUG_PRINT_V(1, spin1_direct, fracubd1, "lambda1, fracubd1")

    x_, v_ = modify_IC(x, v, N_iter=1, is_center_v=True) #1 #2 #6 #np.inf
    # x_, v_ = modify_IC(x, v, N_iter=2, is_center_v=True) #1 #2 #6 #np.inf
    # spin2_direct = tg.spin_lambda_Nbody(x_, v_, mass=m, pot=pot0)
    spin2_direct, R_max, Menc = tg.spin_bullock_lambda_prime(x_, v_, m)
    fracubd2 = tg.frac_unbounded_particles(v_, pot0)
    ads.DEBUG_PRINT_V(1, spin2_direct, fracubd2, "lambda2, fracubd2")
    
    data[:, 0:3] = x_ #at end of prog
    data[:, 3:6] = v_
    np.savetxt(folder_IC_trans+ggg1, data)
    print("Modify spin of initial condotion, done.")
