import numpy as np
import mpmath as mm
import matplotlib.pyplot as plt
import analysis_data_distribution as ads

mm.mp.dps = 50 # Set the precision (e.g., 50 decimal places)
G = 43007.1 #kpc, km/s, Gyr

def h_rn(rn):
    # return rn**(-1./D)
    return 1. / ( ( rn / mm.gamma(1+1./D) * D/3. )**D * 4.*np.pi/3 )

def rm_h_D(h, D):
    return 3./D * (3./(4.*np.pi*h))**(1./D) * mm.gamma(1.+1./D)

if __name__ == "__main__":

    # ## old number density
    # r3 = 1.
    # # r3 = 2.41
    # n3 = 0.1 #pc^(-3)
    # # print(n3)

    # D = 1.23
    # # rD = 0.48
    # rD = 2.41
    # rdm3 = rD**(D-3.)
    # # print(rdm3)
    
    # rnD = 2.41
    # # rnD = 0.48
    # h = h_rn(rnD) #r^(-D) pc
    # print(h)

    # zeta_tftp = n3/(h*rdm3)
    # zeta_tftp1 = n3*rdm3/h
    # print(zeta_tftp)
    # print(zeta_tftp1)

    ## in unit of {pc, pc^(-3)}
    r3 = 1.
    n3 = 0.1
    h3 = n3
    r3_corr = None

    D = 1.23
    rD = 0.48
    hD = 1.644
    rD_corr = 2.41
    rD_eq1 = 3.
    rD_eq = 300.

    # zeta_tftp = n3/hD * rD_corr**(3.-D)
    zeta_tftp = n3/hD * rD**(3.-D)
    # zeta_tftp = n3/hD * r3**3./rD**D
    # zeta_tftp = n3/hD * r3**3./rD_corr**D
    zeta_Diffu = 1./zeta_tftp
    ads.DEBUG_PRINT_V(1, 
        n3/h3, n3/hD, 
        rD**(3.-D), rD_corr**(3.-D), rD_eq1**(3.-D), rD_eq**(3.-D), 
        r3**3./rD**D, r3**3./rD_corr**D
    )
    ads.DEBUG_PRINT_V(0, zeta_tftp, zeta_Diffu)
