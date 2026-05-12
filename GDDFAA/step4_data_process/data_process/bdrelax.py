import numpy as np
# from scipy.special import gamma, hyp2f1
import mpmath as mm
import matplotlib.pyplot as plt

mm.mp.dps = 50 # Set the precision (e.g., 50 decimal places)
G = 43007.1 #kpc, km/s, Gyr

def activate_range(x, xm=1.):
    return np.tanh(x/xm)
    # return np.log10(np.abs(x))

def hDmu(D, mu):
    '''
    D>0 && D<3 && mu>1
    '''
    D = mm.mpf(D)
    mu = mm.mpf(mu)
    return mm.hyp2f1( (3. - D)/2, (3. - 3.*D)/2, (5. - 3.*D)/2, -mu**2 )

def delta_coef_h(D, lam, l1, l2):
    '''
    D>0 && D<3 && lam>1 && N>1
    '''
    D = mm.mpf(D)
    lam = mm.mpf(lam)
    l1 = mm.mpf(l1)
    l2 = mm.mpf(l2)
    return l2**(3.-3.*D) * hDmu(D, l2*lam) - l1**(3.-3.*D) * hDmu(D, l1*lam)

def Feq_scr(D, lam, N, l1, l2, R=-1., v=-1., m=-1.):
    D = mm.mpf(D)
    lam = mm.mpf(lam)
    l1 = mm.mpf(l1)
    l2 = mm.mpf(l2)
    N = mm.mpf(N)
    # ksi_D = np.pi**4 * 2**(14.-3.*D) / ( D*(3.*D-3.) )
    # smallgamma_D = ( gamma(3./2)*gamma(1.-D/2.) / ( gamma(D/2.)*gamma(3./2-D/2.) ) )**2
    # etaM3_D = D
    ksi_smallgamma_etaM3_D = np.pi*D**2 / ( 6.*(D-1.) ) * (mm.gamma(D/2.)*\
        mm.gamma(1.-D/2.)**2) / (mm.gamma(3./2)*mm.gamma(3./2-D/2.)**2)
    # print(ksi_D*smallgamma_D, ksi_smallgamma_etaM3_D); exit(0)
    Dch = delta_coef_h(D, lam, l1, l2)
    NvRp = N**1. #-1, -1.e-8, 0., 1.e-8, 0.2, 1., 2.
    # NvRp = N**0.5
    return ksi_smallgamma_etaM3_D * NvRp * lam * Dch - 1.



if __name__=="__main__":

    ## calculate
    D = 1.5
    # D = 1.5+1.5j
    N = 1e11
    # N = 1e20
    N_p = 100
    rate_min = 1e-20
    # rate_min = 1e0
    # rate_max = 1e6
    rate_max = 1e10
    # print(N**(3.-D) * hDmu(D, N*rate_max)); exit(0)

    mu = np.geomspace(rate_min, rate_max, N_p)
    hDmu_value = np.zeros_like(mu)
    for i in range(N_p):
        hDmu_value[i] = hDmu(D, mu[i])

    l1 = N/2
    # l1 = N/2e8
    # l1 = N/2e-8
    l2 = 1.
    # l2 = 1.e4
    lam = np.geomspace(rate_min, rate_max, N_p)
    Dch = np.zeros_like(lam)
    Fscr = np.zeros_like(lam)
    for i in range(N_p):
        Dch[i] = delta_coef_h(D, lam[i], l1, l2)
        Fscr[i] = Feq_scr(D, lam[i], N, l1, l2)

    lam_old = N/(6*np.log(N/2.))
    print("lam_old: ", lam_old)
    Fscr_tcross = Feq_scr(D, 1., N, l1, l2)
    print("Fscr_tcross: ", Fscr_tcross)

    print(hDmu_value)
    print(Dch)
    print(activate_range(Dch))
    print(Fscr)
    mask = Dch>0. #the fractor 1./(3.*D-3) is in ksi_smallgamma_etaM3_D
    print("mask: ", mask)

    ## plot
    # plt.figure()
    # plt.plot(mu, hDmu_value, marker=".")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()
    # plt.close()

    # plt.figure()
    # plt.plot(lam, Dch, marker=".")
    # # plt.plot(activate_range(lam), activate_range(Dch), marker=".")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()
    # plt.close()

    plt.figure()
    plt.plot([np.min(lam), np.max(lam)], [0., 0.], color="k")
    # plt.plot(lam, Fscr, marker=".")
    plt.plot(lam, activate_range(Fscr), marker=".", label="not mask")
    plt.plot(lam[mask], activate_range(Fscr[mask]), marker=".", label="positive dv2 mask")
    plt.xlabel(r"$\lambda \equiv t_{\mathrm{relax}}/t_{\mathrm{cross}}$")
    plt.ylabel(r"$\arctan{[\mathcal{F}(\lambda)]}$ to solve")
    plt.xscale("log")
    plt.legend()
    plt.show()
    plt.close()
