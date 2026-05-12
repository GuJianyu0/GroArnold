import numpy as np
# from scipy.special import gamma, hyp2f1
import mpmath as mm
import cmath
import matplotlib.pyplot as plt

mm.mp.dps = 50 # Set the precision (e.g., 50 decimal places)
G = 43007.1 #kpc, km/s, Gyr

def activate_range(x, xm=1.):
    return np.tanh(x/xm)
    # return np.log10(np.abs(x))

def afDtheta(alpha, beta, lams, theta):
    '''
    D>0 && D<3 && lams>1 && theta>1
    '''
    alpha = mm.mpf(alpha)
    beta = mm.mpf(beta)
    lams = mm.mpf(lams)
    theta = mm.mpf(theta)
    return mm.appellf1(alpha+beta, alpha, beta, 1.+alpha+beta, -theta**-2, -lams*theta**-2)

def delta_coef_h(D, lam, l1, l2, beta, s):
    '''
    D>0 && D<3 && lam>0
    '''
    D = mm.mpf(D)
    alpha = (3.-D)/2.
    beta = mm.mpf(beta)
    s = mm.mpf(s)
    lams = 1. + (lam*s)**-2
    ab2 = 2.*(alpha + beta)

    lam = mm.mpf(lam)
    l1 = mm.mpf(l1)
    l2 = mm.mpf(l2)
    theta1 = l1/lam
    theta2 = l2/lam
    dch = l2**(-ab2) * afDtheta(alpha, beta, lams, theta2) - l1**(-ab2) * afDtheta(alpha, beta, lams, theta1)
    return dch / ab2

def Feq_scr(D, lam, l1, l2, N3, ND, beta, s):
    '''
    D>0 && D<3 && lam>0 && beta>=0 && s>=1
    Only for numerical density profile of galaxy 
    $n_{3}(r) = n_{3,0} [(r/rscale)^2+1]^(-\beta)$, 
    where $rscale=R_bound/s$, $\beta=0$ for homogeneously medium, 
    and $n_{3,0}$ can be normalized to 1.
    Set $n_{D}(r) = n_{3}(r)$ and $n_{D,0} = n_{3,0}$ now.
    '''
    D = mm.mpf(D)
    lam = mm.mpf(lam)
    l1 = mm.mpf(l1)
    l2 = mm.mpf(l2)
    N3 = mm.mpf(N3)
    ND = mm.mpf(ND)
    beta = mm.mpf(beta)
    s = mm.mpf(s)

    H0 = mm.hyp2f1(beta, D/2., (2.+D)/2., s**2)
    other_D = D/(H0*s**(2*beta))
    Npl = ND/N3**2
    # Npl = N3**0.1
    # Npl = N3**1.
    Dch = delta_coef_h(D, lam, l1, l2, beta, s)
    F = other_D * Npl * lam * Dch - 1.
    return F.real #for numerical small img

def delta_coef_h_theta(D, lam, l1, l2, beta, s):
    D = mm.mpf(D)
    alpha = (3.-D)/2.
    beta = mm.mpf(beta)
    s = mm.mpf(s)
    lams = 1. + (lam*s)**-2
    ab2 = 2.*(alpha + beta)

    lam = mm.mpf(lam)
    l1 = mm.mpf(l1)
    l2 = mm.mpf(l2)
    theta1 = l1/lam
    theta2 = l2/lam
    dch = theta2**(-ab2) * afDtheta(alpha, beta, lams, theta2) - theta1**(-ab2) * afDtheta(alpha, beta, lams, theta1)
    return dch / ab2

def Feq_scr_theta(D, lam, l1, l2, N3, ND, beta, s):
    D = mm.mpf(D)
    lam = mm.mpf(lam)
    l1 = mm.mpf(l1)
    l2 = mm.mpf(l2)
    N3 = mm.mpf(N3)
    ND = mm.mpf(ND)
    beta = mm.mpf(beta)
    s = mm.mpf(s)

    H0 = mm.hyp2f1(beta, D/2., (2.+D)/2., s**2)
    other_D = D/(H0*s**(2*beta))
    Npl = ND/N3**2
    # Npl = N3**0.1
    # Npl = N3**1.
    Dch = delta_coef_h_theta(D, lam, l1, l2, beta, s)
    # print(other_D, H0, s**(2*beta)); exit(0)
    # F = Npl * lam**(D-3.-2.*beta+1.) #scale + 1.
    F = other_D * Npl * lam * lam**(D-3.-2.*beta) * Dch - 1.
    return F.real #for numerical small img

def ksi_example_old_function(N, Nr):
    # Nr = N/2.
    Nrs = N/2.
    D = 3
    X = np.log(Nr)
    ksi = 2./(N*D) * Nrs**2 / X
    # return 1./X
    return ksi

def ksi_onetimes_example_v1ori_D1p5(N, Nr):
    # Nr = N/2.
    Nrs = N/2.
    # D = 1.5
    # FX = mm.hyp2f1(0.75, 0.75, 1.75, -Nr**2)
    # X = -0.525 + 0.667 * Nr**(3.-D) * FX
    D = 1.23
    FX = mm.hyp2f1(0.885, 0.885, 1.885, -Nr**2)
    X = -0.416 + 0.565 * Nr**(3.-D) * FX
    # D = 1.9
    # FX = mm.hyp2f1(0.55, 0.55, 1.55, -Nr**2)
    # X = -0.785 + 0.909 * Nr**(3.-D) * FX
    ksi = 2./(N*D) * Nrs**2 / X
    # return 1./X
    return ksi



if __name__=="__main__":

    ## calculate
    D = 1.5
    # D = 1.5+1.5j
    alpha = (3.-D)/2.
    N = 1e11
    # N = 1e6
    # N = 1e66
    # N2H = N/2
    Nr = 4e4
    # N2H = 4e6
    # Nr = 1e22
    N3 = N
    ND = N
    N_p = 100
    # rate_min = 1e-20
    rate_min = 1e0
    rate_max = 1e10
    # rate_max = 1e30
    beta = 0.
    # beta = 1.
    s = 5.
    # print(N**(3.-D) * hDmu(D, N*rate_max)); exit(0)

    print("example:")
    ksi_old = ksi_example_old_function(N, Nr)
    print("ksi_old:     ", ksi_old)
    # Fscr_tcross = Feq_scr(D, ksi_old, l1, l2, N3, ND, beta, s)
    # print("Fscr_tcross: ", Fscr_tcross)
    ksi_example = ksi_onetimes_example_v1ori_D1p5(N, Nr) #fiexed nearly 3 times to the old
    print("ksi_example: ", float(ksi_example))
    N_space_example = np.geomspace(1, N, N_p)
    Nr_space_example = np.geomspace(1, N, N_p)
    ksi_space_old = np.zeros_like(N_space_example)
    ksi_space_example = np.zeros_like(N_space_example)
    for i in range(N_p):
        # ksi_space_old[i] = ksi_example_old_function(N_space_example[i], N2H)
        # ksi_space_example[i] = ksi_onetimes_example_v1ori_D1p5(N_space_example[i], N2H)
        ksi_space_old[i] = ksi_example_old_function(N, Nr_space_example[i])
        ksi_space_example[i] = ksi_onetimes_example_v1ori_D1p5(N, Nr_space_example[i])
    plt.figure()
    plt.plot(N_space_example, ksi_space_old, marker=".", label="traditional")
    plt.plot(N_space_example, ksi_space_example, marker=".", label="fractal")
    plt.xlabel(r"$\Lambda \equiv b_+/b_-$, fixed $N_D$ and $N_{rs}=R/r_s$")
    plt.ylabel(r"$\xi \equiv t_{\mathrm{relax}}/t_{\mathrm{cross}}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
    plt.close()
    exit(0)

    print("theta:")
    theta = np.geomspace(rate_min, rate_max, N_p)
    hDmu_value = np.zeros_like(theta)
    for i in range(N_p):
        lams = 1. + (1.*s)**-2 #set 1.
        hDmu_value[i] = afDtheta(alpha, beta, lams, theta[i])

    l1 = N/2
    # l1 = N/2e2
    # l1 = N/2e8
    # l1 = N/2e-8
    l2 = 1.
    # l2 = 1.e-20
    # l2 = 1.e20

    print("lam:")
    lam = np.geomspace(rate_min, rate_max, N_p)
    Dch = np.zeros_like(lam)
    Fscr = np.zeros_like(lam)
    Dcht = np.zeros_like(lam)
    Fscrt = np.zeros_like(lam)
    for i in range(N_p):
        Dch[i] = delta_coef_h(D, lam[i], l1, l2, beta, s)
        Dcht[i] = delta_coef_h_theta(D, lam[i], l1, l2, beta, s)
        Fscr[i] = Feq_scr(D, lam[i], l1, l2, N3, ND, beta, s)
        Fscrt[i] = Feq_scr_theta(D, lam[i], l1, l2, N3, ND, beta, s)

    # print(hDmu_value)
    print(Dch)
    print(Dcht)
    # print(activate_range(Dch))
    mask = Dch>0. 
    # print("mask: ", mask)
    print(Fscr)
    print(Fscrt)

    ## plot
    # plt.figure()
    # plt.plot(theta, hDmu_value, marker=".")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()
    # plt.close()

    # plt.figure()
    # # plt.plot(lam, Dch, marker=".")
    # plt.plot(lam, Dcht, marker=".")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()
    # plt.close()

    plt.figure()
    # plt.plot([np.min(lam), np.max(lam)], [0., 0.], color="k")
    # # plt.plot(lam, Fscr, marker=".")
    # plt.plot(lam, activate_range(Fscr), marker=".", label="not mask")
    # plt.plot(lam[mask], activate_range(Fscr[mask]), marker=".", label="positive dv2 mask")
    plt.plot(lam, Fscrt, marker=".")
    # plt.plot(lam, activate_range(Fscrt), marker=".", label="not mask")
    # plt.plot(lam[mask], activate_range(Fscrt[mask]), marker=".", label="positive dv2 mask")
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel(r"$\lambda \equiv t_{\mathrm{relax}}/t_{\mathrm{cross}}$")
    plt.ylabel(r"$\arctan{[\mathscr{F}(\lambda)]}$ to solve")
    plt.legend()
    plt.show()
    plt.close()
