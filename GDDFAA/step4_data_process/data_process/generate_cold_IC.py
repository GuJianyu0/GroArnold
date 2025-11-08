import numpy as np
import numpy.random as rnd

import change_params_galaxy_init as rsi

def direction():
    '''
    三维球面上的Marsaglia方法
    返回一个模长为1的矢量
    '''
    r2=2
    while(r2>1.0):
        tmp = -1.0+2.0*rnd.rand(2)
        u=tmp[0]
        v=tmp[1]
        r2 = u*u + v*v
    return 2.*u*np.sqrt(1-r2), 2.*v*np.sqrt(1-r2), 1.0-2.0*r2

def sample_r(cold_alpha, cold_alphamax, rmin=5.0e-2, rmax=1.0e+2):
    if( cold_alpha >= cold_alphamax):
        print("Error: a>=3.0")
        exit()
    s = rnd.rand()
    # PDF 为 r^{-a}
    # CDF 为 r^{3-a}/(3-a), 最小半径设为rmin{kpc}，最大半径设为rmax{kpc}
    const = (cold_alphamax-cold_alpha)
    const_inv = 1.0/const
    smax = const_inv * rmax**const
    smin = rmin*const_inv
    s = smin + (smax-smin)*s
    return (s*const)**(const_inv)


def sample_pos(cold_alpha, cold_alphamax):
    '''
    在 [-100，100] 范围内抽取x, y, z。密度分布服从r^{-a}律
    '''
    r = sample_r(cold_alpha, cold_alphamax)
    thx,thy,thz = direction()
    return np.array([r*thx,r*thy,r*thz])

def sample_vel(v_sigma):
    '''
    根据 Aguilar & Merritt (1990)，速度弥散与位置无关
    '''
    return rnd.randn(3)*v_sigma



if __name__ == '__main__':
    
    ## read
    filename = "../step1_set_IC_DDFA/IC_param.txt"
    pl = rsi.read_IC_settings(filename)
    print(pl)

    ## galaxy with one component
    # total number and mass
    Mass_total = 137
    N_total = 10000

    # 一维速度弥散，单位km/s，与Gadget设置相同
    cold_alpha   = 1.0
    cold_alphamax = 3.0
    v_sigma = 30.0 #sigma??

    rnd.seed(487543) #seed 1
    #rnd.seed(666666) #seed 2
    #rnd.seed(888888) #seed 3
    #rnd.seed(123456) #seed 4

    mass1, N_comp1, v_sigma, cold_alpha, cold_alphamax, seed = pl
    mass = mass1 #only one component galaxy IC provided
    N_total = N_comp1

    ## generate
    print("Start to generate cold IC...")
    outdata = np.zeros((N_total, 7))
    n = 0
    while(n<N_total):
        outdata[n, 0:3] = sample_pos(cold_alpha, cold_alphamax)
        outdata[n, 3:6] = sample_vel(v_sigma)
        # outdata[n, 6] = ID #not about mass
        # outdata[n, 7] = typep #not about mass
        outdata[n, 6] = mass #not about mass
        # print("%d "%(n))
        n += 1

    np.savetxt("galaxy_general.txt", outdata, fmt="%.6e")
    print("Done to generate cold IC.")
