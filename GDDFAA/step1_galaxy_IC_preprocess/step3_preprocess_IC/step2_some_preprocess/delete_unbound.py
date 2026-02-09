import numpy as np
from sklearn.neighbors import KDTree

def DensityCenter(xyz, v, mp):
    tree = KDTree(xyz, leaf_size=40)
    ntot, itmp = xyz.shape
    # mp = 1.37e-2
    xCOD = np.zeros(3)
    vCOD = np.zeros(3)
    cumu = 0.0
    k = 32
    for ii in range(ntot):
        target = [xyz[ii]]  #query要求传入2-D数组，因此有两个方括号
        distances, indices = tree.query(target, k=k) #两个返回值都是2-D数组
        rmax = distances[0][k-1]
        r = np.sqrt(xyz[ii][0]**2.+xyz[ii][1]**2.+xyz[ii][2]**2.)
        vol = (4./3.)*np.pi*rmax**3.0
        mass = float(k)*mp[ii]
        dens = mass/vol
        xCOD += xyz[ii,:]*dens
        vCOD +=   v[ii,:]*dens
        cumu += dens
    
    return xCOD/cumu, vCOD/cumu



if __name__ == '__main__':
    
    ## read file which contain info x, v, mass and potential
    filename = "galaxy_general.txt"
    print("Change energy for unbound particles ...")
    raw = np.loadtxt(filename)
    x = raw[:,0:3]
    v = raw[:,3:6]
    mass = raw[:,8]
    pot = raw[:,14]
    N_ptcs = len(x)

    ## 计算各个粒子的总能量
    v2 = v[:,0]**2 + v[:,1]**2 + v[:,2]**2
    Etot = 0.5*v2 + pot

    ## 挑出束缚粒子
    mask = (Etot < 0.0)
    xout = x[mask]
    vout = v[mask]

    ub = (Etot > 0.0)
    # print(ub)
    x_ub = x[ub]
    v_ub = v[ub]

    if(len(Etot[mask])==0 ):
        print("No bound particles, so it might wrong. Please check code and data.")
        exit(0)

    nline, ncol = xout.shape
    nline1, ncol1 = x_ub.shape
    # r = (x[:,0]**2+x[:,1]**2+x[:,2]**2)**0.5
    # rs = np.median(r)
    for i in np.arange(N_ptcs):
        if ub[i]:
            v[i] /= 2.# (1.+(r[i]/rs))

    # 计算这些束缚粒子的密度中心的坐标和速度
    # xCOD, vCOD = DensityCenter(xout, vout, mass)
    xCOD, vCOD = DensityCenter(x, v, mass)

    # 修正束缚粒子的坐标和速度
    # xout[:,0:3] -= xCOD[0:3]
    # vout[:,0:3] -= vCOD[0:3]

    xout1 = x - xCOD[0:3]
    vout1 = v - vCOD[0:3]

    # out = []
    # for i in range(nline):
    #     out=np.append(out,xout[i])
    #     out=np.append(out,vout[i])
    # out = out.reshape(nline,6)

    out1 = np.zeros((N_ptcs, 7))
    out1[:, 0:3] = xout1[:]
    out1[:, 3:6] = vout1[:]
    out1[:, 6] = mass[:]

    # np.savetxt("cold_IC_bound.txt", out, fmt="%.6e")
    np.savetxt("galaxy_general.txt", out1, fmt="%.6e")
    print("There are totally %d unbound particles (count)." %(nline))
    print("Change energy for unbound particles. Done.")
