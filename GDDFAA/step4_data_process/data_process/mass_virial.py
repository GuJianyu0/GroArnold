# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

G=43009.15694 #unit of 1e10Msol
rR=200. #unit of kpc
#M_vir #unit of km/s
#v_vir #unit of 1e10Msol

def M_vir(v_vir):
    return v_vir**2*rR/G

def v_vir(M_vir):
    return np.sqrt(G*M_vir/rR)

def _get_median_idx(self, X, idxs, feature):
    n = len(idxs)
    k = n // 2
    col = map(lambda i: (i, X[i][feature]), idxs)
    sorted_idxs = map(lambda x: x[0], sorted(col, key=lambda x: x[1]))
    median_idx = list(sorted_idxs)[k]
    return median_idx


if __name__ == '__main__':
    r    = np.linspace(0,2,100)
    index_aim = np.where(r==0)[0][0] #np.where(r==0) return (array[index],), so add [0][0] to get index
    r    = np.delete(r, index_aim)
    c    = 1.1
    psi0 = -c/r
    psi1 = np.sqrt(r**2+c**2)
    psi2 = 1./psi1
    psi3 = r*np.log(r/c)
    psi4 = r**2*np.log(r/c)
    psi5 = np.exp(-c**2*r**2)

    plt.figure()
    # plt.plot(r, psi0, color="black",  label="individual potential func")
    plt.plot(r, psi1, color="red",    label="rbf of multiquadric")
    plt.plot(r, psi2, color="orange", label="rbf of inverse multiquadric")
    plt.plot(r, psi3, color="yellow", label="rbf of thin-thin-plate spline")
    plt.plot(r, psi4, color="green",  label="rbf of thin-plate spline")
    plt.plot(r, psi5, color="blue",   label="rbf of Gauss kernel")
    ## so it might be proper to select inverse multiquadric of Gauss kernel
    plt.grid()
    plt.legend()
    plt.xlabel("distance to target point")
    plt.ylabel("radius basis funtion(RBF)")
    plt.show()



####main():
    #### test //
    # arr=np.array([3,5,7,3,6,7,9,3,4,9,3])
    # a=11//2.01
    # print(a)

    #### units
    # v=200.
    # M=M_vir(v) #186.006901069

    # M=200.
    # v=v_vir(M) #207.386491700882 #but dice 

    # k1=(182.8/207.386491700882)**2
    # k2=G/1

    # print(k1,k1*k2,k1/k2)