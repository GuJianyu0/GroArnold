#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.neighbors import KDTree
from scipy.interpolate import RBFInterpolator

import analysis_data_distribution as add
# import RW_data_CMGD as rdc
# import triaxialize_galaxy as tg
# import galaxy_models as gm

# The tasks:
# download AGAMA, GALA, galpy, about orbit fractional
# python files to wrap: 
# RW data of CMGD, models of CMGD, deal with snapshot, simple transformations, 
# simple statistics, simple algorithm such as KDTree and interpolation, 
# fit, plot and compare, calculate small categories about CMGD
# wrap D-TACT-DF, \partial s, progs

## funcs
class KDTree_galaxy_particles:
    '''
    The method is from sklearn.KDTree
    [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html]. 
    Here one uses knn. 
    首先, 测试仅查询一个点的情况, 当查询数据点也是tree节点时, 该节点也在knn列表里, 要想办法去掉它. 
    比如说查询k=11个点, 若第一个邻居的距离为0, 则跳过它; 最终, 仍然有10个邻居.
    For data 1e6*3, it use less than 30s.
    '''

    def __init__(self, data, weight_extern_instinct=None, weight_kernal_width=None, leaf_size=40):
        self.data = data #tree, get #example: xyz by filename = "../result/x_COM.txt"
        self.weight_extern_instinct = np.ones(len(self.data)) #like mass of particle
        if weight_extern_instinct is not None: #[learn code]: use "is" (object) instead of "==" (return an array of bool)
            self.weight_extern_instinct = weight_extern_instinct
        self.weight_kernal_width = 1.
        if weight_kernal_width is not None: #like max distance or other weight of each dimension
            self.weight_kernal_width = weight_kernal_width
        self.tree = KDTree(self.data, leaf_size=leaf_size) #default to be Euclidean in the metric space
        # a = KDTree(self.data, leaf_size=leaf_size)
        # a.get_arrays()

        # # #: KDTree 自带的密度计算方法
        # # density = self.tree.kernel_density(targets, h=0.5)
        # # print(density)

        # ntot, itmp = np.shape(self.data)
        # output = []
        # k = 20
        # for ii in range(ntot):
        #     targets = [self.data[ii]]
        #     distances, indices = self.tree.query(targets, k=k)
        #     rmax = distances[0][k-1]
        #     r = np.sqrt(self.data[ii][0]**2.+self.data[ii][1]**2.+self.data[ii][2]**2.)
        #     vol = (4./3.)*np.pi*rmax**3.0
        #     mass = float(k)*1.37e-2
        #     dens = mass/vol
        #     output = np.append(output, self.data[ii])
        #     output = np.append(output, r)
        #     output = np.append(output, dens)
        # output = output.reshape(ntot,5)
        # np.savetxt("../result/density.txt",output,fmt="%.6e")
        # print("Done.")

        # # 按密度排序后输出到文件
        # list=output[:,4]
        # den_max = list.max()
        # den_min = list.min()
        # #print(den_min,den_max)
        # sorted_idx = np.argsort(list)
        # sorted_output = output[sorted_idx]

        # # 输出所有数据
        # np.savetxt("../result/density-sorted.txt",output[sorted_idx],fmt="%.6e")
        # # 仅输出给定密度范围内的数据
        # #mask = (sorted_output[:,4]>1e-3) & (sorted_output[:,4]<1)
        # #np.savetxt("../result/density-sorted.txt",sorted_output[mask],fmt="%.6e")

        return 

    def Weight_SPHSmoothingKernel(self, dr, h):
        '''
        W(r,h) for density and W_2(r,h) for potential in gadget1 paper.
        '''
        rh = np.array([dr[i]/h[i] for i in np.arange(len(h))])
        W1 = np.zeros(np.shape(rh))
        # mask0 = (rh<0.) #dr>0.
        mask1 = (rh<=0.5) #[learn code]: mask = (bool) & (bool)
        mask2 = (rh>0.5) & (rh<=1.)
        # mask3 = (rh>1.) #dr<h
        W1[mask1] = 1.-6.*np.power(rh[mask1],2)+6.*np.power(rh[mask1],3)
        W1[mask2] = 2.*np.power(1.-rh[mask2],3)
        # add.DEBUG_PRINT_V(1, dr, h, mask1, mask2, W1)
        return np.array([8./np.pi/np.power(h[i],3)*W1[i] for i in np.arange(len(h))])

    def query(self, targets, k=32):
        distances, indices = self.tree.query(targets, k=k) #the targets should be a 2d-array
        return distances, indices

    def density_SPH(self, targets, k=32):
        distances, indices = self.tree.query(targets, k=k) #the targets should be a 2d-array
        # values_knn = self.data[indices]
        # print("the targets:     ", targets, "\n")
        # print("the distances:   ", distances)
        # print("the indices:   ", indices)
        # print("the k-nns value: ", values_knn, "\n")
        
        dr = distances
        h = [distances[:, -1]] #the set of the max distance of each target
        # add.DEBUG_PRINT_V(1, np.shape(dr), np.shape(h), np.shape(self.weight_extern_instinct))
        if self.weight_kernal_width is not None:
            dr = distances*self.weight_kernal_width #multiplied by weight of each dimension
            h = distances[:, -1]*self.weight_kernal_width #not adress, deepcopy
        
        DF_nm = self.Weight_SPHSmoothingKernel(dr, h)
        # add.DEBUG_PRINT_V(1, np.shape(indices), np.shape(self.weight_extern_instinct), 
        #     np.shape(self.weight_extern_instinct[indices]), np.shape(DF_nm))
        return np.sum(self.weight_extern_instinct[indices]*DF_nm, axis=1)

    def neighbour_average_and_scatter(self, x, k=16):
        self.tree = KDTree(x)
        distances, indices = self.tree.query(x, k=k)
        # print(indices, x[indices], x[indices].shape)
        x_mean = np.mean(x[indices], axis=1) #axis=1s
        x_std = np.std(x[indices], axis=1)
        # print(x_mean, x_std)
        return x_mean, x_std

    def density_1d(self, targets, k=32):
        '''
        \param targets should be shape of (1, N).
        '''
        distances, indices = self.tree.query(targets, k=k) #the targets should be a 2d-array
        print(np.shape(indices), "shape")
        dr = distances #N*k
        h = [distances[:, -1]] #N #the set of the max distance of each target
        if self.weight_kernal_width is not None:
            dr = distances*self.weight_kernal_width #multiplied by weight of each dimension
            h = distances[:, -1]*self.weight_kernal_width #not adress, deepcopy
        N_targets = len(targets)
        density = np.zeros(N_targets)
        for i in np.arange(N_targets):
            density[i] = np.sum(self.weight_extern_instinct[indices[i]])/(2.*h[i])
        return density

    def RBF_interp_1d(self, k=32):
        return 

    def RBF_interp_nd(self, k=32):
        # KD = kdtp.KDTree_galaxy_particles(JO, weight_extern_instinct=mass)
        # distances, indices = KD.query(tgts)
        # # add.DEBUG_PRINT_V(1, indices, "indices")
        # for i in np.arange(len(tgts)):
        #     # add.DEBUG_PRINT_V(1, np.shape(indices[i]), np.shape(indices), "ii")
        #     rbffunc = RBFInterpolator(JO[indices[i]], xv[indices[i],:], neighbors=32, kernel="thin_plate_spline")
        #     fitintpy = rbffunc([tgts_varational[i]])
        #     add.DEBUG_PRINT_V(1, i, xv[indices[i][0],:], fitintpy[0], "xf")
        return 0

    def density_by_number_per_volumn_2d(self, targets, k=32):
        distances, indices = self.tree.query(targets, k=k)
        # add.DEBUG_PRINT_V(1, np.shape(targets), np.shape(distances), "target shape")
        h_all = distances[:, -1]
        N_targets = len(targets)
        density = np.zeros(N_targets)
        for i in np.arange(N_targets):
            density[i] = np.sum(self.weight_extern_instinct[indices[i]])/(np.pi*h_all[i]**2)
        return density

    def density_by_number_per_volumn_2d_simple(self, targets, k=32):
        distances, indices = self.tree.query(targets, k=k)
        # add.DEBUG_PRINT_V(1, np.shape(targets), np.shape(distances), "target shape")
        # add.DEBUG_PRINT_V(0, distances, indices)
        h = distances[:, -1]
        density = k/(np.pi*h**2)
        # add.DEBUG_PRINT_V(0, distances, h, density)
        return density

    def RBF_Gauss_kernel_summation_2d(self, k=32):
        return 0



# def CartesianToSpherical(Cartesian):
#     '''
#     Convert from 3d position and 3d velocity in Cartesian coordinate to that in SphericalPolar coordinate.
#     @param Cartesian is an array with shape of (n,3) or (n,6), where n is the count of targets.
#     '''
#     if len(np.shape(Cartesian)) != 2:
#         print("@param Cartesian is an array with shape of (n,3) or (n,6), where n is the count of targets.")
#         exit(0)
#     r = np.sqrt(Cartesian[:,0]*Cartesian[:,0]+Cartesian[:,1]*Cartesian[:,1]+Cartesian[:,2]*Cartesian[:,2])
#     SPolar = np.zeros_like(Cartesian)
#     SPolar[:,0] = r #r
#     SPolar[:,1] = np.arctan2(Cartesian[:,1],Cartesian[:,0]) #phi
#     SPolar[:,2] = np.arccos(Cartesian[:,2]/r) #theta
#     if len(Cartesian[0])==3:
#         return SPolar
#     SPolar[:,3] = (Cartesian[:,3]*np.cos(SPolar[:,1])+Cartesian[:,4]*np.sin(SPolar[:,1]))*np.sin(SPolar[:,2])+np.cos(SPolar[:,2])*Cartesian[:,5]
#     SPolar[:,4] = -Cartesian[:,3]*np.sin(SPolar[:,1])+Cartesian[:,4]*np.cos(SPolar[:,1])
#     SPolar[:,5] = (Cartesian[:,3]*np.cos(SPolar[:,1])+Cartesian[:,4]*np.sin(SPolar[:,1]))*np.cos(SPolar[:,2])-np.sin(SPolar[:,2])*Cartesian[:,5]
#     return SPolar

def CartesianToSpherical(Cartesian):
    """
    Convert from 3D position and 3D velocity in Cartesian coordinates to spherical polar coordinates.
    
    Args:
        Cartesian (ndarray): An array with shape (n, 3) for position only or (n, 6) for position and velocity.

    Returns:
        ndarray: An array with shape (n, 3) if only position is provided, or (n, 6) if both position and velocity are provided.
    """
    if len(np.shape(Cartesian)) != 2 or np.shape(Cartesian)[1] not in [3, 6]:
        raise ValueError("@param Cartesian should be an array with shape of (n, 3) or (n, 6), where n is the count of targets.")
    
    r = np.sqrt(Cartesian[:, 0]**2 + Cartesian[:, 1]**2 + Cartesian[:, 2]**2)
    SPolar = np.zeros_like(Cartesian)
    SPolar[:, 0] = r  # r
    SPolar[:, 1] = np.arctan2(Cartesian[:, 1], Cartesian[:, 0])  # phi

    # Handling theta calculation to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        SPolar[:, 2] = np.where(r == 0, 0, np.arccos(Cartesian[:, 2] / r))  # theta, default to 0 if r == 0

    # If velocity components are present
    if Cartesian.shape[1] == 6:
        vx, vy, vz = Cartesian[:, 3], Cartesian[:, 4], Cartesian[:, 5]
        phi, theta = SPolar[:, 1], SPolar[:, 2]
        # Radial velocity (v_r)
        SPolar[:, 3] = (vx * np.cos(phi) + vy * np.sin(phi)) * np.sin(theta) + vz * np.cos(theta)
        # Azimuthal velocity (v_phi)
        SPolar[:, 4] = -vx * np.sin(phi) + vy * np.cos(phi)
        # Polar velocity (v_theta)
        SPolar[:, 5] = (vx * np.cos(phi) + vy * np.sin(phi)) * np.cos(theta) - vz * np.sin(theta)
    return SPolar

def CartesianToPolar(Cartesian):
    '''
    Convert from 3d position and 3d velocity in Cartesian coordinate to that in Polar coordinate.
    @param Cartesian is an array with shape of (n,3) or (n,6), where n is the count of targets.
    '''
    if len(np.shape(Cartesian)) != 2:
        print("@param Cartesian is an array with shape of (n,3) or (n,6), where n is the count of targets.")
        exit(0)
	# x,y,z -> R,phi,z
    Polar = np.zeros_like(Cartesian)
    Polar[:,0] = np.sqrt(Cartesian[:,0]*Cartesian[:,0]+Cartesian[:,1]*Cartesian[:,1])
    Polar[:,1] = np.arctan2(Cartesian[:,1],Cartesian[:,0])
    Polar[:,2] = Cartesian[:,2]
    if len(Cartesian[0])==3:
        return Polar
	# vx,vy,vz -> vR,vphi,vz
    else:
        cp = np.cos(Polar[:,1])
        sp = np.sin(Polar[:,1])
        Polar[:,3] = Cartesian[:,3]*cp+Cartesian[:,4]*sp
        Polar[:,4] = Cartesian[:,4]*cp-Cartesian[:,3]*sp
        Polar[:,5] = Cartesian[:,5]
        return Polar

def velocity_dispersion_knn(x_target, xv_particles, mass=None, weight_str="Nothing", coordinate_str="Spherical"):
    '''
    Convert from 3d position and 3d velocity in Cartesian coordinate to that in Polar coordinate.
    And then calculate the velocity dispersion at position x_target by the knn xv_particles.
    @param x_target or @param xv_particles must be an array with shape of (n,3) or (n,6), where n is the count of targets.
    If x_target is None, it will return the total velocity dispersion.
    '''
    x_particles = xv_particles[:, 0:3]
    nknn = 32
    KD = KDTree_galaxy_particles(x_particles, weight_extern_instinct=mass)
    distances, indices = None, None
    if x_target is not None:
        if not (len(np.shape(x_target)) == 2 and len(np.shape(xv_particles)) == 2):
            print("@param x_target or @param xv_particles must be an array with shape of (n,3) or (n,6), where n is the count of targets.")
            exit(0)
        distances, indices = KD.query(x_target, k=nknn)
    else:
        indices = [0]
    N = len(indices)
    mo1 = np.zeros((N,3)) #in order of mean of (R, p, z)
    mo2 = np.zeros((N,6)) #in order of mean of (RR, Rp, Rz, pp, pz, zz)
    sig2 = np.zeros((N,6))
    for i in np.arange(N):
        xv_knn = None
        if x_target is not None:
            xv_knn = xv_particles[indices[i]]
        else:
            xv_knn = xv_particles[:]
        xvcyl_knn = np.zeros_like(xv_knn)
        if coordinate_str=="Spherical":
            xvcyl_knn = CartesianToSpherical(np.array(xv_knn))
        elif coordinate_str=="Cylindrel":
            xvcyl_knn = CartesianToPolar(np.array(xv_knn))
        else: #Cartesian
            print("The default value of @param coordinate_str is Cartesian coordinate.")
            xvcyl_knn = np.array(xv_knn)
        vcyl_knn = xvcyl_knn[:, 3:6]
        weight = np.ones_like(vcyl_knn) #?? only weight of unit
        mo1[i,0] = np.mean(vcyl_knn[:,0])
        mo1[i,1] = np.mean(vcyl_knn[:,1])
        mo1[i,2] = np.mean(vcyl_knn[:,2])
        mo2[i,0] = np.mean(vcyl_knn[:,0]*vcyl_knn[:,0]) #\bar v_1^2
        mo2[i,1] = np.mean(vcyl_knn[:,0]*vcyl_knn[:,1])
        mo2[i,2] = np.mean(vcyl_knn[:,0]*vcyl_knn[:,2])
        mo2[i,3] = np.mean(vcyl_knn[:,1]*vcyl_knn[:,1]) #\bar v_2^2
        mo2[i,4] = np.mean(vcyl_knn[:,1]*vcyl_knn[:,2])
        mo2[i,5] = np.mean(vcyl_knn[:,2]*vcyl_knn[:,2]) #\bar v_3^2
        sig2[i,0] = (mo2[i,0]-mo1[i,0]*mo1[i,0])
        sig2[i,1] = (mo2[i,1]-mo1[i,0]*mo1[i,1])
        sig2[i,2] = (mo2[i,2]-mo1[i,0]*mo1[i,2])
        sig2[i,3] = (mo2[i,3]-mo1[i,1]*mo1[i,1])
        sig2[i,4] = (mo2[i,4]-mo1[i,1]*mo1[i,2])
        sig2[i,5] = (mo2[i,5]-mo1[i,2]*mo1[i,2])
        # add.DEBUG_PRINT_V(1, mo2[i,0], mo1[i,0]*mo1[i,0])
        # add.DEBUG_PRINT_V(1, mo2[i,1], mo1[i,0]*mo1[i,1])
        # add.DEBUG_PRINT_V(1, mo2[i,2], mo1[i,0]*mo1[i,2])
        # add.DEBUG_PRINT_V(1, mo1[i])
        # add.DEBUG_PRINT_V(1, mo2[i])
        # add.DEBUG_PRINT_V(0, sig2[i])
    return mo1, mo2, sig2

# def velocity_dispersion_total(xv_particles, mass=None, weight="Nothing"):
#     mo1 = np.zeros(3) #in order of mean of (R, p, z)
#     mo2 = np.zeros(6) #in order of mean of (RR, Rp, Rz, pp, pz, zz)
#     sig = np.zeros(6)
#     xvcyl_total = CartesianToPolar(np.array(xv_particles))
#     vcyl_total = xvcyl_total[:, 3:6]
#     mo1[0] = np.mean(vcyl_total[:,0])
#     mo1[1] = np.mean(vcyl_total[:,1])
#     mo1[2] = np.mean(vcyl_total[:,2])
#     mo2[0] = np.mean(vcyl_total[:,0]*vcyl_total[:,0])
#     mo2[1] = np.mean(vcyl_total[:,0]*vcyl_total[:,1])
#     mo2[2] = np.mean(vcyl_total[:,0]*vcyl_total[:,2])
#     mo2[3] = np.mean(vcyl_total[:,1]*vcyl_total[:,1])
#     mo2[4] = np.mean(vcyl_total[:,1]*vcyl_total[:,2])
#     mo2[5] = np.mean(vcyl_total[:,2]*vcyl_total[:,2])
#     sig[0] = np.sqrt(mo2[0]-mo1[0]*mo1[0])
#     sig[1] = np.sqrt(mo2[1]-mo1[0]*mo1[1])
#     sig[2] = np.sqrt(mo2[2]-mo1[0]*mo1[2])
#     sig[3] = np.sqrt(mo2[3]-mo1[1]*mo1[1])
#     sig[4] = np.sqrt(mo2[4]-mo1[1]*mo1[2])
#     sig[5] = np.sqrt(mo2[5]-mo1[2]*mo1[2])
#     return mo1, mo2, sig

def spin_L_parameter_from_xv(xv, mass1, pot):
    # N = len(xv)
    # mass = np.ones(N)
    # L = np.zeros(( N, 6 ))
    # if mass1 is not None:
    #     mass = mass1
    # x = xv[:, 0:3]
    # v = xv[:, 3:6]
    # vnorm = add.norm_l(v, axis=1)
    # L[:, 0] = mass[:]*(x[:, 1]*v[:, 2] - x[:, 2]*v[:, 1])
    # L[:, 1] = mass[:]*(x[:, 2]*v[:, 0] - x[:, 0]*v[:, 2])
    # L[:, 2] = mass[:]*(x[:, 0]*v[:, 1] - x[:, 1]*v[:, 0])
    # Lt = np.sum( add.norm_l(L, axis=1) )
    # E = mass[:]*(0.5*vnorm**2+pot)
    # Et = np.sum(E)
    # G = 43007.1
    # Mt = np.sum(mass)
    # return (Lt*np.abs(Et)**0.5)/(G*Mt**2.5)
    return -1.

def rotate_parameter_cylindrel_from_xv(x_target, xv_particles):
    mo1, mo2, sig2 = velocity_dispersion_knn(x_target, xv_particles, coordinate_str="Cylindrel")
    vp1mean_total = mo1[:,1][0]
    vR2mean_total = mo2[:,0][0]
    vp2mean_total = mo2[:,3][0]
    # vz2mean_total = sig[:,5][0]
    delta = vp2mean_total-vR2mean_total
    sign_delta = np.sign(delta) #?? with sign
    return vp1mean_total / np.sqrt( np.abs(delta) ) * sign_delta

# def beta_parameter_spherical_from_xv(x_target, xv_particles):
#     x = xv[:,0:3]
#     v = xv[:,3:6]
#     r = add.norm_l(x, axis=1)+1e-40
#     rxy = np.sqrt(x[:,0]**2+x[:,1]**2)
#     vr = (x[:,0]*v[:,0]+x[:,1]*v[:,1]+x[:,2]*v[:,2])/r
#     sigr_total = np.std(vr)
#     vtheta = (x[:,2]*v[:,0]-x[:,0]*v[:,2])/(r*np.sqrt(1-(x[:,2]/r)**2))
#     sigtheta_total = np.std(vtheta)
#     vphi = ( (x[:,0]*v[:,1]-x[:,1]*v[:,0])*x[:,0]**2/(x[:,0]**2+x[:,1]**2+1e-40) - (x[:,0]*v[:,0]+x[:,1]*v[:,1])*x[:,2]/r )/rxy
#     sigphi_total = np.std(vphi)
#     return 1.-(sigtheta_total**2+sigphi_total**2)/sigr_total**2

def beta_parameter_spherical_from_xv(x_target, xv_particles):
    mo1, mo2, sig2 = velocity_dispersion_knn(x_target, xv_particles, coordinate_str="Spherical")
    # vr2mean_total = mo2[:,0][0] #??
    # vp2mean_total = mo2[:,3][0]
    # vt2mean_total = mo2[:,5][0]
    vr2mean_total = sig2[:,0][0]
    vp2mean_total = sig2[:,3][0]
    vt2mean_total = sig2[:,5][0]
    return 1. - (vt2mean_total + vp2mean_total) / vr2mean_total

def beta_z_parameter_cylindrel_from_xv(x_target, xv_particles):
    mo1, mo2, sig2 = velocity_dispersion_knn(x_target, xv_particles, coordinate_str="Cylindrel")
    vR2mean_total = mo2[:,0][0]
    # vp2mean_total = mo2[:,3][0]
    vz2mean_total = mo2[:,5][0]
    return 1. - (vz2mean_total + 0.) / vR2mean_total



def RBF_interp_wrap(xdata, ydata, neighbors=32, targets=None):
    '''
    Wrap function to interpolate frequencies from (actions, frequencies) data points, 
    or to interpolate xv from (actions, frequencies) data points.
    @param targets: J or OJ grid list.
    @param xdata: J or OJ.
    @param ydata: O or xv.
    @param mass: mass.
    '''
    # KD = KDTree_galaxy_particles(JO, weight_extern_instinct=mass_one)
    # distances, indices = KD.query(targets)
    # # add.DEBUG_PRINT_V(1, indices, "indices")
    # for i in np.arange(len(targets)):
    #     # add.DEBUG_PRINT_V(1, np.shape(indices[i]), np.shape(indices), "ii")
    #     rbffunc = RBFInterpolator(JO[indices[i]], xv[indices[i],:], neighbors=32, kernel="thin_plate_spline")
    #     fitintpy = rbffunc([tgts_varational[i]])
    #     add.DEBUG_PRINT_V(1, i, xv[indices[i][0],:], fitintpy[0], "xf")
    # return rbffunc, KD

    rbf_func = RBFInterpolator(xdata, ydata, neighbors=neighbors, kernel="thin_plate_spline")
    y_at_targets = None
    if targets is not None:
        y_at_targets = rbf_func(targets)
    return rbf_func, y_at_targets

from scipy.interpolate import griddata
def calculate_mass_within_radii_cylindrical_non_axisymmetric(dens_data, radii, num_points_R=50, num_points_z=50, num_points_phi=10):
    """
    Calculate the mass enclosed within given radii using cylindrical coordinates (non-axisymmetric).
    
    Args:
        dens_data (ndarray): An (N, 4) array where the first 3 columns are Cartesian coordinates
                             (x, y, z) and the 4th column is the mass density value.
        radii (float or array-like): The radius or array of radii of the spheres within which to calculate the mass.
        num_points_R (int): Number of points to sample along the radial (R) direction.
        num_points_z (int): Number of points to sample along the vertical (z) direction.
        num_points_phi (int): Number of points to sample along the azimuthal (\(\phi\)) direction.
        
    Returns:
        float or ndarray: The mass enclosed within the sphere(s) of the given radius/radii.
    """
    # Ensure radii is an array
    radii = np.atleast_1d(radii)

    # Extract the Cartesian coordinates and density
    x, y, z, density = dens_data[:, 0], dens_data[:, 1], dens_data[:, 2], dens_data[:, 3]

    # Convert Cartesian coordinates to cylindrical coordinates
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Define the cylindrical grid for integration
    R_min, R_max = 0, radii.max()
    z_min, z_max = z.min(), z.max()
    phi_min, phi_max = 0, 2 * np.pi
    R_grid = np.linspace(R_min, R_max, num_points_R)
    z_grid = np.linspace(z_min, z_max, num_points_z)
    phi_grid = np.linspace(phi_min, phi_max, num_points_phi)
    R_mesh, z_mesh, phi_mesh = np.meshgrid(R_grid, z_grid, phi_grid, indexing="ij")

    # Convert cylindrical grid to Cartesian for interpolation
    x_mesh = R_mesh * np.cos(phi_mesh)
    y_mesh = R_mesh * np.sin(phi_mesh)

    # Stack grid into points for interpolation
    cylindrical_points = np.column_stack((x, y, z))
    cylindrical_grid_points = np.column_stack((x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()))

    # Interpolate density onto the cylindrical grid
    interpolated_density = griddata(points=cylindrical_points, values=density, xi=cylindrical_grid_points, method='linear', fill_value=0)
    interpolated_density = interpolated_density.reshape(R_mesh.shape)

    # Compute the cylindrical volume elements (dV = R dR dphi dz)
    dR = (R_max - R_min) / (num_points_R - 1)
    dphi = (phi_max - phi_min) / (num_points_phi - 1)
    dz = (z_max - z_min) / (num_points_z - 1)
    dV = R_mesh * dR * dphi * dz

    # Compute cumulative mass for each radial distance
    cumulative_mass = np.cumsum(interpolated_density * dV, axis=0)  # Integrate along z and phi for each R
    total_mass = cumulative_mass.sum(axis=(1, 2))  # Sum over z and phi to get cumulative mass for all radii

    # Interpolate mass as a function of radius
    mass_interpolator = np.interp(radii, R_grid, total_mass)

    return mass_interpolator[0] if len(radii) == 1 else mass_interpolator

def calculate_cylindrical_accelerations(potential_positions, pos_target, dR=1e-3, dz=1e-3, dphi=1e-3):
    """
    Calculate the accelerations in cylindrical coordinates for a target array of positions.
    
    Args:
        potential_positions (function): A function that calculates the potential at given positions.
                                        Input is an (N, 3) array of cylindrical coordinates (R, z, phi).
                                        Returns a (N,) array of potential values.
        pos_target (ndarray): An (N, 3) array of cylindrical coordinates (R, z, phi) for the target positions.
        dR (float): Small step size for numerical differentiation in the radial direction.
        dz (float): Small step size for numerical differentiation in the vertical direction.
        dphi (float): Small step size for numerical differentiation in the azimuthal direction (radians).
    
    Returns:
        ndarray: An (N, 3) array of accelerations in cylindrical coordinates (a_R, a_z, a_phi).
    """
    # Initialize accelerations array
    accelerations = np.zeros_like(pos_target)
    
    # Numerical differentiation for each direction
    for i, (R, z, phi) in enumerate(pos_target):
        # Radial (R) acceleration
        pos_forward_R = np.array([[R + dR, z, phi]])
        pos_backward_R = np.array([[R - dR, z, phi]])
        Phi_forward_R = potential_positions(pos_forward_R)
        Phi_backward_R = potential_positions(pos_backward_R)
        a_R = -(Phi_forward_R - Phi_backward_R) / (2 * dR)
        
        # Vertical (z) acceleration
        pos_forward_z = np.array([[R, z + dz, phi]])
        pos_backward_z = np.array([[R, z - dz, phi]])
        Phi_forward_z = potential_positions(pos_forward_z)
        Phi_backward_z = potential_positions(pos_backward_z)
        a_z = -(Phi_forward_z - Phi_backward_z) / (2 * dz)
        
        # Azimuthal (phi) acceleration
        pos_forward_phi = np.array([[R, z, phi + dphi]])
        pos_backward_phi = np.array([[R, z, phi - dphi]])
        Phi_forward_phi = potential_positions(pos_forward_phi)
        Phi_backward_phi = potential_positions(pos_backward_phi)
        a_phi = -(Phi_forward_phi - Phi_backward_phi) / (2 * dphi * R)
        
        # Store the accelerations
        accelerations[i, 0] = a_R
        accelerations[i, 1] = a_z
        accelerations[i, 2] = a_phi
    
    return accelerations

# NFW halo and PDM acceleration: potential PCDM and PQUMOND, and then difference
# baryons acceleration: potential PN, and then difference



## main
if __name__ == '__main__':

    x = np.linspace(100., 1., 100)
    x = np.array([x]).T
    y = x*x
    KD = KDTree_galaxy_particles(x)
    xm, xs = KD.neighbour_average_and_scatter(x)

    '''
    ####[] data
    path_file = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/snapshot_3.action.method_all.bak.txt"
    # path_file = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/snapshot/snapshot_003.txt"
    # path_file = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
    #     +"galaxy_general_4_EinastoUsual_triaxial_soft5.0_count1e4/txt/snapshot_5000.txt"
    RG = rdc.Read_galaxy_data(path_file)
    # RG.data = RG.data[0:100] #cut data
    RG.AAAA_set_particle_variables(col_particle_x_coordinates=0, 
        col_particle_v_velocities=3, col_particle_mass=7
    )

    x = RG.particle_x_coordinates
    v = RG.particle_v_velocities
    mass = RG.particle_mass
    N_dim = RG.system_space_dimension
    N_ptc = RG.system_particles_count
    add.DEBUG_PRINT_V(1, np.shape(x), np.shape(mass), np.shape(N_ptc))
    
    data = RG.data
    Dim = gm.Dim
    iast = 28
    adur = 10
    AA_TF_FP = data[:, iast+adur*0:iast+adur*0+Dim]
    AA_OD_FP = data[:, iast+adur*1:iast+adur*1+Dim] #none
    AA_GF_FP = data[:, iast+adur*2:iast+adur*2+Dim] #none
    iast += adur*5 # = 78
    AA_TF_DP = data[:, iast+adur*0:iast+adur*0+Dim]
    AA_OD_DP = data[:, iast+adur*1:iast+adur*1+Dim]
    AA_GF_DP = data[:, iast+adur*2:iast+adur*2+Dim] #none

    AA_TF_FP = add.merge_array_by_hstack([AA_TF_FP, np.sum(AA_TF_FP, axis=1)])
    AA_OD_FP = add.merge_array_by_hstack([AA_OD_FP, np.sum(AA_OD_FP, axis=1)])
    AA_GF_FP = add.merge_array_by_hstack([AA_GF_FP, np.sum(AA_GF_FP, axis=1)])
    AA_TF_DP = add.merge_array_by_hstack([AA_TF_DP, np.sum(AA_TF_DP, axis=1)])
    AA_OD_DP = add.merge_array_by_hstack([AA_OD_DP, np.sum(AA_OD_DP, axis=1)])
    AA_GF_DP = add.merge_array_by_hstack([AA_GF_DP, np.sum(AA_GF_DP, axis=1)])
    add.DEBUG_PRINT_V(1, np.shape(AA_GF_DP))

    ####[] DF_x_mass
    KD = KDTree_galaxy_particles(x, weight_extern_instinct=mass)
    # targets = [[0.,0.,0.], [1e2, 1e2, 1e2]]
    targets = x
    DF_x_mass = KD.density_SPH(targets)
    print("DF_x_mass: ", DF_x_mass)

    # # record:
    # path_write = path_file+".DF_x_mass"
    # data_write = add.merge_array_by_hstack([x, v, mass, DF_x_mass])
    # RG.write_numpy_savetxt(path_write, data_write)

    ####[] DF_action_one
    AA_method = AA_TF_DP
    Act = AA_method[:, 0:3+1]
    Ang = AA_method[:, 4:7]
    Fre = AA_method[:, 7:10]

    KD = KDTree_galaxy_particles(Act, weight_extern_instinct=None)
    targets = Act
    DF_action_one = KD.density_SPH(targets)
    print("DF_action_one: ", DF_action_one)

    # record:
    path_write = path_file+".DF"
    data_write = add.merge_array_by_hstack([mass, DF_x_mass, DF_action_one])
    # data_write = add.merge_array_by_hstack([x, v, mass, Act, Fre, Ang, DF_x_mass, DF_action_one]) #??
    RG.write_numpy_savetxt(path_write, data_write)
    '''