# -*- coding: utf-8 -*-
import numpy as np
# import pandas as pd
# import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import scipy.optimize as spopt
import cv2



# def plot_plane_for_b2():

#     #######################################

#     tmp = np.loadtxt("a2b2c2.txt")
#     a2 = tmp[0]
#     b2 = tmp[1]
#     c2 = tmp[2]

#     t_max = 10.0

#     theta_ell=np.linspace(0,2.0*np.pi,60)
#     theta_hyp1=np.linspace(-0.499*np.pi,0.499*np.pi,60)
#     theta_hyp2=np.linspace( 0.501*np.pi,1.499*np.pi,60)

#     bnd=10.0

#     n_la = 10
#     n_mu = 10
#     n_nu = 10

#     # 焦点位置, +/- Delta_2
#     f_bc = [np.sqrt(b2-c2),-np.sqrt(b2-c2)]

#     # 焦点位置, +/- Delta_1
#     f_ab = [np.sqrt(a2-b2),-np.sqrt(a2-b2)]

#     # 焦点位置, +/- sqrt(Delta_2^2+Delta_1^2)
#     f_ac = [np.sqrt(a2-c2),-np.sqrt(a2-c2)]

#     print("Plotting: a2 = %.2f b2 = %.2f c2 = %.2f" %(a2,b2,c2))

#     ######################################## 函数

#     def Equi_la_on_XY(la,theta):
#         x = np.sqrt(la-a2)*np.cos(theta)
#         y = np.sqrt(la-b2)*np.sin(theta)
#         return x,y

#     def Equi_mu_on_XY(mu,theta):
#         x = np.sqrt(a2-mu)*np.tan(theta)
#         y = np.sqrt(mu-b2)/np.cos(theta)
#         return x,y

#     #########
#     def Equi_la_on_XZ(la,theta):
#         x = np.sqrt(la-a2)*np.cos(theta)
#         z = np.sqrt(la-c2)*np.sin(theta)
#         return x,z


#     def Equi_mu_on_XZ(mu,theta):
#         x = np.sqrt(a2-mu)*np.tan(theta)
#         z = np.sqrt(mu-c2)/np.cos(theta)
#         return x,z


#     def Equi_nu_on_XZ(nu,theta):
#         x = np.sqrt(a2-nu)*np.tan(theta)
#         z = np.sqrt(nu-c2)/np.cos(theta)
#         return x,z

#     #########
#     def Equi_la_on_YZ(la,theta):
#         y = np.sqrt(la-b2)*np.cos(theta)
#         z = np.sqrt(la-c2)*np.sin(theta)
#         return y,z

#     def Equi_mu_on_YZ(mu,theta):
#         y = np.sqrt(mu-b2)*np.cos(theta)
#         z = np.sqrt(mu-c2)*np.sin(theta)
#         return y,z

#     def Equi_nu_on_YZ(nu,theta):
#         y = np.sqrt(b2-nu)*np.tan(theta)
#         z = np.sqrt(nu-c2)/np.cos(theta)
#         return y,z


#     #######################################

#     #######################################


#     '''
#     #        画图：y轴= 3个度规系数, x轴= x
#     '''
#     fig = plt.figure(figsize=(12,12))  #创建图像对象，尺寸24*24英寸
#     #fig = plt.figure(figsize=(24,13.5))  #创建图像对象，尺寸24*13.5英寸(4:3)
#     #plt.title(r"$a^2$ = %.2f $b^2$ = %.2f $c^2$ = %.2f" %(a2,b2,c2), fontsize=18)
#     '''
#     #################################### 图1 : XY
#     fig2 = fig.add_subplot(1,3,1)
#     plt.xlabel('X [kpc]',fontsize=18)
#     plt.ylabel('Y [kpc]',fontsize=18)

#     raw=np.loadtxt("CP-XY.dat")
#     t=raw[:,0]
#     x=raw[:,1]
#     y=raw[:,2]

#     if (x.max()>y.max()):
#         bnd = x.max()*1.1
#     else:
#         bnd = y.max()*1.1

#     plt.xlim([-bnd,bnd])
#     plt.ylim([-bnd,bnd])

#     plt.scatter(x,y,c=t,s=10,marker='o',cmap='rainbow')

#     #XY平面上
#     #等la线为椭圆
#     beg=np.log10(a2)
#     end=np.log10(10*a2)
#     for la in np.logspace(beg,end,n_la):
#         x,y=Equi_la_on_XY(la,theta_ell)
#         plt.plot(x,y,c='red',linestyle='--',linewidth=1)

#     #等mu线双曲线
#     beg=np.log10(b2)
#     end=np.log10(a2)
#     for mu in np.logspace(beg,end,n_mu):
#         x,y=Equi_mu_on_XY(mu,theta_hyp1)
#         plt.plot(x,y,c='blue',linestyle='--',linewidth=1)
#         x,y=Equi_mu_on_XY(mu,theta_hyp2)
#         plt.plot(x,y,c='blue',linestyle='--',linewidth=1)

#     # 焦点位置, +/- Delta_1
#     xf = np.zeros_like(f_ab)
#     yf = f_ab
#     plt.scatter(xf,yf,marker='o',s=100,c='black')

#     ################################ 图2 : XZ
#     fig2 = fig.add_subplot(1,3,2)
#     plt.xlabel('X [kpc]',fontsize=18)
#     plt.ylabel('Z [kpc]',fontsize=18)

#     raw=np.loadtxt("CP-XZ.dat")
#     t=raw[:,0]
#     x=raw[:,1]
#     z=raw[:,2]

#     if (x.max()>z.max()):
#         bnd = x.max()*1.1
#     else:
#         bnd = z.max()*1.1

#     plt.xlim([-bnd,bnd])
#     plt.ylim([-bnd,bnd])

#     plt.scatter(x,z,c=t,s=10,marker='o',cmap='rainbow')

#     #XZ平面上
#     #等 la 线为椭圆
#     beg=np.log10(a2)
#     end=np.log10(10*a2)
#     for la in np.logspace(beg,end,n_la):
#         x,z=Equi_la_on_XZ(la,theta_ell)
#         plt.plot(x,z,c='red',linestyle='--',linewidth=1)

#     #等mu 线双曲线
#     beg=np.log10(b2)
#     end=np.log10(a2)
#     for mu in np.logspace(beg,end,n_mu):
#         x,z=Equi_mu_on_XZ(mu,theta_hyp1)
#         plt.plot(x,z,c='blue',linestyle='--',linewidth=1)
#         x,z=Equi_mu_on_XZ(mu,theta_hyp2)
#         plt.plot(x,z,c='blue',linestyle='--',linewidth=1)

#     #等nu 线双曲线
#     beg=np.log10(c2)
#     end=np.log10(b2)
#     for nu in np.logspace(beg,end,n_nu):
#         x,z=Equi_nu_on_XZ(nu,theta_hyp1)
#         plt.plot(x,z,c='black',linestyle='--',linewidth=1)
#         x,z=Equi_nu_on_XZ(nu,theta_hyp2)
#         plt.plot(x,z,c='black',linestyle='--',linewidth=1)

#     # 焦点位置
#     xf = np.zeros_like(f_bc)
#     zf = f_bc
#     plt.scatter(xf,zf,marker='o',s=100,c='blue')
#     xf = np.zeros_like(f_ac)
#     zf = f_ac
#     plt.scatter(xf,zf,marker='o',s=100,c='red')
#     '''

#     #################################### 图3 : YZ
#     fig2 = fig.add_subplot(1,1,1)
#     plt.xlabel('Y [kpc]',fontsize=18)
#     plt.ylabel('Z [kpc]',fontsize=18)

#     raw=np.loadtxt("Orbit.dat")
#     t=raw[:,0]
#     y=raw[:,2]
#     z=raw[:,3]

#     if (y.max()>z.max()):
#         bnd = y.max()*1.1
#     else:
#         bnd = z.max()*1.1

#     bnd=5
#     plt.xlim([-bnd,bnd])
#     plt.ylim([-bnd,bnd])

#     plt.scatter(y,z,c=t,s=5,marker='o',cmap='rainbow')

#     #YZ平面上
#     #等 la 线为椭圆
#     beg=np.log10(a2)
#     end=np.log10(10*a2)
#     for la in np.logspace(beg,end,n_la):
#         y,z=Equi_la_on_YZ(la,theta_ell)
#         plt.plot(y,z,c='red',linestyle='--',linewidth=1)

#     #等mu 线为椭圆
#     beg=np.log10(b2)
#     end=np.log10(a2)
#     for mu in np.logspace(beg,end,n_mu):
#         y,z=Equi_mu_on_YZ(mu,theta_ell)
#         plt.plot(y,z,c='blue',linestyle='--',linewidth=1)

#     #等nu 线双曲线
#     beg=np.log10(c2)
#     end=np.log10(b2)
#     for nu in np.logspace(beg,end,n_nu):
#         y,z=Equi_nu_on_YZ(nu,theta_hyp1)
#         plt.plot(y,z,c='black',linestyle='--',linewidth=1)
#         y,z=Equi_nu_on_YZ(nu,theta_hyp2)
#         plt.plot(y,z,c='black',linestyle='--',linewidth=1)

#     # 焦点位置
#     yf = np.zeros_like(f_bc)
#     zf = f_bc
#     plt.scatter(yf,zf,marker='o',s=100,c='blue')
#     yf = np.zeros_like(f_ac)
#     zf = f_ac
#     plt.scatter(yf,zf,marker='o',s=100,c='red')
#     yf = f_ab
#     zf = np.zeros_like(f_ac)
#     plt.scatter(yf,zf,marker='o',s=100,c='black')


#     ####################################
#     plt.savefig('YZ-plane.png')
#     exit()

# def plot_plane_for_a2():

#     #######################################

#     tmp = np.loadtxt("a2b2c2.txt")
#     a2 = tmp[0]
#     b2 = tmp[1]
#     c2 = tmp[2]

#     t_max = 10.0

#     theta_ell=np.linspace(0,2.0*np.pi,60)
#     theta_hyp1=np.linspace(-0.499*np.pi,0.499*np.pi,60)
#     theta_hyp2=np.linspace( 0.501*np.pi,1.499*np.pi,60)

#     bnd=10.0

#     n_la = 10
#     n_mu = 10
#     n_nu = 10

#     # 焦点位置, +/- Delta_2
#     f_bc = [np.sqrt(b2-c2),-np.sqrt(b2-c2)]

#     # 焦点位置, +/- Delta_1
#     f_ab = [np.sqrt(a2-b2),-np.sqrt(a2-b2)]

#     # 焦点位置, +/- sqrt(Delta_2^2+Delta_1^2)
#     f_ac = [np.sqrt(a2-c2),-np.sqrt(a2-c2)]

#     print("Plotting: a2 = %.2f b2 = %.2f c2 = %.2f" %(a2,b2,c2))

#     ######################################## 函数

#     def Equi_la_on_XY(la,theta):
#         x = np.sqrt(la-a2)*np.cos(theta)
#         y = np.sqrt(la-b2)*np.sin(theta)
#         return x,y

#     def Equi_mu_on_XY(mu,theta):
#         x = np.sqrt(a2-mu)*np.tan(theta)
#         y = np.sqrt(mu-b2)/np.cos(theta)
#         return x,y

#     #########
#     def Equi_la_on_XZ(la,theta):
#         x = np.sqrt(la-a2)*np.cos(theta)
#         z = np.sqrt(la-c2)*np.sin(theta)
#         return x,z


#     def Equi_mu_on_XZ(mu,theta):
#         x = np.sqrt(a2-mu)*np.tan(theta)
#         z = np.sqrt(mu-c2)/np.cos(theta)
#         return x,z


#     def Equi_nu_on_XZ(nu,theta):
#         x = np.sqrt(a2-nu)*np.tan(theta)
#         z = np.sqrt(nu-c2)/np.cos(theta)
#         return x,z

#     #########
#     def Equi_la_on_YZ(la,theta):
#         y = np.sqrt(la-b2)*np.cos(theta)
#         z = np.sqrt(la-c2)*np.sin(theta)
#         return y,z

#     def Equi_mu_on_YZ(mu,theta):
#         y = np.sqrt(mu-b2)*np.cos(theta)
#         z = np.sqrt(mu-c2)*np.sin(theta)
#         return y,z

#     def Equi_nu_on_YZ(nu,theta):
#         y = np.sqrt(b2-nu)*np.tan(theta)
#         z = np.sqrt(nu-c2)/np.cos(theta)
#         return y,z


#     #######################################

#     #######################################


#     '''
#     #        画图：y轴= 3个度规系数, x轴= x
#     '''
#     fig = plt.figure(figsize=(12,12))  #创建图像对象，尺寸24*24英寸
#     #fig = plt.figure(figsize=(24,13.5))  #创建图像对象，尺寸24*13.5英寸(4:3)
#     #plt.title(r"$a^2$ = %.2f $b^2$ = %.2f $c^2$ = %.2f" %(a2,b2,c2), fontsize=18)

#     #################################### 图1 : XY
#     fig2 = fig.add_subplot(1,1,1)
#     plt.xlabel('X [kpc]',fontsize=18)
#     plt.ylabel('Y [kpc]',fontsize=18)

#     raw=np.loadtxt("Orbit.dat")
#     t=raw[:,0]
#     x=raw[:,1]
#     y=raw[:,2]

#     if (x.max()>y.max()):
#         bnd = x.max()*1.1
#     else:
#         bnd = y.max()*1.1

#     plt.xlim([-bnd,bnd])
#     plt.ylim([-bnd,bnd])

#     plt.scatter(x,y,c=t,s=3,marker='o',cmap='rainbow')

#     #XY平面上
#     #等la线为椭圆
#     beg=np.log10(a2)
#     end=np.log10(10*a2)
#     for la in np.logspace(beg,end,n_la):
#         x,y=Equi_la_on_XY(la,theta_ell)
#         plt.plot(x,y,c='red',linestyle='--',linewidth=1)

#     #等mu线双曲线
#     beg=np.log10(b2)
#     end=np.log10(a2)
#     for mu in np.logspace(beg,end,n_mu):
#         x,y=Equi_mu_on_XY(mu,theta_hyp1)
#         plt.plot(x,y,c='blue',linestyle='--',linewidth=1)
#         x,y=Equi_mu_on_XY(mu,theta_hyp2)
#         plt.plot(x,y,c='blue',linestyle='--',linewidth=1)

#     # 焦点位置, +/- Delta_1
#     xf = np.zeros_like(f_ab)
#     yf = f_ab
#     plt.scatter(xf,yf,marker='o',s=100,c='black')

#     '''
#     ################################ 图2 : XZ
#     fig2 = fig.add_subplot(1,3,2)
#     plt.xlabel('X [kpc]',fontsize=18)
#     plt.ylabel('Z [kpc]',fontsize=18)

#     raw=np.loadtxt("CP-XZ.dat")
#     t=raw[:,0]
#     x=raw[:,1]
#     z=raw[:,2]

#     if (x.max()>z.max()):
#         bnd = x.max()*1.1
#     else:
#         bnd = z.max()*1.1

#     plt.xlim([-bnd,bnd])
#     plt.ylim([-bnd,bnd])

#     plt.scatter(x,z,c=t,s=10,marker='o',cmap='rainbow')

#     #XZ平面上
#     #等 la 线为椭圆
#     beg=np.log10(a2)
#     end=np.log10(10*a2)
#     for la in np.logspace(beg,end,n_la):
#         x,z=Equi_la_on_XZ(la,theta_ell)
#         plt.plot(x,z,c='red',linestyle='--',linewidth=1)

#     #等mu 线双曲线
#     beg=np.log10(b2)
#     end=np.log10(a2)
#     for mu in np.logspace(beg,end,n_mu):
#         x,z=Equi_mu_on_XZ(mu,theta_hyp1)
#         plt.plot(x,z,c='blue',linestyle='--',linewidth=1)
#         x,z=Equi_mu_on_XZ(mu,theta_hyp2)
#         plt.plot(x,z,c='blue',linestyle='--',linewidth=1)

#     #等nu 线双曲线
#     beg=np.log10(c2)
#     end=np.log10(b2)
#     for nu in np.logspace(beg,end,n_nu):
#         x,z=Equi_nu_on_XZ(nu,theta_hyp1)
#         plt.plot(x,z,c='black',linestyle='--',linewidth=1)
#         x,z=Equi_nu_on_XZ(nu,theta_hyp2)
#         plt.plot(x,z,c='black',linestyle='--',linewidth=1)

#     # 焦点位置
#     xf = np.zeros_like(f_bc)
#     zf = f_bc
#     plt.scatter(xf,zf,marker='o',s=100,c='blue')
#     xf = np.zeros_like(f_ac)
#     zf = f_ac
#     plt.scatter(xf,zf,marker='o',s=100,c='red')


#     #################################### 图3 : YZ
#     fig2 = fig.add_subplot(1,1,1)
#     plt.xlabel('Y [kpc]',fontsize=18)
#     plt.ylabel('Z [kpc]',fontsize=18)

#     raw=np.loadtxt("Orbit.dat")
#     t=raw[:,0]
#     y=raw[:,2]
#     z=raw[:,3]

#     if (y.max()>z.max()):
#         bnd = y.max()*1.1
#     else:
#         bnd = z.max()*1.1

#     plt.xlim([-bnd,bnd])
#     plt.ylim([-bnd,bnd])

#     plt.scatter(y,z,c=t,s=10,marker='o',cmap='rainbow')

#     #YZ平面上
#     #等 la 线为椭圆
#     beg=np.log10(a2)
#     end=np.log10(10*a2)
#     for la in np.logspace(beg,end,n_la):
#         y,z=Equi_la_on_YZ(la,theta_ell)
#         plt.plot(y,z,c='red',linestyle='--',linewidth=1)

#     #等mu 线为椭圆
#     beg=np.log10(b2)
#     end=np.log10(a2)
#     for mu in np.logspace(beg,end,n_mu):
#         y,z=Equi_mu_on_YZ(mu,theta_ell)
#         plt.plot(y,z,c='blue',linestyle='--',linewidth=1)

#     #等nu 线双曲线
#     beg=np.log10(c2)
#     end=np.log10(b2)
#     for nu in np.logspace(beg,end,n_nu):
#         y,z=Equi_nu_on_YZ(nu,theta_hyp1)
#         plt.plot(y,z,c='black',linestyle='--',linewidth=1)
#         y,z=Equi_nu_on_YZ(nu,theta_hyp2)
#         plt.plot(y,z,c='black',linestyle='--',linewidth=1)

#     # 焦点位置
#     yf = np.zeros_like(f_bc)
#     zf = f_bc
#     plt.scatter(yf,zf,marker='o',s=100,c='blue')
#     yf = np.zeros_like(f_ac)
#     zf = f_ac
#     plt.scatter(yf,zf,marker='o',s=100,c='red')
#     yf = f_ab
#     zf = np.zeros_like(f_ac)
#     plt.scatter(yf,zf,marker='o',s=100,c='black')
#     '''

#     ####################################
#     plt.savefig('XY-plane.png')
#     exit()



def plot_orbit_3d(x, savename, is_show=False):
    fig = plt.figure()
    pointsize = 1.
    fontsize = 2.
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.grid(True)
    ax.scatter(x[:,0], x[:,1], x[:,2], s=pointsize, marker="+")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig_tmp = plt.gcf()
    fig_tmp.savefig(savename+"_3d.png", format="png", dpi=300, bbox_inches='tight')
    if is_show:
        plt.show()
    plt.close("all")
    
def plot_orbit_each_2d(x, savename, is_show=False):
    fig = plt.figure()
    pointsize = 0.2
    fontsize = 2.
    xyz_name = ["x (kpc)", "y (kpc)", "z (kpc)"]
    for i in np.arange(3):
        ax = fig.add_subplot(2,2,i+1)
        ax.grid(True)
        ax.scatter(x[:,i%3], x[:,(i+1)%3], s=pointsize, marker="+")
        ax.set_xlabel(xyz_name[i%3])
        ax.set_ylabel(xyz_name[(i+1)%3])
    fig_tmp = plt.gcf()
    fig_tmp.savefig(savename+".png", format="png", dpi=300, bbox_inches='tight')
    if is_show:
        plt.show()
    plt.close("all")

def is_status_bad_orbit_ellip_at_plane(
    x, swit_axis_along, dimension=3, percent_acceptence=0.1
):
    dim0 = (swit_axis_along)%dimension
    dim1 = (swit_axis_along+1)%dimension
    dim2 = (swit_axis_along+2)%dimension
    dl0 = (np.max(x[:, dim0])-np.min(x[:, dim0]))/2.
    dl1 = (np.max(x[:, dim1])-np.min(x[:, dim1]))/2.
    dl2 = (np.max(x[:, dim2])-np.min(x[:, dim2]))/2.
    if (dl0>np.min([dl1, dl2])*percent_acceptence):
        return 0
    else:
        return 1 

def foci_by_shape_simpleminmax(x, swit_axis_along, dimension=3):
    dim1 = (swit_axis_along+1)%dimension
    dim2 = (swit_axis_along+2)%dimension
    dl1 = (np.max(x[:, dim1])-np.min(x[:, dim1]))/2.
    dl2 = (np.max(x[:, dim2])-np.min(x[:, dim2]))/2.
    print("dim, dl: ", dim1, dl1, dim2, dl2)
    D = abs(dl2**2-dl1**2)
    return D

def dim_by_swit_axis_along_3_to_2(swit_axis_along):
    dim1 = (swit_axis_along+1)%3
    dim2 = (swit_axis_along+2)%3
    return dim1, dim2

def fit_ellip_2d(points):
    points_int = (points*1000).astype(int)
    center, axes, angle = cv2.fitEllipse(points_int)
    return (center[0]/1000., center[1]/1000.), \
        (axes[0]/2000., axes[1]/2000.), \
        angle #(center_x, center_y), (semi_short_axis, long), angle_from_x??

def func_line_be_1d(x, p_0, p_1):
    return x*p_0 + p_1

def fit_line_be_1d(x, y, p0):
    p = spopt.curve_fit(func_line_be_1d, x, y, p0=p0)
    return p

def load_orbit_stream(filename, ncol=8):
    """Robust loader for orbit_*.dat that tolerates 'fat' lines.

    It reads the file as a flat stream of floats and then reshapes
    into (N, ncol). Any trailing incomplete record is discarded.
    """
    # vals = []
    # with open(filename, "r") as f:
    #     for line in f:
    #         line = line.strip()
    #         if not line:
    #             continue
    #         for tok in line.split():
    #             try:
    #                 vals.append(float(tok))
    #             except ValueError:
    #                 # skip any non-numeric junk
    #                 continue
    # if not vals:
    #     raise RuntimeError(f"No numeric data found in {filename}")
    # nrec = len(vals) // ncol
    # if len(vals) % ncol != 0:
    #     print(f"Warning: {filename} has {len(vals)} floats, "
    #           f"truncating to {nrec*ncol}.")
    # arr = np.array(vals[:nrec*ncol], dtype=float).reshape(nrec, ncol)
    
    vals = []
    idx = 0
    with open(filename, "r") as f:
        for line in f:
            # line = (line.strip()).split()
            line = (line).split()
            if not line or len(line) != ncol:
                continue
            try:
                for tok in line:
                    vals.append(float(tok))
            except ValueError:
                continue
            idx += 1
    arr = np.array(vals, dtype=float).reshape(idx, 8)
    return arr

def remake_foci_by_lin_interp_first_several(f, n_start, b2_0=-1.05, a2_0=-1.2, ycut_0=0.5):
    if f[0,4]<ycut_0-1e-8:
        print("The minimum ycut is too less. Exit.")
        exit(0)
    f1 = f
    ycut_remake = f[0:n_start, 4]
    kb = (f[n_start,1]-b2_0)/(f[n_start,4]-ycut_0)
    b2_remake = (ycut_remake-ycut_0)*kb+b2_0
    ka = (f[n_start,2]-a2_0)/(f[n_start,4]-ycut_0)
    a2_remake = (ycut_remake-ycut_0)*ka+a2_0
    f1[0:n_start, 1] = b2_remake
    f1[0:n_start, 2] = a2_remake
    return f1

def plot_foci_table_compare_old():
    ## old code to see various method
    filename_foci_debug = "foci_example.txt"
    write_content1 = np.loadtxt(filename_foci_debug) #TACT formula
    filename_foci_debug = "some_lmn_foci_Pot_old.txt"
    write_content2 = np.loadtxt(filename_foci_debug)
    filename_foci_debug = "foci_debug.txt"
    write_content3 = np.loadtxt(filename_foci_debug)
    filename_foci_debug = "../some_lmn_foci_Pot.txt"
    write_content9 = np.loadtxt(filename_foci_debug) #newest GSS

    y_init1 = np.linspace(0.5, 300., N_grid)
    y_init2 = write_content2[:,4]
    y_init3 = write_content2[:,4]
    # y_init1 = np.abs(write_content1[:,0])
    # y_init2 = np.abs(write_content2[:,0])
    # y_init3 = np.abs(write_content2[:,0])
    #?? when yinit<60.kpc, use the example foci
    y_init9 = np.linspace(0.5, 250., N_grid)

    E_1 = write_content1[:,0]
    E_9 = write_content9[:,0]

    n_start = 7
    y_init4 = y_init3
    # y_init4[0:n_start] = y_init1[0:n_start]
    ab4 = np.zeros((N_grid, 2))
    ab4[0:n_start] = write_content1[0:n_start,1:3]
    ab4[n_start:] = write_content3[n_start:,1:3]

    n_start = 0
    pbeta = fit_line_be_1d(y_init3[n_start:], write_content3[n_start:,1], [-1.,-1.])[0]
    print(pbeta)
    palpha = fit_line_be_1d(y_init3[n_start:], write_content3[n_start:,2], [-1.,-1.])[0]

    ab1 = write_content1[:,1:3]
    ab9 = write_content9[:,1:3]



    # plt.plot(y_init1, -write_content1[:,1], label="beta  by TACT_example", color="red", marker=".")
    # plt.plot(y_init2, -write_content2[:,1], label="beta  by do_rotate (not check)", color="green", marker=".")
    # plt.plot(y_init3, -write_content3[:,1], label="beta  by not_rotate", color="blue", marker=".")
    # plt.plot(y_init3, -func_line_be_1d(y_init3, *pbeta), label="beta  by line fit", color="k")
    # plt.plot(y_init4, -ab4[:,0], label="beta  by comp", color="purple", marker=".")
    # plt.plot(y_init1, -write_content1[:,2], label="alpha by TACT_example", color="red", marker="+")
    # plt.plot(y_init2, -write_content2[:,2], label="alpha by do_rotate (not check)", color="green", marker="+")
    # plt.plot(y_init3, -write_content3[:,2], label="alpha by not_rotate", color="blue", marker="+")
    # plt.plot(y_init3, -func_line_be_1d(y_init3, *palpha), label="alpha  by line fit", color="k")
    # plt.plot(y_init4, -ab4[:,1], label="alpha by comp", color="purple", marker=".")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # # plt.show()
    # plt.close()

    plt.plot(E_1, -ab1[:,0], label="b2 by TACT_formula", color="red", marker=".")
    plt.plot(E_1, -ab1[:,1], label="a2 by TACT_formula", color="red", marker="+")
    plt.plot(E_9, -ab9[:,0], label="b2 by SCF_GSS", color="blue", marker=".")
    plt.plot(E_9, -ab9[:,1], label="a2 by SCF_GSS", color="blue", marker="+")
    plt.legend()
    # plt.show()
    plt.close()

    plt.plot(y_init1, -ab1[:,0], label="b2 by TACT_formula", color="red", marker=".")
    plt.plot(y_init1, -ab1[:,1], label="a2 by TACT_formula", color="red", marker="+")
    plt.plot(y_init9, -ab9[:,0], label="b2 by SCF_GSS", color="blue", marker=".")
    plt.plot(y_init9, -ab9[:,1], label="a2 by SCF_GSS", color="blue", marker="+")
    plt.plot(y_init9, np.abs(-ab1[:,1]+ab1[:,0]), label="a2mb2 by TACT_formula", color="g", marker="+")
    plt.plot(y_init9, np.abs(-ab9[:,1]+ab9[:,0]), label="a2mb2 by SCF_GSS", color="k", marker="+")
    plt.legend()
    plt.show()
    plt.close()

    # filename = "some_lmn_foci_Pot.txt"
    # arr = np.zeros_like(write_content3)
    # arr[:,0] = y_init4
    # arr[:,1:3] = ab4
    # np.savetxt(filename, arr)
    print("Plot, done.")



if __name__=="__main__":

    ## to re-search the foci
    snapshot = int(sys.argv[1])

    #1. #rewrite the original foci table, by shape from C++
    foci_table_the_old_path = "../../../GDDFAA/"\
        +"step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
        +"galaxy_general/intermediate/snapshot_%d_lmn_foci_Pot.old.txt"%(snapshot)
    foci_table_init = np.loadtxt(foci_table_the_old_path) #init
    foci_table_init_sortidx = np.argsort(foci_table_init[:, 4])
    print("re-sort index: ", foci_table_init_sortidx)
    foci_table_init = foci_table_init[foci_table_init_sortidx]

    foci_table_by_minlam_path = "../../../GDDFAA/"\
        +"step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
        +"galaxy_general/intermediate/snapshot_%d_lmn_foci_Pot.by_minlam.txt"%(snapshot)
    foci_table_by_minlam = foci_table_init*1.
    np.savetxt(foci_table_by_minlam_path, foci_table_by_minlam)

    #2. calculate foci table by fitellip
    foci_table_by_fitellip = foci_table_init*1.
    N_grid = len(foci_table_by_fitellip)
    # is_show = True
    is_show = False
    write_content = []

    is_plot_orbit = True
    # is_plot_orbit = False
    orbit_snapshot = "../../../GDDFAA/"\
        +"step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
        +"galaxy_general/intermediate/orbit_%d/"%(snapshot)
    if is_plot_orbit:
        for i in np.arange(N_grid):
            print("plot orbit_%d ..."%(i))

            #|| swit 0
            swit = 0 #orbit rotates along x-axis (swit 0) and is at yz plane
            filename = orbit_snapshot+"orbit_%d_b2.dat"%(i)
            savename = orbit_snapshot+"orbit_%d_b2"%(i)
            # data = np.loadtxt(filename) #cols: time, x, y, z, vx, vy, vz, pot
            data = load_orbit_stream(filename) #cols: time, x, y, z, vx, vy, vz, pot
            x = data[:, 1:4]
            v = data[:, 4:7]
            # plot_orbit_3d(x, savename, is_show=is_show)
            plot_orbit_each_2d(x, savename)

            status_bad_yz = is_status_bad_orbit_ellip_at_plane(x, swit)
            D2_yz = foci_by_shape_simpleminmax(x, swit)
            beta = -1.-D2_yz
            print("status_bad_yz = %d, D2_yz = %f"%(status_bad_yz, D2_yz))

            dim1, dim2 = dim_by_swit_axis_along_3_to_2(swit)
            ba = fit_ellip_2d(x[:,[dim1,dim2]])[1]
            # plt.plot(x[:, dim1], x[:, dim2])
            # plt.show()
            D2_yz = np.abs(ba[0]**2-ba[1]**2)
            beta = -1.-D2_yz
            print(ba, D2_yz)

            #|| swit 2
            swit = 2 #orbit rotates along z-axis (swit 2) and is at xy plane
            filename = orbit_snapshot+"orbit_%d_a2.dat"%(i)
            savename = orbit_snapshot+"orbit_%d_a2"%(i)
            # data = np.loadtxt(filename) #cols: time, x, y, z, vx, vy, vz, pot
            data = load_orbit_stream(filename) #cols: time, x, y, z, vx, vy, vz, pot
            x = data[:, 1:4]
            v = data[:, 4:7]
            # plot_orbit_3d(x, savename, is_show=is_show)
            plot_orbit_each_2d(x, savename)
            
            status_bad_zx = is_status_bad_orbit_ellip_at_plane(x, swit)
            D2_zx = foci_by_shape_simpleminmax(x, swit)
            alpha = beta-D2_zx
            print("status_bad_zx = %d, D2_zx = %f"%(status_bad_zx, D2_zx))
            
            dim1, dim2 = dim_by_swit_axis_along_3_to_2(swit)
            ba = fit_ellip_2d(x[:,[dim1,dim2]])[1]
            # plt.plot(x[:, dim1], x[:, dim2])
            # plt.show()
            D2_zx = np.abs(ba[0]**2-ba[1]**2)
            alpha = beta-D2_zx
            print(ba, D2_zx)

            write_content.append([0., beta, alpha, status_bad_yz, D2_yz, status_bad_zx, D2_zx])
        
        # comment_header = "# E, beta, alpha, status_bad_yz, D2_yz, status_bad_zx, D2_zx"
        write_content = np.array(write_content)
        foci_table_by_fitellip[:,1] = write_content[:,1]*1.
        foci_table_by_fitellip[:,2] = write_content[:,2]*1.
        
    foci_table_by_fitellip_path = "../../../GDDFAA/"\
        +"step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
        +"galaxy_general/intermediate/snapshot_%d_lmn_foci_Pot.by_fitellip.txt"%(snapshot)
    np.savetxt(foci_table_by_fitellip_path, foci_table_by_fitellip)

    #3. or by minlam from C++
    foci_table_by_shape_path = "../../../GDDFAA/"\
        +"step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
        +"galaxy_general/intermediate/snapshot_%d_lmn_foci_Pot.by_minlam.txt"%(snapshot)
    foci_table_by_shape = foci_table_init*1.
    foci_table_by_shape[:,1] = foci_table_init[:,5]*1.
    foci_table_by_shape[:,2] = foci_table_init[:,6]*1.
    np.savetxt(foci_table_by_shape_path, foci_table_by_shape)

    #4. linear interpolate the first several oscillate value in foci table
    foci_table_by_firstseveral_path = "../../../GDDFAA/"\
        +"step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
        +"galaxy_general/intermediate/snapshot_%d_lmn_foci_Pot.by_firstseveral.txt"%(snapshot)
    n_start = 6 #the minimum ycut that less than 35.0 kpc
    if N_grid-1<=n_start:
        n_start = N_grid-1
    foci_table_by_firstseveral = remake_foci_by_lin_interp_first_several(foci_table_by_fitellip*1., n_start)
    np.savetxt(foci_table_by_firstseveral_path, foci_table_by_firstseveral)
    # is_lin_interp_first_several = True
    is_lin_interp_first_several = False
    foci_table_to_use = None
    if is_lin_interp_first_several:
        foci_table_to_use = foci_table_by_firstseveral
    else:
        # foci_table_to_use = foci_table_init
        # foci_table_to_use = foci_table_by_fitellip
        foci_table_to_use = foci_table_by_shape
    
    print("Rewrite the foci table to use.")
    foci_table_to_use_path = "../../../GDDFAA/"\
        +"step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
        +"galaxy_general/intermediate/snapshot_%d_lmn_foci_Pot.txt"%(snapshot)
    np.savetxt(foci_table_to_use_path, foci_table_to_use)



    ## plot foci table
    plt.figure()
    plt.plot(foci_table_to_use[:,4], foci_table_to_use[:,1], marker=".", label=r"$-b^2$")
    plt.plot(foci_table_to_use[:,4], foci_table_to_use[:,2], marker=".", label=r"$-a^2$")
    plt.xlabel(r"y cut ($\mathrm{kpc}$)")
    plt.ylabel(r"minus semi-axis square ($\mathrm{kpc}^2$)")
    plt.legend()
    fig_tmp = plt.gcf()
    fig_tmp.savefig(orbit_snapshot+"foci_y0.png", format="png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    plt.figure()
    plt.plot(foci_table_to_use[:,0], foci_table_to_use[:,1], marker=".", label=r"$-b^2$")
    plt.plot(foci_table_to_use[:,0], foci_table_to_use[:,2], marker=".", label=r"$-a^2$")
    plt.xlabel(r"energy ($(\mathrm{km/s})^2$)")
    plt.ylabel(r"minus semi-axis square ($\mathrm{kpc}^2$)")
    plt.legend()
    fig_tmp = plt.gcf()
    fig_tmp.savefig(orbit_snapshot+"foci_E.png", format="png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    print("Plot the foci table to use. Done.")

    plot_foci_table_compare = True
    # plot_foci_table_compare = False
    if plot_foci_table_compare:
        plt.figure()
        idx = 4 #y_cut
        plt.plot(foci_table_by_shape[:,idx], foci_table_by_shape[:,1], marker=".", color="red", label=r"$-b^2$ by_shape")
        plt.plot(foci_table_by_shape[:,idx], foci_table_by_shape[:,2], marker="+", color="red", label=r"$-a^2$ by_shape")
        plt.plot(foci_table_by_minlam[:,idx], foci_table_by_minlam[:,1], marker=".", color="yellow", label=r"$-b^2$ by_mintau")
        plt.plot(foci_table_by_minlam[:,idx], foci_table_by_minlam[:,2], marker="+", color="yellow", label=r"$-a^2$ by_mintau")
        plt.plot(foci_table_by_fitellip[:,idx], foci_table_by_fitellip[:,1], marker=".", color="blue", label=r"$-b^2$ by_fitellip")
        plt.plot(foci_table_by_fitellip[:,idx], foci_table_by_fitellip[:,2], marker="+", color="blue", label=r"$-a^2$ by_fitellip")
        # plt.plot(foci_table_by_firstseveral[:,idx], foci_table_by_firstseveral[:,1], marker=".", color="green", label=r"$-b^2$ by_interpfirst")
        # plt.plot(foci_table_by_firstseveral[:,idx], foci_table_by_firstseveral[:,2], marker="+", color="green", label=r"$-a^2$ by_interpfirst")
        plt.xlabel(r"y cut ($\mathrm{kpc}$)")
        plt.ylabel(r"minus semi-axis square ($\mathrm{kpc}^2$)")
        plt.legend()
        fig_tmp = plt.gcf()
        fig_tmp.savefig(orbit_snapshot+"foci_y0.compare.png", format="png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

        plt.figure()
        idx = 0 #energy
        plt.plot(foci_table_by_shape[:,idx], foci_table_by_shape[:,1], marker=".", color="red", label=r"$-b^2$ by_shape")
        plt.plot(foci_table_by_shape[:,idx], foci_table_by_shape[:,2], marker="+", color="red", label=r"$-a^2$ by_shape")
        plt.plot(foci_table_by_minlam[:,idx], foci_table_by_minlam[:,1], marker=".", color="yellow", label=r"$-b^2$ by_mintau")
        plt.plot(foci_table_by_minlam[:,idx], foci_table_by_minlam[:,2], marker="+", color="yellow", label=r"$-a^2$ by_mintau")
        plt.plot(foci_table_by_fitellip[:,idx], foci_table_by_fitellip[:,1], marker=".", color="blue", label=r"$-b^2$ by_fitellip")
        plt.plot(foci_table_by_fitellip[:,idx], foci_table_by_fitellip[:,2], marker="+", color="blue", label=r"$-a^2$ by_fitellip")
        # plt.plot(foci_table_by_firstseveral[:,idx], foci_table_by_firstseveral[:,1], marker=".", color="green", label=r"$-b^2$ by_interpfirst")
        # plt.plot(foci_table_by_firstseveral[:,idx], foci_table_by_firstseveral[:,2], marker="+", color="green", label=r"$-a^2$ by_interpfirst")
        plt.xlabel(r"energy ($(\mathrm{km/s})^2$)")
        plt.ylabel(r"minus semi-axis square ($\mathrm{kpc}^2$)")
        plt.legend()
        fig_tmp = plt.gcf()
        fig_tmp.savefig(orbit_snapshot+"foci_E.compare.png", format="png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        print("Plot the foci tables to compare. Done.")
