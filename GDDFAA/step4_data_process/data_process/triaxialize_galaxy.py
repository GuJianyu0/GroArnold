#!/usr/bin/env python
# -*- coding:utf-8 -*-

# ===================================================================================
# Description: To make a galaxy steady and fix a triaxial-like galaxy
# to the directions of xyz-axis:
#   centralize
#   eliminate_totalRotation
#   estimate_axisRatio
#   estimate_mainAxisDirection
#   triaxialize
# ===================================================================================

# In[]
# from ast import operator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import scipy.optimize as spopt
from scipy.optimize import curve_fit
# from re import DEBUG
# from IPyhton.display import Latex

import affine_trans_group as atg
import galaxy_models as gm
import analysis_data_distribution as ads
import fit_rho_fJ as fff

# In[]
# calculator

def load_snapshot(
    basepath="galaxy_general/txt/", snapshot=0,
    startIndex_xv=0, startIndex_t=9, startIndex_mass=8, startIndex_potential_and_forces=14
):
    datapath = basepath+"snapshot_%d.txt" % (snapshot)
    data = np.loadtxt(datapath, dtype=float)
    N_ptcs = len(data)
    x = data[:, startIndex_xv:startIndex_xv+3]
    v = data[:, startIndex_xv+3:startIndex_xv+6]
    m = data[:, startIndex_mass]
    print("Loaded %s, done." % (datapath))
    return N_ptcs, x, v, m

def is_correct_shape_motion_data(x, v, m):
    n, l = x.shape
    n1, l1 = v.shape
    n2 = m.shape[0]
    if not(n1 == n and l1 == l and n2 == n):
        print(r"Wrong shape of input argument! Now exit.")
        print(r"The lengths are: %d %d, %d %d, %d.", n, l, n1, l1, n2)
        return False, 0, 0
    else:
        return True, n, l

def rewrite_snapshot(x, v, m, comp_gal=0, 
    basepath="galaxy_general/aa/",  
    snapshot=0, suffix=""
):
    datapath = basepath+"snapshot_%d.%s.txt" % (snapshot, suffix)
    is_correct, n, l = is_correct_shape_motion_data(x, v, m)
    if not is_correct:
        exit(0)
    file_handle = open(datapath, mode="w")
    for i in np.arange(n):
        file_handle.write("%e %e %e %e %e %e     %e %d \n" 
            % (x[i, 0], x[i, 1], x[i, 2], v[i, 0], v[i, 1], v[i, 2], 
            m[i], comp_gal)
        )
    file_handle.close()
    print("Rewrite %s, done." % (datapath))
    return 0

def write_preprocessed_SCF(x, v, mass, 
    path_file="galaxy_general/txt/galaxy_general.SCF.txt"
):
    is_correct, n, l = is_correct_shape_motion_data(x, v, mass)
    if not is_correct:
        exit(0)
    file_handle = open(path_file, mode="w")
    file_handle.write("%d \n%d \n%le \n"%(100, n, 0.01))
    for i in np.arange(n):
        file_handle.write("%d %e    %e %e %e %e %e %e\n" #??
            % (i+1, mass[i], 
            x[i, 0], x[i, 1], x[i, 2], v[i, 0], v[i, 1], v[i, 2])
        )
    file_handle.close()
    print("python: write_preprocessed_SCF(): write %s ... Done."%(path_file));
    return 0

def mean_coordinate(x, v, m):
    is_correct, n, l = is_correct_shape_motion_data(x, v, m)
    if not is_correct:
        exit(0)
    x_mean_old, v_mean_old = np.zeros(l), np.zeros(l)
    for i in np.arange(l):
        x_mean_old[i] = np.sum(m[:]*x[:, i])/np.sum(m[:])
        v_mean_old[i] = np.sum(m[:]*v[:, i])/np.sum(m[:])
    return x_mean_old, v_mean_old

def angularMoment(x, v, m_input=None):
    '''
    angularMoment of each particle
    '''
    m = None
    n = len(x) #particle count
    if m_input is None: #if all the particles has unit mass
        m = np.ones(n)
    else:
        m = m_input
    l = 3
    L = np.zeros((n, l))
    L[:, 0] = m[:]*(x[:, 1]*v[:, 2] - x[:, 2]*v[:, 1])
    L[:, 1] = m[:]*(x[:, 2]*v[:, 0] - x[:, 0]*v[:, 2])
    L[:, 2] = m[:]*(x[:, 0]*v[:, 1] - x[:, 1]*v[:, 0])
    return L

def angularMoment_total(x, v, m, r_range=None):
    '''
    total angularMoment
    '''
    L = angularMoment(x, v, m)
    if r_range is not None: #select particles in certain radius range
        # r = ads.norm_l(x, axis=1, l=2)
        r = np.linalg.norm(x, axis=1, ord=2)
        mask = None
        mask = (r>=r_range[0])
        mask = mask & (r<=r_range[1])
        L = L[mask]
    return np.sum(L, axis=0)

def interiaMoment(x, v, m):  # tensor of interiaMoment
    '''
    # is_correct, n, l = is_correct_shape_motion_data(x, v, m)
    # if not is_correct:
    #     exit(0)
    # if 1:
        # cl = np.where(r<r_range[0])[0]
        # L = L[cl]
    '''
    is_correct, n, l = is_correct_shape_motion_data(x, v, m)
    if not is_correct:
        exit(0)
    I = np.zeros((n, l, l))
    for i in np.arange(l):
        for j in np.arange(l):
            I[:, i, j] = m[:]*x[:, i]*x[:, j]
    return I

def interiaMoment_total(x, v, m, r_range=None):
    is_correct, n, l = is_correct_shape_motion_data(x, v, m)
    if not is_correct:
        exit(0)
    IA = np.zeros((l, l))
    I = interiaMoment(x, v, m)
    if r_range is not None: #select particles in certain radius range
        # r = ads.norm_l(x, axis=1, l=2)
        r = np.linalg.norm(x, axis=1, ord=2)
        mask = None
        mask = (r>=r_range[0])
        mask = mask & (r<=r_range[1])
        I = I[mask]
    return np.sum(I, axis=0)

def frequency_by_interiaMoment_each(I, L):  # I \Omega = L
    sI0, sI1 = I.shape
    sL = L.shape[0]
    if not(sI0 == sI1 and sI0 == sL):
        print(r"Wrong shape of input argument! Now exit.")
        print(r"The lengths are: %d %d, %d.", sI0, sI1, sL)
        exit(0)
    O = np.linalg.solve(np.matrix(I), np.matrix(L).T)
    return O

def operator_mainAxises(W):
    '''
    计算转动惯量MOI及其特征向量. 
    **注意: $\texttt{eigenvectors}$矩阵的列是三个特征向量. **
    使用另一种方法求新坐标系下的坐标, 即把向量$\mathbf{r}$
    向三个特征向量($\mathbf{l,m,n}$)投影得到其坐标. 
    '''
    l, l1 = W.shape
    if l != l1:
        print(r"Wrong shape of input argument! Now exit.")
        print(r"The lengths are: %d %d.", l, l1)
        exit(0)

    # #debug1
    # x = np.matrix([[1.1],[2.1],[3.1]])
    # A = W
    # C = eigenvalues
    # # evals =   [ 32447.72078194  49223.01569287 136828.03904702]
    # B = eigenvectors
    # # evecs =  [[ 0.32400089  0.36242551 -0.87388281]
    # #           [ 0.48939778  0.72630543  0.4826699 ]
    # #           [ 0.80963772 -0.58406178  0.05795337]]
    # C1 = eigenvectors[0]
    # C2 = eigenvectors[:,0]
    # a = np.dot(W, x)
    # c = C[0]*x[0]
    # b1 = np.dot(B, x)
    # b2 = np.dot(B.T, x)
    # ads.DEBUG_PRINT_V(0, eigenvalues, eigenvectors, "eigen", a, c, b1, b2) #, c1, c2)
    
    # #debug2
    # MOI = np.matrix(W)
    # print('MOI (debug): \n',MOI)
    # eigenvalues, eigenvectors = np.linalg.eigh(MOI)
    # ads.DEBUG_PRINT_V(1, "eigen before: ", eigenvalues, eigenvectors)
    # #提取三个特征向量，为列向量
    # vl = eigenvectors[:,2] #should sort
    # vm = eigenvectors[:,1]
    # vn = eigenvectors[:,0]
    # print("vl_original = ",vl )
    # print("vm_original = ",vm )
    # print("vn_original = ",vn )
    # #特征向量正交性检查
    # print( np.matmul(vl.T,vn) )
    # print("vl_T = ",vl.T ) #gjy add
    # print("vm_T = ",vm.T )
    # print("vn_T = ",vn.T )

    TI = np.matrix(np.zeros((l, l)))
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.matrix(W))  # vectors has been normalized to 1
    # ads.DEBUG_PRINT_V(1, "eigen before: ", eigenvalues, eigenvectors)
    sortlist = np.argsort(-np.abs(eigenvalues)) #resort by ev from large to small
    eigenvalues = eigenvalues[sortlist]
    eigenvectors = eigenvectors[:, sortlist]
    for i in np.arange(l):
        TI[i] = eigenvectors[i]
    # print(TI, np.linalg.det(TI), np.linalg.det(TI.I))
    # ads.DEBUG_PRINT_V(0, "operator_mainAxises(): ", TI, eigenvalues, eigenvectors)
    #: to return rotation operater T (x_new = x_old \dot T), EVA, EVE
    return TI.I, eigenvalues, eigenvectors

def estimate_axisRatio(x, v, m, tag=0):  # ??
    is_correct, n, l = is_correct_shape_motion_data(x, v, m)
    if not is_correct:
        exit(0)
    q1, q2 = 0., 0.
    if tag == 0:  # esitimate by interiaMoment
        q1, q2 = q1, q2
    elif tag == 1:  # esitimate by (mean $mr$)
        q1, q2 = q1, q2
    else:  # esitimate by fitting mass density
        print("Please fitting and loading paramters.")
    return q1, q2

# dealing with
def centralize_coordinate(x, v, m):
    is_correct, n, l = is_correct_shape_motion_data(x, v, m)
    if not is_correct:
        exit(0)
    x_ = x*1.
    v_ = v*1.
    xMEAN, vMEAN = mean_coordinate(x, v, m)
    for i in np.arange(l):
        x_[:, i] -= xMEAN[i]
        v_[:, i] -= vMEAN[i]
    operators = [xMEAN, vMEAN]
    return x_, v_, operators


def eliminate_totalRotation(x, v, m, tag=4, r_range=None):  # same with rotate main axis ??
    is_correct, n, l = is_correct_shape_motion_data(x, v, m)
    if not is_correct:
        exit(0)
    # axis=0: shrink particle index; axis=1: shrink coordinate index
    r = ads.norm_l(x, axis=1)
    L = angularMoment(x, v, m)
    v_ = np.zeros((n, l))
    operators = None
    if tag == 0:  # indent mean angularMoment (and invert solve velocity)
        L_indent = 0.
    # if tag==1: #indent mean relative angularMoment #centralization has done this
    #     L_indent = 0.
    # if tag==2: #indent mean angularMoment by mean velocity #centralization has done this
    #     L_indent = 0.
    elif tag == 3:  # indent mean frequency to angularMoment by inertiaMoment and angularMoment
        #: it is wrong
        I = interiaMoment(x, v, m)
        O = np.zeros(I.shape)
        O_ = np.zeros(I.shape)
        LA = angularMoment_total(x, v, m, r_range=r_range)
        IA = interiaMoment_total(x, v, m, r_range=r_range)
        OA = frequency_by_interiaMoment_each(IA, LA)
        for j in np.arange(n):
            O[j] = frequency_by_interiaMoment_each(I[j], L[j])
            O_[j] = O[j]-OA/n
        operators = [OA]
    elif tag == 4:  # indent mean frequency directly by inertiaMoment and angularMoment
        LA = angularMoment_total(x, v, m, r_range=r_range) #1./10^3
        IA = interiaMoment_total(x, v, m, r_range=r_range) #1./10^4
        OA = frequency_by_interiaMoment_each(IA, LA)
        for j in np.arange(n):
            v_[j] = v[j]-np.cross((np.array(OA).T)[0], x[j])
            # ads.DEBUG_PRINT_V(1, LA, IA, (np.array(OA).T)[0], x[j])
            # ads.DEBUG_PRINT_V(0, v[j], np.cross((np.array(OA).T)[0], x[j]), v_[j])
        # for i in np.arange(l):
        #     v_[:, i] = v[:, i]-OA[i]*x[:, i]
        operators = [OA]
    elif tag == 5:  # indent scaled mean centripetal force, as (mean $mv^2/r$)
        cf_indent = 0.
    # indent scaled mean circular velocity, as (mean $\sqrt{r|d\Phi/dr|}$)
    elif tag == 6:
        cf_indent = 0.
    else:
        print("No such tag, do nothing.")
    # Besides v_, there might be also x_
    x_ = x*1.
    return x_, v_, operators


def rotate_mainAxisDirection(x, v, m, tag=0, r_range=None):  # ??
    is_correct, n, l = is_correct_shape_motion_data(x, v, m)
    if not is_correct:
        exit(0)
    # ads.check_x_un_finite(x, "rotate_mainAxisDirection()")
    x_ = x*1.
    v_ = v*1.
    operators = None
    T = np.zeros((n, l))
    if tag == 0:  # esitimate by three main axises of interiaMoment
        IA = interiaMoment_total(x, v, m, r_range=r_range)
        # ads.check_x_un_finite(x, "IA")
        T, eigenvalues, eigenvectors = operator_mainAxises(IA)
        for j in np.arange(n):
            x_[j] = np.dot(np.matrix(x[j]), T)  # rank*mat
            v_[j] = np.dot(np.matrix(v[j]), T)
        operators = [T, eigenvalues, eigenvectors]
    else:  # esitimate by fitting mass density
        print("Please fitting and loading paramters.")
        print("Not provided. Exit.")
        exit(0)
    return x_, v_, operators


# In[]
# for actions
## the update one
def all_triaxial_process(
    x, v, m, is_centralize_coordinate=True, 
    is_rotate_mainAxisDirection=True, r_range=None, 
    is_eliminate_totalRotation=False, 
    is_by_DF=False, DF=None
):
    '''
    Warning: It is not deep copy and it may change the input.
    '''
    # ads.DEBUG_PRINT_V(1, x[0], v[0], "xv 0")
    # ads.check_x_un_finite(x)

    oper0 = [np.zeros(3), np.zeros(3)] #no change
    oper1 = [np.zeros((3,3)), np.zeros(3), np.zeros((3,3))] #not default value
    oper2 = [np.zeros(3)] #no change
    if is_by_DF==True:
        # method by density: ...
        operators = [oper0, oper1, oper2]
        return x, v, operators

    if is_centralize_coordinate:
        x, v, oper0 = centralize_coordinate(x, v, m)
    # ads.DEBUG_PRINT_V(1, x[0], v[0], "xv 1")
    # ads.check_x_un_finite(x)

    if is_rotate_mainAxisDirection:
        x, v, oper1 = rotate_mainAxisDirection(x, v, m, 0, r_range=r_range)
    # ads.DEBUG_PRINT_V(1, x[0], v[0], "xv 2")
    # ads.check_x_un_finite(x)

    if is_eliminate_totalRotation: #Warning: only used in the IC before simulation for stability
        # LA1 = angularMoment_total(x, v, m, r_range=r_range) #after main axis and before frequency
        # IA1 = interiaMoment_total(x, v, m, r_range=r_range)
        x, v, oper2 = eliminate_totalRotation(x, v, m, 4, r_range=r_range)
        # LA2 = angularMoment_total(x, v, m, r_range=r_range)
        # IA2 = interiaMoment_total(x, v, m, r_range=r_range)
        # ads.DEBUG_PRINT_V(0, LA1, IA1, LA2, IA2, "LA1, IA1, LA2, IA2")
    # ads.DEBUG_PRINT_V(1, oper2, "rotation")
    # ads.DEBUG_PRINT_V(0, x[0], v[0], "xv 3")
    # ads.check_x_un_finite(x)

    #: do not need the second time of centralization because there conserve center
    
    operators = [oper0, oper1, oper2]
    # ads.DEBUG_PRINT_V(0, oper0, oper1, oper2)
    # unpack:
    # x_mean_old, v_mean_old = operators[0][0], operators[0][1] #arrays 3 and 3 for translation
    # T = operators[1][0] #array 3*3 for rotation
    # OA = operators[2][0] #array 3 for elimination #only in IC before simulation
    return x, v, operators

# def all_triaxial_process1(
#     x_inp, v_inp, m,
#     is_eliminate_totalRotation=True, 
#     is_rotate_mainAxisDirection=True, r_range=None, 
#     is_by_DF=False, DF = None
# ):
#     if (is_by_DF==True) and (DF is not None):
#         return 

#     # ads.DEBUG_PRINT_V(1, mean_coordinate(x_inp, v_inp, m))
#     # centralize2 #comflict with other operations??
#     x, v = centralize_coordinate(x_inp, v_inp, m)

#     #debug
#     # LA = angularMoment_total(x, v, m, r_range=r_range)
#     # IA = interiaMoment_total(x, v, m, r_range=r_range)
#     # OA = frequency_by_interiaMoment_each(IA, LA)
#     # T, eigenvalues, eigenvectors = operator_mainAxises(IA)
#     # #把质心系里的坐标变换为MOI主轴坐标系里的坐标，输出到文件
#     # ntot, itmp = x.shape
#     # x_MOI = np.zeros_like(x)
#     # for ii in range(ntot):
#     #     #vec = np.matrix([x_inp[ii,0],x_inp[ii,1],x_inp[ii,2]]) # 行向量
#     #     vec = x_inp[ii,0:3]
#     #     #print("粒子 ",ii)
#     #     #print(" 原质心系坐标：",vec)
#     #     x_MOI[ii,0] = np.matmul(vec,vl) #should sort
#     #     x_MOI[ii,1] = np.matmul(vec,vm)
#     #     x_MOI[ii,2] = np.matmul(vec,vn)
#     #     # print(vec.shape, vl.shape, vec, vl)
#     #     #print(" MOI坐标：",vec_MOI)
#     # np.savetxt("../result/x_MOI.txt",x_MOI,fmt="%.6e")

#     if is_eliminate_totalRotation:
#         x, v = eliminate_totalRotation(x, v, m, 4, r_range=r_range)

#     eigenvalues = np.ones(3)
#     if is_rotate_mainAxisDirection:
#         x, v, eigenvalues, eigenvectors = rotate_mainAxisDirection(x, v, m, 0, r_range=r_range)
        
#     x, v = centralize_coordinate(x, v, m)  # centralize2
#     # ads.DEBUG_PRINT_V(1, mean_coordinate(x, v, m))

#     OA, T, eigenvalues = None, None, None #??
#     return x, v, OA, T, eigenvalues

def galaxy_Nbody_triaxialize_comparison(x0, v0, x1, v1,
    is_histogram=True, is_scatter=True, r_range=None
):
    PLOT = fff.Plot_model_fit()
    bd = [125., 1000.]
    if r_range != None:
        bd = r_range
    if is_histogram:
        name = r"${x, y, z}$"
        x_list = [[x0[:, 0], x1[:, 0]], [
            x0[:, 1], x1[:, 1]], [x0[:, 2], x1[:, 2]]]
        N_bins = 41
        lim = bd[0]/2
        bins_list = [
            [np.linspace(-lim, lim, N_bins), np.linspace(-lim, lim, N_bins)],
            [np.linspace(-lim, lim, N_bins), np.linspace(-lim, lim, N_bins)],
            [np.linspace(-lim, lim, N_bins), np.linspace(-lim, lim, N_bins)]
            # [np.linspace(np.min(x0[:,0]), np.max(x0[:,0]), N_bins), np.linspace(np.min(x0[:,0]), np.max(x0[:,0]), N_bins)],
            # [np.linspace(np.min(x0[:,1]), np.max(x0[:,1]), N_bins), np.linspace(np.min(x0[:,1]), np.max(x0[:,1]), N_bins)],
            # [np.linspace(np.min(x0[:,2]), np.max(x0[:,2]), N_bins), np.linspace(np.min(x0[:,2]), np.max(x0[:,2]), N_bins)]
        ]
        label_list = [[r"x0, $x$", r"x1, $x$"], [
            r"x0, $y$", r"x1, $y$"], [r"x0, $z$", r"x1, $z$"]]
        PLOT.plot_histogram(x_list=x_list, bins_list=bins_list,
                            label_list=label_list, name=name)

        name = r"${v_x, v_y, v_z}$"
        x_list = [[v0[:, 0], v1[:, 0]], [
            v0[:, 1], v1[:, 1]], [v0[:, 2], v1[:, 2]]]
        N_bins = 41
        lim = bd[1]
        bins_list = [
            [np.linspace(-lim, lim, N_bins), np.linspace(-lim, lim, N_bins)],
            [np.linspace(-lim, lim, N_bins), np.linspace(-lim, lim, N_bins)],
            [np.linspace(-lim, lim, N_bins), np.linspace(-lim, lim, N_bins)]
        ]
        label_list = [[r"v0, $v_x$", r"v1, $v_x$"], [
            r"v0, $v_y$", r"v1, $v_y$"], [r"v0, $v_z$", r"v1, $v_z$"]]
        PLOT.plot_histogram(x_list=x_list, bins_list=bins_list,
                            label_list=label_list, name=name)

    if is_scatter:
        name = r"${x, y, z}$"
        x_list = [x0, x1]
        lim = bd[0]
        f_list = None
        # f_list = [ads.norm_l(v0, axis=1), ads.norm_l(v1, axis=1)]
        label_list = ["x0", "x1"]
        limit_list = [-lim, lim, -lim, lim, -lim, lim]
        PLOT.plot_x_scatter3d_general(x_list=x_list, f_list=f_list,
                                      label_list=label_list, name=name, limit_list=limit_list)

        name = r"${v_x, v_y, v_z}$"
        x_list = [v0, v1]
        lim = bd[1]
        f_list = None
        # f_list = [ads.norm_l(x0, axis=1), ads.norm_l(x1, axis=1)]
        label_list = ["v0", "v1"]
        limit_list = [-lim, lim, -lim, lim, -lim, lim]
        PLOT.plot_x_scatter3d_general(x_list=x_list, label_list=label_list,
                                      f_list=f_list, name=name, limit_list=limit_list)

    #:: judge histogram
    #:: fit by Gaussian
    #:: compare half widths
    return 0

def generate_usual_coordinate_samples(xs=1., vs=1., k1=1e-3, k2=5e0, N_each=10, is_triaxial=True, method=0):
    Dim = 3  # Here space_dimension can only be 3
    if not (xs>0. and vs>0. and k1>0. and k2>0.):
        print("Wrong settings of scaled length and scaled velocity! Exit now.")
        exit(0)
    N = np.int(N_each**(Dim*2))
    N_each_x = N_each
    if is_triaxial:
        # N_each_x = int(N_each_x/2)
        N = np.int(N_each**Dim * N_each_x**Dim)
    if N > 2**32-1:
        print("Worning: The count of samples in each dimesion is %ld. "
              "It is might too large for C/C++ <int> directly. "
              "Do not let it be larger than 35.9(=int_Max**(1./(Dim*2))), please." % (N))
    x = np.array([])
    v = np.array([])
    if method == 0:
        xb = xs/np.sqrt(3.)*k2
        vb = vs/np.sqrt(3.)*k2**0.5
        xbD = -xb
        vbD = -vb
        if is_triaxial:
            xbD = xs*k1
            # vbD = vs*k1 #??
        for x0 in np.linspace(xbD, xb, N_each_x):
            for x1 in np.linspace(xbD, xb, N_each_x):
                for x2 in np.linspace(xbD, xb, N_each_x):
                    print(x0, x1, x2, xb, vb)
                    for v0 in np.linspace(-vbD, vb, N_each):
                        # print(v0, vb)
                        for v1 in np.linspace(-vbD, vb, N_each):
                            for v2 in np.linspace(-vbD, vb, N_each):
                                xp = np.array([x0, x1, x2])
                                vp = np.array([v0, v1, v2])
                                x = np.append(x, xp)  # , axis=0
                                v = np.append(v, vp)
        x = x.reshape(N, Dim)
        v = v.reshape(N, Dim)
    else:
        print("No such method provided! Exit now.")
        exit(0)
    return x, v

def potential_and_action_comparison():
    return

def action_state_density():
    return

def potential_direct_summation(x_tgt, x, m1=None, soft=0.02):
    m = None
    N = len(x)
    if m1 is not None:
        m = m1
    else:
        m = np.ones(N)
    pot = 0.
    for i in np.arange(N):
        deltar2 = np.sum( (x_tgt-x[i])*(x_tgt-x[i]) )
        if deltar2>1e-20:
            deltar2_ = deltar2 + soft*soft
            deltar = np.sqrt(deltar2_)
            pot += m[i]/deltar
        else:
            pot += 0.
    print("potential = ", pot)
    return pot*gm.G

def frac_unbounded_particles(v, pot):
    if np.sum(np.abs(pot))<1e-20:
        print("There is not potential data. Exit.")
    vnorm = ads.norm_l(v, axis=1)
    # ads.DEBUG_PRINT_V(0, 0.5*vnorm**2, pot)
    E = 0.5*vnorm**2 + pot
    mask = (E>0)
    Eubd = E[mask]
    frac = len(Eubd)/len(E)
    return frac, Eubd

def spin_lambda_Nbody(x, v, mass, pot=None):
    vnorm = ads.norm_l(v, axis=1)
    Lt_xyz = angularMoment_total(x, v, mass) #has mass
    Lt = ads.norm_l(Lt_xyz)
    N = len(x)
    pot_each = np.zeros(N)
    if pot is not None:
        pot_each = pot
    else:
        for i in np.arange(N):
            pot_each[i] = potential_direct_summation(x[i], x)
        # pot_each *= m[:]
    E = mass[:]*(0.5*vnorm**2+pot_each) #has mass
    Et = np.sum(E)
    G = gm.G
    Mt = np.sum(mass)
    lambdat = (Lt*np.abs(Et)**0.5)/(G*Mt**2.5)
    ads.DEBUG_PRINT_V(1, Mt, Et, Lt_xyz, Lt, lambdat, "Mt, Et, Lt_xyz, Lt, lambdat")
    return lambdat

#angular moment
def total_angular_momentum(x, v, m):
    # J vector (a.k.a. L_tot)
    return np.sum(m[:, None] * np.cross(x, v), axis=0)

def spin_bullock_lambda_prime(x, v, m, R=None, frac_enclosed=0.95, G=gm.G):
    """
    Bullock+01 spin proxy: lambda' = J / (sqrt(2) M(<R) Vc R),
    choosing R as the radius enclosing a given mass fraction if not supplied.
    Returns (lambda_prime, R_used, M_enclosed).
    """
    r = ads.norm_l(x, axis=1)
    Mtot = np.sum(m)
    if R is None:
        idx = np.argsort(r)
        rc = r[idx]
        mc = np.cumsum(m[idx])
        j = np.searchsorted(mc, frac_enclosed * Mtot, side="left")
        j = min(max(j, 1), len(rc) - 1)
        R = rc[j]
        Menc = mc[j]
    else:
        Menc = np.sum(m[r <= R])
        if Menc <= 0:
            raise ValueError("spin_bullock_lambda_prime(): M(<R) == 0; choose a larger R.")

    Jvec = total_angular_momentum(x, v, m)
    J = ads.norm_l(Jvec)
    Vc = np.sqrt(G * Menc / max(R, 1e-30))
    lam_p = J / (np.sqrt(2.0) * Menc * Vc * R)
    return lam_p, R, Menc

def spin_peebles_lambda_virial_proxy(x, v, m, G=gm.G):
    """
    Peebles lambda using |E| ≈ T (virial proxy): lambda ≈ J*sqrt(T)/(G M^(5/2)).
    Fast, potential-free; assumes near-equilibrium.
    """
    Jvec = total_angular_momentum(x, v, m)
    J = ads.norm_l(Jvec)
    v2 = np.sum(v**2, axis=1)
    T = 0.5 * np.sum(m * v2)
    M = np.sum(m)
    if M <= 0:
        raise ValueError("Total mass M = 0.")
    return J * np.sqrt(max(T, 0.0)) / (G * (M ** 2.5))

#fast potential via FFT/PM (isolation approximated by padding)
def _hist3d_mass(x, m, bounds, ngrid):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    H, edges = np.histogramdd(
        x, bins=(ngrid, ngrid, ngrid),
        range=((xmin, xmax), (ymin, ymax), (zmin, zmax)),
        weights=m
    )
    cell_vol = ((xmax - xmin) / ngrid) * ((ymax - ymin) / ngrid) * ((zmax - zmin) / ngrid)
    rho = H / cell_vol
    return rho, edges

def _k2_rfftn(nx, ny, nz, lx, ly, lz):
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, d=lz / nz)
    kx2 = kx[:, None, None] ** 2
    ky2 = ky[None, :, None] ** 2
    kz2 = kz[None, None, :] ** 2
    return kx2 + ky2 + kz2

def potential_PM_fft(x, m, ngrid=128, pad=2.0, G=gm.G):
    """
    Quick 3D Poisson solve:
      - Put mass on grid (NGP histogram)
      - FFT solve Phi_k = -4 pi G rho_k / k^2 (k=0 -> 0)
      - Return gridded Phi and a helper to interpolate back to particle positions.
    """
    # box with padding
    mins = np.min(x, axis=0); maxs = np.max(x, axis=0)
    size = (maxs - mins)
    ctr  = 0.5 * (maxs + mins)
    half = 0.5 * pad * np.max(size)
    bounds = ((ctr[0]-half, ctr[0]+half),
              (ctr[1]-half, ctr[1]+half),
              (ctr[2]-half, ctr[2]+half))
    Lx = bounds[0][1] - bounds[0][0]
    Ly = bounds[1][1] - bounds[1][0]
    Lz = bounds[2][1] - bounds[2][0]

    rho, edges = _hist3d_mass(x, m, bounds, ngrid)
    rho_k = np.fft.rfftn(rho)
    k2 = _k2_rfftn(ngrid, ngrid, ngrid, Lx, Ly, Lz)
    with np.errstate(divide='ignore', invalid='ignore'):
        green = np.zeros_like(k2)
        mask = k2 > 0.0
        green[mask] = -4.0 * np.pi * G / k2[mask]
    phi_k = rho_k * green
    phi = np.fft.irfftn(phi_k, s=rho.shape)

    # trilinear interpolator
    dx = Lx / ngrid; dy = Ly / ngrid; dz = Lz / ngrid
    x0, y0, z0 = bounds[0][0], bounds[1][0], bounds[2][0]

    def interp_phi(pos):
        # Convert positions -> fractional indices
        fx = (pos[:, 0] - x0) / dx
        fy = (pos[:, 1] - y0) / dy
        fz = (pos[:, 2] - z0) / dz
        i = np.clip(np.floor(fx).astype(int), 0, ngrid - 2)
        j = np.clip(np.floor(fy).astype(int), 0, ngrid - 2)
        k = np.clip(np.floor(fz).astype(int), 0, ngrid - 2)
        wx = fx - i; wy = fy - j; wz = fz - k

        # gather 8 corners
        def at(ii, jj, kk): return phi[ii, jj, kk]
        c000 = at(i,   j,   k  ); c100 = at(i+1, j,   k  )
        c010 = at(i,   j+1, k  ); c110 = at(i+1, j+1, k  )
        c001 = at(i,   j,   k+1); c101 = at(i+1, j,   k+1)
        c011 = at(i,   j+1, k+1); c111 = at(i+1, j+1, k+1)

        # trilinear
        c00 = c000*(1-wx) + c100*wx
        c01 = c001*(1-wx) + c101*wx
        c10 = c010*(1-wx) + c110*wx
        c11 = c011*(1-wx) + c111*wx
        c0  = c00*(1-wy) + c10*wy
        c1  = c01*(1-wy) + c11*wy
        return c0*(1-wz) + c1*wz

    return phi, bounds, interp_phi



# In[]
if __name__ == '__main__':

    # boundary_rotate_refer = [80., 1000.] #kpc, km/s
    # x = np.array([[-53.3303, -75.9465, 15.7593]])
    # v = np.array([[-178.072, 40.2511, 56.5314]])
    # mass = [1.37e-4]
    # x, v, operators = all_triaxial_process(
    #     x, v, mass, is_centralize_coordinate=True, 
    #     is_rotate_mainAxisDirection=True, boundary_rotate_refer=boundary_rotate_refer, 
    #     is_eliminate_totalRotation=False, is_by_DF=False, DF=None
    # )

    galaxy_model_name = "galaxy_general"
    basepath = galaxy_model_name+"/txt/"
    snapshot = 5000
    N_ptcs, x_inp, v_inp, m = load_snapshot(basepath=basepath, snapshot=snapshot)
    x_inp, v_inp = centralize_coordinate(x_inp, v_inp, m) #??

    boundary = [60., 1000.]
    x, v, OA, T, eigenvectors = all_triaxial_process(x_inp, v_inp, m, False, True, boundary=boundary)
    # is_better_triaxial = galaxy_Nbody_triaxialize_comparison(x_inp, v_inp, x, v, is_histogram=True, is_scatter=True, boundary=boundary)

    # #:: eliminate_totalRotation() donot almost change
    # x1, v1, OA, T = all_triaxial_process(x_inp, v_inp, m, False, True, boundary=boundary)
    # ads.DEBUG_PRINT_V(1, mean_coordinate(x1, v1, m), OA, T)
    # is_better_triaxial = galaxy_Nbody_triaxialize_comparison(x1, v1, x, v, is_histogram=True, is_scatter=False, boundary=boundary)

    # #:: rotate_mainAxisDirection() do change, bad
    # x2, v2, OA, T = all_triaxial_process(x_inp, v_inp, m, True, False, boundary=boundary)
    # ads.DEBUG_PRINT_V(1, mean_coordinate(x2, v2, m), OA, T)
    # is_better_triaxial = galaxy_Nbody_triaxialize_comparison(x2, v2, x, v, is_histogram=True, is_scatter=False, boundary=boundary)

    # D = np.matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    # theta = 4.304433e+00, 9.737117e-01, 4.775786e+00
    # D1 = atg.SO3(D, theta)
    # ads.DEBUG_PRINT_V(1, mean_coordinate(x, v, m), OA, T, eigenvectors, D1)

    # is_rewroten = rewrite_snapshot(x, v, m=m, comp_gal=0, basepath=basepath, snapshot=snapshot)
    


    rs = 20. #kpc
    vs = np.median(ads.norm_l(v, axis=1))
    print(rs, vs)
    x_samples, v_samples = generate_usual_coordinate_samples(rs, vs, N_each=10)
    m_samples = np.ones(len(x_samples))*m[0]
    comp_gal = 0
    is_rewroten = rewrite_snapshot(x_samples, v_samples, m_samples, comp_gal=comp_gal, 
        basepath=basepath, snapshot=snapshot, suffix="samples")
    ads.DEBUG_PRINT_V(1, x_samples.shape, v_samples.shape)


