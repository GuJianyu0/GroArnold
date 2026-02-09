#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.interpolate import Rbf



def norm_l(a,l=2):
    x=0
    for e in a:
        x += e**l
    return x**(1./l)

def put_certain_3ddatapoints_inaline(x1_points,x2_points,x3_points,fxxx_points, i_x1i,i_x1f,i_x2i,i_x2f,i_x3i,i_x3f):
    if not( len(x1_points.shape)==1 and len(x2_points.shape)==1 and len(x2_points.shape)==1 ):
        print(r"Bad shape of independent variables of data points! Please 1d only.")
        sys.exit(0)
    dim = 3
    if len(fxxx_points.shape)!=dim:
        print(r"Bad shape of dependent variables of data points! Please 3d only.")
        sys.exit(0)
    lx1 = len(x1_points)
    lx2 = len(x2_points)
    lx3 = len(x3_points)
    ly1,ly2,ly3 = fxxx_points.shape
    if not( lx1==ly1 and lx2==ly2 and lx3==ly3 ):
        print(r"Inconsistent lengthes of independent variables (x1,x2,x3) and dependent variables (f(x1,x2,x3)) of data points! Please equally only.")
        sys.exit(0)
    if not( type(i_x1i)==int and type(i_x1f)==int and type(i_x2i)==int and type(i_x2f)==int and type(i_x3i)==int and type(i_x3f)==int ):
        print(r"Bad type of appointment indexes of data points! Please int only.")
        sys.exit(0)
    if not( 1<=i_x1f-i_x1i<=lx1 and 1<=i_x2f-i_x2i<=lx2 and 1<=i_x3f-i_x3i<=lx3 ):
        print(r"Too large length of appointment indexes of data points! Please 1<=xf-xi<=x_points.shape only.")
        sys.exit(0)

    nx1 = i_x1f-i_x1i+1
    nx2 = i_x2f-i_x2i+1
    nx3 = i_x3f-i_x3i+1
    n_data_inaline = nx1*nx2*nx3
    xxx_data_inaline = np.zeros((n_data_inaline,3))
    fxxx_data_inaline = np.zeros(n_data_inaline)

    n = 0
    for n1 in range(nx1):
        for n2 in range(nx2):
            for n3 in range(nx3):
                xxx_data_inaline[n] = np.array([ x1_points[n1], x2_points[n2], x3_points[n3] ])
                fxxx_data_inaline[n] = fxxx_points[ n1, n2, n3 ]
                n+=1

    if not( n==n_data_inaline ):
        print(r"Wrong count caused by this function! Please check.")
        sys.exit(0)
    return xxx_data_inaline, fxxx_data_inaline, n_data_inaline

def divided_bins_123(D, colD=0, nmesh=0,nmesh2=0,nmesh3=0, whatbin=1, datamin=1.e-8,datamax=1.e8, param1=0,param2=19.6,param3="purple",param4=1.):

    ## judge input params
    shape1_0 = 40 #80 #the least data points
    shape2_0 = 3 #space dim
    if type(colD)!=int or type(nmesh)!=int or type(nmesh2)!=int or type(nmesh3)!=int:
        print(r"False type of some input arguments!")
        sys.exit(0)
    if len(D.shape)!=2:
        print(r"Input data should be a 2d-vector!")
        sys.exit(0)
    shape1,shape2 = D.shape
    print(r"The shape of 2d-data = %d %d, colD = %d." % (shape1, shape2, colD) )
    if shape1<shape1_0:
        print(r"Too less data points!")
        sys.exit(0)
    if colD<0 or colD+3>shape2:
        print(r"Bad rank appointment!")
        sys.exit(0)

    ## set inf and nan to 0.
    D_num = np.where(D==np.inf, 0., D)
    D = np.array(D_num)
    D0 = D[:, colD:colD+3] #data in step 1, none inf or nan
    description = "_halo_"

    ## some params
    # J0 = galaxy_models.G*galaxy_models.M_total*r0 #as total scale

    ## conditions
    err = 1e-10 #should: datamin<=err
    spr = 1e10 #should: datamax>=err
    D0 = abs(D0)
    print("min: ", min(D0[0]))
    LD = len(D0) #the outest dim
    A_judge = np.zeros(LD)
    for i in range(LD):
        # t = norm_l(J0[i])
        # if t>1.e-2 and t<1.e6: #no +-
        t = D0[i]
        # if 1:
        # if t[0]>0 and t[1]>0 and t[2]>0:
        if t[0]>datamin and t[1]>datamin and t[2]>datamin and t[0]<datamax and t[1]<datamax and t[2]<datamax:
            A_judge[i] = 1
    N_eff = int(sum(A_judge))
    N_record = 0

    ## screen data
    A_nonezero = np.zeros((N_eff,3)) #data in step 2, none zero
    idx = 0
    for i in range(LD):
        if A_judge[i]:
            A_nonezero[idx] = D0[i]
            idx += 1
    rate = float(N_eff)/float(LD)
    print(r"The rate of reliable actions = %d %f." % (N_eff, rate))

    ## devided bins
    if nmesh==0:
        nmesh = int((LD*1./shape1_0)**(1./1))
    x_points = np.zeros((nmesh, 3)) #the ,3 represents 3 indepandent x values in C_3^2=3 combination of 3 coordinates
    fx_points = np.zeros((nmesh, 3)) #corresponded y values
    x_output = np.zeros((nmesh, 3)) #put each data points in a line
    fx_output = np.zeros((nmesh, 3)) #put each data points in a line
    if nmesh2==0:
        nmesh2 = int((LD*1./shape1_0)**(1./2))
    xx_points = np.zeros((nmesh2,nmesh2, 3)) #the ,3 represents 3 x-x values in C_3^2=3 combination of 3 coordinates
    fxx_points = np.zeros((nmesh2,2, 3)) #corresponded y values
    xx_output = np.zeros((nmesh2**2,2, 3)) #put each data points in a line
    fxx_output = np.zeros((nmesh2**2, 3)) #put each data points in a line
    if nmesh3==0:
        nmesh3 = int((LD*1./shape1_0)**(1./3)) #to let the count of data points in a cell not too less
    xxx_points = np.zeros((nmesh3,3)) #the x-x-x values in C_3^3=1 combination of 3 coordinates
    fxxx_points = np.zeros((nmesh3,nmesh3,nmesh3)) #corresponded y values
    xxx_output = np.zeros((nmesh3**3,3)) #put each data points in a line
    fxxx_output = np.zeros(nmesh3**3) #put each data points in a line
    print(r"cell count of 1d (nmesh**1) = %d, of 2d (nmesh2**2) = %d, of 3d (nmesh3**3) = %d." % (nmesh**1, nmesh2**2, nmesh3**3))



    ## 1 dim
    ## y = f1(x1), f1(x2), f1(x3), where $f1(x_i) = \int f3(x_1,x_2,x_3) \mathrm{d}J_\mathrm{i\,other\,1} \mathrm{d}J_\mathrm{i\,other\,2}$, counts 3
    for k in range(3):
        A = A_nonezero[:,k] #data in step 2, each
        A_mesh = np.zeros(nmesh+1) #defined as mesh, counts (nmesh+1)
        DA = np.zeros(nmesh) #difined as bin
        A_count = np.zeros(nmesh) #defined
        A_frequency = np.zeros(nmesh) #defined
        A_distribution = np.zeros(nmesh) #defined

        A_min = min(A)
        A_max = max(A)
        A_median = np.median(A)
        if whatbin==1:
            A_mesh = np.linspace(A_min-err, A_max+err, nmesh+1) #linear, from 0 to max
        elif whatbin==2:
            A_mesh = np.logspace(np.log10(A_min-err), np.log10(A_max+err), nmesh+1) #logarithmic, from 0 to max
        elif whatbin==3:
            A_mesh = np.linspace(A_min-err, A_median+err, nmesh+1) #line, from 0 to median, more data points will not be recorded
        elif whatbin==4:
            A_mesh = np.logspace(np.log10(A_min-err), np.log10(A_median+err), nmesh+1) #logarithmic, from 0 to median, more data points will not be recorded
        elif whatbin==5:
            A_mesh = np.linspace(A_min-err, (param4+err), nmesh+1) #line
        elif whatbin==6:
            A_mesh = np.logspace(np.log10(A_min-err), np.log10((param4+err)), nmesh+1) #logarithmic
        else:
            print(r"No such bin provided! Exit.")
            sys.exit(0)
        DA = A_mesh[1:]-A_mesh[:-1] #the next difference
        print(r"data A: min max median = %f %f %f" % (A_min, A_max, A_median))
        # print(r"data A: DJ1 = ", DA)

        N_record = 0
        for j in A:
            for n in range(nmesh):
                if j>=A_mesh[n] and j<A_mesh[n+1]: #remove zeros before
                    A_count[n]+=1
                    N_record+=1

        A_count*=1.0 #to let int -> float
        A_frequency = A_count/N_record
        A_distribution = A_frequency/DA
        # print(r"count in each bin: ", A_count)
        print(r"the N_record = %d" % N_record)
        print(r"the sum of A_count = %f" % sum(A_count))
        print(r"the sum of A_frequency = %f", sum(A_frequency))

        x_points[:,k] = A_mesh[:-1]+DA/2 #we make a dislocation to set the output datapoint are in the center of the cell
        fx_points[:,k] = A_distribution #it has devided the volumne of the cell so as be a probability density distribution
    x_output = x_points
    fx_output = fx_points


    ## 2 dim
    ## y = f2(x2,x3), f2(x1,x3), f2(x1,x2), where $f2(x_i,x_j) = \int f3(x_1,x_2,x_3) \mathrm{d}x_\mathrm{i,j\,other}$, counts 3
    # #...



    ## 3 dim
    ## y = f3(x1,x2,x3), where $f3(x_1,x_2,x_3) = f3(x_1,x_2,x_3)$, counts 1
    AAA = A_nonezero #data in step 2, each dim -> 3 dim
    A3_mesh = np.zeros((nmesh3+1,3)) #defined as mesh, each dim -> 3 dim
    DA3 = np.zeros((nmesh3,3)) #difined as bin, each dim -> 3 dim
    DADADA = np.zeros((nmesh3,nmesh3,nmesh3)) #defined
    # AAA_mesh = np.zeros((nmesh3+1,nmesh3+1,nmesh3+1))
    AAA_count = np.zeros((nmesh3,nmesh3,nmesh3)) #defined
    AAA_frequency = np.zeros((nmesh3,nmesh3,nmesh3)) #defined
    AAA_distribution = np.zeros((nmesh3,nmesh3,nmesh3)) #defined

    AAA_min = np.array([min(AAA[:,0]), min(AAA[:,1]), min(AAA[:,2])])
    AAA_max = np.array([max(AAA[:,0]), max(AAA[:,1]), max(AAA[:,2])])
    AAA_median = np.array([np.median(AAA[0]), np.median(AAA[1]), np.median(AAA[2])])
    for k in range(3):
        if whatbin==1:
            A3_mesh[:,k] = np.linspace(AAA_min[k]-err, AAA_max[k]+err, nmesh3+1) #linear, from 0 to max
        elif whatbin==2:
            A3_mesh[:,k] = np.logspace(np.log10(AAA_min[k]-err), np.log10(AAA_max[k]+err), nmesh3+1) #logarithmic, from 0 to max
        elif whatbin==3:
            A3_mesh[:,k] = np.linspace(AAA_min[k]-err, AAA_median[k]+err, nmesh3+1) #line, from 0 to median, more data points will not be recorded
        elif whatbin==4:
            A3_mesh[:,k] = np.logspace(np.log10(AAA_min[k]-err), np.log10(AAA_median[k]+err), nmesh3+1) #logarithmic, from 0 to median, more data points will not be recorded
        elif whatbin==5:
            A3_mesh[:,k] = np.linspace(AAA_min[k]-err, (param4+err), nmesh3+1) #line
        elif whatbin==6:
            A3_mesh[:,k] = np.logspace(np.log10(AAA_min[k]-err), np.log10((param4+err)), nmesh3+1) #logarithmic
        else:
            print(r"No such bin provided! Exit.")
            sys.exit(0)
        DA3[:,k] = A3_mesh[1:,k]-A3_mesh[:-1,k] #the next difference
        print(r"data AAA's one dim: min max median = %f %f %f" % (AAA_min[k], AAA_max[k], AAA_median[k]))
        # print(r"data A: DJ1 = ", DA3[k])

    N_record = 0
    for aaa in AAA: #fxxx to traverse each data point with reliable actions #remove zeros before #the order is AAA[x1,x2,x3]
        for n1 in range(nmesh3): #x1 ~ k=0
            if aaa[0]>=A3_mesh[n1,0] and aaa[0]<A3_mesh[n1+1,0]:
                for n2 in range(nmesh3): #x2 ~ k=1
                    if aaa[1]>=A3_mesh[n2,1] and aaa[1]<A3_mesh[n2+1,1]:
                        for n3 in range(nmesh3): #x3 ~ k=2
                            if aaa[2]>=A3_mesh[n3,2] and aaa[2]<A3_mesh[n3+1,2]:
                                AAA_count[n1,n2,n3] += 1
                                N_record+=1
            # else ... #else continue is no use

    for n1 in range(nmesh3): #dxdxdx
        for n2 in range(nmesh3):
            for n3 in range(nmesh3):
                DADADA[n1,n2,n3] = DA3[n1,0]*DA3[n2,1]*DA3[n3,2]

    AAA_count*=1.0 #to let int -> float
    AAA_frequency = AAA_count/N_record
    AAA_distribution = AAA_frequency/DADADA
    print(r"the N_record = %d" % N_record)
    print(r"the sum of AAA_count = %f" % sum(sum(sum(AAA_count))))
    print(r"the sum of AAA_frequency = %f" % sum(sum(sum(AAA_frequency))))

    xxx_points = A3_mesh[:-1,:]+DA3/2 #we make a dislocation to set the output datapoint are in the center of the cell
    fxxx_points = AAA_distribution #it has devided the volumne of the cell so as be a probability density distribution
    xxx_output, fxxx_output, N3 = put_certain_3ddatapoints_inaline(xxx_points[:,0],xxx_points[:,1],xxx_points[:,2],fxxx_points, 0,len(xxx_points[0]),0,len(xxx_points[1]),0,len(xxx_points[2]))

    ## a sheer
    # nnx = 0
    # sheer = 1
    # for n1 in range(nmesh_output): #not nmesh_out-1, the remaining 0 value should be dealed with
    #     for n2 in range(nmesh_output):
    #         JJ_all[nnx,0] = J1mesh[n1]
    #         JJ_all[nnx,1] = J2mesh[n2]
    #         fJJ_all[nnx] = JJJdistribution[n1,n2,sheer]
    #         nnx += 1
    # # print("f: ", JJJdistribution[:,:,sheer])
    # # print("f 00: ", JJ_all[0,0],JJ_all[0,1],fJJ_all[0])
    # # print("J1J2: ", J1mesh, J2mesh)
    # # print("J1J2: ", JJ_all[:,0], JJ_all[:,1])
    # # print("f 00: ", J1mesh[0],J2mesh[0],JJJdistribution[0,0,sheer])
    # # print("f 11: ", J1mesh[1],J2mesh[1],JJJdistribution[1,1,sheer])
    # # print("f -1-1: ", J1mesh[-1],J2mesh[-1],JJJdistribution[-1,-1,sheer])
    # # print("J1J2J3: ", J1mesh, J2mesh, J3mesh)
    # # print("J1J2J3: ", JJJ_all[:,0], JJJ_all[:,1], JJJ_all[:,2])

    return x_output, fx_output, xx_output, fxx_output, xxx_output, fxxx_output

def interpolation_Rbf_xxx_yyy_xxx0_k(xxx_knn, fxxx_knn, xxx0, k=10, funcname="gaussian"):

    # ## finding K neighbors
    # tree = KDTree(xxx, leaf_size=40)
    # indices, distances = tree.query(xxx0, k=k)

    # ## judge input params
    # we do not do it for efficiency=

    func1 = Rbf(xxx[:,0],xxx[:,1],xxx[:,2], fxxx[:,0], function=funcname)
    f1xxx0 = func1(xxx0[0],xxx0[1],xxx0[2])
    func2 = Rbf(xxx[:,0],xxx[:,1],xxx[:,2], fxxx[:,1], function=funcname)
    f2xxx0 = func2(xxx0[0],xxx0[1],xxx0[2])
    func3 = Rbf(xxx[:,0],xxx[:,1],xxx[:,2], fxxx[:,2], function=funcname)
    f3xxx0 = func3(xxx0[0],xxx0[1],xxx0[2])

    return np.array([f1xxx0,f2xxx0,f3xxx0])





def f_J__plot(ff, snapid, colJ = 12, nmesh = 100, whatbin=1):

    ##load
    if type(snapid)!=list:
        print("f_J__plot() parameter should be a one dim list.")
        return 0

    # coor = [r'\lambda',r'\mu',r'\nu']
    # fig = plt.figure(figsize=(16, 12), dpi=200, facecolor=(0.0, 0.0, 0.0))

    ##each snapshot
    if 1:
        # filename = ff+"allID_%03d.txt" % (snapid)
        filename = ff
        data3 = np.loadtxt(filename, dtype=float)
        data_num3 = np.where(data3==np.inf, 0., data3)
        data3 = np.array(data_num3)
        Jdata =data3

        #select #gas 0~2000, halo 2000~12000, disk 12000~17000, bulge 17000~18000 #ASF FP 0:3, TSF FP 3:6, ASF DP 7:10, TSF DP 10:13
        # J0 = Jdata[0:2000, 10:13]
        # description = "_gas_"
        # J0 = Jdata[0:10000, 10:13]
        J0 = Jdata[:, colJ:colJ+3]
        description = "_halo_"
        # # J0 = Jdata[:, 0:3]
        # J0 = Jdata[12000:17000, 7:10]
        # # # J0 = Jdata[12000:17000,0]*Jdata[12000:17000,4] -Jdata[12000:17000,1]*Jdata[12000:17000,3] #direct Lz
        # description = "_disk_"
        # J0 = Jdata[17000:18000, 7:10]
        # description = "_bulge_"

        # J0 = Jdata[0:2000, 10:13]
        # description = "_gas_"
        # J0 = Jdata[2000:12000, 10:13]
        # description = "_halo_"
        # J0 = Jdata[24000:34000, 0:3] #, 10:13]
        # description = "_disk_"
        # J0 = Jdata[17000:18000, 7:10]
        # description = "_bulge_"

        #calculation continue...
        J0 = abs(J0)
        LJ = len(J0) #the outest dim
        print("LJ = ", LJ)
        Jjudge = np.zeros(LJ)
        for i in range(LJ):
            # t = norm_l(J0[i])
            # if t>1.e-2 and t<1.e6: #no +-
            t = J0[i]
            # if 1:
            # if t[0]>1.e-6 and t[1]>1.e-6 and t[2]>1.e-6 and t[0]<1.e8 and t[1]<1.e8 and t[2]<1.e8:
            if t[0]>0 and t[1]>0 and t[2]>0:
                Jjudge[i] = 1
        N_eff = int(sum(Jjudge))

        Jnz = np.zeros((N_eff,3))
        idx = 0
        for i in range(LJ):
            if Jjudge[i]:
                Jnz[idx] = J0[i]
                idx += 1
        rate = float(N_eff)/float(LJ)
        print("The rate of reliable actions = %d %f." % (N_eff, rate))

        # nmesh = 100
        J_output = np.zeros((nmesh,3))
        fJ_output = np.zeros((nmesh,3))
        fJJJ_output = np.zeros(nmesh)



        ##each dim
        for k in range(3):
            J = Jnz[:,k]

            Jmedian = np.median(J) #as scale
            if whatbin==1:
                Jmesh = np.linspace(0., max(J), nmesh) #line
            if whatbin==2:
                Jmesh = np.logspace(0., np.log(max(J))*2.7/10, nmesh) #log
            else:
                Jmesh = np.linspace(0., Jmedian, nmesh) #line
            Jbin = (max(J)-0.)/nmesh
            # Jmesh = np.logspace( np.log(min(J)), np.log(max(J)) ,nmesh ) #for log
            # Jbin = ... #for log
            print(r"data A: min max median bin_linear = %f %f %f %f" % (min(J), max(J), Jmedian, Jbin))

            Jdistribution = np.zeros(nmesh)
            for j in J:
                for n in range(nmesh-1):
                    # if j>Jmesh[n] and j<Jmesh[n+1]: #remove zeros before
                    if j>=Jmesh[n] and j<Jmesh[n+1]+1e-20: #remove zeros before
                        Jdistribution[n] += 1
            JdistributionOne = Jdistribution/N_eff
            Jdistribution = JdistributionOne/Jbin
            # print("rate in each bin: ", JdistributionOne)
            # print("sum: ", sum(JdistributionOne))

            # plt.subplot(2,2,k+1)
            # # plt.axes(yscale = "log")
            # plt.scatter(Jmesh, Jdistribution, s=2., label="snapshot at %.1f(Gyr)"%(s/100))
            # # plt.plot(nmeshmesh, Jdistribution)

            # plt.xlabel(r'$J_'+coor[k]+r'\quad (\mathrm{kpc}\cdot\mathrm{km}\cdot s^{-1})$', fontsize=20) #M_\odot
            # plt.ylabel(r'number density profile of  $J_'+coor[k]+r'$', fontsize=20)
            # plt.ylim(0., max(Jdistribution)*1.2)
            # plt.legend()
            # if len(snap)==1:
            #     plt.savefig(filename+description+".png")

            J_output[:,k] = Jmesh
            fJ_output[:,k] = Jdistribution

        # plt.show()
        # plt.close("all")



        ##here we statistic fJ3
        J1 = Jnz[:,0]
        J2 = Jnz[:,1]
        J3 = Jnz[:,2]

        J1median = np.median(J1) #as scale
        J1mesh = np.linspace(0., max(J1), nmesh) #line
        J1bin = (max(J1)-0.)/nmesh
        print("J_ max min mm bin = ", max(J1), min(J1), J1median, J1bin)
        J2median = np.median(J2) #as scale
        J2mesh = np.linspace(0., max(J2), nmesh) #line
        J2bin = (max(J2)-0.)/nmesh
        J3median = np.median(J3) #as scale
        J3mesh = np.linspace(0., max(J3), nmesh) #line
        J3bin = (max(J3)-0.)/nmesh

        JJJdistribution = np.zeros((nmesh,nmesh,nmesh))

        for jjj in Jnz: #to traverse each particle with reliable actions #remove zeros before
            for n1 in range(nmesh-1):
                if jjj[0]>=J1mesh[n1] and jjj[0]<J1mesh[n1+1]+1e-20: #each colomn in J1
                    for n2 in range(nmesh-1):
                        if jjj[1]>=J1mesh[n2] and jjj[1]<J1mesh[n2+1]+1e-20: #each colomn in J2 in J1
                            for n3 in range(nmesh-1):
                                if jjj[2]>=J1mesh[n3] and jjj[2]<J1mesh[n3+1]+1e-20: #each colomn in J2 in J1
                                    JJJdistribution[n1,n2,n3] += 1
                # else ... #else continue is no use

        # for j1 in J1:
        #     for n in range(nmesh-1):
        #         if j1>Jmesh[n] and j1<Jmesh[n+1]:
        #             Jdistribution[n] += 1
        JJJdistribution = JJJdistribution/N_eff/J1bin/J2bin/J3bin
        print("\ntotal sum of JJJdistribution = %f\n" % ( sum(sum(sum(JJJdistribution)))*J1bin*J2bin*J3bin ) )
        fJJJ_output = JJJdistribution



        ##to put all these points in a line
        nmesh_output = int(nmesh/10) #the first several bins, to do not output near 0 distribution value #100 points for 1d, 25 points for 2d, 10 points for 3d
        JJJ_all = np.zeros((nmesh_output**3, 3))
        fJJJ_all = np.zeros(nmesh_output**3)
        JJ_all = np.zeros((nmesh_output**2, 2))
        fJJ_all = np.zeros(nmesh_output**2)

        nn = 0
        dJ1 = 0
        for n1 in range(nmesh_output):
            for n2 in range(nmesh_output):
                for n3 in range(nmesh_output):
                    JJJ_all[nn,0] = J1mesh[n1]
                    JJJ_all[nn,1] = J2mesh[n2]
                    JJJ_all[nn,2] = J3mesh[n3]
                    fJJJ_all[nn] = JJJdistribution[n1,n2,n3]
                    # print(nn, JJJ_all[nn,0],J1mesh[n1])
                    dJ1 += (JJJ_all[nn,0]-J1mesh[n1])
                    nn += 1
        # print(nn, dJ1, JJJ_all[-1,0], J1mesh[-1])
        # print("J1: ", JJJ_all[:,0], J1mesh)

        nnx = 0
        sheer = 1
        for n1 in range(nmesh_output): #not nmesh_out-1, the remaining 0 value should be dealed with
            for n2 in range(nmesh_output):
                JJ_all[nnx,0] = J1mesh[n1]
                JJ_all[nnx,1] = J2mesh[n2]
                fJJ_all[nnx] = JJJdistribution[n1,n2,sheer]
                nnx += 1
        # print("f: ", JJJdistribution[:,:,sheer])
        # print("f 00: ", JJ_all[0,0],JJ_all[0,1],fJJ_all[0])
        # print("J1J2: ", J1mesh, J2mesh)
        # print("J1J2: ", JJ_all[:,0], JJ_all[:,1])
        # print("f 00: ", J1mesh[0],J2mesh[0],JJJdistribution[0,0,sheer])
        # print("f 11: ", J1mesh[1],J2mesh[1],JJJdistribution[1,1,sheer])
        # print("f -1-1: ", J1mesh[-1],J2mesh[-1],JJJdistribution[-1,-1,sheer])
        # print("J1J2J3: ", J1mesh, J2mesh, J3mesh)
        # print("J1J2J3: ", JJJ_all[:,0], JJJ_all[:,1], JJJ_all[:,2])

    # if len(snap)>1:
    #     # plt.title(r"Plummer halo with $r_s=1.kpc, N_{particles}=1e4$ and so on", fontsize=20) #bad positopn
    #     plt.savefig(ff+description+"many.action"+".png", dpi=200)

    return J_output, fJ_output, JJ_all, fJJ_all, JJJ_all, fJJJ_all



def J_scatter__plot(J0, col1, col2):

    ##vector only
    if len(J0.shape)!=2:
        print("analysis_data_distribution.py: plot_all_DF(): Input argument 1 should be a 2d-array!")
        sys.exit(0)
        return 0

    ##this is to judge the reliable data for extra aim and store the id in original data list
    A1 = abs(J0[:, col1:col1+3])
    A2 = abs(J0[:, col2:col2+3])
    LJ = len(A1) #the outest dim
    print("length of data = ", LJ)
    Jjudge = np.zeros(LJ)
    for i in range(LJ):
        t1 = A1[i]
        t2 = A2[i]
        if t1[0]>0 and t1[1]>0 and t1[2]>0 and t2[0]>0 and t2[1]>0 and t2[2]>0:
            Jjudge[i] = 1
    N_eff = int(sum(Jjudge))

    J_reliable1 = np.zeros((N_eff,3))
    J_reliable2 = np.zeros((N_eff,3))
    idx = 0
    for i in range(LJ):
        if Jjudge[i]:
            J_reliable1[idx] = A1[i]
            J_reliable2[idx] = A2[i]
            idx += 1
    rate = float(N_eff)/float(LJ)
    print("The rate of reliable actions = %d %f." % (N_eff, rate))

    return J_reliable1, J_reliable2



def devide_bin(J0, nmesh = 100, axisscale = 1):

    ##vector only
    L = len(J0.shape)
    if L!=1:
        print("analysis_data_distribution.py: plot_DF(): Input argument should be a vector!")
        sys.exit(0)
        return 0

    ##this is to judge the reliable data for extra aim and store the id in original data list
    J0 = abs(J0)
    N = len(J0) #the outest dim
    Jjudge = np.zeros(N)
    for i in range(N):
        t = J0[i]
        if t>0: #aim none
        # if t[0]>1.e-2 and t[1]>1.e-6 and t[2]>1.e-6 and t[0]<1.e8 and t[1]<1.e8 and t[2]<1.e8: #aim 1
            Jjudge[i] = 1
    N_eff = int(sum(Jjudge))

    ##resore the reliable data
    Jreliable = np.zeros(N_eff) #np.zeros((N_eff,3))
    idx = 0
    for i in range(N):
        if Jjudge[i]:
            Jreliable[idx] = J0[i]
            idx += 1
    rate = float(N_eff)/float(N)
    print("The rate of reliable data A: %f = (float)(%d/%d)." % (rate, N_eff, N))

    ##classifiy bins
    J = Jreliable #Jreliable[:,k]
    Jmedian = np.median(J) #as scale



    # # Jmesh = np.linspace(0., max(J), nmesh) #for line relative to data
    # Jmesh = np.linspace(0., 1.e4, nmesh) #for line compulsely
    # Jbin = (Jmesh[-1]-Jmesh[0])/nmesh #for line

    # ##ship to each bin
    # Jdistributionfunction = np.zeros(nmesh)
    # for j in J:
    #     for n in range(nmesh-1):
    #         if j>=Jmesh[n] and j<Jmesh[n+1]+1e-20: #remove zeros before
    #             Jdistributionfunction[n] += 1
    # JdistributionOne = Jdistributionfunction/N_eff
    # Jdistributionfunction = JdistributionOne/Jbin
    # # Jmesh /= Jmedian
    # print("rate in each bin: ", JdistributionOne)
    # print("sum: ", sum(JdistributionOne))

    # ##plot
    # if axisscale==1:
    #     aaaa = 0
    #     logJmesh = (Jmesh+(Jmesh[1]-Jmesh[0])/2)/r_scale #set to middle to avoid log(0)
    #     M_mesh = Jdistributionfunction *Jbin*N_eff *M_total/N #here N==N_eff
    #     rho_mesh = np.zeros(nmesh)
    #     V_mesh = np.zeros(nmesh)
    #     log_rho_mesh = np.zeros(nmesh)
    #     print(Jdistributionfunction.shape, rho_mesh.shape, log_rho_mesh.shape)
    #     for nn in range(nmesh-1):
    #         V_mesh[nn] = 4./3*np.pi* (Jmesh[nn+1]**3 - Jmesh[nn]**3)
    #         rho_mesh = M_mesh/V_mesh
    #     log_rho_mesh = rho_mesh/rho_s
    #     print(Jdistributionfunction.shape, rho_mesh.shape, log_rho_mesh.shape)

    # return logJmesh[0:-1], log_rho_mesh[0:-1]



    # Jmesh = np.logspace( np.log(min(J)), np.log(max(J)) ,nmesh ) #for log
    Jmesh = np.logspace( -3., 3. ,nmesh ) #for log

    ##ship to each bin
    Jdistributionfunction = np.zeros(nmesh)
    for j in J:
        for n in range(nmesh-1):
            if j>=Jmesh[n] and j<Jmesh[n+1]+1e-20: #remove zeros before
                Jdistributionfunction[n] += 1 #number of partcles in each bin
    # JdistributionOne = Jdistributionfunction/N_eff
    # Jdistributionfunction = JdistributionOne/Jbin

    ##plot
    relative_r_mesh = (Jmesh+(Jmesh[1]-Jmesh[0])/2)/r_scale #set to middle to avoid log(0)
    M_mesh = Jdistributionfunction/N *M_total #here N==N_eff
    rho_mesh = np.zeros(nmesh)
    V_mesh = np.zeros(nmesh)
    relative_rho_mesh = np.zeros(nmesh)
    print(Jdistributionfunction.shape, rho_mesh.shape, relative_rho_mesh.shape)
    for nn in range(nmesh-1):
        V_mesh[nn] = 4./3*np.pi* (Jmesh[nn+1]**3 - Jmesh[nn]**3)
        rho_mesh = M_mesh/V_mesh

    relative_r_mesh = Jmesh/r_scale
    relative_rho_mesh = rho_mesh/rho_s

    return relative_r_mesh, relative_rho_mesh



## action DF models
# import sympy
# xx,yy,zz = sympy.symbols("xx,yy,zz")
def func_poly5(x, a,b,c,d,e,f): #good3
    return a+b*x+c*x**2+d*x**3+e*x**4+f*x**5

def func_powerlaw_M1(x, a,c):
    return a*x**(-c) #donot add x+b, *+c

def func_powerlaw(x, a,b,c,d): #good1
    return a*(x+b)**(-c)+d

def func_exp(x, a,b,c): #good2
    return a*np.exp(-b*x)+c
    # if inx>=0:
    #     return 1.0/(1+exp(-inx))
    # else:
    #     return exp(inx)/(1+exp(inx))

def func_Gau(x, a,m,s):
    return a*np.exp(-((x-m)/s)**2) +0. #-, cannot from + #b??

def func_Gau2(x, a,b,c,d,e,f):
    return a*np.exp(-(x-b)**2)/(2.*c**2) +d*np.exp(-(x-e)**2)/(2.*f**2)

def func_doublepowerlaw(x, a,b,c):
    return (x-a)**b +c

def func_doublepowerlaw_Posti(x, a,b,c):
    return a*(1+b*x)**1.67/(1+c*x)**2.9

def func_doublepowerlaw_down(x, a,b,c):
    J0 = np.sqrt(G*M_total*r_scale)
    xx = x/J0
    return a*xx**b*(1.+xx)**c

def func_Posti15(x):
    return x
    
def func_WE15_1(x, a,b,c): #bad
    J0 = np.sqrt(G*M_total*r_scale)
    NJ2 = 1.*a #
    k = NJ2*M_total/(2*np.pi)**3
    TD = x/J0*b #
    LL = 2*TD*c #
    A = TD*LL**-(5./3)
    B = (J0**2+LL**2)**(5./3)
    return k*A/B

def func2_exp(x, a,k1,k2):
    return a*np.exp(-k1*x[:,0]-k2*x[:,1])

def func2_shadow_exp(x,y, a,k1,k2):
    return a*np.exp(-k1*x-k2*y)

def func3_exp(x, a,k1,k2,k3):
    return a*np.exp(-k1*x[:,0]-k2*x[:,1]-k3*x[:,2])
    # return a*np.exp(k1*x[0]+k2*x[1]+k3*x[2])

## mass DF models
def rho_r_ExpAndPowerlaw(r, a,b,k1,k2):
    return a *np.exp(-abs(b)*r**k1)/r**k2
def rho_r_GaussAndPowerlaw(r, a,b):
    return a *np.exp(-abs(b)*r**2)/r**2

def rho_r_singlepowerlaw(r, a,b,k1):
    return a / (r/abs(b))**k1
def rho_r_King(r, a,b):
    return a / (r/abs(b))**2

def rho_r_doublepowerlaw(r, a,b,k1,k2):
    return a / ( (r/abs(b))**k1 *(1+r/abs(b))**k2 )
def rho_r_NFW(r, a,b):
    return a / ( (r/abs(b))**1 *(1+r/abs(b))**2 )
def rho_r_Hernquist(r, a,b):
    return a / ( (r/abs(b))**1 *(1+r/abs(b))**3 )

def rho_r_BurkertIndex(r, a,b,k1,k2):
    return a*abs(b)**(k1+k2) / ( (r+abs(b))**k1 *(r**k2+abs(b)**k2) )
def rho_r_Burkert(r, a,b):
    return a*abs(b)**3 / ( (r+abs(b))**1 *(r**2+abs(b)**2) )

def rho_r_cLB(r, a):
    return 0



from scipy.optimize import curve_fit
def f_J__fit(func, Jmesh, fJmesh): #func??
    if func==func_exp:
        popt, pcov = curve_fit(func, Jmesh, fJmesh, p0=[1., 1., 1.])
        # popt, pcov = curve_fit(func, Jmesh, fJmesh)
        # popt, pcov = curve_fit(func, Jmesh, fJmesh, p0=[0.01, 1., 0.001])

    elif func==func_Gau:
        m = np.mean(Jmesh)
        s = np.std(Jmesh)
        popt, pcov = curve_fit(func, Jmesh, fJmesh, p0=[1., m, s])

    elif func==func_Gau2:
        popt, pcov = curve_fit(func, Jmesh, fJmesh, p0=[0.1, 10., 1.,    0.1, 15., 0.1])

    elif func==func_powerlaw:
        popt, pcov = curve_fit(func, Jmesh, fJmesh, p0=[1., 1., 1., 0.1])
        # popt, pcov = curve_fit(func, Jmesh, fJmesh,p0=[0.14225543, 0.157705, 0.36341783, -0.03731371])

    elif func==func_poly5:
        popt, pcov = curve_fit(func, Jmesh, fJmesh)

    else:
        #exit(0)
        # popt, pcov = curve_fit(func, Jmesh, fJmesh)
        popt, pcov = curve_fit(func, Jmesh, fJmesh , p0=[2., 1., 1.])

    return popt, pcov





if __name__ == '__main__':

    ## finding K neighbors
    pcloud = np.random.random((10000,3))
    # PP = np.array([0.5, 0.5, 0.5])
    # P = np.array([PP])
    # P = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    P = np.array([[0.5, 0.5, 0.5]])
    # P = np.array([[0.5, 0.5, 0.5], [0., 0.5, 0.5]]) #at least 2D but each individual is independant
    tree = KDTree(pcloud, leaf_size=40)
    N = 10000
    for p in range(10000):
        P += np.ones(P.shape)/float(N)
        indices, distances = tree.query(P, k=10)
        if p%100==0:
            print("p: ", p)
    # print(pcloud)
    # print(indices)
    # print(distances)

    # # ## plot data rho_r
    # # r__D = np.array([0.4, 0.9, 1.3, 1.8, 2.0, 2.1, 2.5, 2.7, 2.9, 3.1, 3.3, 4., 5., 7., 10., 15., 20.])
    # # Nr_deriv_N__D = np.array([0.05*0.75, 0.096, 0.17, 0.24, 0.25+0.05*0.67, 0.25+0.05*0.33, 0.2-0.05*0.33, 0.11, 0.05*1.33, 0.055, 0.045, 0.041, 0.05*0.67, 0.029, 0.024, 0.016, 0.01])
    # # rho_r__D = 1./(4*np.pi*r__D**2) *Nr_deriv_N__D
    # # print(len(r__D), len(rho_r__D))
    # # print(r__D)
    # # print(rho_r__D)
    # # x0 = r__D[1:]
    # # y0 = rho_r__D[1:]
    # # plt.scatter(x0,y0,s=10., label="simu data:    from L08, virialrate=1.9")
    # # plt.plot(x0,y0)

    # # # plot simu rho_r
    # # id_snapshot = 000
    # # ff = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galaxy_general_1_NFW/snaps/txt/snapshot_%03d.txt"%(id_snapshot)
    # # data3 = np.loadtxt(ff, dtype=float)
    # # A0 = data3[:, 0:3]
    # # l = len(A0[:,0])
    # # r = np.zeros(l)
    # # for i in np.arange(0,l):
    # #     r[i] = norm_l(A0[i])
    # # x0, y0 = devide_bin(r, nmesh=100, axisscale=2)
    # # wy0 = np.where((y0!=0)) #&(y0<max(y0)) & (y0>min(y0)) & (y0!=np.NaN) & (y0!=np.Inf))
    # # x0 = x0[wy0]
    # # y0 = y0[wy0]
    # # print("simu data: ", x0, y0, len(x0), len(y0))
    # # plt.scatter(x0,y0,s=10., label="simu data:    from NFW, virial= , snapshot=%03d"%(id_snapshot))
    # # plt.plot(x0,y0)

    # # id_snapshot = 000
    # # ff = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galaxy_general_2_Hernquist/snaps/txt/snapshot_%03d.txt"%(id_snapshot)
    # # # ff = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galaxy_general_1_NFW_X/snaps/txt/snapshot_%03d.txt"%(id_snapshot)
    # # data3 = np.loadtxt(ff, dtype=float)
    # # A0 = data3[:, 0:3]
    # # l = len(A0[:,0])
    # # r = np.zeros(l)
    # # for i in np.arange(0,l):
    # #     r[i] = norm_l(A0[i])
    # # x0, y0 = devide_bin(r, nmesh=100, axisscale=2)
    # # wy0 = np.where((y0!=0)) #&(y0<max(y0)) & (y0>min(y0)) & (y0!=np.NaN) & (y0!=np.Inf))
    # # x0 = x0[wy0]
    # # y0 = y0[wy0]
    # # print("simu data: ", x0, y0, len(x0), len(y0))
    # # plt.scatter(x0,y0,s=10., label="simu data:    from Hernquist, virial= , snapshot=%03d"%(id_snapshot))
    # # plt.plot(x0,y0)

    # id_snapshot = 000
    # ff = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galsbox/galaxy_general_2_Hernquist_1e4/snaps/txt/snapshot_%03d.txt"%(id_snapshot)
    # data3 = np.loadtxt(ff, dtype=float)
    # A0 = data3[:, 0:3]
    # l = len(A0[:,0])
    # r = np.zeros(l)
    # for i in np.arange(0,l):
    #     r[i] = norm_l(A0[i])
    # x0, y0 = devide_bin(r, nmesh=100, axisscale=2)
    # wy0 = np.where((y0!=0)) #&(y0<max(y0)) & (y0>min(y0)) & (y0!=np.NaN) & (y0!=np.Inf))
    # x0 = x0[wy0]
    # y0 = y0[wy0]
    # print("simu data: ", x0, y0, len(x0), len(y0))
    # plt.scatter(x0,y0,s=10., label="simu data:    from Hernquist, virial= , snapshot=%03d"%(id_snapshot))
    # plt.plot(x0,y0)
    # plt.show()

    # id_snapshot = 100
    # ff = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galsbox/galaxy_general_2_Hernquist_1e4/snaps/txt/snapshot_%03d.txt"%(id_snapshot)
    # data3 = np.loadtxt(ff, dtype=float)
    # A0 = data3[:, 0:3]
    # l = len(A0[:,0])
    # r = np.zeros(l)
    # for i in np.arange(0,l):
    #     r[i] = norm_l(A0[i])
    # x0, y0 = devide_bin(r, nmesh=100, axisscale=2)
    # wy0 = np.where((y0!=0)) #&(y0<max(y0)) & (y0>min(y0)) & (y0!=np.NaN) & (y0!=np.Inf))
    # x0 = x0[wy0]
    # y0 = y0[wy0]
    # print("simu data: ", x0, y0, len(x0), len(y0))
    # plt.scatter(x0,y0,s=10., label="simu data:    from Hernquist, virial= , snapshot=%03d"%(id_snapshot))
    # plt.plot(x0,y0)

    # id_snapshot = 200
    # ff = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galsbox/galaxy_general_2_Hernquist_1e4/snaps/txt/snapshot_%03d.txt"%(id_snapshot)
    # data3 = np.loadtxt(ff, dtype=float)
    # A0 = data3[:, 0:3]
    # l = len(A0[:,0])
    # r = np.zeros(l)
    # for i in np.arange(0,l):
    #     r[i] = norm_l(A0[i])
    # x0, y0 = devide_bin(r, nmesh=100, axisscale=2)
    # wy0 = np.where((y0!=0)) #&(y0<max(y0)) & (y0>min(y0)) & (y0!=np.NaN) & (y0!=np.Inf))
    # x0 = x0[wy0]
    # y0 = y0[wy0]
    # print("simu data: ", x0, y0, len(x0), len(y0))
    # plt.scatter(x0,y0,s=10., label="simu data:    from Hernquist, virial= , snapshot=%03d"%(id_snapshot))
    # plt.plot(x0,y0)

    # id_snapshot = 800
    # ff = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galsbox/galaxy_general_2_Hernquist_1e4/snaps/txt/snapshot_%03d.txt"%(id_snapshot)
    # data3 = np.loadtxt(ff, dtype=float)
    # A0 = data3[:, 0:3]
    # l = len(A0[:,0])
    # r = np.zeros(l)
    # for i in np.arange(0,l):
    #     r[i] = norm_l(A0[i])
    # x0, y0 = devide_bin(r, nmesh=100, axisscale=2)
    # wy0 = np.where((y0!=0)) #&(y0<max(y0)) & (y0>min(y0)) & (y0!=np.NaN) & (y0!=np.Inf))
    # x0 = x0[wy0]
    # y0 = y0[wy0]
    # print("simu data: ", x0, y0, len(x0), len(y0))
    # plt.scatter(x0,y0,s=10., label="simu data:    from Hernquist, virial= , snapshot=%03d"%(id_snapshot))
    # plt.plot(x0,y0)



    # ## fit rho_r
    # x1 = np.linspace(min(x0),max(x0), 1000)
    # func = rho_r_NFW
    # y1 = func(x1, 1., 1.)
    # plt.plot(x1,y1, label="rho_r_NFW:    a / ( (r/b)**1 *(1+r/b)**2 )")
    # func = rho_r_Hernquist
    # y1 = func(x1, 1., 1.)
    # plt.plot(x1,y1, label="rho_r_Hernquist:    a / ( (r/b)**1 *(1+r/b)**3 )")

    # # # func = rho_r_ExpAndPowerlaw
    # # # popt, pcov = curve_fit(func, x0, y0 , p0=[1.e-5,1.e-10, 1.,2.])
    # # # y1 = func(x1, *popt)
    # # # print("fit param1: ", popt, pcov)
    # # # plt.plot(x1,y1, label="rho_r_ExpAndPowerlaw:    a *np.exp(-b*r**k1)/r**k2; fit: k1=%.3g, k2=%.3g" % (popt[2], popt[3]))
    # # func = rho_r_GaussAndPowerlaw
    # # popt, pcov = curve_fit(func, x0, y0 , p0=[1.e-5,2.])
    # # # popt, pcov = curve_fit(func, x0, y0 , p0=[1.e-4,2.])
    # # y1 = func(x1, *popt)
    # # print("fit param: ", popt, pcov)
    # # plt.plot(x1,y1, label="rho_r_GaussAndPowerlaw:    a *np.exp(-b*r**2)/r**2")

    # # # func = rho_r_singlepowerlaw
    # # # popt, pcov = curve_fit(func, x0, y0 , p0=[1.e-5,1., 2.])
    # # # y1 = func(x1, *popt)
    # # # print("fit param2:", popt, pcov)
    # # # plt.plot(x1,y1, label="rho_r_singlepowerlaw:    a / (r/b)**k1; fit: k1=%.3g" % (popt[2]))
    # # func = rho_r_King
    # # popt, pcov = curve_fit(func, x0, y0 , p0=[1.e-5,1.])
    # # y1 = func(x1, *popt)
    # # print("fit param:", popt, pcov)
    # # plt.plot(x1,y1, label="rho_r_King:    a / (r/b)**2")

    # # # func = rho_r_doublepowerlaw
    # # # popt, pcov = curve_fit(func, x0, y0 , p0=[1.e-5,1., 1.,2.])
    # # # y1 = func(x1, *popt)
    # # # print("fit param3:", popt, pcov)
    # # # plt.plot(x1,y1, label="doublepowerlaw:    a / ( (r/b)**k1 *(1+r/b)**k2 ); fit: k1=%.3g, k2=%.3g" % (popt[2], popt[3]))
    # # func = rho_r_NFW
    # # popt, pcov = curve_fit(func, x0, y0 , p0=[1.e-5,1.])
    # # print("fit param:", popt, pcov)
    # # y1 = func(x1, *popt)
    # # plt.plot(x1,y1, label="rho_r_NFW:    a / ( (r/b)**1 *(1+r/b)**2 )")

    # # # func = rho_r_BurkertIndex
    # # # popt, pcov = curve_fit(func, x0, y0 , p0=[1.e-5,1., 1.,2.])
    # # # y1 = func(x1, *popt)
    # # # print("fit param4:", popt, pcov)
    # # # plt.plot(x1,y1, label="rho_r_BurkertIndex:    a*b**(k1+k2) / ( (r+b)**k1 *(r**k2+b**k2) ); fit: k1=%.3g, k2=%.3g" % (popt[2], popt[3]))
    # # func = rho_r_Burkert
    # # popt, pcov = curve_fit(func, x0, y0 , p0=[1.e-5,10.])
    # # y1 = func(x1, *popt)
    # # print("fit param:", popt, pcov)
    # # plt.plot(x1,y1, label="rho_r_Burkert:    a*b**3 / ( (r+abs(b))**1 *(r**2+b**2) )")

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend(fontsize=10)
    # plt.savefig("rho_r__fit.png")
    # plt.show()
    # plt.close("all")
    # print("Figs done.")






    # ##plot xyz scatter
    # ff = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galaxy_general_2_Hernquist/snaps/txt/snapshot_900.txt"
    # # ff = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galsbox/galaxy_Bovy13_disk/snaps/snapshot_000.txt"
    # JJJ = np.loadtxt(ff, dtype=float)
    # JJJ_num = np.where(JJJ==np.inf, 0., JJJ)
    # JJJ = np.array(JJJ_num)
    # J_F0 = JJJ[:, 0:3]
    # fig = plt.figure(figsize=(16, 12), dpi=200, facecolor=(0.0, 0.0, 0.0))
    # ax = Axes3D(fig)
    # ax.grid(True)
    # ax.scatter(J_F0[:,0], J_F0[:,1], J_F0[:,2], color="blue", s=0.2, label="actions by formula potential (%f points)" % (len(J_F0)))
    # xl = max(J_F0[:,0])
    # klim = 0.05
    # ax.set_xlim(-xl*klim, xl*klim)
    # ax.set_ylim(-xl*klim, xl*klim)
    # ax.set_zlim(-xl*klim, xl*klim)
    # ax.set_title(r'scattered particle positions in one snapshot', fontsize=10)
    # plt.legend(fontsize=8)
    # # ax.view_init(elev = 90., azim = 90.)
    # plt.show()
    # plt.close("all")
    # print("Figs scatter done.")





    # ##plot JJJ scatter
    # ff = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galaxy_general_1_NFW/snaps/aa/allID_0.txt"
    # JJJ = np.loadtxt(ff, dtype=float)
    # JJJ_num = np.where(JJJ==np.inf, 0., JJJ)
    # JJJ = np.array(JJJ_num)
    # J_F, J_D = J_scatter__plot(JJJ, 0, 7)
    # J_F0 = JJJ[:, 0:3]

    # fig = plt.figure(figsize=(16, 12), dpi=200, facecolor=(0.0, 0.0, 0.0))
    # ax = Axes3D(fig)
    # ax.grid(True)
    # # ax.plot([1,2], [1,2], [1,2], color="green", lw=1., label="difference of actions between by data and by formula")
    # # ax.plot([1,4], [2,4], [3,4], color="green", lw=1., label="difference of actions between by data and by formula")
    # ax.scatter(J_F[:,0], J_F[:,1], J_F[:,2], color="blue", s=0.2, label="actions by formula potential (%f points)" % (len(J_F0)))
    # ax.scatter(J_D[:,0], J_D[:,1], J_D[:,2], color="red",  s=0.2, label="actions by data potential (%f points)" % (len(J_D)))
    # ax.plot([J_D[0,0], J_F[0,0]], [J_D[0,1], J_F[0,1]], [J_D[0,2], J_F[0,2]], color="green", lw=0.5, 
    #             label="differences of actions between by data and by formula"+'\n'+'(only 1/5 of these differences are displayed for clarity)')
    # for i in np.arange(8000, len(J_D)): # (7000,8000): 
    #     ax.plot([J_D[i,0], J_F[i,0]], [J_D[i,1], J_F[i,1]], [J_D[i,2], J_F[i,2]], color="green", lw=0.5)
    # klim = 12
    # ax.set_xlim(0., np.median(J_D[:,0])*klim)
    # ax.set_ylim(0., np.median(J_D[:,1])*klim)
    # ax.set_zlim(0., np.median(J_D[:,2])*klim)
    # ax.set_xlabel(r'scaled R-action $J_R\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=10)
    # ax.set_ylabel(r'scaled $\phi$-action $L_z\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=10)
    # ax.set_zlabel(r'scaled z-action $J_z\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=10)
    # ax.set_title(r'scattered actions-points in one snapshot', fontsize=10)
    # plt.legend(fontsize=8)
    # # ax.view_init(elev = 90., azim = 90.)
    # plt.savefig("/home/darkgaia/0prog/data_process/0arrangement/analysis_data_ps__.png")
    # plt.show()

    # plt.close("all")
    # print("Figs scatter done.")





    # ## f1, f2, f3 of one halo
    # galbox = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galsbox/"
    # cl = ["purple", "blue", "green", "orange", "red"]
    # model = "galaxy_general"
    # # model = "galaxy_general_center"
    # # model = "galaxy_general_centerno"
    # # model = "galaxy_general_1_NFW_1e4"
    # # model = "galaxy_general_2_Hernquist_1e4"
    # # model = "galaxy_general_3_Burkert_center"
    # # model = "galaxy_general_4_Einasto_center"
    # # model = "galaxy_general_5_isothermal_center"
    # # model = "galaxy_general_6_Plummer_center"

    # # em = [0]
    # em = [0, 1, 2, 3]
    # start = 22
    # nmesh = 200
    # for j_ss in range(len(em)):
    #     for i_00X in range(1):
    #         j_X00 = em[j_ss]
    #         snapid = j_X00*100 + i_00X
    #         ff = galbox+model+"/snaps/aa"
    #         filename = ff+"/allID_%03d.txt" % (snapid)
    #         print(filename)
    #         J_discrete, fJ_discrete, JJJ_all, fJJJ_all, JJ_all, fJJ_all = f_J__plot(filename, [snapid], colJ = start, nmesh = nmesh)

    #         #1d distribution
    #         for k in range(3):
    #             plt.subplot(2,2,k+1)

    #             func = func_powerlaw #func_WE15_1 #func_exp #func_poly5 #func_powerlaw_M1 #func_doublepowerlaw_down
    #             J_scale = 1e4
    #             x = J_discrete[:,k].T /J_scale #to rescale, or it will have bad numerical value and stop to calculation
    #             y = fJ_discrete[:,k].T *1e0
    #             # x *= J_scale
    #             # y /= scale
    #             # y1 /= scale
    #             x1 = np.linspace(min(x),max(x), 1000)

    #             # popt, pcov = f_J__fit(func, x, y)
    #             # y1 = func(x1, *popt) #a*(x+b)**(-c)+d
    #             # print("fit params: ", popt, pcov)
    #             # plt.plot(x1,y1, color=cl[j_ss], label="fJ1 of snapshot_%03d (0.01 Gyr/snapshot)" % (snapid))

    #             plt.plot(x,y, color=cl[j_ss], label="fJ1 of snapshot_%03d (0.01 Gyr/snapshot)" % (snapid))
    #             plt.scatter(x,y,s=2.)
    #             # plt.bar(x,y, width=(max(x)-min(x))/nmesh/2)
    #             plt.legend()
    #             plt.xlim(-np.median(x)/100, np.median(x)/2)
    #             plt.ylim(0., max(y)*1.1)
    #             if k==0:
    #                 plt.xlabel(r'scaled R-action $J_R\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=10)
    #                 plt.ylabel(r'distribution function (divided by bin) $f_1(J_R)\, \times 1$'+'\n'+r'($L_z$ and $J_z$ are intergrated)', fontsize=10)
    #                 # plt.text(max(x)/5, max(y), r'$f_1 = a\times (J_R+b)^{-c} +d$'
    #                 #             +'\n'+r'$a=%.3g,\, b=%f\,\mathrm{kpc}\cdot\mathrm{km/s},\, c=%f,\, d=%f$'%(popt[0],popt[1],popt[2],popt[3]), fontsize=10)
    #             if k==1:
    #                 plt.xlabel(r'scaled $\phi$-action $L_z\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=10)
    #                 plt.ylabel(r'distribution function (divided by bin) $f_1(L_z)\, \times 1$'+'\n'+r'($J_R$ and $J_z$ are intergrated)', fontsize=10)
    #                 # plt.text(max(x)/5, max(y), r'$f_1 = a\times (L_z+b)^{-c} +d$'
    #                 #             +'\n'+r'$a=%.3g,\, b=%f\,\mathrm{kpc}\cdot\mathrm{km/s},\, c=%f,\, d=%f$'%(popt[0],popt[1],popt[2],popt[3]), fontsize=10)
    #             if k==2:
    #                 plt.xlabel(r'scaled z-action $J_z\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=10)
    #                 plt.ylabel(r'distribution function (divided by bin) $f_1(J_z)\, \times 1$'+'\n'+r'($J_R$ and $L_z$ are intergrated)', fontsize=10)
    #                 # plt.text(max(x)/5, max(y), r'$f_1 = a\times (J_z+b)^{-c} +d$'
    #                 #             +'\n'+r'$a=%.3g,\, b=%f\,\mathrm{kpc}\cdot\mathrm{km/s},\, c=%f,\, d=%f$'%(popt[0],popt[1],popt[2],popt[3]), fontsize=10)
    #         # plt.show()
    #         # plt.savefig("fit_action1"+".png", dpi=200)
    #         # plt.close("all")
    #         # print("Figs1 done.")
    # plt.show()



    # #rho 000
    # # model = "galaxy_general"
    # # model = "galaxy_general_center"
    # # model = "galaxy_general_centerno"
    # model = "galaxy_general_1_NFW_1e4"
    # # model = "galaxy_general_2_Hernquist_1e4"
    # # model = "galaxy_general_3_Burkert_center"
    # # model = "galaxy_general_5_isothermal_center"
    # # model = "galaxy_general_6_Plummer_center"

    # snapshot = [0, 100, 200, 300]
    # nmesh = 100
    # for s in range(len(snapshot)):
    #     path_detail = "/snaps/txt/snapshot_%03d.txt" % (s)
    #     filename = galbox+model+path_detail
    #     data3 = np.loadtxt(filename, dtype=float)
    #     A0 = data3[:, 0:3]
    #     l = len(A0[:,0])
    #     r = np.zeros(l)
    #     for i in np.arange(0,l):
    #         r[i] = norm_l(A0[i])
    #     x0, y0 = devide_bin(r, nmesh=nmesh, axisscale=2)
    #     wy0 = np.where((y0!=0)) #| (y0!=np.NaN) | (y0!=np.Inf)) #&(y0<max(y0)) & (y0>min(y0)) & (y0!=np.NaN) & (y0!=np.Inf))
    #     x0 = x0[wy0]
    #     y0 = y0[wy0]
    #     x0 = x0[0:-1]
    #     y0 = y0[0:-1]
    #     print("simu data: ", x0, y0, len(x0), len(y0))
    #     plt.scatter(x0,y0,s=10., label="rho(r)")
    #     # plt.plot(x0,y0)

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()



            # ##2d distribution
            # nl = len(fJJ_all)
            # J_scale = 1e4
            # x3d = JJ_all/J_scale
            # y = fJJ_all*1e0
            # func = func2_exp
            # popt, pcov = curve_fit(func, x3d, y, p0=[1e-10, 1., 1.])
            # yy = func(x3d, *popt)
            # sx = x3d.shape
            # sy = y.shape
            # syy = yy.shape
            # print("sum: ", sum(fJJ_all), sum(y), sum(yy), sum((y-yy)**2))
            # print(popt,pcov)
            # # x3d *= J_scale
            # # y /= scale
            # # yy /= scale
            # # print("at 00: ", x3d[0,0], y[0], yy[0])
            # # print(sx,sy,syy)
            # # print(x3d[:,0],x3d[:,1])
            # # print("y yy: ", y, yy)
            # # print("where 0: ", np.where(x3d[:,0]==0))
            # # print(np.where(x3d[:,1]==0))
            # # print(np.where(y==0))

            # fig = plt.figure(figsize=(16, 12), dpi=80, facecolor=(0.0, 0.0, 0.0))
            # ax = Axes3D(fig)
            # ax.grid(True) # ax.set_axis_off() #or to remove all relevent axis
            # nmeshmesh = len(J_discrete[:,0])
            # J1mesh = np.linspace(0., J_discrete[10,0], 2*nmeshmesh)*1e-4
            # J2mesh = np.linspace(0., J_discrete[10,1], 2*nmeshmesh)*1e-4
            # x_mesh, y_mesh = np.meshgrid(J1mesh, J2mesh, indexing='ij') #gen mesh is to product each other with dims
            # z_mesh = func2_shadow_exp(x_mesh, y_mesh, *popt)*1e0
            # ax.plot_surface(x_mesh, y_mesh, z_mesh, color="blue")
            # ax.scatter(x3d[:,0], x3d[:,1], yy[:], color="blue", s=0.1, label="fitted function value") #fitted function points
            # ax.scatter(x3d[:,0], x3d[:,1], y[:], color="red", s=5.0, label="bin-number data (%d points)"%(nl)) #data points
            # # ax.set_zlim(0., 1e-10)
            # ax.set_xlabel(r'scaled R-action $J_R\, \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=16)
            # ax.set_ylabel(r'scaled $\phi$-action $L_z\, \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=16)
            # ax.set_zlabel(r'scaled distribution function (divided by bin) $f_2(J_R,L_z)\, \times 1$'+'\n'+r'($J_z$ are intergrated)', fontsize=16)
            # ax.set_title(r'$f_2 = a\times\exp(-k_1 J_R -k_2 L_z)$'
            #                 +'\n'+r'$a=%.3g,\, k_1=%f\,\mathrm{kpc}^{-1}\cdot\mathrm{km/s}^{-1},\, k_2=%f\,\mathrm{kpc}^{-1}\cdot\mathrm{km/s}^{-1}$'%(popt[0],popt[1],popt[2]), fontsize=16)
            # ax.legend()
            # ax.view_init(elev = 0., azim = 90.)
            # # ax.set_xscale('log')
            # # ax.set_yscale('log')
            # plt.legend()
            # plt.show()
            # plt.savefig("fit_action2"+".png", dpi=200)
            # plt.close("all")
            # print("Figs2 done.")



            # ##3d distribution
            # # nl = 1000
            # # x3d = np.zeros((nl,3))
            # # y = np.zeros(nl)
            # # nums = random.sample(range(len(fJJJ_all)), nl)
            # # x3d = JJJ_all[nums]*1e-4
            # # y = fJJJ_all[nums]*1e10
            # nl = len(fJJJ_all)
            # J_scale = 1e4
            # x3d = JJJ_all/J_scale
            # y = fJJJ_all*1e0
            # func = func3_exp
            # popt, pcov = curve_fit(func, x3d, y, p0=[1e-10, 1., 1., 1.])
            # yy = func(x3d, *popt)
            # sx = x3d.shape
            # sy = y.shape
            # syy = yy.shape
            # print("sum: ", sum(fJJJ_all), sum(y), sum(yy), sum((y-yy)**2))
            # print(popt,pcov)
            # # x3d *= J_scale
            # # y /= scale
            # # yy /= scale
            # # print(sx,sy,syy)
            # # print("at 00: ", x3d[0,0], y[0], yy[0])
            # # print("x1x2x3: ", x3d[:,0],x3d[:,1],x3d[:,2])
            # # print("y yy: ", y, yy)
            # # print("where 0: ", np.where(x3d[:,0]==0))
            # # print(np.where(x3d[:,1]==0))
            # # print(np.where(y==0))

            # fig = plt.figure(figsize=(16, 12), dpi=80, facecolor=(0.0, 0.0, 0.0))
            # ax = Axes3D(fig)
            # ax.grid(True) # ax.set_axis_off() #or to remove all relevent axis
            # ax.scatter(x3d[:,0], x3d[:,1], y[:], color="red", s=0.5, label="samples (%d points)"%(nl))
            # ax.scatter(x3d[:,0], x3d[:,1], yy[:], color="blue", s=0.5, label="fitted (%d points)"%(nl))
            # # ax.scatter(x3d[0:1000,0], x3d[0:1000,2], y[0:1000], color="red", s=0.5) #, label="%f"%(PotMeanDiff))
            # # ax.scatter(x3d[0:1000,0], x3d[0:1000,2], yy[0:1000], color="blue", s=0.5)
            # # ax.plot_surface(x3d[0:1000,0], x3d[0:1000,2], yy[0:1000], color="blue") #convert to 2d
            # # ax.set_zlim(0., 1e-10)
            # ax.set_xlabel(r'scaled R-action $J_R\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=16)
            # ax.set_ylabel(r'scaled $\phi$-action $L_z\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=16)
            # ax.set_zlabel(r'scaled distribution function (divided by bin) $f(J_R,L_z,J_z)\, \times 1$'
            #                 +'\n'+r'(here only $J_R$ and $L_z$ axis displayed, $J_z$ axis are overlapped)', fontsize=16)
            # ax.set_title(r'$f = a\times\exp(-k_1 J_R -k_2 L_z -k_3 J_z)$'
            #                 +'\n'+r'$a=%.3g,\, k_1=%f\,\mathrm{kpc}^{-1}\cdot\mathrm{km/s}^{-1},\, k_2=%f\,\mathrm{kpc}^{-1}\cdot\mathrm{km/s}^{-1},\, k_3=%f\,\mathrm{kpc}^{-1}\cdot\mathrm{km/s}^{-1}$'%(popt[0],popt[1],popt[2],popt[3]), fontsize=16)
            # ax.legend()
            # ax.view_init(elev = 0., azim = 90.)
            # plt.legend()
            # plt.show()
            # plt.savefig("fit_action3"+".png", dpi=200)
            # plt.close("all")
            # print("Figs3 done.")