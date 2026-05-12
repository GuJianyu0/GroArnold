#!/usr/bin/env python
# -*- coding:utf-8 -*-

# ============================================================================================
# Description: A wrapper to simply estimate the multifractional parameters 
# of galaxy distribution function.
# Author: Jianyu Gu
# ============================================================================================

##[] modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as spopt
from scipy.fftpack import fft, ifft

import RW_data_CMGD as rdc
import analysis_data_distribution as add
import galaxy_models as gm
import triaxialize_galaxy as tg
import KDTree_python as kdtp
import fit_galaxy_wrapper as fgw
import scipy.integrate as spitg
import advection_diffusion_scale_simple as adss



##[] functions
def search_particles_in_cubic_cell(data, boundaries):
    return 0

def multifractional_statistics(data, data_bound_r=100., smin=0.1, smax=10., N_grid_1d=20, dim=2):
    mesh1d = add.logspace_1d_by_boundaries(smin, smax, N_grid_1d)
    mesh = np.meshgrid(mesh1d, mesh1d) #?? dim=[any]
    N_block_1d = data_bound_r/N_grid_1d
    #block for for
    #deviate
    for i in mesh1d:
        P = search_particles_in_cubic_cell(data, [[0.-0.5, 1.-0.5]])
        # P_mean, P_std
    #for mesh2d
    #statistic powerlaw to compare
    return 0

def multifractional_spectrum():
    return 0

def multifractional_restore_by_fitting():
    return 0

def diffution_coeff_by_DF_with_time():
    return 0 #

def diffusion_coeff_simply(DF):
    return 0

def search_multifractional_parameter_and_diffusion_island_scale():
    return 0 #

def plot_multifractional_and_diffusion():
    return 0

def GFGU_MFDMA_2D(X, n_min , n_max , N, theta , q): #from code GFGU_MFDMA_2D.m of Gao-Feng Gu (2010)
    '''
    function [ n , Fq , tau , alpha , f ] = GFGU_MFDMA_2D(X, n_min , n_max , N, theta , q)
    %
    % The procedure GFGU_MFDMA_2D
    % is used to calculate the multifractal properties of two-dimensional
    % multifractal measures.
    %
    % Input :
    % X: the two-dimensional multifractal measures we considered.
    % n_min: the lower bound of the segment size n
    % n_max: the upper bound of the segment size n
    % N: the length of n , that is , the data points in the plot of Fq VS n
    % theta: the position parameter of the moving window
    % q: multifractal order
    % 
    % Output :
    % n: segment size series
    % Fq: q-th order fluctuation function
    % tau: multifractal scaling exponent
    % alpha: multifractal singularity strength function
    % f: multifractal spectrum
    % 
    % The procedure works as follows:
    % 1) For each n , construct the cumulative sum Y in a moving window .
    % 2) C a l c u l a t e  t h e   moving  a v e r a g e   f u n c t i o n \ w i d e   t i l d e   {Y} .
    % 3) Determine t h e r e s i d u a l e by d e t r e n d i n g \ w i d e t i l d e {Y} from Y.
    % 4) E s t i m a t e t h e r o o t-mean-s q u a r e f u n c t i o n F .
    % 5) C a l c u l a t e t h e q-t h o r d e r o v e r a l l f l u c t u a t i o n f u n c t i o n Fq .
    % 6) C a l c u l a t e t h e m u l t i f r a c t a l s c a l i n g e x p o n e n t t a u ( q ) .
    % 7) C a l c u l a t e t h e s i n g u l a r i t y s t r e n g t h f u n c t i o n a l p h a ( q ) and spect ru m f ( a l p h a ) .
    % 
    % Note :
    % 1) The window size and the segment size must be identical.
    % 2) The lower bound n_min would better be selected around 10 .
    % 3) The upper bound n_max would better be selected around 10% of min(size(X)).
    % 4) N would better be seleceted in the range [ 2 0 , 4 0 ] .
    % 5) The parameter theta varies in the range [ 0 , 1 ] , and we have
    % theta=theta1=theta2. Theta = 0 correponds to backward MFDMA, and
    % theta=0.5 corresponds to the centered MFDMA, and theta=1 corresponds to
    % theforward MFDMA. We recommend theta=0.
    % 6) In the procedure, we have n=n1=n2 for the segment size.
    % 
    % Example :
    % X = rand(200, 200);
    % q = -4:0.1:4;
    % [n, Fq, tau, alpha, f] = GFGU_MFDMA_2D( X, 10 , round( min( size(X) )/10 ), 30, 0, q );
    % 

    %% main
    N1 = size(X, 1);
    N2 = size(X, 2);
    MIN = log10(n_min);
    MAX = log10(n_max);
    n = ( unique( round( logspace(MIN,MAX,N) ) ) )';

    for i = 1: length(n)
        lgth = n(i, 1) ;

        Y = zeros(N1-lgth+1, N2-lgth+1) ;
        Y1 = zeros(N1-lgth+1, N2-lgth+1) ;
        for j = 1 : N1-lgth+1
            for k = 1 : N2-lgth+1
                Z = X( j : j+lgth-1 , k : k+lgth-1) ;
                Z1 = ( cumsum( ( cumsum(Z) )' ) )' ;
                % Construct the cumulative sum Y
                Y(j, k) = Z1(end, end) ;
                % Calculate the moving average function \ wide tilde {Y}
                Y1(j, k) = mean( Z1(:) ) ;
            end
        end

        % Determine the residual e
        x0 = 1: size(Y, 1) - min( floor( lgth*theta ), lgth-1);
        y0 = 1: size(Y, 2) - min( floor( lgth*theta ), lgth-1);
        x1 = size (Y1, 1) - length ( x0 ) +1: size (Y1 , 1 ) ;
        y1 = size (Y1, 2) - length ( y0 ) +1: size (Y1 , 2 ) ;
        e = Y(x0, y0) - Y1( x1 , y1 ) ;

        % Estimate the root-mean-square function F
        for k1 =1: floor( size ( e , 1 ) / lgth )
            for k2 =1: floor( size( e , 2 ) / lgth )
                E=e ( ( k1-1)*lgth +1: k1*lgth , ( k2-1)*lgth +1: k2*lgth ) ;
                F{ i } ( k1 , k2 )=sqrt(mean(E ( : ).^2 ) ) ;
            end
        end
    end

    % Calculate the q-th order overall fluctuation function Fq
    for i =1 : length( q )
        for j =1 : length(F)
            f = F{ j } ( : ) ;
            if q( i ) == 0
                Fq( j , i ) = exp( 0.5*mean( log( f.^2 ) ) ) ;
            else
                Fq( j , i ) = (mean( f.^q ( i ) ) ) ^(1 / q ( i ) ) ;
            end
        end
    end

    % Calculate the multifractal scaling exponent tau( q )
    for i = 1: size( Fq , 2 )
        fq = Fq( : , i ) ;
        r = regstats( log( fq ) , log( n ) , 'linear' , { 'tstat' } ) ;
        k = r . tstat . beta( 2 ) ;
        h( i , 1 ) = k ;
    end
    tau=h.*q'-2 ;

    % Calculate the singularity strength function alpha( q ) and spectrum f( alpha )
    dx =7;
    dx = fix ( ( dx-1) / 2 ) ;
    for i = dx +1: length( tau )-dx
        xx = q( i-dx : i+dx ) ;
        yy = tau( i-dx : i+dx ) ;
        r = regstats( yy , xx , 'linear' , { 'tstat' } ) ;
        alpha( i , 1 ) = r . tstat . beta( 2 ) ;
    end
    alpha = alpha(dx+1:end);
    f = q(dx+1:end-dx)'.*alpha - tau(dx+1:end-dx);


    %% debug
    % n=0
    % Fq=1
    % tau=2
    % alpha=3
    % f=4

    %% plot
    % size(n), size(Fq), size(tau), size(alpha), size(f)
    % tau_th = 0 %?? p1, p2, p3, p4
    % scatter(n, Fq(1,:))
    % scatter(q, tau)
    % scatter(q, delta_tau)
    % scatter(alpha, f)
    '''
    return 0

def MFDMA_pre_all():
    filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/"\
        +"snapshot_160.action.method_all.txt"
    # filename1 = filename+".small"
    data = np.loadtxt(filename)
    # np.savetxt(filename1, data[::1000, :])
    # data = np.loadtxt(filename1)
    xv = data[:, 0:6]
    x = data[:,0:3]
    rx = add.norm_l(x, axis=1)
    v = data[:,3:6]
    rv = add.norm_l(v, axis=1)
    IDs = data[6]
    mass = np.ones(len(data))
    potd = data[:,11]
    energy = 0.5*add.norm_l(v,axis=1)**2 + potd
    energy_T = np.array([energy]).T
    percentiles = add.percentiles_by_xv_data(xv)
    add.DEBUG_PRINT_V(1, percentiles, "percentiles")

    # data = np.random.random((100,100))
    # multifractional_statistics(data)

    # N_particle = 100
    # rx = np.random.random(N_particle)
    # rv = np.random.random(N_particle)
    # # rx = np.linspace(0., 1., N_particle)
    # # rv = np.linspace(0., 1., N_particle)

    rxv_for_kernel = add.merge_array_by_hstack([rx/np.mean(rx), rv/np.mean(rv)]) #for dimonsionless and kernel
    add.DEBUG_PRINT_V(1, np.mean(rx), np.max(rxv_for_kernel[:,0]), np.max(rxv_for_kernel[:,1]))
    mass = np.ones(len(rxv_for_kernel))
    KD_rxv = kdtp.KDTree_galaxy_particles(rxv_for_kernel, weight_extern_instinct=mass)
    N_mesh1d = 200
    mesh1d = np.linspace(0., 3., N_mesh1d) #3
    density_mesh2d = np.zeros((N_mesh1d, N_mesh1d))
    # mesh2d = np.meshgrid(mesh1d, mesh1d)
    for i in np.arange(N_mesh1d):
        for j in np.arange(N_mesh1d):
            # print(np.array([mesh1d[i], mesh1d[j]]), "np.array([mesh1d[i], mesh1d[j]])")
            # density_mesh2d[i,j] = KD_rxv.density_by_number_per_volumn_2d_simple([[1.,1.]])
            density_mesh2d[i,j] = KD_rxv.density_by_number_per_volumn_2d_simple([np.array([mesh1d[i], mesh1d[j]])])
    add.DEBUG_PRINT_V(1, np.shape(density_mesh2d), "target shape 2")
    np.savetxt("./proj2/galaxy_small_snapshot_160_mf2d.txt", density_mesh2d)



##[] main
if __name__ == "__main__":

    filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/"\
        +"snapshot_160.action.method_all.txt"
    # filename1 = filename+".small"
    data = np.loadtxt(filename)
    # np.savetxt(filename1, data[::1000, :])
    # data = np.loadtxt(filename1)
    xv = data[:, 0:6]
    x = data[:,0:3]
    rx = add.norm_l(x, axis=1)
    v = data[:,3:6]
    rv = add.norm_l(v, axis=1)
    IDs = data[6]
    mass = np.ones(len(data))
    potd = data[:,11]
    energy = 0.5*add.norm_l(v,axis=1)**2 + potd
    energy_T = np.array([energy]).T
    percentiles = add.percentiles_by_xv_data(xv)
    add.DEBUG_PRINT_V(1, percentiles, "percentiles")
    # data = np.random.random((100,100))
    # multifractional_statistics(data)

    # N_particle = 1000
    # rx = np.random.random(N_particle)
    # rv = np.random.random(N_particle)
    # # rx = np.linspace(0., 1., N_particle)
    # # rv = np.linspace(0., 1., N_particle)

    rxv_for_kernel = add.merge_array_by_hstack([rx/np.mean(rx), rv/np.mean(rv)]) #for dimonsionless and kernel
    add.DEBUG_PRINT_V(1, np.mean(rx), np.max(rxv_for_kernel[:,0]), np.max(rxv_for_kernel[:,1]))
    mass = np.ones(len(rxv_for_kernel))
    plt.scatter(rxv_for_kernel[:,0], rxv_for_kernel[:,1], s=0.5)
    plt.xlabel("r/r_mean")
    plt.ylabel("vtot/vtot_mean")
    plt.title("galaxy_Sersic_snapshot_160_particlecount_1e6")
    plt.show()
    exit(0)
    
    N_split = 3
    rxvs = add.split_mesh_by_percentile_2d(rxv_for_kernel, N_split)
    
    for io in np.arange(N_split):
        for jo in np.arange(N_split):
            add.DEBUG_PRINT_V(1, io, jo, np.shape(rxvs))#, np.shape(rxvso))
            rxvso = rxvs[io][jo]
            suffix = "_x%dv%d"%(io,jo)

            KD_rxv = kdtp.KDTree_galaxy_particles(rxvso, weight_extern_instinct=mass)
            N_mesh1d = 200
            low0 = np.max([0., np.min(rxvso[:,0])])
            up0 = np.min([3., np.max(rxvso[:,0])])
            low1 = np.max([0., np.min(rxvso[:,1])])
            up1 = np.min([3., np.max(rxvso[:,1])])
            mesh1d0 = np.linspace(low0, up0, N_mesh1d)
            mesh1d1 = np.linspace(low1, up1, N_mesh1d)
            density_mesh2d = np.zeros((N_mesh1d, N_mesh1d))
            # mesh2d = np.meshgrid(mesh1d, mesh1d)
            for i in np.arange(N_mesh1d):
                for j in np.arange(N_mesh1d):
                    density_mesh2d[i,j] = KD_rxv.density_by_number_per_volumn_2d_simple([np.array([mesh1d0[i], mesh1d1[j]])])
            add.DEBUG_PRINT_V(1, np.shape(density_mesh2d), "target shape 2")
            np.savetxt("./proj2/galaxy_small_snapshot_160_mf2d"+suffix+".txt", density_mesh2d)




    # filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/"\
    #     +"snapshot_160.action.method_all.txt"
    # filename1 = filename+".small"
    # # data = np.loadtxt(filename)
    # # np.savetxt(filename1, data[::1000, :])
    # data = np.loadtxt(filename1)
    # xv = data[:, 0:6]
    # x = data[:,0:3]
    # rx = add.norm_l(x, axis=1)
    # v = data[:,3:6]
    # rv = add.norm_l(v, axis=1)
    # IDs = data[6]
    # mass = np.ones(len(data))
    # potd = data[:,11]
    # energy = 0.5*add.norm_l(v,axis=1)**2 + potd
    # energy_T = np.array([energy]).T
    # percentiles = add.percentiles_by_xv_data(xv)
    # add.DEBUG_PRINT_V(1, percentiles, "percentiles")
    
    # '''
    # KD_x = kdtp.KDTree_galaxy_particles(x, weight_extern_instinct=mass)
    # xm_1d = np.linspace(-150., 150., N_mesh_eachdim)
    # xm = np.meshgrid(xm_1d, xm_1d, xm_1d)
    # xm_stroke_1 = mesh_cubic_3dN_to_1dN(xm[0])
    # xm_stroke_2 = mesh_cubic_3dN_to_1dN(xm[1])
    # xm_stroke_3 = mesh_cubic_3dN_to_1dN(xm[2])
    # xm_stroke = vector_1d_to_vector_nd_h((xm_stroke_1, xm_stroke_2, xm_stroke_3))
    # DF_xm_stroke = KD_x.density_SPH(xm_stroke)
    # DF_xm = mesh_cubic_1dN_to_3dN(DF_xm_stroke, (N_mesh_eachdim, N_mesh_eachdim, N_mesh_eachdim))
    # # KD_v = kdtp.KDTree_galaxy_particles(v, weight_extern_instinct=mass)
    # # vm = xm
    # # DF2m = KD_v.density_SPH(vm)
    # KD_e = kdtp.KDTree_galaxy_particles(energy_T, weight_extern_instinct=mass)
    # em = energy
    # DF_em = KD_e.density_1d(energy_T)
    # add.DEBUG_PRINT_V(1, np.shape(DF_xm), "np.shape(DF_xm)")
    # add.DEBUG_PRINT_V(1, np.shape(em), "np.shape(em)")
    # add.DEBUG_PRINT_V(1, np.shape(DF_em), "np.shape(DF_em)")
    # # exit(0)

    # p0_em = [1., 1., 1., 1.]
    # bounds_em = ([1e-10, 1e-10, 1e-10, 1e-10], [1e10, 1e10, 1e10, 1e10])
    # par_LB, cov_LB = fit_DF_leastsq(DF_LB1_versus_energy, em, DF_em, p0=p0_em, bounds=bounds_em)
    # beta, eta0, lam, epslm = par_LB #??
    # DF_em_fit = DF_LB1_versus_energy(em, beta, eta0, lam, epslm)
    # add.DEBUG_PRINT_V(1, par_LB, "par_LB")
    # plt.scatter(em, DF_em, marker=".", label="em~DF_em_data")
    # plt.scatter(em, DF_em_fit, marker="x", label="em~DF_em_fit")
    # plt.ylim(-0.01, 0.07)
    # plt.legend()
    # # plt.show()
    # plt.close()
    # eta = eta0 #??
    # xs = np.mean(rx)
    # vs = np.mean(rv)
    # vcg = vs #??> hypothesis
    # # exit(0)

    # N_xcg = 10
    # xcg_trys = np.logspace(-2., 1.5, N_xcg)
    # deltas = np.ones_like(xcg_trys)*1e10
    # d_xm_1d = xm_1d[1]-xm_1d[0]
    # for i in np.arange(N_xcg):
    #     deltas[i] = rootfunc_coarse_grained_scale(xcg_trys[i], vcg, xs, vs, eta, beta, lam, xm, DF_xm, d_xm_1d)
    #     add.DEBUG_PRINT_V(1, i, xcg_trys[i], deltas[i], "xcg_trys~deltas")
    # print("xcg_trys : ", xcg_trys)
    # print("deltas   : ", deltas)
    # # xcg_trys = [
    # #     1.00000000e-02, 2.44843675e-02, 5.99484250e-02, 1.46779927e-01, 
    # #     3.59381366e-01, 8.79922544e-01, 2.15443469e+00, 5.27499706e+00, 
    # #     1.29154967e+01, 3.16227766e+01
    # # ]
    # # deltas = [
    # #     -3.72964373e+17, -4.91632207e+15, -7.10641939e+13, -1.06377700e+12, 
    # #     -1.61011581e+10, -2.22294174e+08, -8.21070493e+04,  2.20975369e+02, 
    # #     2.20975369e+02,  2.20975369e+02
    # # ]
    # xplot = add.log_abs_P1(xcg_trys)
    # yplot = add.log_abs_P1(deltas)
    # plt.plot(xplot, yplot, marker=".", label="xcg_trys~deltas")
    # # plt.xscale("log") #??
    # # plt.yscale("log")
    # plt.legend()
    # plt.show()
    # plt.close()
    # exit(0)
    # # '''
    
    # filename_after = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/"\
    #     +"snapshot_80.action.method_all.txt"
    # filename_after1 = filename_after+".small"
    # # data_after1 = np.loadtxt(filename_after)
    # # np.savetxt(filename_after1, data_after1[::1000, :])
    # data_after = np.loadtxt(filename_after1)
    # IDs_after = data_after[:,6]
    # x_after = data_after[:,0:3]

    # x_a = x_after #(3,N)
    # IDs_x_a = data_after[:,6]
    # x_a_by_x_a_0 = x_a[np.argsort(x_a[:,0])] #(3,N)
    # color_arr = set_color_by_init(len(x_a), 4)
    # x_a_supper_by_IDs = np.hstack((x_a_by_x_a_0, transpose_1d_to_column(IDs_x_a), transpose_1d_to_column(color_arr)))
    # x_a_supper_by_IDs = x_a_supper_by_IDs[np.argsort(IDs_x_a)] #(5,N)
    
    # x_b = x #(3,N)
    # IDs_x_b = data[:,6]
    # IDs_x_b_by_IDs = IDs_x_b[np.argsort(IDs_x_b)]
    # x_b_by_IDs = x_b[np.argsort(IDs_x_b)] #(3,N)
    # color_arr_by_IDs = color_arr[np.argsort(IDs_x_b)]
    # x_b_supper_by_IDs = np.hstack((x_b_by_IDs, transpose_1d_to_column(IDs_x_b_by_IDs), transpose_1d_to_column(color_arr_by_IDs)))

    # cm = plt.cm.get_cmap("gist_rainbow") #rainbow
    # plt.subplot(1,2,1)
    # f_color = x_a_supper_by_IDs[:,-1]
    # plt.scatter(x_a_supper_by_IDs[:,0], x_a_supper_by_IDs[:,1], s=0.2, c=f_color, cmap=cm, label="t = 0.8 Gyr")
    # plt.xlabel("x (kpc)")
    # plt.ylabel("y (kpc)")
    # plt.xlim(-600., 600.)
    # plt.ylim(-600., 600.)
    # plt.legend()
    # plt.subplot(1,2,2)
    # f_color = x_b_supper_by_IDs[:,-1]
    # plt.scatter(x_b_supper_by_IDs[:,0], x_b_supper_by_IDs[:,1], s=0.2, c=f_color, cmap=cm, label="t = 1.6 Gyr")
    # plt.xlabel("x (kpc)")
    # plt.ylabel("y (kpc)")
    # plt.xlim(-600., 600.)
    # plt.ylim(-600., 600.)
    # plt.legend()
    # plt.show()
    # plt.close()
    # exit(0)





    # ##examples
    # N = 100
    # x = np.linspace(0., 2., N)
    # y = np.linspace(0., 2., N)
    # xm, ym = np.meshgrid(x, y)
    # freq_x = np.arange(len(x))
    # freq_y = np.arange(len(y))
    # freq_xm, freq_ym = np.meshgrid(freq_x, freq_y)
    # add.DEBUG_PRINT_V(1, np.shape(xm), "a")

    # def func1(x1, y1=0., args=None):
    #     return 1.5*np.sin(( x1+y1 )*20.*np.pi) + 0.5 #period 20., total 100.
    # um = func1(xm, ym)
    # print(um)
    # add.DEBUG_PRINT_V(1, np.shape(um), "b")
    # F_um = fft(um)/N
    # add.DEBUG_PRINT_V(1, np.shape(F_um), "c")
    # abs_um=np.abs(F_um)
    # ang_um=np.angle(F_um)
    # add.DEBUG_PRINT_V(1, np.shape(abs_um), "d")
    # add.DEBUG_PRINT_V(1, np.shape(ang_um), "e")
    
    # vw = func1(x)
    # F_vw = fft(vw)/N
    # abs_vw = np.abs(F_vw)
    # ang_vw = np.angle(F_vw)
    # # add.DEBUG_PRINT_V(0, np.shape(x), np.shape(F_vw), np.shape(abs_vw))

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection=None)
    # # ax.scatter(x, F_vw, label="Original")
    # ax.scatter(freq_x, abs_vw, label="Fourior: abs")
    # ax.scatter(freq_x, ang_vw, label="Fourior: ang")
    # plt.legend()
    # plt.show()
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection="3d")
    # # ax.scatter(xm, ym, um, label="Original: u=u(x,y)")
    # # ax.scatter(freq_xm, freq_ym, F_um, label="Fourior: F_u=abs_u(k_x,k_y)", color="red")
    # ax.scatter(freq_xm, freq_ym, abs_um, label="Fourior: abs_u=abs_u(k_x,k_y)")
    # ax.scatter(freq_xm, freq_ym, ang_um, label="Fourior: ang_u=abs_u(k_x,k_y)")
    # plt.legend()
    # plt.show()
    # plt.close()
    # exit(0)



    # ##examples
    # def func1(x, p):
    #     return p[0]*x*x + p[1]*x + p[2]
    # def func1r(p, x, y):
    #     return func1(x, p) - y
    # def func1_log(x, p):
    #     return np.log10(func1(x, p))
    # def func1r_log(p, x, y):
    #     return np.log10(func1(x, p)) - y
    
    # x = np.linspace(0.1, 10.0, 100)
    # y = 0.5*x*x +10. + 0.2*(np.random.random(100)-0.5)
    # y_log = np.log10(y)
    # print(x, y, y_log)

    # res = spopt.leastsq(func1r, x0=(0.4, 0.1, 0.), args=(x, y))
    # print(res[0])

    # res = spopt.leastsq(func1r_log, x0=(0.4, 0.1, 0.), args=(x, y_log))
    # print(res[0])

    # CFMM = fgw.Minimize_fit()
    # res = CFMM.leastsq_residual_log(func1_log, x0=(0.4, 0.1, 0.), args=(x, y_log))
    # print(res[0])

    # ##other
    # #采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
    # x=np.linspace(0,1,1400) 
    
    # #设置需要采样的信号，频率分量有180，390和600
    # y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)
    
    # yy=fft(y)      #快速傅里叶变换
    # yreal = yy.real    # 获取实数部分
    # yimag = yy.imag    # 获取虚数部分
    
    # yf=abs(fft(y))    # 取绝对值
    # yf1=abs(fft(y))/len(x)   #归一化处理
    # yf2 = yf1[range(int(len(x)/2))] #由于对称性，只取一半区间
    
    # xf = np.arange(len(y))  # 频率
    # xf1 = xf
    # xf2 = xf[range(int(len(x)/2))] #取一半区间
    
    # plt.subplot(221)
    # plt.plot(x[0:50],y[0:50])
    # plt.title('Original wave')
    
    # plt.subplot(222)
    # plt.plot(xf,yf,'r')
    # plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B') #注意这里的颜色可以查询颜色代码表
    
    # plt.subplot(223)
    # plt.plot(xf1,yf1,'g')
    # plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')
    
    # plt.subplot(224)
    # plt.plot(xf2,yf2,'b')
    # plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')
    # plt.show()

    # Fs = 150.0;     # sampling rate采样率
    # Ts = 1.0/Fs;    # sampling interval 采样区间
    # t = np.arange(0,1,Ts)  # time vector,这里Ts也是步长
    
    # ff = 25;     # frequency of the signal
    # y = np.sin(2*np.pi*ff*t)
    
    # n = len(y)     # length of the signal
    # k = np.arange(n)
    # T = n/Fs
    # frq = k/T     # two sides frequency range
    # frq1 = frq[range(int(n/2))] # one side frequency range
    
    # YY = np.fft.fft(y)   # 未归一化
    # Y = np.fft.fft(y)/n   # fft computing and normalization 归一化
    # Y1 = Y[range(int(n/2))]
    
    # fig, ax = plt.subplots(4, 1)
    
    # ax[0].plot(t,y)
    # ax[0].set_xlabel('Time')
    # ax[0].set_ylabel('Amplitude')
    
    # ax[1].plot(frq,abs(YY),'r') # plotting the spectrum
    # ax[1].set_xlabel('Freq (Hz)')
    # ax[1].set_ylabel('|Y(freq)|')
    
    # ax[2].plot(frq,abs(Y)) # plotting the spectrum
    # ax[2].set_xlabel('Freq (Hz)')
    # ax[2].set_ylabel('|Y(freq)|')
    
    # ax[3].plot(frq1,abs(Y1)) # plotting the spectrum
    # ax[3].set_xlabel('Freq (Hz)')
    # ax[3].set_ylabel('|Y(freq)|')
    
    # plt.show()
    # exit(0)
