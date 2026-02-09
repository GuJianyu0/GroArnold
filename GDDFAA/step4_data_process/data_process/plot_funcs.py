#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import analysis_data_distribution as ads
import galaxy_models as gm



#### sklearn symbolic regresstion
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from gplearn.genetic import SymbolicRegressor

def sksr(xdata, ydata):
    # Assuming you have your data stored in arrays h_values and log_f_values
    h_values, log_f_values = xdata, ydata

    # Concatenate x_values and s_values into a single input array
    X = h_values.reshape(-1,1)
    # s_values = 1./h_values #cannot use parameterized sub-function now
    # X = np.column_stack((h_values, s_values))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, log_f_values, test_size=0.2, random_state=42)

    # Define a set of mathematical functions or primitives
    # function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos', 'exp']
    function_set = ['add', 'sub', 'mul', 'div', 'log']

    # Create a symbolic regressor with the specified function set
    est_gp = SymbolicRegressor(population_size=5000, generations=20, stopping_criteria=0.01,
                            p_crossover=0.7, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, p_point_mutation=0.1,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient=0.01, random_state=0,
                            function_set=function_set)
    
    # Fit the regressor to the training data
    est_gp.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = est_gp.predict(X_test)

    # Evaluate the performance
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Print the symbolic expression of the best fit
    print("Symbolic Expression:", est_gp._program)

    # import graphviz
    # dot_data = est_gp._program.export_graphviz()
    # graph = graphviz.Source(dot_data)
    # print(graph)
    return est_gp._program



#### main
if __name__ == '__main__':

    ## data
    data_cal = np.loadtxt("tmp_data/2.txt")
    hdata = data_cal[:,0]
    fdata = data_cal[:,1]

    # xmin = 1e1
    xmin = np.min(hdata)
    # xmax = 1e6
    xmax = np.max(hdata)
    # h = np.linspace(xmin, xmax, 100)
    h = np.geomspace(xmin, xmax, 100)
    h_log = np.log10(h)

    nbins = 200
    hbins = np.linspace(np.min(hdata), np.max(hdata), nbins) #?? better for xdata distribution but need fit
    # hbins = np.geomspace(np.min(hdata), np.max(hdata), nbins)
    cout_bin, hdata_bin = np.histogram(hdata, hbins)
    fdata_bin = None
    # plt.figure()
    # plt.step(hbins[:-1], cout_bin, where='post')
    # plt.show()
    # plt.close()
    # ads.DEBUG_PRINT_V(0, hbins, hdata_bin)

    # # hbd, fbd = ads.get_mean_ydata_by_xdata_1d_bin(hdata, fdata, hdata_bin)
    # hbd, fbd, xsbs, nxss = ads.split_various_bin_1d_percentile(hdata, fdata, [0., 0.99, 1.], [100, 10]) #curve
    # # hbd, fbd, xsbs, nxss = ads.split_various_bin_1d_percentile(hdata, fdata, [0., 0.99, 1.], [1000, None]) #normal
    hbd, fbd, xsbs, nxss = ads.split_various_bin_1d_percentile(hdata, fdata, [0., 0.01, 0.99, 1.], [10, 100, 10]) #curve
    # plt.figure()
    # plt.scatter(hdata, fdata, s=1.)
    # plt.scatter(hbd, fbd, s=0.5)
    # plt.show()
    # plt.close()
    # ads.DEBUG_PRINT_V(0, xsbs, nxss, len(hbd))

    hbd_log = np.log10(hbd)
    fbd_notlog = np.exp(fbd)



    ### symbolic
    # ep = sksr(hbd, fbd)
    # # ep = sksr(hbd_log, fbd)
    # ads.DEBUG_PRINT_V(0, ep)

    # import lagueree_expand as lag
    # # lagcoef = lag.get_coef_of_laguerre_expansion_1d(hbd, fbd, n_terms=7, is_plot=False)
    # lagcoef = lag.get_coef_of_laguerre_expansion_1d(hbd[:-5], fbd[:-5], n_terms=5, is_plot=False)
    # f = lag.get_lagval(hbd, lagcoef)
    # plt.figure()
    # # plt.scatter(hbd, fbd, color='red', label='Data Points')
    # # plt.plot(hbd, f, color='blue', label='Laguerre fit (xdata)')
    # plt.scatter(hbd[:-5], fbd[:-5], color='red', label='Data Points')
    # plt.plot(hbd[:-5], f[:-5], color='blue', label='Laguerre fit (xdata[:-5])')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Laguerre Expansion of Data Points')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # plt.close()
    # ads.DEBUG_PRINT_V(0, lagcoef)



    #### calculate ymodel
    # # J0, Jc, Jt, p1, p2, p3 = 1., 1., 0., 1., 1., 1.
    # # J0, Jc, Jt, p1, p2, p3 = 8000., 10., 0., 0.5, 1., 0.99 #mannually
    # # J0, Jc, Jt, p1, p2, p3 = \
    # #     1.02606956e+04, 6.92760248e+01, 0., \
    # #     1.98773999e-01, 1.00000000e+01, 2.25088849e+00 #MTPL mpfit
    # J0, Jc, Jt, p1, p2, p3 = \
    #     1.07e4, 2.7e3, 0., 0.569, 2.53, 0.942 #MTPL curvefit
    # f = gm.fh_MTPL_log10(h, J0, Jc, Jt, p1, p2, p3)
    
    # J0, p1, p2, C1 = 1., 1., 3., 1.e4 #MDPT
    # f = gm.fh_MDPL_log10(h, J0, p1, p2, C1)

    # J0, p1, p2 = 8000., 1., 2. #DPL
    # f = gm.fh_DPL_log10(h, J0, p1, p2)

    # J0, J1, J2, J3, J4, p1, p2, p3, p4 = 2e3, 9e2, 2e4, 4.5e4, 6e4, 2.1, 0.25, 1.5, 0.18 #TPL1
    # f = gm.fh_TPL1_log10(h, J0, J1, J2, J3, J4, p1, p2, p3, p4)
    # J0, J1, J2, J3, p1, p2, p3 = 2e3, 5e2, 2e3, 2e4, 1.5, 0.2, 1.7 #TPL1
    # J0, J1, J2, J3, p1, p2, p3 = 2e3, 9e2, 6e4, 9e4, 2.1, 1.8, 3.0 #TPL1
    # J0, J1, J2, J3, p1, p2, p3 = 2e3, 9e2, 2e4, 4.5e4, 2.1, 0.25, 1.5 #TPL1 #latest
    # J0, J1, J2, J3, p1, p2, p3 = \
    #     9.82004101e+02, 2.40234371e+02, 9.03267757e+03, 3.07763951e+04, \
    #     1.98914935e+00, 5.00000000e-01, 1.78578679e+00 #mpfit1
    # J0, J1, J2, J3, p1, p2, p3 = \
    #     1.49038322e+03, 1.11689037e+03, 3.70986027e+04, 3.47622578e+04, \
    #     2.88084618e+00, 1.05813842e+01, 7.75357167e+00
    # f = gm.fh_TPL1_log10(h, J0, J1, J2, J3, p1, p2, p3)

    # J0, J1, J2, p1, p2, k21, k22 = 1700, 5.0e2, 2.1e4, 1.5, 2.05, 2.1, 1.5 #TPL
    # J0, J1, J2, p1, p2, k21, k22 = 6.0e3, 5.0e2, 2.1e4, 1.1, 1.90, 2.1, 1.5 #TPL
    # J0, J1, J2, p1, p2, k21, k22 = 3.0e4, 2.0e3, 5.1e4, 1.1, 1.90, 0.1, 0.15 #TPL #bad #latest
    # J0, J1, J2, p1, p2, k21, k22 = \
    #     4.37842598e+03, 1.10259569e+03, 1.16910242e+05, 2.92628384e+00, \
    #     1.24905654e+00, 4.82531784e-01, 2.77228084e-01 #mpfit
    # J0, J1, J2, p1, p2, k21, k22 = \
    #     4.02165602e+04, 2.00000000e+04, 6.06867397e+04, 1.10000000e-02, \
    #     2.74932409e+00, 6.0e-01, 5.18155268e-01
    # J0, J1, J2, p1, p2, k21, k22 = \
    #     9.0e+04, 4.16535972e+07, 7.66709069e+06, 1.73e+03, \
    #     4.43936491e-01, 2.99213499e-03, 1.51905893e-03
    # f = gm.fh_TPL_log10(h, J0, J1, J2, p1, p2, k21, k22)

    J0, J1, J2, J3, p1, p2, p3, k21, k22, k31, k32, k33= \
        1.02165602e+04, 2.00000000e+04, 6.06867397e+04, 8.06867397e+05, \
        1.1e-01, 2.75e+00, 1.09e-01, \
        5.0e-1, 5.e-1, 5.e-1, 5.e-1, 5.e-1
    # E0:
    # 6.14924545e-01  6.67956258e+04  3.44854228e+04  1.48668126e+06
    # 2.47531140e+00  5.70636050e-01 -4.46157694e+01  3.26900331e-01
    # 1.74106818e-01 -4.38730318e-01  4.99872027e-01  5.07635510e-01
    # E2:
    # 4.84930339e+02  1.72055631e-11  2.62703040e+04  1.88254743e+06
    # -5.90561066e-01  5.63513269e-01 -2.94804368e+01  7.29923013e-01
    # 3.74915762e-01 -2.39945033e+00  1.16327954e+00  3.50070162e-01
    f = gm.fh_SPL_log10(h, J0, J1, J2, J3, p1, p2, p3, k21, k22, k31, k32, k33)

    # # J0, Jc, Jt, p1, p2, p3, p4, k1, ym = 1., 1., 1., \
    # #     0., 5./3, 2.9, 5./6, 1., 0. #P18
    # J0, Jc, Jt, p1, p2, p3, p4, k1, ym = 1., 1., 1., \
    #     0., 5./3, 2.9, 5./6, 1., -25.
    # f = gm.fh_P18_log10(h, J0, Jc, Jt, p1, p2, p3, p4, k1, ym)

    # J0, p1 = 0., 0.
    # # f = gm.fh_sr1(h)
    # # f = gm.fh_loglog_sr1(h_log)
    # f = gm.fh_sr2(h)



    #### plot
    ## plot one
    # print("f: ", f)
    plt.figure()
    plt.scatter(hbd, fbd, label="data_cal_meanbin", s=16.)
    # plt.scatter(hbd, fbd_notlog, label="data_cal_meanbin")
    plt.plot(h, f, label="%d"%(0), lw=2., color="orange")
    # plt.scatter(h, f, label="%d"%(0), s=8.)
    # plt.xlim(xmin, xmax/10)
    plt.xscale("log")
    # plt.yscale("log")
    plt.legend()
    plt.show()

    # ##plot many
    # nl = 3
    # # par_ = [1., 10., 20.] #p2
    # par_ = [1, 1, 1] #p2
    # f_ = list(range(nl))
    # for i in np.arange(nl):
    #     f_[i] = fh_MTPL_log10(h, J0, Jc, Jt, p1, par_[i], p3)
    #     print(f_[i])

    # plt.figure()
    # for i in np.arange(nl):
    #     plt.scatter(h, f_[i], label="%d"%(i))
    # plt.xscale("log")
    # # plt.yscale("log")
    # plt.show()
