#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
# import random
# from numpy.lib.function_base import average
import scipy.stats as spstat

import analysis_data_distribution as add
import galaxy_models as gm
# import fit_rho_fJ as fff

def var_x(A,x):
    if not len(A.shape)==1:
        print("Unsupported length!")
        return 0.
    # m = np.mean(A)
    v = sum((A-x)**2)
    return v

if __name__ == '__main__':

    ##unit convertion
    G_international = 6.67e-11
    G_astronamical = 4.3009e4
    G_dsMOND = 4.493531540994E-12

    convert_potential_km_per_s_to_units_GeV = 3.797641792E-08
    convert_potential2 = 1./convert_potential_km_per_s_to_units_GeV
    print(convert_potential2)



    # ##chi2
    # ps1 = spstat.chi2.cdf(1*1,1)
    # ps2 = spstat.chi2.cdf(2*2,1)
    # ps3 = spstat.chi2.cdf(3*3,1)
    # print(ps1, ps2, ps3)
    # cs1 = spstat.chi2.ppf(ps1,1)
    # cs2 = spstat.chi2.ppf(ps2,1)
    # cs3 = spstat.chi2.ppf(ps3,1)
    # print(cs1, cs2, cs3)



    # ##ks test
    # N_sample = 15
    # x = np.random.randn(N_sample)
    # m = np.mean(x)
    # s = np.std(x)
    # x1 = x+0.5#*2.#+1.
    # print(x, m, s)
    # kst = spstat.kstest(x1, "norm",args=(m, s))
    # print(kst)



    # ##population and its standard deviation
    # N_total = 1000
    # A = np.random.randn(N_total) #the statistic population
    # m = np.mean(A) #the mean value of the population
    # sigma = np.std(A) #the standard deviation of the population
    
    # # ##the set of samples and its standard error set
    # # N_samplings = 500
    # # N_sampleElements = 100
    # # Sample_set = list(range(N_samplings)) #the set of samples
    # # m_sample_set = list(range(N_samplings)) #the set of mean values of the set of samples
    # # sigma_sample_set = list(range(N_samplings)) #the set of mean values of the set of samples
    # # for i in np.arange(N_samplings):
    # #     # print(i)
    # #     Sample_set[i] = random.sample(list(A), N_sampleElements) #to sample the population for many times, then we get each sample, here we sample randomly
    # #     m_sample_set[i] = np.mean(Sample_set[i]) #the mean value of each sample
    # #     sigma_sample_set[i] = np.std(Sample_set[i]) #standard error of the mean of the samples set
    # # m_sample_set = np.array(m_sample_set) #to convert type of data in python
    # # sigma_sample_set = np.array(sigma_sample_set) #to convert type of data in python
    # # # sigma_sample_set = np.std(m_samples)
    # # sigma_sample_set_mean = np.sum(sigma_sample_set)/N_samplings #the mean of standard errors of all samples

    # # ##relation of standard deviation of population and standard error of a kind of statistic sample, here the the statistic sample is arithmetic mean
    # # something = sigma/np.sqrt(N_sampleElements*1.) #the relation of the two kind of "standard"
    # # print(sigma, something, sigma_sample_set_mean)

    # ##sample one time
    # N_sampleElements = 100
    # Sample = random.sample(list(A), N_sampleElements)
    # m_sample = np.mean(Sample)
    # var_sample = np.var(Sample)
    # # print(var_sample)
    # # v = list(range(N_sampleElements))
    # # for i in Sample:
    # #     v = sum((A-i)**2)
    # # sigma_sample = np.std(Sample)
    # var__mean_sample = np.sum(abs(A-m_sample)**2) /(len(A)-1)
    # # var__mean_sample = np.mean(np.abs(A-m_sample)**2)
    # # print(np.sum(abs(A-m_sample)**2), len(A)-1, len(np.abs(A-m_sample)**2))
    # sigma__mean_sample = np.sqrt(var__mean_sample)
    # something = sigma/np.sqrt(N_sampleElements*1.)
    # print(sigma, something, sigma__mean_sample)


    
    # a = np.array([[2,3],[4,2],[1,1],[3,4]])*1.
    # b = np.ones(4)
    # c = add.merge_array_by_hstack([a,b])
    # print(a,b,c)
    # d, a_ = add.neighbourAverage_bin(c,1,2)
    # print(d, a_)



    # # a = 124.239582905290682
    # # b = 1.95836829048239620
    # # c = a*b
    # # print(c)

    # # d = np.array([1,2,3,4,6])
    # # e = d*0.5
    # # print(e)

    # # score
    # Chinese = 100.
    # Math = 99.
    # English = 0.
    # avrg = (Chinese+Math+English)/3
    # print(avrg)



    # # f_ds_1 = np.array([-77680.1, -37493.1, 104732])
    # # f_gg_1 = np.array([1689.932251, 718.906433, -2183.781006])
    # # f_ds_1 = np.array([-54620.1, -155903, -26480.9])
    # # f_gg_1 = np.array([980.379761, 2268.240234, 362.070038])
    # f_ds_1 = np.array([-131590, 96119.8, -53574.9])
    # f_gg_1 = np.array([1633.221191, -1239.839233, 638.134277])
    # add.DEBUG_PRINT_V(1, f_ds_1/f_gg_1, "ds/gg")



    # ## used
    # l = 10000
    # k = 10.
    # b = 2.
    # x = np.arange(l)
    # y  = k*x/l+b*np.random.random(l)
    # ym = k*x/l+b*0.5
    # plt.scatter(x,y,label="samples",s=0.5)
    # plt.plot(x,ym,label="model")
    # fig_tmp = plt.gcf()
    # fig_tmp.savefig("savefig/0a.png", dpi=200)
    # plt.show()
    # fig_tmp.savefig("./data_process/0a.png", dpi=300)

    #     A = np.array([[1,3],[2,4]])
    #     B = np.linalg.inv(A) #inv of mat
    #     print(B)

    # xdata = np.random.random((8,3))+10.
    # print(xdata)
    # m = list(range(3))
    # pmc = m
    # for i in range(3):
    #     m[i] = np.percentile(xdata[:, i], [16, 50, 84])
    #     print(m[i], np.diff(m[i]))
    #     pmc[i] = m[i][1]
    # print(pmc)
