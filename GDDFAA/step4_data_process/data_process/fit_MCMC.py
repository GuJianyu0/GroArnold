#!/usr/bin/env python
# -*- coding:utf-8 -*-

# https://www.cnblogs.com/yymn/p/4716107.html

import numpy as np
import matplotlib.pyplot as plt
import emcee
 
np.random.seed(42)
theta_true = (25, 0.5)
xdata = 100 * np.random.random(20)
ydata = theta_true[0] + theta_true[1] * xdata

# add scatter to points
xdata = np.random.normal(xdata, 10)
ydata = np.random.normal(ydata, 10)



# Create some convenience routines for plotting
# 让我们做一些辅助工作来可视化数据

def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]

    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)

def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
    """Plot traces and contours"""
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace[0], trace[1], ',k', alpha=0.1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')

def plot_MCMC_model(ax, xdata, ydata, trace):
    """Plot the linear model and 2sigma contours"""
    ax.plot(xdata, ydata, 'ok')

    alpha, beta = trace[:2]
    xfit = np.linspace(-20, 120, 10)
    yfit = alpha[:, None] + beta[:, None] * xfit #the most expanded arrays
    mu = yfit.mean(0) #the 0 - indices
    print("shapes: ")
    print(alpha.shape)
    print(alpha[:, None].shape)
    print(beta.shape)
    print(beta[:, None].shape)
    print(xfit.shape)
    print(yfit.shape)
    print(mu.shape)
    sig = 2 * yfit.std(0)

    ax.plot(xfit, mu, '-k')
    ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def plot_MCMC_results(xdata, ydata, trace, colors='k'):
    """Plot both the trace and the model together"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_MCMC_trace(ax[0], xdata, ydata, trace, True, colors=colors)
    plot_MCMC_model(ax[1], xdata, ydata, trace)



# Define our posterior using Python functions
# for clarity, I've separated-out the prior and likelihood
# but this is not necessary. Note that emcee requires log-posterior

# 用Python函数定义后验，我把先验和似然估计分开写了，其实没必要，主要是显得更简洁。
# 注意emcee需要对数后验证

import pdb
def log_prior(theta):
    alpha, beta, sigma = theta
    if sigma < 0:
        return -np.inf  # log(0)
    else:
        return -1.5 * np.log(1 + beta ** 2) - np.log(sigma)

def log_likelihood(theta, x, y):
    alpha, beta, sigma = theta
    # print("shapes likelihood: ")
    # print(alpha)
    # print(beta)
    # print(sigma)
    # print("sadasasdffsdgfgr")
    # pdb.set_trace()
    y_model = alpha + beta * x
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_posterior(theta, x, y):
    return log_prior(theta) + log_likelihood(theta, x, y)



# Here we'll set up the computation. emcee combines multiple "walkers",
# each of which is its own MCMC chain. The number of trace results will
# be nwalkers * nsteps

# 设置计算参数。emcee组合了多个"walkers"，每个都有自己的MCMC链。
# 跟踪结果的数量为 nwalkers * nsteps

ndim = 3  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nburn = 1000  # "burn-in" period to let chains stabilize
nsteps = 2000  # number of MCMC steps to take

# set theta near the maximum likelihood, with 
np.random.seed(0)
starting_guesses = np.random.random((nwalkers, ndim))



# Here's the function call where all the work happens:
# we'll time it using IPython's %time magic

# 这就是所有调用内容。
# 这里用IPythn的%time方法计时。

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xdata, ydata])
sampler.run_mcmc(starting_guesses, nsteps)
print("done")

# sampler.chain is of shape (nwalkers, nsteps, ndim)
# we'll throw-out the burn-in points and reshape:

# sampler.chain返回数组维度为(nwalkers, nsteps, ndim)

sampler.chain
print(sampler.chain.shape)
emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T #-1 is to automacally calculation the rest of shape
print(emcee_trace.shape)
print(xdata.shape)
plot_MCMC_results(xdata, ydata, emcee_trace)
plt.show()