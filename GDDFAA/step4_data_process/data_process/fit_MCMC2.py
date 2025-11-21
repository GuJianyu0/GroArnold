#!/usr/bin/env python
# -*- coding:utf-8 -*-

import emcee
import corner
def N_Scl_invChi2ln(x, mu,nu,kappa,sigma2):
    if x[1]<0:
        return -1e50
    else:
        return Scl_InvChi2ln(x[1],\
            nu,sigma2)+stats.norm.logpdf(x[0],\
            loc=mu, scale=x[1]/kappa)

n = np.float(len(y))
#sample mean and variance
y_bar = np.sum(y)/n
s2 = np.sum((y-y_bar)**2)/(n-1)
#parameters of the prior density
nu0 = 1.
sigma02 = 40**2
kappa0 = 50
mu0 = 3.
#parameters of the posterior density
mun = kappa0*mu0/(kappa0+n)+n/(kappa0+n)*y_bar
kappan = kappa0+n
nun = nu0+n
nunsigman2 = nu0*sigma02+(n-1)*s2+kappa0*n*\
(y_bar-mu0)**2/(kappa0+n)
sigman2 = nunsigman2/nun
ndim = 2
nwalkers = 50
p0=np.zeros((nwalkers,ndim))
p0[:,0] = np.random.rand(nwalkers)*30.-15.
p0[:,1] = np.random.rand(nwalkers)*50.+10.
sampler = emcee.EnsembleSampler(nwalkers, \
    ndim, N_Scl_invChi2ln, \
    args=[mun,nun,kappan,sigman2])
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
sampler.run_mcmc(pos, 1000)
samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
fig = corner.corner(samples,\
    labels=[r"$\mu$", r"$\sigma^2$"],\
    quantiles=[0.16, 0.5, 0.84],\
    show_titles=True, \
    title_kwargs={"fontsize": 12})