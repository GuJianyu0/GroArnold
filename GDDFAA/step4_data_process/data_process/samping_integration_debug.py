#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

def continuous_integral(f, g, x_min, x_max):
    from scipy.integrate import quad
    return quad(lambda x: f(x) * g(x), x_min, x_max)[0]

def sampled_integral_know_DF(g, N_samples, f, x_min, x_max):
    samples = np.random.uniform(x_min, x_max, N_samples)
    # f_values = f(samples)
    # weights = f_values / f_values.sum()  # Normalize weights
    # return np.sum(weights * g(samples)) * (x_max - x_min)
    return np.sum(f(samples)*g(samples)) * (x_max-x_min)/N_samples

def sampled_integral_know_samples(g, samples):
    weights = N_0/len(samples)
    return weights * np.sum(g(samples))

if __name__ == "__main__":

    N_0 = 20.
    # N_0 = 0.02
    x_0 = 1.
    x_epsilon = 1e-4
    x_min, x_max = -5., 5.
    loc = 0.
    scale = 1.
    f = lambda x: N_0 * np.exp(-x**2 / 2.) / np.sqrt(2. * np.pi)  # Gaussian density
    g = lambda x: 1. / (x_epsilon + (x-x_0)**2)  # Concerned variable

    I_continuous = continuous_integral(f, g, x_min, x_max)
    print(f"Continuous Integral: {I_continuous}")
    # N_samples_list = [1000, 10000, 100000]
    N_samples_list = [1000, 10000, 100000, 1000000]
    # N_samples_list = [1000, 10000, 100000, 1000000, 10000000, 100000000]
    for (i, N_samples) in enumerate(N_samples_list):
        I_sampled_weight = sampled_integral_know_DF(g, N_samples, f, x_min, x_max)
        print(f"Sampled Integral weight: {I_sampled_weight}, N_samples = {N_samples}")
        
        f_samples = np.random.normal(loc=0., scale=1., size=(N_samples))
        I_sampled_f = sampled_integral_know_samples(g, f_samples)
        print(f"Sampled Integral f: {I_sampled_f}, N_samples = {N_samples}")
