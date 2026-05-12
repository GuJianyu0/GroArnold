#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import numpy as np

def sort_xy_by_x(x, y):
    # sorted_indices = np.argsort(XY[:,0], axis=0)
    sorted_indices = np.argsort(x)
    sorted_x_ = x[sorted_indices]
    sorted_y_ = y[sorted_indices]
    return sorted_x_, sorted_y_

def integrate_samples_trapezoid_1d(x_seq, y_seq, x_down=-np.inf, x_up=np.inf):
    # x_seq = xy_seq[:,0]
    # y_seq = xy_seq[:,1]
    x, y = sort_xy_by_x(x_seq, y_seq)
    N = len(x)
    I = 0.
    for i in np.arange(N-1):
    # for i in np.arange(N-2):
        # print(x[i], y[i])
        if x[i]>=x_down and x[i+1]<=x_up:
            I += (x[i+1]-x[i])*(y[i+1]+y[i])/2
    return I
