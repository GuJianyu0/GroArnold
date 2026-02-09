#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
# from scipy.special import genlaguerre
from numpy.polynomial.laguerre import lagfit, lagval
# x = np.linspace(0, 10)
# err = np.random.randn(len(x))/10
# y = lagval(x, [1, 2, 3]) + err
# lagfit(x, y, 2) #array([ 0.96971004,  2.00193749,  3.00288744]) # may vary
import matplotlib.pyplot as plt

def get_lagval(x, coef):
    return lagval(x, coef)

def get_coef_of_laguerre_expansion_1d(x_data, y_data, n_terms=3, is_plot=False):

    coefficients = lagfit(x_data, y_data, n_terms-1)

    if is_plot:
        # Generate the Laguerre expansion for a range of x values
        x_range = np.linspace(np.min(x_data), np.max(x_data), 100)  # Define the range of x values for evaluation
        y_range = lagval(x_range, coefficients)

        # Plot the original data points and the Laguerre expansion

        plt.scatter(x_data, y_data, color='red', label='Data Points')
        plt.plot(x_range, y_range, color='blue', label='Laguerre fit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Laguerre Expansion of Data Points')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print the computed coefficients
        print("Expansion coefficients:", coefficients)

    return coefficients
