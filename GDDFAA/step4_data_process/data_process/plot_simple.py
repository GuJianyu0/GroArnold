#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys
import analysis_data_distribution as ads



def plot_normalized_df(filename, suffix="default", geom_q=np.geomspace(0.01, 0.99, 50)):
    """
    Plot the normalized distribution function (DF) based on percentile data.

    Args:
        filename (str): Path to the file containing the (N, 2)-array data. 
                        The first column is the percentiles, the second column is the values.
        suffix (str): Suffix for the saved plot filename.
        geom_q (array): Geometric sequence for percentile values (0 to 1, exclusive).
                        Defaults to np.geomspace(0.01, 0.99, 50).
    """
    # Load the percentile data (N, 2) from the file
    data = np.loadtxt(filename)
    q = data[:, 0]  # Percentiles
    values = data[:, 1]  # Corresponding values
    
    # Normalize the distribution function
    df = np.diff(values) / np.diff(q)  # Density is the difference in values divided by difference in q
    df = df / np.sum(df)  # Normalize the DF to 1

    # Midpoints of q for plotting (as geometric bin centers)
    q_mid = (q[:-1] + q[1:]) / 2.0

    # Plotting settings
    fontsize = 40
    pointsize = 3.2
    figsize = (20, 15)  # For 3x3 layout
    dpi = 400

    plt.figure(figsize=figsize, dpi=dpi)

    # Plot the normalized DF
    plt.plot(q_mid, df, label="Normalized DF", marker="o", markersize=pointsize)

    # Set logarithmic scale for x and y axes
    plt.xscale("log")
    plt.yscale("log")

    # Title and labels
    plt.title(r"Histogram of main diffusion coefficient of each particle", fontsize=fontsize)
    plt.xlabel(r"Diffusion, $D_\mathrm{position, main}$ ($\mathrm{(km/s)^3/kpc}$)", fontsize=fontsize)
    plt.ylabel(r"Distribution, $f$ ($\mathrm{kpc/(km/s)^3}$)", fontsize=fontsize)

    # Legend and ticks
    plt.legend(fontsize=fontsize * 0.6, loc=0)
    plt.tick_params(which='major', length=0, labelsize=fontsize * 0.8)

    # Layout and save the plot
    plt.tight_layout()
    plt.savefig(f"../data/small/diffur_percentile_{suffix}.png", format="png", bbox_inches='tight')

    print(f"Saved fig of diffur_percentile.")
    return 0



if __name__ == '__main__':

    ## 20241224
    filename = sys.argv[1]
    suffix = sys.argv[2]
    # filename = "./proj2/MFRT/diffu_r_simple_each/simulated/galaxy_general.g1.txt"
    ads.DEBUG_PRINT_V(1, filename, "filename to plot")
    data = np.loadtxt(filename)
    positions = data[:, 0:3]
    velocities = data[:, 3:6]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=0.1, alpha=0.6)
    ax.set_title("positions")
    plt.savefig("../data/small/plot_simple_positions_{}.png".format(suffix), format="png", bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(velocities[:, 0], velocities[:, 1], velocities[:, 2], s=0.1, alpha=0.6)
    ax.set_title("velocities")
    plt.savefig("../data/small/plot_simple_velocities_{}.png".format(suffix), format="png", bbox_inches="tight")
    plt.close()
    


    # ## 20221201
    # filename = "~/workroom/0prog/GroArnold_framework/GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general_DPL_less_20221107/aa/snapshot_40_lmn_foci_Pot.txt"
    # data = np.loadtxt(filename)
    # ycut = data[:,4]
    # a2 = -data[:,2]
    # b2 = -data[:,1]
    # plt.scatter(ycut, a2, label="ycut~a2")
    # plt.scatter(ycut, b2, label="ycut~b2")
    # plt.legend()
    # plt.show()
