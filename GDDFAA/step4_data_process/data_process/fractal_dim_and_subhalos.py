#!/usr/bin/env python
# -*- coding:utf-8 -*-
#In[] modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
# from scipy.spatial import KDTree
# from Corrfunc.theory.DD import DD

import analysis_data_distribution as ads
import KDTree_python as kdtp

#In[] settings

Dim = 3
col_actions = 78
col_frequencies = col_actions+7

# galaxy_name = sys.argv[1]
# snapshot_ID = int(sys.argv[2])
snapshot_ID = 10 #fixed

#In[] functions
# Function to recognize subhalos using DBSCAN
def recognize_subhalos_dbscan(positions, distance_threshold, min_particles):
    """
    Recognize subhalos using DBSCAN clustering algorithm.
    
    Parameters:
    positions: array of shape (N, 3) representing the positions of N particles.
    distance_threshold: Maximum distance between particles to be considered part of the same subhalo.
    min_particles: Minimum number of particles in a neighborhood to form a subhalo.
    
    Returns:
    labels: Array of subhalo labels for each particle (-1 for noise points).
    N_sub: Number of subhalos detected.
    """
    # DBSCAN clustering based on particle positions
    db = DBSCAN(eps=distance_threshold, min_samples=min_particles).fit(positions)
    labels = db.labels_
    
    # The number of subhalos is the number of unique labels (excluding noise, which is labeled as -1)
    N_sub = len(set(labels)) - (1 if -1 in labels else 0)
    ptcs_in_sub = np.where(labels!=-1)[0] #not noise particle
    N_ptcs_in_sub = len(ptcs_in_sub)
    
    return labels, N_sub, ptcs_in_sub, N_ptcs_in_sub

# Sample function to estimate the two-point correlation function
def correlation_function_distmat(positions, r_max=100.0, num_bins=50, is_enable_normalization=True):
    """
    Compute the two-point correlation function for particle positions.
    Need much memory.
    
    Parameters:
    positions: array of shape (N, 3) representing the positions of N particles.
    num_bins: Number of bins for the distance histogram.
    
    Returns:
    r_bins: The bin centers for the distances.
    xi: The estimated two-point correlation function values.
    """
    # Compute pairwise distances between particles
    from scipy.spatial import distance_matrix
    dist_matrix = distance_matrix(positions, positions)
    distances = dist_matrix[np.triu_indices(len(positions), k=1)]
    # ads.DEBUG_PRINT_V(1, len(distances))
    distances = distances[distances<r_max] # mask by max distance
    N = len(positions)
    # ads.DEBUG_PRINT_V(0, len(distances))
    
    # Create a histogram of distances
    hist, bin_edges = np.histogram(distances, bins=num_bins, density=True)
    r_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if not is_enable_normalization:
        # Estimate two-point correlation function (xi ~ 1/r^gamma)
        xi = hist / np.mean(hist) - 1
    else:
        # Normalize the histogram
        # Calculate the volume of each shell
        shell_volumes = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        # Number density
        n = N / ((4/3) * np.pi * (r_max)**3)
        # Expected number of pairs in each shell for a random distribution
        expected = shell_volumes * n
        # Compute xi
        xi = hist / expected - 1
    
    return r_bins, xi

# Function to compute the two-point correlation function (TPCF)
def correlation_function_scipyckdtree(positions, r_max=50.0, num_bins=50, is_enable_normalization=True):
    """
    Compute the two-point correlation function for particle positions using cKDTree.
    Cost much time.

    Using module in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.count_neighbors.html

    Parameters:
    positions: array of shape (N, 3) representing the positions of N particles.
    num_bins: Number of bins for the distance histogram.
    r_max: Maximum distance to consider for correlation function.
    
    Returns:
    r_bins: The bin centers for the distances.
    xi: The estimated two-point correlation function values.
    """
    N = len(positions)
    
    # Build a cKDTree from the particle positions
    kdtree = cKDTree(positions)
    
    # Create bins for distances
    bin_edges = np.linspace(0, r_max, num_bins + 1)
    r_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Count the number of neighbors within each bin for each particle
    counts = np.zeros(num_bins)
    for i in range(num_bins):
        r_min = bin_edges[i]
        r_max_bin = bin_edges[i + 1]
        # Count neighbors in the range (r_min, r_max_bin)
        count_min = kdtree.count_neighbors(kdtree, r_min)
        count_max = kdtree.count_neighbors(kdtree, r_max_bin)
        # The number of pairs in this bin is the difference between the two counts
        counts[i] = count_max - count_min
    
    if not is_enable_normalization:
        # Compute xi (TPCF)
        xi = counts / np.mean(counts) - 1 #??
    else:
        # Normalize the counts
        # Shell volumes for each bin
        shell_volumes = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        
        # Number density
        n = N / ((4/3) * np.pi * (r_max)**3)
        
        # Expected number of pairs in each shell for a random distribution
        expected_counts = shell_volumes * n
        
        # Compute xi (TPCF)
        xi = counts / expected_counts - 1
    
    return r_bins, xi

# Function to compute the two-point correlation function (TPCF)
def count_radius_scipyckdtree(positions, r_max=50.0, num_bins=50, is_enable_normalization=True):
    """
    Compute the two-point correlation function for particle positions using cKDTree.
    Cost much time.

    Using module in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.count_neighbors.html

    Parameters:
    positions: array of shape (N, 3) representing the positions of N particles.
    num_bins: Number of bins for the distance histogram.
    r_max: Maximum distance to consider for correlation function.
    
    Returns:
    r_bins: The bin centers for the distances.
    xi: The estimated two-point correlation function values.
    """
    N = len(positions)
    
    # Build a cKDTree from the particle positions
    kdtree = cKDTree(positions)
    
    # Create bins for distances
    bin_edges = np.linspace(0, r_max, num_bins + 1)
    r_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Count the number of neighbors within each bin for each particle
    counts = np.zeros(num_bins)
    for i in range(num_bins):
        r_min = bin_edges[i]
        r_max_bin = bin_edges[i + 1]
        # Count neighbors in the range (r_min, r_max_bin)
        count_min = kdtree.count_neighbors(kdtree, r_min)
        count_max = kdtree.count_neighbors(kdtree, r_max_bin)
        # The number of pairs in this bin is the difference between the two counts
        counts[i] = count_max - count_min
    
    if not is_enable_normalization:
        # Compute xi (TPCF)
        xi = counts / np.mean(counts) - 1 #??
    else:
        # Normalize the counts
        # Shell volumes for each bin
        shell_volumes = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        
        # Number density
        n = N / ((4/3) * np.pi * (r_max)**3)
        
        # Expected number of pairs in each shell for a random distribution
        expected_counts = shell_volumes * n
        
        # Compute xi (TPCF)
        xi = counts / expected_counts - 1
    
    return r_bins, xi

def fracal_dim_by_boxcounting(positions, order_2=10):
    from boxcounting import boxCount
    cdim = len(np.shape(positions))
    # ads.DEBUG_PRINT_V(0, cdim, positions[0], "box")
    boxCountObj = boxCount(positions)
    n, r, df = boxCountObj.calculateBoxCount()
    ads.DEBUG_PRINT_V(1, n, r, df, "box")
    D_embeding = np.shape(positions)[1]
    df_mean = np.mean(df[0:order_2])
    return n, r, df, D_embeding, df_mean

# Function to fit the correlation function to a power law
def powerlaw_1_log10(r, gamma, c):
    return np.log10(r) * (-gamma) + c

def powerlaw_1_translation_log10(r, gamma, rb, c):
    return np.log10(r+rb) * (-gamma) + c

def estimate_fractal_dimension_single(r_bins, xi, func, suffix="suffix"):
    """
    Estimate the fractal dimension D from particle positions.
    
    Parameters:
    r_bins: The bin centers for the distances.
    xi: The estimated two-point correlation function values.

    Returns:
    D: Estimated fractal dimension.
    """
    
    # Fit the two-point correlation function to a power law
    popt, pcov = curve_fit(func, r_bins, xi, maxfev=50000)
    gamma = popt[0]
    yfit = func(r_bins, *popt)
    print("popt: ", popt)
    
    # Fractal dimension is D = 3 - gamma
    D = 3. - gamma
    
    # Plot the correlation function and the power law fit
    plt.figure()
    plt.plot(r_bins, xi, label='Correlation Function')
    plt.plot(r_bins, yfit, label=f'Fit by powerlaw: alpha={gamma:.4e}')
    # plt.loglog(r_bins, xi, label='Correlation Function')
    # plt.loglog(r_bins, yfit, label=f'Fit by powerlaw: alpha={gamma:.2f}')
    plt.xlabel('Distance (r)')
    plt.ylabel('Correlation Function (xi)')
    plt.xscale("log")
    # plt.yscale("log")
    plt.legend()
    plt.savefig('../data/frac_dim_fit_{}.png'.format(suffix))
    plt.show()
    return D

def broken_power_law(r, gamma1, gamma2, rb, c1, c2):
    """
    Piecewise power law with a break at rb.
    """
    return np.where(r < rb, np.log10(r) * (-gamma1) + c1, np.log10(r) * (-gamma2) + c2)

# Fit the correlation function with the broken power law
def estimate_fractal_dimension_broken_powerlaw(r_bins, xi, func=None, suffix="suffix"):
    """
    Estimate the fractal dimension using a broken power law fit.
    
    Parameters:
    r_bins: The bin centers for the distances.
    xi: The estimated two-point correlation function values.
    
    Returns:
    D_large: Estimated fractal dimension at large scales.
    D_small: Estimated fractal dimension at small scales.
    """
    # Fit the TPCF to a broken power law
    popt, _ = curve_fit(broken_power_law, r_bins, xi, p0=[1, 2, 10, 1, 0.5], maxfev=50000)
    gamma1, gamma2, rb, c1, c2 = popt
    yfit = broken_power_law(r_bins, *popt)
    
    # Fractal dimensions
    D_small = 3 - gamma1
    D_large = 3 - gamma2
    
    # Plot the correlation function and fit
    plt.figure()
    plt.plot(r_bins, xi, label='Correlation Function')
    plt.plot(r_bins, yfit, label=f'Fit: gamma1={gamma1:.2f}, gamma2={gamma2:.2f}')
    plt.xlabel('Distance (r)')
    plt.ylabel('Correlation Function (xi)')
    plt.legend()
    plt.savefig('../data/frac_dim_fit_{}.png'.format(suffix))
    plt.show()
    return D_small, D_large

def hierarchical_clustering():
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.cluster.hierarchy import fcluster
    # Example data: create a dataset with two clusters
    np.random.seed(42)
    data1 = np.random.normal(loc=[2, 2], scale=0.5, size=(100, 2))
    data2 = np.random.normal(loc=[8, 8], scale=0.5, size=(100, 2))
    data = np.vstack((data1, data2))

    # Perform hierarchical clustering using the 'ward' linkage method
    Z = linkage(data, method='ward')

    # Plot the dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.show()

    # Cut the dendrogram to form flat clusters at a specific distance threshold
    clusters = fcluster(Z, t=5, criterion='distance')

    # Plot the clustered data
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis')
    plt.title('Hierarchical Clustering')
    plt.show()

def fuzzy_clustering(data):
    '''
    Fuzzy clustering by skfuzzy.
    https://scikit-fuzzy.github.io/scikit-fuzzy/api/index.html
    https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html#example-plot-cmeans-py
    '''
    
    import skfuzzy as skf
    # Example data: create a dataset with two clusters
    # np.random.seed(42)
    # data1 = np.random.normal(loc=[2, 2], scale=0.5, size=(100, 2))
    # data2 = np.random.normal(loc=[8, 8], scale=0.5, size=(100, 2))
    # data = np.vstack((data1, data2))

    # Fuzzy C-Means clustering
    cntr, u, u0, d, jm, p, fpc = skf.cluster.cmeans(data.T, c=1000, m=2., error=0.005, maxiter=1000, init=None)

    # Hard clustering result by assigning data points to the cluster with the highest membership
    cluster_membership = np.argmax(u, axis=0)
    ads.DEBUG_PRINT_V(1, np.mean(cluster_membership), np.min(cluster_membership), np.max(cluster_membership), np.shape(u))

    # Plotting
    plt.scatter(data[:, 0], data[:, 1], s=10., c=cluster_membership, cmap='viridis')
    plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', s=10., c='red', label='Centroids')
    plt.title('Fuzzy C-Means Clustering')
    plt.legend()
    plt.show()

    return cluster_membership

def func_adjust_mesh_poly3_1d(x):
    return x**3/np.max(x)**2

def density_from_descrete_data(x, n, savepath, 
    m=None, bound_percent=98., func_adjust_mesh=None, 
    is_log10_dens=False
):
    '''
    To calculate density from descrete data and save to txt.

    @param x: an (n,l)-array.
    @param n_grid: int or an l-array.
    '''
    import mytools_simple as mts
    
    N, L = np.shape(x)
    n_grids = None
    # if type(n) == int:
    #     n_grids = np.ones(L)*n
    # else:
    #     n_grids = n
    n_grids = (np.ones(L)*n).astype(int)
    grids = list(range(L))
    # ads.DEBUG_PRINT_V(0, N, L, n, n_grids)
    
    bounds = np.zeros((L,2))
    for i in np.arange(L):
        # bd = np.max( np.abs(x[:,i]) )
        bd = np.percentile( np.abs(x[:,i]), bound_percent )
        bounds[i] = np.array([-bd, bd])
    for i in np.arange(L):
        grids[i] = np.linspace(bounds[i][0], bounds[i][1], n_grids[i])
        if func_adjust_mesh is not None:
            grids[i] = func_adjust_mesh(grids[i])
    
    if L != 3:
        print("Other dimesions are not provided now. Exit.")
        exit(0)
    mgs = np.meshgrid(grids[0], grids[1], grids[2])
    ads.DEBUG_PRINT_V(1, bounds, np.shape(mgs[0]), np.shape(grids))
    
    rho = np.zeros_like(mgs[0])
    KD = mts.KDTree_galaxy_particles(x, weight_extern_instinct=m)
    for i in np.arange(n_grids[0]):
        for j in np.arange(n_grids[1]):
            for k in np.arange(n_grids[2]):
                targets = [[grids[0][i], grids[1][j], grids[2][k]]]
                rho[j,i,k] = KD.density_SPH(targets)[0] #Note: It is i,j,k -> j,i,k in meshgrid()

    if is_log10_dens:
        # rho = np.random.random((N_grid_x, N_grid_y, N_grid_z))*-10.
        # rho = np.log10 ( 1. / (mg1**2+mg2**2) )

        # rho0 = KD.density_SPH([[0., 0., 0.]])
        # rho = np.log10(rho/rho0)
        rho = np.log10(rho)
    ads.DEBUG_PRINT_V(1, np.shape(x), np.shape(rho), np.shape((mgs[0]).reshape(-1)))

    dens_grids = np.hstack( (
        np.array([ ( mgs[0] ).reshape(-1) ]).T, 
        np.array([ ( mgs[1] ).reshape(-1) ]).T, 
        np.array([ ( mgs[2] ).reshape(-1) ]).T, 
        np.array([ ( rho    ).reshape(-1) ]).T
    ) )
    np.savetxt(savepath, dens_grids, header="## pos_x, pos_y, pos_z, num_dens")
    print("Save dens to {}.".format(savepath))
    return 0

def wavelet_position_versus_density(data, grid_size):
    '''
    An wavelet example.
    https://blog.csdn.net/m0_37605642/article/details/135598057
    '''
    import pywt
    from sklearn.preprocessing import StandardScaler

    # Extract the positions and density values
    positions = data[:, 0:3]  # (x, y, z) coordinates
    densities = data[:, 3]  # Density values

    # Assume the positions (x, y, z) are on a regular grid (reshape to a grid if needed)
    # Here we assume that the positions can form a 3D grid. For simplicity, let's assume 
    # the grid is (10, 10, 10)
    grid_size = grid_size  # Example 3D grid size

    # Reshape densities to fit into the grid
    density_grid = densities.reshape(grid_size) #Note: The array is reshaped from 3d to 1d, then from 1d to 3d

    # Apply 3D wavelet transform to the density grid
    wavelet = 'db4'  # Daubechies 4 wavelet
    coeffs = pywt.wavedecn(density_grid, wavelet)
    # ads.DEBUG_PRINT_V(0, coeffs)

    # Extract the approximation coefficients (at the coarsest scale)
    approx = coeffs[0]

    # Visualize a slice of the approximation coefficients (e.g., at z=5)
    plt.imshow(approx[:, :, 10], cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title('Wavelet Approximation Coefficients (z={} slice)'.format(10))
    plt.show()

    # Reconstruct the approximation at the highest level (coarse approximation)
    reconstructed_density = pywt.waverecn(coeffs, wavelet)
    
    # Now, let's visualize the original and the reconstructed density for one slice of the grid
    plt.figure(figsize=(10, 5))

    # Visualize original density slice (e.g., z=5 slice)
    plt.subplot(1, 2, 1)
    plt.imshow(density_grid[:, :, 5], cmap='viridis', aspect='auto')
    plt.title('Original Density (z=5 slice)')
    plt.colorbar()

    # Visualize reconstructed density slice (e.g., z=5 slice)
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_density[:, :, 5], cmap='viridis', aspect='auto')
    plt.title('Reconstructed Density (z=5 slice)')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return coeffs

def fracal_analysis_by_MFDFA(signal_1d):
    from MFDFA import MFDFA

    # Assuming you have an (n, 4)-array where the first 3 columns are positions (x, y, z)
    # and the 4th column is the density at those positions.
    # For simplicity, let's assume we are working with the 1D density data already

    # Example: Synthetic 1D density signal (distribution function) derived from 3D data
    # n = 8000
    # density_signal = np.random.randn(n)
    density_signal = signal_1d

    # Define lag values (window sizes) and moments for the analysis
    lag_values = np.logspace(1, 3, num=50, base=10).astype(int)
    order_values = np.hstack((np.linspace(-5., -1., num=5), np.linspace(1., 5., num=5)))

    # Perform MFDFA on the density signal
    lag, fluct = MFDFA(density_signal, lag=lag_values, q=order_values)

    # Compute the multifractal spectrum h(q) by fitting log-log of F_q(s) vs lag
    hq = []
    for i in range(len(order_values)):
        # ads.DEBUG_PRINT_V(1, np.shape(order_values), np.shape(np.log(lag)), np.shape(np.log(fluct[:, i])))
        coeffs = np.polyfit(np.log(lag), np.log(fluct[:, i]), 1)
        hq.append(coeffs[0])  # Slope gives h(q)
    
    # Plot the multifractal spectrum (h(q) vs q)
    plt.plot(order_values, hq, 'bo-', label='Multifractal Spectrum h(q)')
    plt.xlabel('q')
    plt.ylabel('h(q)')
    plt.title('Multifractal Spectrum h(q)')
    plt.legend()
    plt.show()

    # Optional: Plot the fluctuation functions Fq(s) for different q
    for i, q in enumerate(order_values):
        plt.loglog(lag, fluct[:, i], label='q={:}'.format(q))
    plt.xlabel('Scale (log(s))')
    plt.ylabel('Fluctuation Function Fq(s)')
    plt.legend()
    plt.title('Fluctuation Functions for Different q')
    plt.show()

    return hq, lag, fluct

def t_relax_rate(labels):
    N = len(labels)
    N_cl = np.max(labels) + 1
    N_noise = len(np.where(labels==-1)[0])
    
    N_at_single = 1
    N_at_halo = np.zeros(N_cl)
    for i in np.arange(N_cl):
        N_at_halo[i] = len(np.where(labels==i)[0])
    N_sum = int(N_noise + np.sum(N_at_halo))
    ads.DEBUG_PRINT_V(1, N_cl, N_sum)
    ads.DEBUG_PRINT_V(1, N, N_noise, N_at_halo)
    A, V, m = 1., 1., 1.
    tr_all_noise = A*V/m**2 / ( N_at_single**2 * N_sum)
    tr_all_cluster = A*V/m**2 / ( N**2 * 1.)
    tr = A*V/m**2 / ( N_at_single**2 * N_noise + np.sum(N_at_halo**2 * 1.) )
    ads.DEBUG_PRINT_V(1, tr, tr_all_noise, tr_all_cluster)
    return tr, tr_all_noise, tr_all_cluster

# Calculate velocity dispersion
def velocity_dispersion(x, xv):
    # beta = kdtp.velocity_dispersion_knn()
    beta_sph = kdtp.beta_parameter_spherical_from_xv(x, xv)
    beta_cyl = kdtp.beta_z_parameter_cylindrel_from_xv(x, xv)
    return beta_sph, beta_cyl

# Define the distribution function f_a(v_a) - using an example (Maxwellian)
def fa_MB(va):
    return np.exp(-va**2)

def fa_MB_fluctuation_and_long_tail(va):
    return fa_MB(va) * (1. + 1e-1 * np.sin(va * 2.*np.pi) + 1e-2 * np.sin(va/10. * 2.*np.pi))

def fa_fluctuation_and_long_tail(va):
    return va**-4 * (1. + 1e-2 * np.sin(va * 2.*np.pi))

# Calculate numberical integration in diffusion coef about velocity DF
def diffusion_coef_about_velocity_DF(fa, fa_args=None, va_min=0., va_max=np.inf, v_substract=220.):
    from scipy.integrate import quad

    # Define constants
    M = 137. #1e10 MSun
    N = 1e11
    G = 43007.1 #unit as gadget2
    ma = M/N
    log_Lambda = np.log10(N/2.) #np.log10(R_max/b_90)
    v = v_substract #km/s

    # Define the first integrand
    def integrand1(va, fa_args):
        return (va**4 / v**3) * fa(va, *fa_args) / (4.*np.pi*va**2)

    # Define the second integrand
    def integrand2(va, fa_args):
        return va * fa(va, *fa_args) / (4.*np.pi*va**2)

    # Perform the numerical integration
    subdivisions = 1000
    integral1, error1 = quad(integrand1, va_min, v, args=fa_args, limit=subdivisions)
    integral2, error2 = quad(integrand2, v, va_max, args=fa_args, limit=subdivisions)

    # Calculate the full expression for D[(Delta v_parallel)^2]
    coef_const = 32./3. * np.pi**2 * G**2 * ma**2 * log_Lambda
    D_DeltaV2 = coef_const * (integral1 + integral2)

    return D_DeltaV2

# # Maxwellian-Boltzmann function definition
# def fva_speed_Maxwellian_Boltzmann(v, A, sigma_0):
#     # B = 1 / (2 * sigma_0**2)
#     # return A * np.exp(-B * (v / sigma_0)**2)
#     # return A * np.exp(-B * (v / sigma_0)**2) * (v/sigma_0)**2
#     # return A * np.exp(-B * (v / sigma_0)**2) * (v/1.)**2
#     return A * np.exp(-(v / sigma_0)**2) * (v/sigma_0)**2

# # Power-law function definition
# def fva_speed_power_law(v, A, sigma_0, alpha, beta):
#     # return A * (v / sigma_0)**(-alpha) * (1 + v / sigma_0)**(-beta)
#     # return A * (1 + v / sigma_0)**(-beta)
#     # return A * (v / sigma_0)**(-alpha)
#     # return A * (v / sigma_0)**(-alpha+2.)
#     return A * (v / sigma_0)**(-alpha+2.) * (1 + v / sigma_0)**(-beta)

# def g_combine_weight(v, vt, vw):
#     return 1. / ( 1. + np.exp(-(v-vt)/vw) )

# # Maxwellian-Boltzmann DF with a powerlaw tail function definition
# def fva_speed_MB_with_tail(v, sigma, sigma_2, A, alpha, vt, vw):
#     # A = 1.
#     fva1 = A*(2./np.pi)**0.5/sigma**3 * v**2 * np.exp(-(v/sigma)**2)
#     fva2 = 1./sigma_2**3 * v**2. * (v/sigma_2+1.e-1)**(-alpha)
#     fva = fva1*(1.-g_combine_weight(v, vt, vw)) + fva2*g_combine_weight(v, vt, vw)
#     return fva

# def fit_velocity_DF(v_size):
#     import scipy.optimize as opt
#     from sklearn.neighbors import KernelDensity

#     # Fit function
#     def fit_distribution(velocities, vel_DF_data, fit_func, p0, bounds):
#         # Perform the curve fit using non-linear least squares
#         popt, _ = opt.curve_fit(fit_func, velocities, vel_DF_data, p0=p0, bounds=bounds, maxfev=5000)
#         return popt

#     # # Main part of the code
#     # if __name__ == "__main__":
#     # Step 1: Load or generate velocity data
#     # N = 1000
#     # velocities = np.random.uniform(10, 200, N)  # Example data (velocity values in arbitrary units)
#     velocities = v_size

#     # Step 2: Kernel Density Estimation (KDE) to estimate the velocity distribution function
#     kde = KernelDensity(kernel='gaussian', bandwidth=5.0).fit(velocities[:, np.newaxis])
#     log_density = kde.score_samples(velocities[:, np.newaxis])
#     vel_DF_data = np.exp(log_density)

#     # Step 3: Define bounds and initial parameters for both models
#     v_min, v_max = np.min(velocities), np.max(velocities)

#     # Maxwell-Boltzmann initial guesses and bounds
#     # p0_MB = [1.0, 50.0]  # Initial guess for [A, sigma_0]
#     p0_MB = [1.0, 200.0]  # Initial guess for [A, sigma_0]
#     # bounds_MB = ([0, 0], [np.inf, np.inf])  # Bounds for [A > 0, sigma_0 > 0]
#     bounds_MB = ([1e-6, 1e-6], [1e6, 1e6])  # Bounds for [A > 0, sigma_0 > 0]

#     # Power-law initial guesses and bounds
#     p0_PL = [1.0, 50.0, 2.0, 2.0]  # Initial guess for [A, sigma_0, alpha, beta]
#     bounds_PL = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])  # Bounds for [A, sigma_0, alpha, beta]
#     # bounds_PL = ([0, 0, -1000., 1000.], [np.inf, np.inf, 1000., 1000.])  # Bounds for [A, sigma_0, alpha, beta]

#     # # MB with tail initial guesses and bounds
#     # p0_MBtail = [1.0, 200.0, 1.0, 4.]  # Initial guess for [A, sigma_0, B, alpha]
#     # bounds_MBtail = ([1e-6, 1e-6, 1e-6, 1e-6], [1e6, 1e6, 1e6, 1e6])  # Bounds for [A > 0, sigma_0 > 0]

#     # fva_speed_MB_with_tail
#     p0_MBtail = [180., 200., 1., 4., 500., 100.]  # Initial guess
#     bounds_MBtail = ([1e-6, 1e2, 1e-6, 3e0, 1e2, 1e-6], [1e6, 1e6, 1e6, 1e6, 1e6, 1e6])  # Bounds

#     # Step 3: Fit both distributions
#     params_MBtail = fit_distribution(velocities, vel_DF_data, fva_speed_MB_with_tail, p0_MBtail, bounds_MBtail)
#     params_MB = fit_distribution(velocities, vel_DF_data, fva_speed_Maxwellian_Boltzmann, p0_MB, bounds_MB)
#     params_PL = fit_distribution(velocities, vel_DF_data, fva_speed_power_law, p0_PL, bounds_PL)

#     # Step 4: Plot the original velocity data and fitted functions
#     v_fit = np.linspace(v_min, v_max, 1000)

#     plt.figure(figsize=(10, 6))
#     # params_MB = [ 6423.07640959, 200.] #debug
#     # plt.hist(velocities, bins=50, density=True, alpha=0.5, label='Velocity Data')
#     plt.plot(velocities, vel_DF_data, 'k*', label='data')
#     plt.plot(v_fit, fva_speed_Maxwellian_Boltzmann(v_fit, *params_MB), 'r-', label='Maxwell-Boltzmann Fit')
#     plt.plot(v_fit, fva_speed_power_law(v_fit, *params_PL), 'b--', label='Power-Law Fit')
#     plt.plot(v_fit, fva_speed_MB_with_tail(v_fit, *params_MBtail), 'g--', label='MB-tail Fit')
#     plt.xlabel('Velocity Size')
#     plt.ylabel('Density')
#     # plt.yscale("log")
#     plt.title('Velocity Distribution Fitting')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('../data/velocity_distribution_fit.png')
#     plt.show()
#     return params_MB, params_PL, params_MBtail

# Maxwellian-Boltzmann function definition
def fva_speed_Maxwellian_Boltzmann(v, A, sigma_0):
    # B = 1 / (2 * sigma_0**2)
    # return A * np.exp(-B * (v / sigma_0)**2)
    # return A * np.exp(-B * (v / sigma_0)**2) * (v/sigma_0)**2
    # return A * np.exp(-B * (v / sigma_0)**2) * (v/1.)**2
    return A * np.exp(-(v / sigma_0)**2) * (v/sigma_0)**2

# Power-law function definition
def fva_speed_power_law(v, A, sigma_0, alpha, beta):
    # return A * (v / sigma_0)**(-alpha) * (1 + v / sigma_0)**(-beta)
    # return A * (1 + v / sigma_0)**(-beta)
    # return A * (v / sigma_0)**(-alpha)
    # return A * (v / sigma_0)**(-alpha+2.)
    return A * (v / sigma_0)**(-alpha+2.) * (1 + v / sigma_0)**(-beta)

def g_combine_weight(v, vt, vw):
    return 1. / ( 1. + np.exp(-(v-vt)/vw) )

# Maxwellian-Boltzmann DF with a powerlaw tail function definition
# def fva_speed_MB_with_tail(v, sigma, sigma_2, A_pl, epsilon, alpha, vt, vw):
def fva_speed_MB_with_tail(v, sigma, sigma_2, A_pl, alpha, vt, vw):
    A = 1.
    # A_pl = 1.
    epsilon = 1e-1
    fva1 = A*(2./np.pi)**0.5/sigma**3 * v**2 * np.exp(-(v/sigma)**2)
    # fva2 = 1./sigma_2**3 * v**2. * (v/sigma_2+1.e-1)**(-alpha)
    fva2 = 1./sigma_2**3 * v**2. * (v/sigma_2+epsilon)**(-alpha)
    # fva = vw*fva1 + A_pl*fva2
    fva = fva1*(1.-g_combine_weight(v, vt, vw)) + A_pl*fva2*g_combine_weight(v, vt, vw)
    return fva

def fit_velocity_DF(v_size):
    import scipy.optimize as opt
    from sklearn.neighbors import KernelDensity

    # Fit function
    def fit_distribution(velocities, vel_DF_data, fit_func, p0, bounds):
        # Perform the curve fit using non-linear least squares
        popt, _ = opt.curve_fit(fit_func, velocities, vel_DF_data, p0=p0, bounds=bounds, maxfev=5000)
        return popt

    # # Main part of the code
    # if __name__ == "__main__":
    # Step 1: Load or generate velocity data
    # N = 1000
    # velocities = np.random.uniform(10, 200, N)  # Example data (velocity values in arbitrary units)
    velocities = v_size

    # Step 2: Kernel Density Estimation (KDE) to estimate the velocity distribution function
    kde = KernelDensity(kernel='gaussian', bandwidth=5.0).fit(velocities[:, np.newaxis])
    log_density = kde.score_samples(velocities[:, np.newaxis])
    vel_DF_data = np.exp(log_density)

    # Step 3: Define bounds and initial parameters for both models
    v_min, v_max = np.min(velocities), np.max(velocities)

    # Maxwell-Boltzmann initial guesses and bounds
    # p0_MB = [1.0, 50.0]  # Initial guess for [A, sigma_0]
    p0_MB = [1.0, 200.0]  # Initial guess for [A, sigma_0]
    # bounds_MB = ([0, 0], [np.inf, np.inf])  # Bounds for [A > 0, sigma_0 > 0]
    bounds_MB = ([1e-6, 1e-6], [1e6, 1e6])  # Bounds for [A > 0, sigma_0 > 0]

    # Power-law initial guesses and bounds
    p0_PL = [1.0, 50.0, 2.0, 2.0]  # Initial guess for [A, sigma_0, alpha, beta]
    bounds_PL = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])  # Bounds for [A, sigma_0, alpha, beta]
    # bounds_PL = ([0, 0, -1000., 1000.], [np.inf, np.inf, 1000., 1000.])  # Bounds for [A, sigma_0, alpha, beta]

    # # MB with tail initial guesses and bounds
    # p0_MBtail = [1.0, 200.0, 1.0, 4.]  # Initial guess for [A, sigma_0, B, alpha]
    # bounds_MBtail = ([1e-6, 1e-6, 1e-6, 1e-6], [1e6, 1e6, 1e6, 1e6])  # Bounds for [A > 0, sigma_0 > 0]

    # fva_speed_MB_with_tail
    p0_MBtail = [180., 200., 1., 4., 500., 100.]  # Initial guess
    # p0_MBtail = [180., 200., 1., 1e-1, 4., 500., 100.]  # Initial guess
    bounds_MBtail = ([1e-6, 1e2, 1e-6, 3e0, 1e2, 1e-6], [1e6, 1e6, 1e6, 1e6, 1e6, 1e6])  # Bounds
    # bounds_MBtail = ([1e-6, 1e2, 1e-6, 1e-6, 3e0, 1e2, 1e-6], [1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6])  # Bounds

    # Step 3: Fit both distributions
    params_MBtail = fit_distribution(velocities, vel_DF_data, fva_speed_MB_with_tail, p0_MBtail, bounds_MBtail)
    params_MB = fit_distribution(velocities, vel_DF_data, fva_speed_Maxwellian_Boltzmann, p0_MB, bounds_MB)
    params_PL = fit_distribution(velocities, vel_DF_data, fva_speed_power_law, p0_PL, bounds_PL)

    # Step 4: Plot the original velocity data and fitted functions
    v_fit = np.linspace(v_min, v_max, 1000)

    plt.figure(figsize=(10, 6))
    # params_MB = [ 6423.07640959, 200.] #debug
    # plt.hist(velocities, bins=50, density=True, alpha=0.5, label='Velocity Data')
    plt.plot(velocities, vel_DF_data, 'k*', label='data')
    plt.plot(v_fit, fva_speed_Maxwellian_Boltzmann(v_fit, *params_MB), 'r-', label='Maxwell-Boltzmann Fit')
    plt.plot(v_fit, fva_speed_power_law(v_fit, *params_PL), 'b--', label='Power-Law Fit')
    plt.plot(v_fit, fva_speed_MB_with_tail(v_fit, *params_MBtail), 'g--', label='MB-tail Fit')
    plt.xlabel('Velocity Size')
    plt.ylabel('Density')
    # plt.yscale("log")
    plt.title('Velocity Distribution Fitting')
    plt.legend()
    plt.grid(True)
    plt.savefig('../data/velocity_distribution_fit.png')
    plt.show()
    return params_MB, params_PL, params_MBtail



#In[] main
if __name__ == "__main__":

    ## data
    data_path = "../data/samples_simulated/snapshot_010_example_NFW.txt"
    # data_path = "../data/samples_simulated/snapshot_080_example_merge.txt"
    # data_path = "../data/snapshot_%03d_big.txt"%(snapshot_ID)
    data = np.loadtxt(data_path, dtype=float)
    N_particles = len(data)
    positions = data[:, 0:3]
    # positions = np.random.random((N, 3))
    # positions = np.random.normal(0, 1, (N, 3))
    velocities = data[:, 3:6]
    m = data[:, 8]
    pers = [50., 90., 95., 97., 98., 99., 99.5]
    meds = ads.percentiles_by_xv_data(data[:,0:6], pers=pers)
    ads.DEBUG_PRINT_V(1, pers, meds[3], meds[7], "xvmeds")
    r_max = 100.
    n_average = N_particles/(4.*np.pi/3*r_max**3) #0.238
    print("mean numbder density per kpc: {}".format(n_average))
    # beta_sph, beta_cyl = velocity_dispersion(None, xv)
    # ads.DEBUG_PRINT_V(1, beta_sph, beta_cyl, "beta_sph, beta_cyl")
    positions -= np.mean(positions, axis=0) #translate x center
    velocities -= np.mean(velocities, axis=0) #translate v center
    r_data = ads.norm_l(positions, axis=1)
    V_data = ads.norm_l(velocities, axis=1)
    r_mean = np.mean(r_data)
    V_max = np.max(V_data)
    t_cross = np.mean(r_data/V_data)

    is_screen_radius = True
    # is_screen_radius = False
    if is_screen_radius: #to screen particles whose radius within 0.5 ~ 1.5 r_mean
        idx_range = np.where((r_data>0.5*r_mean) & (r_data<1.5*r_mean))[0]
        xv_range = data[idx_range]
        positions = xv_range[:, 0:3]
        velocities = xv_range[:, 3:6]
        r_data = ads.norm_l(positions, axis=1)
        V_data = ads.norm_l(velocities, axis=1)
        ads.DEBUG_PRINT_V(1, len(idx_range), "idx_range")
    
    # ##1. Estimate the fractal dimension using TPCF with KDTree# Estimate the fractal dimension
    # r_bins, xi = correlation_function_distmat(positions, r_max=r_max, num_bins=50)
    # # r_bins, xi = correlation_function_scipyckdtree(positions, r_max=r_max, num_bins=50)
    
    # func = powerlaw_1_log10
    # # func = powerlaw_1_translation_log10
    # D_list = estimate_fractal_dimension_single(r_bins, xi, func)
    # # D_list = estimate_fractal_dimension_broken_powerlaw(r_bins, xi, func)
    # print(f"Estimated Fractal Dimension (D): {D_list}")
    # # ads.DEBUG_PRINT_V(0, xi, D_list)

    # n, r, df, D_embeding, df_mean = fracal_dim_by_boxcounting(positions)
    # ads.DEBUG_PRINT_V(0, n, r, df, D_embeding, df_mean)

    # ##2. Recognize subhalos using DBSCAN
    # distance_threshold = 1. #kpc # Adjust based on your data scale (e.g., kpc)
    # # distance_threshold = 1.0 #kpc # Adjust based on your data scale (e.g., kpc)
    # # min_particles = int(5*N/(4.*np.pi/3*r_max**3)) # Minimum number of particles to form a subhalo
    # # min_particles = 5
    # min_particles = 32
    # print(f"distance_threshold, min_particles: {distance_threshold}, {min_particles}")
    # labels, N_sub, ptcs_in_sub, N_ptcs_in_sub = recognize_subhalos_dbscan(positions, distance_threshold, min_particles)
    # print(f"N, N_sub, N_ptcs_in_sub: {N}, {N_sub}, {N_ptcs_in_sub}")
    # # ads.DEBUG_PRINT_V(0, np.shape(labels), labels[ptcs_in_sub], N_sub, N_noise)
    # # For a big snapshot:
    # # distance_threshold, min_particles: 1.0, 32
    # # N, N_sub, N_ptcs_in_sub: 1000000, 106, 195860

    # ##3. fractal or fractional relaxation time example
    # tr, tr_all_noise, tr_all_cluster = t_relax_rate(labels)
    # print(f"tr_multi frac: {tr_all_noise/tr}")

    ##4. fuzzy clustering or hierarchical clustering for galaxy center and multi fracal
    # # cluster_membership = hierarchical_clustering()
    # cluster_membership = fuzzy_clustering(positions)

    # signal_1d = np.random.randn(8000)
    # hq, lag, fluct = fracal_analysis_by_MFDFA(signal_1d)

    # savepath_dens = data_path+".dens_grids.txt"
    # grid_size = [20, 20, 20]
    # # density_from_descrete_data(positions, grid_size, savepath=savepath_dens, m=None)
    # pos_and_dens_on_grid = np.loadtxt(savepath_dens, dtype=float)
    # WT_coeffs = wavelet_position_versus_density(pos_and_dens_on_grid, grid_size)

    ##5. velocty DF
    params_MB, params_PL, params_MBtail = fit_velocity_DF(V_data)
    ads.DEBUG_PRINT_V(1, params_MB, params_PL, params_MBtail, "params_fit")

    va_min = 1e-2
    va_max = 1e3
    # va_max = 1e4
    # v_substract = 10.
    # v_substract = 50.
    v_substract = 220.
    # v_substract = 500.
    # Diffu1 = diffusion_coef_about_velocity_DF(fa_MB, va_min=va_min, va_max=va_max)
    # Diffu2 = diffusion_coef_about_velocity_DF(fa_fluctuation_and_long_tail, va_min=va_min, va_max=va_max)
    # print("dd: {}, {}, {}".format(dd1, dd2, dd3))
    # Diffu1 = diffusion_coef_about_velocity_DF(fa_MB_fluctuation_and_long_tail, fa_args=None, va_min=va_min, va_max=va_max, v_substract=v_substract)
    # Diffu2 = diffusion_coef_about_velocity_DF(fva_speed_power_law, fa_args=params_PL, va_min=va_min, va_max=va_max, v_substract=v_substract)
    Diffu1 = diffusion_coef_about_velocity_DF(fva_speed_Maxwellian_Boltzmann, fa_args=params_MB, va_min=va_min, va_max=va_max, v_substract=v_substract)
    Diffu2 = diffusion_coef_about_velocity_DF(fva_speed_MB_with_tail, fa_args=params_MBtail, va_min=va_min, va_max=va_max, v_substract=v_substract)
    ads.DEBUG_PRINT_V(1, Diffu1, Diffu2, "Diffu1, Diffu2")
    print("amplication rate of diffu coef: {}".format(Diffu2/Diffu1))
