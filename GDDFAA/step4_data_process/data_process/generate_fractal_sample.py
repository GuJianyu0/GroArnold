#!/usr/bin/env python
# -*- coding:utf-8 -*-
#In[] modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from scipy.optimize import curve_fit
# from sklearn.cluster import DBSCAN
# from scipy.spatial import cKDTree
# # from scipy.spatial import KDTree
# # from Corrfunc.theory.DD import DD
from scipy.integrate import tplquad
from noise import pnoise3
import vegas
import pandas as pd

import analysis_data_distribution as ads
import KDTree_python as kdtp
import observed_data_process as odp
import fractal_dim_and_subhalos as fds
import generate_fractal_sample_various as gfsv



#In[] settings and functions
def generate_fractal_3D_vertex(num_points, fractal_dim, bounds=(-1, 1)):  
    """  
    Generate a 3D fractal distribution with a given fractal dimension.  
    
    Parameters:  
        num_points (int): Total number of points to generate.  
        fractal_dim (float): Desired fractal dimension (e.g., 1.8).  
        bounds (tuple): Bounds of the cube (x, y, z) in which points are generated.  
    
    Returns:  
        np.array: Array of shape (num_points, 3) with fractal-distributed 3D points.  
    """  
    # Define the initial bounding 3D cube  
    min_bound, max_bound = bounds  
    initial_vertices = np.array([  
        [min_bound, min_bound, min_bound],  
        [max_bound, min_bound, min_bound],  
        [min_bound, max_bound, min_bound],  
        [max_bound, max_bound, min_bound],  
        [min_bound, min_bound, max_bound],  
        [max_bound, min_bound, max_bound],  
        [min_bound, max_bound, max_bound],  
        [max_bound, max_bound, max_bound],  
    ])  

    # Start with one point in the middle  
    points = [np.array([(min_bound + max_bound) / 2] * 3)]  
    
    # Iteratively refine points  
    for _ in range(num_points - 1):  
        # Choose a random vertex (for self-similarity)  
        vertex = initial_vertices[np.random.randint(0, len(initial_vertices))]  
        # Move halfway towards that vertex  
        new_point = points[-1] + (vertex - points[-1]) * (0.5 ** (3 - fractal_dim))  
        points.append(new_point)  

    return np.array(points)  

def generate_fractal_3D_walk(num_steps, dim): 
    '''
    You can use algorithms such as the Random Walk, L-System, or Diffusion-Limited Aggregation (DLA) to create fractals.
    For a desired dimension of 1.8, a random walk or a recursive structure may be best.
    ''' 
    steps = []  
    position = np.zeros(3)  # Starting position at the origin (0, 0, 0)  
    
    for _ in range(num_steps):  
        # Random step direction, with adjustments to get closer to dimension 1.8  
        theta = np.random.uniform(0, 2 * np.pi)  
        phi = np.random.uniform(0, np.pi)  
        
        # Move in 3D  
        x_step = np.sin(phi) * np.cos(theta)  
        y_step = np.sin(phi) * np.sin(theta)  
        z_step = np.cos(phi)  
        
        step_size = np.random.exponential(scale=(2 ** (1 / dim)))  # Scale to approximate dimension  
        position += step_size * np.array([x_step, y_step, z_step])  
        steps.append(position.copy())
    
    return np.array(steps)

def generate_uniform_sample_with_shape(rs, N, bound_shape="spherical"):
    """
    Generate a sample of particle positions.

    Parameters:
    - N (int): The count of particles.
    - rs (float): The mean radius to which the particle distribution should be scaled.

    Returns:
    - samples (numpy.ndarray): An (N, 3) array representing the positions of particles.
    """
    # Generate spherical coordinates with uniform distribution
    samples = None
    # if npseed is not None:
    #     np.random.seed(npseed)
    # else:
    #     # np.random.seed(11)
    #     np.random.seed(65065)

    if bound_shape == "spherical": #?? not uniform
        # r_vals = np.sqrt(np.random.uniform(0., rs**3, N))  # Radial distance (uniform in volume)
        r_vals = np.cbrt(np.random.uniform(0., rs**3, N))  # Radial distance (uniform in volume)
        theta_vals = np.arccos(np.random.uniform(-1, 1, N))  # Polar angle
        # theta_vals = np.random.uniform(0., np.pi, N)  # Polar angle
        phi_vals = np.random.uniform(0., 2.*np.pi, N)  # Azimuthal angle

        # Convert spherical coordinates to Cartesian coordinates
        x_vals = r_vals * np.sin(theta_vals) * np.cos(phi_vals)
        y_vals = r_vals * np.sin(theta_vals) * np.sin(phi_vals)
        z_vals = r_vals * np.cos(theta_vals)
        samples = np.vstack((x_vals, y_vals, z_vals)).T

    elif bound_shape == "cylindrical":
        r_vals = np.sqrt(np.random.uniform(0, rs**2, N))  # Radial distance (uniform in area)
        phi_vals = np.random.uniform(0, 2.*np.pi, N)  # Azimuthal angle
        z_vals = np.random.uniform(0, rs, N)  # Height (uniform in z)

        # Convert cylindrical coordinates to Cartesian coordinates
        x_vals = r_vals * np.cos(phi_vals)
        y_vals = r_vals * np.sin(phi_vals)
        samples = np.vstack((x_vals, y_vals, z_vals)).T

    else: #cubic shape with length rs*2.
        samples = np.random.uniform(0, rs*2., (N, 3))

    # Move samples to the mass center (mean of the positions to be at the origin)
    mass_center = np.mean(samples, axis=0)
    samples -= mass_center

    # Rescale the mean radius of samples to rs
    current_mean_radius = np.mean(np.linalg.norm(samples, axis=1))
    scaling_factor = rs / current_mean_radius
    samples *= scaling_factor
    return samples

from noise import pnoise3
def Perlin_motion(positions, scale=0.1, octaves=4, step_size=0.5):
    new_positions = np.zeros_like(positions)
    
    for i, pos in enumerate(positions):
        x, y, z = pos
        noise_dx, noise_dy, noise_dz = 0.0, 0.0, 0.0

        # Generate multi-layer Perlin noise by iterating through octaves
        for octave in range(octaves):
            frequency = 2 ** octave  # Increasing frequency for each octave
            amplitude = 0.5 ** octave  # Decreasing amplitude for each octave

            noise_dx += amplitude * pnoise3(x * scale * frequency, y * scale * frequency, z * scale * frequency)
            noise_dy += amplitude * pnoise3(y * scale * frequency, z * scale * frequency, x * scale * frequency)
            noise_dz += amplitude * pnoise3(z * scale * frequency, x * scale * frequency, y * scale * frequency)
        
        # Direction vector influenced by multi-layer Perlin noise
        direction = np.array([noise_dx, noise_dy, noise_dz])
        # direction /= np.linalg.norm(direction) #if move with magnitude 1
        
        # Update the position of each particle with direction magnitude influenced by Perlin noise
        new_positions[i] = pos + direction * step_size
    return new_positions

def particles_3d_Perlin_motion_factor_scale(
    initial_positions, N_iter=10, scale=0.21, step_size=5., is_same_mean_radius=True, 
    is_plot=True, is_plot_witin_bound=False, suffix="suffix"
):
    # Step 1: Generate Perlin noise directions to simulate Brownian or Levy motion
    # # r_mean = get_mean_radius(initial_positions, is_center=True)
    # # ferq_fluc_per_dim = 10.
    # # scale = 0.21
    # # step_size = 5.
    # scale = 0.21
    # step_size = 5.
    new_positions = initial_positions*1.
    for i in range(N_iter):
        new_positions = Perlin_motion(new_positions, scale=scale, octaves=4, step_size=step_size)
    new_positions -= np.mean(new_positions, axis=0) #translate to mass center

    # Step 2: Rescale positions to match the initial volume (cylindrical boundary)
    scaling_factor = 1. #when scaling_factor decrease, the ampl rate increase, with not large influence
    R_init = ads.norm_l(initial_positions, axis=1)
    R_new = ads.norm_l(new_positions, axis=1)
    R_init_mean = np.mean(R_init)
    R_new_mean = np.mean(R_new)
    scaling_factor = R_new_mean/R_init_mean
    if is_same_mean_radius:
        new_positions = new_positions/scaling_factor
    ads.DEBUG_PRINT_V(1, R_init_mean, R_new_mean, scaling_factor, "scaling_factor")

    # Step 3: Plot the initial and new positions of particles
    if is_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], c='blue', s=5, alpha=0.5)
        ax1.set_title("Initial Positions of Particles (x, y) Coordinates")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        if is_plot_witin_bound:
            ax1.set_xlim([-R_init_mean, R_init_mean])
            ax1.set_ylim([-R_init_mean, R_init_mean])
        ax1.grid(True)
        ax2.scatter(new_positions[:, 0], new_positions[:, 1], c='red', s=5, alpha=0.5)
        ax2.set_title("New Positions of Particles (x, y) Coordinates")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        if is_plot_witin_bound:
            ax2.set_xlim([-R_init_mean, R_init_mean])
            ax2.set_ylim([-R_init_mean, R_init_mean])
        ax2.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig("../data/Perlin_motion_xy_"+suffix+".png", format="png", bbox_inches='tight')
    return new_positions




def compute_correlation_dimension(points, max_radius):  
    """
    Estimate the fractal dimension using the correlation dimension method.  
    
    Parameters:  
        points (np.array): (N, 3) array of points.  
        max_radius (float): Maximum radius for counting neighbors.  
    
    Returns:  
        float: Estimated fractal dimension.  
    """
    from scipy.spatial import distance  
    pairwise_distances = distance.cdist(points, points)  
    r_values = np.logspace(-3, np.log10(max_radius), 10)  
    counts = np.zeros(len(r_values))  
    
    for i, r in enumerate(r_values):  
        counts[i] = (pairwise_distances < r).sum()  
        
    log_r = np.log(r_values)  
    log_counts = np.log(counts)  
    
    # Fit a line to log(counts) vs log(radius)  
    fit = np.polyfit(log_r, log_counts, 1)  
    return fit[0]  # Slope of the line is the estimated fractal dimension  

def compute_correlation_dimension_volume(points, r_min, r_max):  
    """
    Estimate the fractal dimension using the correlation dimension method.  
    
    Parameters:  
        points (np.array): (N, 3) array of points.  
        r_max (float): Maximum radius for counting neighbors.  
    
    Returns:  
        float: Estimated fractal dimension.  
    """
    from scipy.spatial import distance  
    pairwise_distances = distance.cdist(points, points)  
    # r_values = np.logspace(-3, np.log10(r_max), 10)  
    r_values = np.geomspace(r_min, r_max, 10)  
    counts = np.zeros(len(r_values))  
    
    for i, r in enumerate(r_values):  
        counts[i] = (pairwise_distances < r).sum() #to fit count
        counts[i] /= (4./3.*np.pi*r**3) #to fit count density
        
    log_r = np.log(r_values)  
    log_counts = np.log(counts)  
    
    # Fit a line to log(counts) vs log(radius)  
    fit = np.polyfit(log_r, log_counts, 1)
    h_frac = fit[-1]
    Dim_frac = 3.+fit[0] #alpha = 3.-Dim_frac
    return h_frac, Dim_frac  # Slope of the line is the estimated fractal dimension  

def powerlaw_1_log10(r, h_D, D):
    """
    Step2: Define the function for fitting mean inside density.
    Arguments:
        r : float : Radius.
        h_D : float : Normalization constant.
        D : float : Fractal dimension.
    Returns:
        n_D : float : Mean inside density.
    """
    # return np.log10( h_D/1. * r**(D-3) ) #(4./3*np.pi)
    # return np.log10( h_D/(4./3*np.pi) ) + np.log10(r)*(D-3)
    return np.log10( h_D/(4./3*np.pi) * r**(D-3) )

from scipy.optimize import curve_fit
from sklearn.neighbors import KDTree
def inside_count_with_radius(positions, targets, r_min=1., r_max=100., n_radii=10.):
    """
    Step1: Calculate the count of nearest particles inside spheres with different radii.
    Arguments:
        positions : (N, 3)-array : 3D coordinates of particles.
        targets : (N_target, 3)-array : Target positions for the sphere center.
        r_max : float : Maximum radius.
        n_radii : int : Number of radii to evaluate.
    Returns:
        radii : (n_radii,)-array : Radii used for the count.
        inside_counts : (n_radii,)-array : Count of nearest particles within each radius.
    """
    # Use KDTree for efficient radius search
    tree = KDTree(positions)
    # radii = np.linspace(r_min, r_max, n_radii)
    radii = np.geomspace(r_min, r_max, n_radii)
    N_targets = len(targets)
    inside_counts = np.zeros((N_targets, n_radii))
    
    for i in np.arange(n_radii):
        counts = tree.query_radius(targets, r=radii[i], count_only=True)
        # inside_counts[:, i] = np.array(counts)
        counts = np.array(counts) - 1  # subtract self-count
        counts[counts < 0] = 0        # ensure no negative count
        inside_counts[:, i] = counts
    
    avg_counts = np.mean(inside_counts, axis=0)
    return radii, avg_counts

def fit_fractal_dimension(radii_list, inside_counts_list, suffix="suffix", is_plot=True, savepath=None):
    """
    Step3: Fit the power law function to get the fractal dimension.
    Arguments:
        radii_list : (n_radii,)-array : Radii used for counting.
        inside_counts_list : (n_radii,)-array : Count of nearest particles within each radius.
    Returns:
        popt : array : Optimal values for the parameters.
    """
    # Calculate mean density inside radius
    idx_mask = np.where(inside_counts_list>0)[0]
    radii = radii_list[idx_mask]
    inside_counts = inside_counts_list[idx_mask]
    mean_density_log10 = np.log10( inside_counts / (4./3*np.pi*radii**3) )
    # ads.DEBUG_PRINT_V(1, radii, inside_counts, mean_density_log10)
    
    # Fit the power law function to the mean density data
    # bounds = ([1e-20, 1e-10], [np.inf, 3.])
    bounds = ([1e-20, 1e-10], [np.inf, 1e5])
    popt, _ = curve_fit(powerlaw_1_log10, radii, mean_density_log10, p0=[1.0, 1.5], bounds=bounds)
    
    # Plotting the fit result
    if savepath is None:
        savepath = "../data/examples/"
    if is_plot:
        plt.figure(figsize=(10, 6))
        mean_density_log10_fit_at = powerlaw_1_log10(radii, *popt)
        # plt.scatter(radii, 10.**mean_density_log10, label='Data', color='blue')
        # plt.plot(radii, 10.**mean_density_log10_fit_at, label='Fit: $h_D$={:.2f}, $D$={:.2f}'.format(*popt), color='red')
        # plt.xlabel('Radius')
        # plt.ylabel('Mean Density')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.scatter(np.log10(radii), mean_density_log10, label='Data', color='blue')
        plt.plot(np.log10(radii), mean_density_log10_fit_at, label='Fit: $h_D$={:.2f}, $D$={:.2f}'.format(*popt), color='red')
        plt.xlabel('log10 Radius')
        plt.ylabel('log10 Mean Density')
        plt.legend()
        plt.grid(True)
        # plt.title('Fractal Dimension Fit')
        # plt.show()
        plt.savefig("{}./fractal_dimension_fit_{}.eps".format(savepath, suffix), format="eps", bbox_inches='tight')
        plt.close()
    return popt

import warnings
def calculate_mean_neareast_count(pos, r_min=None, r_max=None, n_radii=None, suffix="suffix", rate_calculate=None):
    N = len(pos)
    mean_radius = ads.get_mean_radius(pos, is_center=True)
    if r_max is None:
        # r_max = mean_radius*0.2
        r_max = mean_radius*1. #the mean radius of system, which is larger than the radius when middle frac dim
    if r_min is None:
        # r_min = 1e-19
        # r_min = r_max/10.
        r_min = mean_radius*(9./2*np.pi/N)**(1./3)*0.5 #the average distance of particles
    if r_min>=r_max:
        print("The current r_min = {}, r_max = {}".format(r_min, r_max))
        warnings.warn("The r_min is larger than r_max. Set r_min to be default.")
        r_min = r_max/10.
    if n_radii is None:
        # n_radii = int(N_sample**(1./3))
        n_radii = 12
    pos_targets = None
    if rate_calculate is None:
        pos_targets = pos
    else: #only calulate N*rate_calculate particles for briefty
        # if not (rate_calculate>=0. and rate_calculate<1.-1e-20):
        if not (rate_calculate>1e-20 and rate_calculate<1.-1e-20):
            warnings.warn("The rate_calculate is not within [0, 1]. Set it to be default.")
            pos_targets = pos
        else:
            N_targets = int(N*rate_calculate)
            sample_indices = np.random.choice(N, size=N_targets, replace=False)
            pos_targets = pos[sample_indices]
    # ads.DEBUG_PRINT_V(1, np.shape(pos_targets), "np.shape(pos_targets)")

    radii, inside_counts = inside_count_with_radius(pos, pos_targets, r_min=r_min, r_max=r_max, n_radii=n_radii)
    # ads.DEBUG_PRINT_V(0, radii, inside_counts, "radii")
    popt = fit_fractal_dimension(radii, inside_counts, suffix=suffix)
    h_frac, Dim_frac = popt
    # print("Optimal Parameters: h_frac = {:.2f}, Dim_frac = {:.2f}".format(h_frac, Dim_frac))
    return h_frac, Dim_frac

def calculate_mean_neareast_count_load(radii, inside_counts, save_path, suffix="suffix", is_plot=True):
    # data = np.loadtxt(filename, dtype=float)
    # radii, inside_counts = data[:,0], data[:,1]
    # ads.DEBUG_PRINT_V(0, radii, inside_counts, "radii")
    popt = fit_fractal_dimension(radii, inside_counts, savepath=save_path, suffix=suffix, is_plot=is_plot)
    h_frac, Dim_frac = popt
    # print("Optimal Parameters: h_frac = {:.2f}, Dim_frac = {:.2f}".format(h_frac, Dim_frac))
    return h_frac, Dim_frac

def sample_maxwellian_isotropic(N, sigma):
    """
    Sample speed from a Maxwellian distribution.
    """
    # Equivalent to norm of 3 independent 1D Gaussians
    v = np.random.normal(0, sigma, size=(N, 3))
    return v

def g_func(v, k, v1):
    """Logistic blending function"""
    return 1.0 / (1.0 + np.exp(-k * (v - v1)))

def sample_velocity_isotropic_df(
    N, sigma, vc, gamma, k, v1, A_pl, epsilon=0.1, vmax=1000.0
    # N, sigma_iso=1.0, A=0.1, vc=3.0, gamma=4.5, vmax=1000.0
):
    def df_velocity(v): #fva_speed_MB_with_tail()
        # Gaussian core
        # f_G = 0.
        # f_G = v**2 * np.exp(-0.5 * (v / sigma_iso)**2)
        f_G = (2.0*np.pi * sigma**2)**(-1.5) * np.exp(-0.5 * (v/sigma)**2)
        
        # Power-law tail (note: translated by epsilon)
        # f_pl = 0.
        # f_pl = v**2 * A * (1 + (v / vc)**2)**(-gamma)
        f_pl = A_pl / vc**3 * ( epsilon + (v/vc)**2 )**(-gamma)
        
        # Blending function: logistic form
        # g = 0.5
        g = g_func(v, k, v1)
        
        # Combined model
        f_combined = 4.0*np.pi * v**2 * ( f_G * (1-g) + f_pl * g )
        
        return f_combined

    samples = []
    # f_max = df_velocity(np.sqrt(2) * sigma_iso)  # maximum occurs at sqrt(2)*sigma
    # vmax = vmax_factor*sigma
    v_grid = np.linspace(0, vmax, 1000)
    f_max = np.max(df_velocity(v_grid))

    while len(samples) < N:
        # Step 1: sample speed by rejection sampling
        v_candidate = np.random.uniform(0, vmax)
        u = np.random.uniform(0, f_max)
        if u > df_velocity(v_candidate):
            continue

        # Step 2: sample isotropic direction
        direction = np.random.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)

        velocity = v_candidate * direction
        samples.append(velocity)

    return np.array(samples)

def sample_isotropic_to_anisotropic(v, sigma_3=[3., 2., 1.]):
    N_samplepoints = len(v)
    v_aniso = np.zeros_like(v)
    for i in np.arange(N_samplepoints):
        # Generate isotropic direction
        direction = np.random.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)

        # Scale by anisotropic dispersions
        scaled_direction = np.array(sigma_3) * direction
        scaled_direction /= np.linalg.norm(scaled_direction)

        # Assign magnitude
        v_aniso[i] = v[i] * scaled_direction
    return v_aniso

def df_anisotropic(
    vx, vy, vz, sigma, vc, gamma, k, v1, A_pl, epsilon=0.1
    # vx, vy, vz, sigma, vc, A, gamma
):
    v = np.sqrt(vx*vx + vy*vy + vz*vz)
    v_fracto_sigma_square = (vx/sigma[0])**2 + (vy/sigma[1])**2 + (vz/sigma[2])**2
    v_fracto_vc_square = (vx/vc[0])**2 + (vy/vc[1])**2 + (vz/vc[2])**2
    sigma_multiply = sigma[0]*sigma[1]*sigma[2]
    vc_multiply = vc[0]*vc[1]*vc[2]
    # ads.DEBUG_PRINT_V(0, v, v_fracto_sigma_square, v_fracto_vc_square, "v")

    # Gaussian core
    # f_G = 0.
    # f_G = np.exp(-0.5*((vx/sigma[0])**2 + (vy/sigma[1])**2 + (vz/sigma[2])**2))
    f_G = (2.0*np.pi)**(-1.5) / sigma_multiply * np.exp(-0.5 * v_fracto_sigma_square)
    
    # Power-law tail (note: translated by epsilon)
    # f_pl = 0.
    # f_pl = A*(1 + (vx/vc[0])**2 + (vy/vc[1])**2 + (vz/vc[2])**2)**(-gamma)
    f_pl = A_pl / vc_multiply * ( epsilon + v_fracto_vc_square )**(-gamma)
    
    # Blending function: logistic form
    # g = 0.5
    g = g_func(v, k, v1)
    
    # Combined model
    f_combined = ( f_G * (1-g) + f_pl * g )
    # ads.DEBUG_PRINT_V(0, f_G, f_pl, g, f_combined, "f_combined")
    
    return f_combined

def sample_anisotropic_df(
    N, sigma, vc, gamma, k, v1, A_pl, epsilon=0.1, vmax_three=[1000., 800., 600.]
    # N, sigma=[1., 1., 1.], vc=[3., 3., 3], A=0.1, gamma=4.5, vmax_factor=5
):
    samples = []
    # f_max = df_anisotropic(0, 0, 0, sigma, vc, A, gamma)
    vmax = vmax_three
    # vmax = vmax_factor * np.array(sigma)
    f_max = df_anisotropic(0, 0, 0, sigma, vc, gamma, k, v1, A_pl, epsilon)

    while len(samples) < N:
        candidate = np.random.uniform(-vmax, vmax)
        # prob = df_anisotropic(*candidate, sigma, vc, A, gamma)
        prob = df_anisotropic(*candidate, sigma, vc, gamma, k, v1, A_pl, epsilon)
        # ads.DEBUG_PRINT_V(0, f_max, prob, "fmax")
        if np.random.uniform(0, f_max) <= prob:
            samples.append(candidate)
    return np.array(samples)

def generate_uniform_sample(
    N, Dim_frac=None, bound_shape="spherical"
    # N, Dim_frac=None, bound_shape="cylindrical"
):
    """
    Generate a sample of particle positions.

    Parameters:
    - N (int): The count of particles.
    - rs (float): The mean radius to which the particle distribution should be scaled.

    Returns:
    - samples (numpy.ndarray): An (N, 3) array representing the positions of particles.
    """
    # Generate spherical coordinates with uniform distribution
    samples = None
    rs = 1.

    if bound_shape == "spherical": #?? not uniform
        # r_vals = np.sqrt(np.random.uniform(0., rs**3, N))  # Radial distance (uniform in volume)
        r_vals = np.cbrt(np.random.uniform(0., rs**3, N))  # Radial distance (uniform in volume)
        theta_vals = np.arccos(np.random.uniform(-1, 1, N))  # Polar angle
        # theta_vals = np.random.uniform(0., np.pi, N)  # Polar angle
        phi_vals = np.random.uniform(0., 2.*np.pi, N)  # Azimuthal angle

        # Convert spherical coordinates to Cartesian coordinates
        x_vals = r_vals * np.sin(theta_vals) * np.cos(phi_vals)
        y_vals = r_vals * np.sin(theta_vals) * np.sin(phi_vals)
        z_vals = r_vals * np.cos(theta_vals)
        samples = np.vstack((x_vals, y_vals, z_vals)).T

    elif bound_shape == "cylindrical":
        r_vals = np.sqrt(np.random.uniform(0, rs**2, N))  # Radial distance (uniform in area)
        phi_vals = np.random.uniform(0, 2.*np.pi, N)  # Azimuthal angle
        z_vals = np.random.uniform(0, rs, N)  # Height (uniform in z)

        # Convert cylindrical coordinates to Cartesian coordinates
        x_vals = r_vals * np.cos(phi_vals)
        y_vals = r_vals * np.sin(phi_vals)
        samples = np.vstack((x_vals, y_vals, z_vals)).T

    else: #cubic shape with length rs*2.
        samples = np.random.uniform(0, rs*2., (N, 3))

    return samples

def generate_samples_from_file(filename):
    samples = np.loadtxt(filename)
    return samples

def load_samples_from_file(file_path, N_particles):
    samples = np.loadtxt(file_path)
    return samples[0:N_particles, 0:3]

def generate_samples_each_steps(
    N_particles, Dim_frac_setting, mean_radius_setting, func_list, is_plot=True
):
    sample_list = []
    Dim_frac_list = []
    for (i, func) in enumerate(func_list):
        print("generating ...")
        sample_name = func[2]
        if func[3]:
            sample = func[0](func[1])[:, 0:3]
        else:
            sample = func[0](N_particles, Dim_frac_setting)
        # ads.DEBUG_PRINT_V(1, sample[0], "samples")
        sample = ads.readjust_positions(sample, mean_radius_setting)
        Dim_frac_calculated = None
        print("calculaitng dim ...")
        # r_max = mean_radius_setting*1.
        # r_max = mean_radius_setting*0.5
        r_max = mean_radius_setting*0.25
        # r_max = mean_radius_setting*0.1
        r_min = None
        # r_min = r_max*0.1
        h_frac, Dim_frac_calculated = calculate_mean_neareast_count(sample, r_min=r_min, r_max=r_max, suffix=sample_name)
        # h_frac, Dim_frac_calculated = compute_correlation_dimension_volume(sample, r_min=r_min, r_max=r_max)
        print("Optimal Parameters: h_frac = {:.2f}, Dim_frac = {:.2f}".format(h_frac, Dim_frac_calculated))
        print("plotting ...")
        if is_plot:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=0.1, alpha=0.6)
            ax.set_title("generated sample: Dim_frac_calculated={}".format(Dim_frac_calculated))
            plt.savefig("../data/small/generated_sample_{}.png".format(sample_name), format="png", bbox_inches="tight")
            plt.close()
        sample_list.append(sample)
        Dim_frac_list.append(Dim_frac_calculated)
        ads.print_sample_info(sample, sample_name)
    return sample_list, Dim_frac_list



#In[] main
if __name__ == "__main__":

    ## Step 1. Settings
    N_particles = 10000
    Dim_frac = 1.8
    # Dim_frac = 2.8
    # Dim_frac = 20.
    mean_radius_setting = 50.
    filename1 = "../data/samples_simulated/uniform.txt"
    filename2 = "../data/samples_simulated/noised.txt"
    func_list = [
        # [generate_uniform_sample, None, "generate_uniform_sample", 0], 
        # [gfsv.generate_random_walk, None, "generate_random_walk", 0], 
        # [gfsv.generate_biased_random_walk, None, "generate_biased_random_walk", 0], 
        # [gfsv.generate_random_walk_from_uniform, None, "generate_random_walk_from_uniform", 0], 
        [generate_samples_from_file, filename1, "generate_samples_filename1", 1], 
        [generate_samples_from_file, filename2, "generate_samples_filename2", 1], 
    ]
    sample_list, Dim_frac_list = generate_samples_each_steps(N_particles, Dim_frac, mean_radius_setting, func_list)



    # ## Step 2. generate fractal sample by vaious method
    # pos_1 = generate_fractal_3D_vertex(N_particles, Dim_frac)
    
    # fig = plt.figure(figsize=(10, 8))  
    # ax = fig.add_subplot(111, projection='3d')  
    # ax.scatter(pos_1[:, 0], pos_1[:, 1], pos_1[:, 2], s=0.1, alpha=0.6)
    # ax.set_title('3D Fractal Distribution (Dimension ~ 1.8)')
    # plt.savefig("../data/small/fractal_sample_vertex.png", format="png", bbox_inches="tight")



    # pos_2 = generate_fractal_3D_walk(N_particles, Dim_frac)  

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(pos_2[:, 0], pos_2[:, 1], pos_2[:, 2], marker='o', markersize=1, alpha=0.6)  
    # ax.set_title(f"3D Fractal Distribution with Fractal Dimension ~ 1.8")  
    # ax.set_xlabel('X-axis')  
    # ax.set_ylabel('Y-axis')  
    # ax.set_zlabel('Z-axis')  
    # plt.savefig("../data/small/fractal_sample_walk_exp.png", format="png", bbox_inches="tight")



    # pos_3 = generate_uniform_sample_with_shape(
    #     mean_radius_setting, N_particles, bound_shape="cylindrical"
    # )
    # pos_3 = particles_3d_Perlin_motion_factor_scale(
    #     pos_3, N_iter=30, scale=0.1, step_size=2., 
    #     is_plot=False, 
    #     is_plot_witin_bound=False, suffix="pos_3"
    # )



    # ## Step 3. estimate fractal dimension
    # rm1 = ads.get_mean_radius(pos_1)
    # rm2 = ads.get_mean_radius(pos_2)
    # # rate_rm = 0.1
    # # rate_rm = 1.
    # rate_rm = 2.
    # ads.DEBUG_PRINT_V(1, np.shape(pos_1), np.shape(pos_2), "shape")
    # ads.DEBUG_PRINT_V(1, rm1, rm2, "rm")

    # max_radius = rm1*rate_rm
    # # fractal_dim_estimate = compute_correlation_dimension(pos_1, max_radius)  
    # _, fractal_dim_estimate = compute_correlation_dimension_volume(pos_1, max_radius)  
    # print(f"Estimated Fractal Dimension: {fractal_dim_estimate:.2f}")  

    # max_radius = rm2*rate_rm
    # # fractal_dim_estimate = compute_correlation_dimension(pos_2, max_radius)
    # _, fractal_dim_estimate = compute_correlation_dimension_volume(pos_2, max_radius)
    # print(f"Estimated Fractal Dimension: {fractal_dim_estimate:.2f}")
