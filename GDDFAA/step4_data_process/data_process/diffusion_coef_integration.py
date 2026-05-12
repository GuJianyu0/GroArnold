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
import generate_fractal_sample as gfs

#In[] settings
Dim = 3
col_actions = 78
col_frequencies = col_actions+7
# galaxy_name = sys.argv[1]

G = 43007.1      # Gravitational constant (kpc, km/s, 1 M_sun)
frac_mass = 1.
# frac_mass = 0.05
M_total_gal_1e10MSun = 137.
N_total_MW = 1e11         # stars count
M_total_MW_b = 1.37e12*frac_mass # total mass of stars (1 M_sun)
Rs = 20.0        # scale length (kpc)
ma = M_total_gal_1e10MSun/N_total_MW         # Mass (arbitrary units) #??
km = 1.0         # mass DF adjustmant
v0 = 230.0       # Initial velocity (km/s) #??
Ek_mean = 10000.
t_during = 0.25  # Time (Gyr)
Vol0 = 4.*np.pi/3.*(Rs*2.)**3 #considered total volume
R_plus = 20.0*2    # Upper bound for R' (kpc)
R_minus =  1e-3    # Lower bound for R' (kpc)
# R_minus =  R_plus/N    # Lower bound for R' (kpc)

#In[] functions
## Data
def load_obs_Zhu():
    # in kpc/Myr
    # read from Huang+2020
    filename = "../dependancies/JeansTest-MW-main/JeansTestPy/binned_results.npz"
    ff = np.load(filename)
    R_bin=ff["R_bin"]
    z_bin=np.abs(ff["z_bin"])
    count=ff["count"]
    v_R_bin=ff["v_R_bin"]
    v_phi_bin=np.abs(ff["v_phi_bin"])
    v_z_bin=ff["v_z_bin"]
    sigma_R_bin=ff["sigma_R_bin"]
    sigma_phi_bin=ff["sigma_phi_bin"]
    sigma_z_bin=ff["sigma_z_bin"]

    try:
        Ev_R_bin=ff["Ev_R_bin"]
        Ev_phi_bin=np.abs(ff["Ev_phi_bin"])
        Ev_z_bin=ff["Ev_z_bin"]
        Esigma_R_bin=ff["Esigma_R_bin"]
        Esigma_phi_bin=ff["Esigma_phi_bin"]
        Esigma_z_bin=ff["Esigma_z_bin"]
    except:
        print("no err")

        R_bin=R_bin.ravel()
        z_bin=z_bin.ravel()
        count=count.ravel()
        v_R_bin=v_R_bin.ravel()
        v_phi_bin=v_phi_bin.ravel()
        v_z_bin=v_z_bin.ravel()
        sigma_R_bin=sigma_R_bin.ravel()
        sigma_phi_bin=sigma_phi_bin.ravel()
        sigma_z_bin=sigma_z_bin.ravel()

        Ev_R_bin=np.zeros(len(v_R_bin))
        Ev_phi_bin=np.zeros(len(v_R_bin))
        Ev_z_bin=np.zeros(len(v_R_bin))
        Esigma_R_bin=np.zeros(len(v_R_bin))
        Esigma_phi_bin=np.zeros(len(v_R_bin))
        Esigma_z_bin=np.zeros(len(v_R_bin))

    return R_bin, z_bin, v_R_bin, v_phi_bin, v_z_bin, sigma_R_bin, sigma_phi_bin, sigma_z_bin

def beta_velocity_dispersion_deviation(sigma_R_bin, sigma_phi_bin, sigma_z_bin):
    # sigma_t_bin2 = (sigma_R_bin**2+sigma_phi_bin**2)/2
    sigma_t_bin2 = ( (sigma_R_bin+sigma_phi_bin)/2 )**2
    sigma_z_bin2 = sigma_z_bin**2
    return (sigma_z_bin2-sigma_t_bin2)/sigma_z_bin2

## Define noise function
def n_noise_nothing(R, phi, z):
    return 0.

def n_noise_sin(R, phi, z):
    ampl = 0.1
    # ampl = 0.5
    # ampl = 5.
    # period = 5. #kpc
    # period = 1. #kpc
    period = 0.1 #kpc
    k = 2.*np.pi/period

    # noise = ampl*(k*R)**-1.7 #rate 1.6
    noise = ampl*np.sin(k*R) #rate 1.01
    # noise = ampl*np.sin(k*R**2) #rate 1.01
    # noise = ampl*np.sin(k*(R+0.*period)*np.cos(phi))*np.sin(k*(R+0.*period)*np.sin(phi))*np.sin(k*(z+0.*period)) #rate 1.0
    # noise = ampl*np.sin(k*(R+0.1*period)*np.cos(phi))*np.sin(k*(R+0.5*period)*np.sin(phi))*np.sin(k*(z+0.5*period)) #rate 1.0
    return noise

def n_noise_sinrandom(R, phi, z):
    N_noise_comp = 5  # Number of noise components
    A_noise_comp = [0.02] * N_noise_comp  # Amplitudes
    k_R = [1.0, 2.0, 3.0, 4.0, 5.0]
    k_phi = [0.5, 1.0, 1.5, 2.0, 2.5]
    k_z = [1.0, 1.5, 2.0, 2.5, 3.0]
    phi_phase = np.random.uniform(0, 2.*np.pi, N_noise_comp)

    noise = 0.0
    for i in range(N_noise_comp):
        noise += A_noise_comp[i] * np.sin(k_R[i] * R + k_phi[i] * phi + k_z[i] * z + phi_phase[i])
    return noise

# Use Perlin noise for smoother noise generation
def n_noise_perlin(R, phi, z):
    '''
    Arguments of noise.pnoise():
    scale: Controls the size of features in the noise. A smaller scale means larger, smoother features, while a larger scale makes the features smaller and more detailed.
    octaves: Controls the number of levels of detail (frequency layers) added to the noise. More octaves mean more detail and complexity, at the cost of increased computation time.
    persistence: Determines the amplitude of each successive octave. Lower persistence means that higher-frequency octaves contribute less to the final noise.
    lacunarity: Controls the frequency multiplier between successive octaves. Higher lacunarity means that each successive octave will have a higher frequency compared to the previous one.
    '''
    # Convert cylindrical to Cartesian coordinates
    x = R * np.cos(phi)
    y = R * np.sin(phi)

    # Perlin noise parameters
    # scale = 0.1  # Adjust scale to control noise characteristics
    scale = 0.5  # Adjust scale to control noise characteristics
    # scale = 1.0  # Adjust scale to control noise characteristics
    noise_value = pnoise3(x * scale, y * scale, z * scale)

    # Normalization step to ensure integrated value matches n0
    noise_amplitude = 0.1  # 10% of n0
    noise_value_normalized = noise_amplitude * (noise_value - 
        np.mean([pnoise3(x * scale, y * scale, z * scale) 
        for x, y, z in zip(np.random.uniform(-Rs, Rs, 1000), 
        np.random.uniform(-Rs, Rs, 1000), 
        np.random.uniform(0, v0 * t_during, 1000))]))

    ads.DEBUG_PRINT_V(1, noise_amplitude)
    return n0 + noise_value_normalized

# Total distribution function with noise
def n_f(R, phi, z, n_noise):
    return n0 + n_noise(R, phi, z)

## diffusion coef
# diffusion coef estimate by distant encounters
def diffusion_distant_encounters_formula(n0):
    t_during = 1.
    I = 2.*np.pi * v0 * t_during * n0 * np.log(R_plus/R_minus)
    D_diffu_value = (4. * G**2 * km * ma**2) / (v0**2 * t_during) * I
    return D_diffu_value

def diffusion_distant_encounters_uniform_truncated(
    N_particles, M_total, r_mean, v_mean, R_minus=0.001, is_3d_cross=False
):
    R_m = R_minus
    R_p = r_mean*2.
    v0 = v_mean
    n0 = N_particles / (2.*np.pi * R_p**2 * v0 * t_during) #using cylindrical
    m = M_total/N_particles
    Aplha = np.log(R_p/R_m)
    D_diffu_value = 8.*np.pi * G**2*m**2 / v0 * n0 * Aplha
    return D_diffu_value

def diffusion_distant_encounters_uniform_bounded(
    N_particles, M_total, r_mean, v_mean, R_minus=0.001, is_3d_cross=False
):
    R_m = R_minus
    R_p = r_mean*2. #note: it should consist with descrete diffusion
    v0 = v_mean
    n0 = N_particles / (2.*np.pi * R_p**2 * v0 * t_during) #using cylindrical
    m = M_total/N_particles
    Aplha = np.log(R_p/R_m)
    D_diffu_value = 8.*np.pi * G**2*m**2 / v0 * n0 * Aplha
    return D_diffu_value

# Define the diffusion coefficient function
def diffusion_pos_homog_quad(n0, n_noise):
    def integrand(z, phi, R):
        return n0 * ( 1. + n_noise(R, phi, z) ) / R

    subdv = 100
    # Triple integral over R, phi, z
    integral, error = tplquad(integrand, 
        R_minus, R_plus,  # Integration bounds for R
        lambda R: 0, lambda R: 2 * np.pi,  # Integration bounds for phi
        lambda R, phi: 0, lambda R, phi: v0 * t_during #, 
        # subdivisions=[subdv, subdv, subdv]
    )  # Integration bounds for z

    # Calculate D_diffu
    D_diffu_value = (4. * G**2 * km * ma**2) / (v0**2 * t_during) * integral
    return D_diffu_value, integral, error

# Monte Carlo integration for the diffusion coefficient
def diffusion_pos_homog_monte_carlo(n0, n_noise, num_samples=100000):
    def n_total(R, phi, z):
        return n0 * ( 1. + n_noise(R, phi, z) ) / R

    # x_samples = np.random.uniform(R_minus, R_plus, num_samples)
    # y_samples = np.random.uniform(R_minus, R_plus, num_samples)
    # z_samples = np.random.uniform(0, v0 * t_during, num_samples)
    # R_samples = np.hypot(x_samples, y_samples)
    # phi_samples = np.arctan2(y_samples, x_samples)
    # z_samples = np.random.uniform(0, v0 * t_during, num_samples)
    
    R_samples = np.random.uniform(R_minus, R_plus, num_samples)
    phi_samples = np.random.uniform(0, 2 * np.pi, num_samples)
    z_samples = np.random.uniform(0, v0 * t_during, num_samples)
    
    # Compute the integrand for each sample
    integrand_values = []
    for R, phi, z in zip(R_samples, phi_samples, z_samples):
        integrand_values.append(n_total(R, phi, z))
    
    # integrand_values = np.zeros(num_samples)
    # i_itg = 0
    # for R, phi, z in zip(R_samples, phi_samples, z_samples):
    #     integrand_values[i_itg] = n_total(R, phi, z)
    #     i_itg += 1
    
    # Average the integrand values to estimate the integral
    volume = (R_plus - R_minus) * (2 * np.pi) * (v0 * t_during)
    integral_estimate = np.mean(integrand_values) * volume
    
    # Calculate the diffusion coefficient
    D_diffu_value = (4 * G**2 * km * ma**2) / (v0**2 * t_during) * integral_estimate
    return D_diffu_value

# Monte Carlo integration for the diffusion coefficient
def normalization_pos_homog_vegas(n0, n_noise, num_samples=100000):
    # Function to be integrated
    def integrand(x):
        R, phi, z = x[0], x[1], x[2]
        return n0 * (1. + n_noise(R, phi, z))

    # Setting up the integration domain
    integral_domain = [[R_minus, R_plus], [0, 2 * np.pi], [0, v0 * t_during]]

    # Vegas Monte Carlo integration
    import vegas
    integrator = vegas.Integrator(integral_domain)

    # Perform the integration
    result = integrator(integrand, nitn=10, neval=num_samples)
    return result
    # D_diffu_value = (4 * G**2 * km * ma**2) / (v0**2 * t_during) * result.mean
    # return D_diffu_value

# Vegas Monte Carlo integration for the diffusion coefficient
def diffusion_pos_homog_vegas(n0, n_noise, num_samples=100000):
    # Function to be integrated
    def integrand(x):
        R, phi, z = x[0], x[1], x[2]
        return n0 * (1. + n_noise(R, phi, z)) / R

    # Setting up the integration domain
    integral_domain = [[R_minus, R_plus], [0, 2 * np.pi], [0, v0 * t_during]]

    # Vegas Monte Carlo integration
    integrator = vegas.Integrator(integral_domain)

    # Perform the integration
    result = integrator(integrand, nitn=10, neval=num_samples)
    D_diffu_value = (4 * G**2 * km * ma**2) / (v0**2 * t_during) * result.mean
    return D_diffu_value

def particles_3d_Perlin_motion(N_particles, R, initial_positions_=None):
    # Step 1: Define initial set of N particles with uniform distribution in 3D space
    initial_positions = None
    if initial_positions_ is not None:
        initial_positions = initial_positions_
    else:
        N = N_particles  # Number of particles
        space_size = R  # Size of the cubic space (10*10*10 units)
        z_m = space_size/2.

        # Generate cylindrical coordinates with uniform distribution
        np.random.seed(65065)
        # initial_positions = np.random.uniform(0, space_size, (N, 3))  # Uniform distribution in a cube of size 10
        r_vals = np.sqrt(np.random.uniform(0, R_plus**2, N))  # Radial distance (uniform in area)
        phi_vals = np.random.uniform(0, 2 * np.pi, N)  # Azimuthal angle
        z_vals = np.random.uniform(0, z_m, N)  # Height (uniform in z)

        # Convert cylindrical coordinates to Cartesian coordinates
        x_vals = r_vals * np.cos(phi_vals)
        y_vals = r_vals * np.sin(phi_vals)
        initial_positions = np.vstack((x_vals, y_vals, z_vals)).T
        ads.DEBUG_PRINT_V(1, initial_positions[0], "initial_positions[0]")

    # Step 2: Generate Perlin noise directions to simulate Brownian or Levy motion
    def perlin_motion(positions, scale=0.1, octaves=4, step_size=0.5):
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

    # Apply the modified motion function
    # scale = 0.02 #rate < 1. 
    # scale = 0.05 #rate < 1. 
    scale = 0.1 #rate < 1.
    # scale = 0.2 #rate > 1.
    # scale = 0.5 #rate < 1.
    # scale = 1. #rate < 1.
    # scale = 5. #rate
    # step_size = 0.1
    # step_size = 0.5
    step_size = 1.
    # step_size = 5.
    # step_size = 20.
    new_positions = initial_positions
    N_iter = 10
    # N_iter = 50
    for i in range(N_iter):
        new_positions = perlin_motion(new_positions, scale=scale, octaves=4, step_size=step_size)

    # # Step 4: Plot the initial and new positions of particles
    # fig = plt.figure(figsize=(12, 6))
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2], c='blue', s=5, alpha=0.5)
    # ax1.set_title("Initial Positions of Particles")
    # # ax1.set_xlim([0, space_size])
    # # ax1.set_ylim([0, space_size])
    # # ax1.set_zlim([0, space_size])

    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.scatter(new_positions[:, 0], new_positions[:, 1], new_positions[:, 2], c='red', s=5, alpha=0.5)
    # ax2.set_title("New Positions of Particles (After Perlin Noise Motion)")
    # # ax2.set_xlim([0, space_size])
    # # ax2.set_ylim([0, space_size])
    # # ax2.set_zlim([0, space_size])

    # plt.tight_layout()
    # # plt.show()
    # plt.savefig("../data/Perlin_motion_xyz.png", format="png", bbox_inches='tight')
    
    # Step 5: Plot the initial and new positions' (x, y) coordinates on 2D figure
    # plt.figure(figsize=(10, 5))
    # # plt.scatter(initial_positions[:, 0], initial_positions[:, 1], c='blue', s=5, alpha=0.5, label='Initial Positions')
    # plt.scatter(new_positions[:, 0], new_positions[:, 1], c='blue', s=5, alpha=0.5, label='New Positions')
    # # plt.scatter(new_positions[:, 0], new_positions[:, 1], c='red', s=5, alpha=0.5, label='New Positions')
    # plt.title("Initial and New Positions of Particles (x, y) Coordinates")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # # plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], c='blue', s=5, alpha=0.5)
    ax1.set_title("Initial Positions of Particles (x, y) Coordinates")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim([-R_size, R_size])
    ax1.set_ylim([-R_size, R_size])
    ax1.grid(True)
    ax2.scatter(new_positions[:, 0], new_positions[:, 1], c='red', s=5, alpha=0.5)
    ax2.set_title("New Positions of Particles (x, y) Coordinates")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim([-R_size, R_size])
    ax2.set_ylim([-R_size, R_size])
    ax2.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/Perlin_motion_xy.png", format="png", bbox_inches='tight')

    return initial_positions, new_positions

def generate_fractal_sample_call():
    return 0

def diffusion_descrete_cyl_softenning(pos_init, pos_field, 
    mass_field, pos_target_=None, mass_target_=None, pos_target_to_center=None
):
    pos_target = pos_target_
    if pos_target_ is None:
        pos_target = np.mean(pos_field, axis=0)
    mass_target = mass_target_
    if mass_target_ is None:
        mass_target = np.mean(mass_field)
    pos = pos_field-pos_target #relative positions with center [0,0,0]
    
    # if pos_target_to_center is not None:
    #     pos -= pos_target_to_center #dens
    # pos -= [-6., -6., 0.] #dens
    # pos -= [6., 6., 0.] #dens

    pos -= [0., 0., 0.] #dens
    # pos -= [3., 0., 0.] #dens
    # pos -= [6., 0., 0.] #dens
    # pos -= [12.5, 0., 0.] #dens
    # pos -= [25., 0., 0.] #dens
    # pos -= [50., 0., 0.] #dens
    # pos -= [75., 0., 0.] #dens
    # pos -= [150., 0., 0.] #dens

    pos_center = np.mean(pos, axis=0)
    pos_xy1 = pos[:, 0:2]
    pos_z1 = pos[:, 2]

    # Rescale positions to match the initial volume (cylindrical boundary)
    distances_init = np.sqrt(pos_init[:, 0]**2 + pos_init[:, 1]**2)
    distances = np.sqrt(pos_xy1[:, 0]**2 + pos_xy1[:, 1]**2)
    scaling_factor = np.mean(distances)/np.mean(distances_init)
    # scaling_factor = 1. #debug: when scaling_factor decrease, the ampl rate increase, with not large influence
    pos_xy2 = pos_xy1/scaling_factor
    # pos_z2 = pos[:, 2]*1.

    R = np.linalg.norm(pos_xy2, axis=1)
    # R += R_plus*10. #debug: ?? far particles, less influence by translation and R_minus
    R[R<1.] = 1. #debug: ?? mask the nearest flucation
    R_soften = R_minus
    # R_soften = np.min(R)
    # ads.DEBUG_PRINT_V(0, R_minus, R_soften)

    k_mass = np.sum(mass_field**2)

    Integrand = ( R / (R**2+R_soften**2) )**2
    # Integrand = mass_field**2/(R+R_minus)**2
    # R_frac_b90 = R/R_minus
    # Integrand = ( (2*mass_field)/(mass_target+mass_field) * R_frac_b90/(1.+R_frac_b90**2) )**2
    I_Diffu = k_mass * np.sum(Integrand) #over 3d space
    ads.DEBUG_PRINT_V(1, pos_target, pos_center, scaling_factor, np.mean(R), "scaling_factor")
    return I_Diffu

def plot_mass_contour_postions_or_velocities(
    x_input, m_input=None, is_pos=True, 
    savename="./savefig/savename.png", 
    is_show=False
):
    #data
    x = x_input
    if m_input is None:
        m_input = 1.
    elif len(np.shape(m_input))==0:
        m_input = np.ones(len(x_input))*m_input

    xyunit = "kpc"
    if not is_pos:
        xyunit = "km/s"

    N_grid_x = 100
    N_grid_y = 80
    N_grid_z = 9
    bounds = np.zeros((3,2))
    for i in np.arange(3):
        # bd = np.max( np.abs(x[:,i]) )
        bd = np.percentile( np.abs(x[:,i]), 98. )
        bounds[i] = np.array([-bd, bd])
    grid_x = np.linspace(bounds[0][0], bounds[0][1], N_grid_x)
    grid_y = np.linspace(bounds[1][0], bounds[1][1], N_grid_y)
    grid_z = np.linspace(bounds[2][0], bounds[2][1], N_grid_z)
    # grid_z = grid_z**3/np.max(grid_z)**2
    # mg1, mg2, mg3 = np.meshgrid(grid_y, grid_x, grid_z) #??
    mg1, mg2, mg3 = np.meshgrid(grid_x, grid_y, grid_z)
    ads.DEBUG_PRINT_V(1, bounds, np.shape(mg1))
    rho = np.zeros_like(mg1)
    rho_activate = np.zeros_like(mg1)
    # rho_activate = np.random.random((N_grid_x, N_grid_y, N_grid_z))*-10.
    # rho_activate = np.log10 ( 1. / (mg1**2+mg2**2) )
    KD = kdtp.KDTree_galaxy_particles(x, weight_extern_instinct=m_input)
    for i in np.arange(N_grid_x):
        for j in np.arange(N_grid_y):
            for k in np.arange(N_grid_z):
                targets = [[grid_x[i], grid_y[j], grid_z[k]]]
                rho[j,i,k] = KD.density_SPH(targets)
    rho0 = KD.density_SPH([[0., 0., 0.]])
    # rho_activate = np.log10(rho/rho0)
    rho_activate = np.log10(rho)
    ads.DEBUG_PRINT_V(1, np.shape(rho_activate))

    #plot
    fontsize = 20.
    pointsize = 0.2
    # figsize = 46, 16 #for 4, 2
    figsize = 20, 15 #for 3, 3
    dpi = 400
    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = None
    # projection = "3d"
    levels = np.linspace(-8., 0., 16) #same range
    contour1 = None
    for k in np.arange(N_grid_z):
        ax = fig.add_subplot(3, 3, k+1, projection=projection)
        import matplotlib.colors
        normc = matplotlib.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        contour1 = plt.contourf(mg1[:,:,k], mg2[:,:,k], rho_activate[:,:,k], cmap="viridis", 
            levels=levels
            # norm=normc
        )
        cbar = plt.colorbar(contour1)
        cbar.set_label(r"$\log_{10}(\rho)$", fontsize=fontsize)
        ax.set_aspect("equal")
        ax.set_title("z = %f $\mathrm{%s}$"%(grid_z[k], xyunit), fontsize=fontsize)
        if 1: #k>=4 and k<=7:
            ax.set_xlabel(r"$x$ ($\mathrm{%s}$)"%(xyunit), fontsize=fontsize)
        if 1: #k==0 or k==4:
            ax.set_ylabel(r"$y$ ($\mathrm{%s}$)"%(xyunit), fontsize=fontsize)
        # ax.legend(fontsize=fontsize, loc=0)
        ax.tick_params(labelsize=fontsize)

    cax = fig.add_axes([0.95, 0.4, 0.015, 0.3]) #same colorbar
    cbar1 = fig.colorbar(contour1, cax=cax) #, orientation='horizontal'
    cbar1.set_label(r"$\log_{10}(\rho)$ ($\mathrm{1e10\,M_\mathrm{Sun}\, %s^{-3}}$)"%(xyunit), fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    # plt.tight_layout() #to let them not cover each other, auto
    fig_tmp = plt.gcf()
    # fig_tmp.savefig(savename+".eps", format="eps", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    fig_tmp.savefig(savename+".png", format="png", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    if is_show==True:
        plt.show()
    plt.close("all")

    rho_flatten = np.array(rho_activate).flatten()
    dens_mean_grid = np.mean(rho_flatten)
    dens_fluc_grid = np.std(rho_flatten)
    return dens_mean_grid, dens_fluc_grid

def generate_initial_positions_crystal_sphere(mean_radius, N_particles, lattice_constant=1., crystal_type="FCC"):
    """
    Generate particles positioned like a crystal lattice inside a spherical shape.

    Arguments:
    - num_particles: int, approximate number of particles.
    - lattice_constant: float, distance between adjacent particles in the lattice.
    - max_radius: float, the maximum radius of the spherical crystal.

    Returns:
    - positions: (N, 3)-array, positions of particles in 3D space.
    """
    def generate_crystal_lattice_sphere(num_particles, lattice_constant, max_radius):
        # Step 1: Generate a crystal lattice (FCC in this example)
        lattice_points = []
        half_max_radius = max_radius / 2
        range_limit = int(half_max_radius / lattice_constant)

        # Define FCC lattice points in the unit cell
        fcc_offsets = [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0]
        ]
        fcc_offsets = np.array(fcc_offsets) * lattice_constant

        # Create lattice points in 3D space
        for x in range(-range_limit, range_limit + 1):
            for y in range(-range_limit, range_limit + 1):
                for z in range(-range_limit, range_limit + 1):
                    origin = np.array([x, y, z]) * lattice_constant
                    for offset in fcc_offsets:
                        lattice_points.append(origin + offset)

        lattice_points = np.array(lattice_points)

        # Step 2: Filter lattice points to create a spherical shape
        distances = np.linalg.norm(lattice_points, axis=1)
        sphere_mask = distances <= max_radius
        spherical_points = lattice_points[sphere_mask]

        # Step 3: Ensure the total number of particles is approximately num_particles
        if len(spherical_points) < num_particles:
            # raise ValueError("Not enough lattice points to generate the desired number of particles. Increase max_radius or reduce lattice_constant.")
            print("Not enough lattice points to generate the desired number of particles. Increase max_radius or reduce lattice_constant.")
            lattice_constant /= 10.
            return generate_crystal_lattice_sphere(num_particles, lattice_constant, max_radius)

        np.random.shuffle(spherical_points)
        spherical_points = spherical_points[:num_particles]

        # Step 4: Resize the crystal to fit within the mean_radius while ensuring the particle count
        spherical_points -= np.mean(spherical_points, axis=0)
        r_points = np.linalg.norm(spherical_points, axis=1)
        mean_distance = np.mean(np.linalg.norm(spherical_points, axis=1))
        scaling_factor = mean_radius / mean_distance
        spherical_points *= scaling_factor
        # ads.DEBUG_PRINT_V(1, np.shape(spherical_points), np.mean(r_points), np.max(r_points), "crystal mean radius")
        return spherical_points

    # Generate crystal lattice positions in a sphere
    num_particles = N_particles
    positions = generate_crystal_lattice_sphere(num_particles, lattice_constant, mean_radius)

    # Plotting the positions in 3D space
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Crystal Lattice with Spherical Shape')
    plt.savefig('../data/crystal_lattice_sphere.png')
    plt.show()
    return positions

def generate_initial_positions_uniform(cylindrical_mean_radius, N_particles):
    # Settings of distribution
    N = N_particles  # Number of particles
    R_m = cylindrical_mean_radius  # Size of the cubic space (10*10*10 units)
    # z_m = R_m/2.
    z_m = R_m/1.
    rs = cylindrical_mean_radius

    # Generate cylindrical coordinates with uniform distribution
    np.random.seed(65065)
    # np.random.seed(11)

    # # initial_positions = np.random.uniform(0, R_m, (N, 3))  # Uniform distribution in a cube of size 10
    # r_vals = np.sqrt(np.random.uniform(0, R_m**2, N))  # Radial distance (uniform in area)
    # phi_vals = np.random.uniform(0, 2.*np.pi, N)  # Azimuthal angle
    # # phi_vals = np.random.uniform(0, 2 * np.pi, N)  # Azimuthal angle
    # z_vals = np.random.uniform(0, z_m, N)  # Height (uniform in z)

    # # Convert cylindrical coordinates to Cartesian coordinates
    # x_vals = r_vals * np.cos(phi_vals)
    # y_vals = r_vals * np.sin(phi_vals)
    # initial_positions = np.vstack((x_vals, y_vals, z_vals)).T

    r_vals = np.sqrt(np.random.uniform(0, rs**2, N))  # Radial distance (uniform in area)
    phi_vals = np.random.uniform(0, 2.*np.pi, N)  # Azimuthal angle
    z_vals = np.random.uniform(0, rs, N)  # Height (uniform in z)

    # Convert cylindrical coordinates to Cartesian coordinates
    x_vals = r_vals * np.cos(phi_vals)
    y_vals = r_vals * np.sin(phi_vals)
    samples = np.vstack((x_vals, y_vals, z_vals)).T

    # Move samples to the mass center (mean of the positions to be at the origin)
    mass_center = np.mean(samples, axis=0)
    samples -= mass_center

    # Rescale the mean radius of samples to rs
    mean_distance = np.mean(np.linalg.norm(samples, axis=1))
    scaling_factor = cylindrical_mean_radius / mean_distance
    samples *= scaling_factor
    return samples

def generate_particle_positions(rs, N, bound_shape="spherical"):
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

def generate_vel_DF_Gaussian(
    N, velmean_xyz=np.array([0., 0., 0.]), sigma_xyz=1., mean_radius=None, vel_bound=None
):
    # Generate 3D Gaussian samples
    samples = np.random.normal(velmean_xyz, sigma_xyz, (N, 3))
    # mass_center = np.mean(samples, axis=0)
    # samples -= mass_center
    if vel_bound is not None:
        samples[samples<vel_bound[0]] = vel_bound[0]
        samples[samples>vel_bound[1]] = vel_bound[1]

    # Rescale the mean radius of samples to rs
    if mean_radius is not None:
        current_mean_radius = np.mean(np.linalg.norm(samples, axis=1))
        scaling_factor = mean_radius / current_mean_radius
        samples *= scaling_factor
    return samples

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

def particles_3d_Perlin_motion_factor(
    initial_positions, N_iter=10, is_same_mean_radius=True, is_plot=True, 
    is_plot_witin_bound=False, suffix="suffix"
):
    # Step 1: Generate Perlin noise directions to simulate Brownian or Levy motion
    # scale = 0.02 #rate < 1.
    # scale = 0.05 #rate < 1.
    scale = 0.1 #rate < 1.
    # scale = 0.2 #rate > 1.
    # scale = 0.5 #rate < 1.
    # scale = 1. #rate < 1.
    # scale = 5. #rate
    # step_size = 0.1
    # step_size = 0.5
    step_size = 1.
    # step_size = 5.
    # step_size = 20.
    new_positions = initial_positions*1.
    # # N_iter = 10
    # N_iter = 50
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
        ax1.set_title("initial")
        ax1.set_xlabel("axis1")
        ax1.set_ylabel("axis2")
        if is_plot_witin_bound:
            ax1.set_xlim([-R_init_mean, R_init_mean])
            ax1.set_ylim([-R_init_mean, R_init_mean])
        ax1.grid(True)
        ax2.scatter(new_positions[:, 0], new_positions[:, 1], c='red', s=5, alpha=0.5)
        ax2.set_title("Perlin noise motion, N_iter={}".format(N_iter))
        ax2.set_xlabel("axis1")
        ax2.set_ylabel("axis2")
        if is_plot_witin_bound:
            ax2.set_xlim([-R_init_mean, R_init_mean])
            ax2.set_ylim([-R_init_mean, R_init_mean])
        ax2.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig("../data/examples_pos/Perlin_motion_xy_"+suffix+".png", format="png", bbox_inches='tight')
    return new_positions

def diffusion_descrete_cyl_softenning_mean( #diffu each
    pos_field, mass_field=None, M_total=None, r_mean=None, v_mean=None, 
    R_minus=0.001, R_local_bound_count=None, 
    is_3d_cross=True, is_save=True, suffix="suffix"
):
    N_particles = len(pos_field)
    radius_to_center = np.zeros(N_particles) #a vector for mean distance to other particles
    mean_distance_with_softenning = np.zeros(N_particles) #a vector for mean distance to other particles
    min_distance_with_softenning = np.zeros(N_particles) #a vector for mean distance to other particles
    Diffu_mainpart = np.zeros(N_particles) #a vector for diffusion coef of each particle
    pos = pos_field-np.mean(pos_field, axis=0) #translate to mass center
    pos_cross = pos
    # pos_cross = None
    # if is_3d_cross:
    #     pos_cross = pos #cross by relative r_xyz, roughly
    # else:
    #     pos_cross = pos[:, 0:2] #cross by relative R_xy, need velocities of targets
    
    if M_total is None:
        M_total = 1.
    if r_mean is None:
        r_mean = get_mean_radius(pos)
    if v_mean is None:
        v_mean = np.sqrt(G*M_total/frac_mass/r_mean)
    # m = 1. #debug diffu by particles count
    m = M_total/N_particles
    lambda1 = 3.
    lambda2 = 0.2
    # b90 = 2.*G*m/v_mean**2
    b90 = r_mean / (lambda2 * N_particles)
    coef_const = lambda1*G**2*m**2/(v_mean*r_mean) #using cylindrical
    # ads.DEBUG_PRINT_V(0, b90, coef_const, "consts")
    # coef_const = 10000/N_particles #debug: sampling
    # ads.DEBUG_PRINT_V(0, M_total, N_particles, m, G, r_mean, v_mean, coef_const)

    #: Alter (1) for each particle
    # for i in np.arange(10): #debug
    for i in np.arange(N_particles):
        pos_xy_some = pos_cross - pos_cross[i]
        pos_xy_some = np.delete(pos_xy_some, i, axis=0) #remove the index that the particle itself, i.e. i=j case
        # ads.DEBUG_PRINT_V(
        #     1, 
        #     i, len(pos_xy_some), np.shape(pos_xy_some), 
        #     np.where(ads.norm_l(pos_xy_some, axis=1)<1e-20)[0], 
        #     np.min(ads.norm_l(pos_xy_some, axis=1)), 
        #     "mindist"
        # )
        # if i>=10:
        #     exit(0)

    # #: Alter (2) for each grid
    # N_grid_diffu = 10
    # pos_xy_grid = np.zeros((N_grid_diffu*N_grid_diffu, 2))
    # bound_grid_diffu = np.mean( ads.norm_l(pos, axis=1) )*2.
    # k_grid = 0
    # for i in np.arange(N_grid_diffu):
    #     for j in np.arange(N_grid_diffu):
    #         xg = bound_grid_diffu*i/N_grid_diffu
    #         yg = bound_grid_diffu*j/N_grid_diffu
    #         pos_xy_grid[k_grid] = np.array([xg, yg])
    #         k_grid += 1
    # for i in np.arange(N_grid_diffu*N_grid_diffu):
    #     pos_xy_some = pos_xy - pos_xy_grid[i]

    #: end alter, start to loop
        R = None
        pos_xy_some_part = None
        if R_local_bound_count is not None:
            tree = KDTree(pos_xy_some, leaf_size=40) #default to be Euclidean in the metric space
            distances, indices = tree.query([[0., 0., 0.]], k=R_local_bound_count)
            pos_xy_some_part = pos_xy_some[indices[0]]
            # pos_xy_some_part = screen_radius_samples(pos_xy_some, radius_bound_up=R_local_bound)[0] #r_mean*0.5
            ads.DEBUG_PRINT_V(0, np.shape(indices), np.shape(pos_xy_some), np.shape(pos_xy_some_part), "pos_xy_some_part")
        else:
            pos_xy_some_part = pos_xy_some
        if is_3d_cross:
            R = np.linalg.norm(pos_xy_some_part, axis=1) #*np.sqrt(2./3.) #its length is N_particles-1
        else:
            R = (
                np.linalg.norm(pos_xy_some_part[:, 0:2], axis=1)
                + np.linalg.norm(pos_xy_some_part[:, 2:3], axis=1)
                + np.linalg.norm(pos_xy_some_part[:, 3:1], axis=1)
            ) / 3. #its length is N_particles-1 #roughly meaning
        
        R_soften = None
        if R_minus is None:
            # R[R<R_soften] = R_soften #debug: this might mask the nearest flucation
            # R += R_plus*10. #debug: far particles, less influence by translation and R_minus
            # R_soften = np.min(R) #debug: min distance
            # R_soften = 1e-10 #fixed
            # R_soften = R_minus #fixed
            # R_soften = r_mean*1e-4 #fixed
            # R_soften = 0. #no softenning
            # R_soften = b90 #b90
            R_soften = r_mean/(0.2*N_particles) #several b90
            # R_soften = 2.*b90 #debug: several b90
            # R_mask = R[R>=b90] #debug
            # R[R<b90] = b90 #debug: mask
        else:
            R_soften = R_minus
        r_to_center = ads.norm_l(pos[i])
        min_dist = np.min(R)
        mean_dist = np.mean(R)

        # R_soften = 0.01 #debug cpp
        Integrand = ( R / (R**2+R_soften**2) )**2
        # Integrand = ( R / (R**2+R_soften**2) ) #debug: sampling
        # Integrand = ( R_mask / (R_mask**2+R_soften**2) )**2 #debug
        # Integrand = ( R / (R**2+R_soften**2) )**2
        # # Integrand = mass_field**2/(R+R_minus)**2
        # # R_frac_b90 = R/R_minus
        # # Integrand = ( (2*mass_field)/(mass_target+mass_field) * R_frac_b90/(1.+R_frac_b90**2) )**2
        Diffu_mainpart_each = coef_const * np.sum(Integrand) #over 3d space

        radius_to_center[i] = r_to_center
        min_distance_with_softenning[i] = min_dist
        mean_distance_with_softenning[i] = mean_dist
        Diffu_mainpart[i] = Diffu_mainpart_each
        if i%1000 == 0: #sample to print
            print("ID = {}, radius_to_center = {}, Diffu = {}".format(i, r_to_center, Diffu_mainpart[i]))
    print("")
    # ads.DEBUG_PRINT_V(0, np.mean(Diffu_mainpart), np.median(Diffu_mainpart), "Diffu_mainpart")
    
    savedata = np.hstack((
        np.array([ np.arange(N_particles) ]).T, 
        np.array([min_distance_with_softenning ]).T, 
        np.array([mean_distance_with_softenning ]).T, 
        np.array([Diffu_mainpart ]).T
    ))
    if is_save:
        np.savetxt("../data/examples_pos/Diffu_mainpart_each_particle_"+suffix+".txt", savedata)
    ads.DEBUG_PRINT_V(1, b90, r_mean, np.percentile(min_distance_with_softenning, q=odp.q_pers_see_more), 
        len(np.where(min_distance_with_softenning<R_soften)[0])*1./N_particles, "pers mindist")
    # # return np.mean(Diffu_mainpart)
    # return np.median(Diffu_mainpart)
    return Diffu_mainpart

def diffusion_reference_value_cylinder(
    N_particles, r_mean, v_mean=None, M_total=None, R_minus=None, 
    is_using_formula=True
):
    if M_total is None:
        M_total = 1.
    if v_mean is None:
        v_mean = np.sqrt(G*M_total/frac_mass/r_mean)
    m = M_total/N_particles
    b90 = 2.*G*m/v_mean**2
    R_soften = None
    if R_minus is not None:
        R_soften = R_minus
    else:
        # R_soften = b90
        R_soften = r_mean/(0.2*N_particles)
        # R_soften = r_mean*1e-4
    log_Alpha = np.log(r_mean/R_soften)
    # n0 = N_particles/(4.*np.pi*r_mean**3) #old cylinder region
    # n0 = N_particles/(2.*np.pi*(r_mean*2.)**3) #cylinder region
    n0 = N_particles/(np.pi*(r_mean*1.5)**2*2.*r_mean) #roughly cylinder region
    # n0 = N_particles/(4.*np.pi*(r_mean*1.5)**3) #?? uniform bound rate cylinder region
    # n0 = N_particles/(4.*np.pi*(r_mean*2.)**3) #larger cylinder region
    # n0 = N_particles/(4./3.*np.pi*r_mean**3) #sphere region
    IRp2 = 8./9.*N_particles*log_Alpha
    coef_const = 27.*np.pi/2.*G**2*M_total**2*n0/(v_mean*N_particles**3)
    Diffu_0 = None
    if is_using_formula:
        Diffu_0 = 8.*np.pi*G**2*M_total**2*n0*log_Alpha/(v_mean*N_particles**2)
    else:
        Diffu_0 = coef_const * IRp2
    return Diffu_0

def rate_t_relax_to_t_cross_diffu(Diffu, r_mean=1., v_mean=1.):
    '''
    Note the mass in calculating Diffu should be m = M_total/N_particles.
    '''
    eta_relax = v_mean**3 / (r_mean*Diffu)
    return eta_relax

def rate_t_relax_to_t_cross_IR2(N_particles, h_frac, Dim_frac, R_p, R_m=None):
    '''
    Note the mass in calculating Diffu should be m = M_total/N_particles.
    '''
    # Nk = N_particles
    Nk = 0.2*N_particles
    if R_m is None:
        # R_m = R_p/Nk
        R_m = R_p*1e-4
    IR2 = None
    if Dim_frac<=0 or Dim_frac>3:
        print("Wrong value of Dim_frac. Please check.")
        IR2 = np.nan
    elif Dim_frac==2:
        IR2 = 4.*np.pi*h_frac*R_p**2*np.log(R_p/R_m)
    elif Dim_frac==3: #??
        IR2 = 4.*np.pi*h_frac*R_p**2*(R_p-R_m)
    elif Dim_frac<2:
        IR2 = 4.*np.pi*h_frac*R_p**Dim_frac / (2.-Dim_frac) * ( (Nk)**(2.-Dim_frac) - 1. )
    else:
        IR2 = 4.*np.pi*h_frac*R_p**Dim_frac / (2.-Dim_frac) * ( (Nk)**(2.-Dim_frac) - 1. )
    eta_relax = 4.*N_particles**2 / (3.*IR2)
    return eta_relax

def rate_t_relax_to_t_cross_count(N_particles, Dim_frac, lambda_Rpm=0.2):
    '''
    Note the mass in calculating Diffu should be m = M_total/N_particles.
    '''
    Nk = lambda_Rpm*N_particles #0.5 for simple distant encounters in BT08 Chapter 7, 0.2 in Chapter 7
    lambda1 = 3. #cylinder
    eta_relax = None
    if Dim_frac<=0 or Dim_frac>3:
        print("Wrong value of Dim_frac. Please check.")
        eta_relax = np.nan
    elif Dim_frac==2:
        eta_relax = N_particles / (Dim_frac*lambda1*np.log(Nk))
    # elif Dim_frac==3: #??
    #     eta_relax = N_particles / (Dim_frac*lambda1)
    else:
        eta_relax = (Dim_frac-2.)/(Dim_frac*lambda1) * N_particles/(1.-Nk**(2.-Dim_frac))
    return eta_relax

# Generate Mass distribution
def generate_mass_distribution(M, N):
    import scipy.stats as stats
    from scipy.optimize import minimize
    from scipy.integrate import quad
    from scipy.stats import rv_continuous

    # Step 1: Define the Gaussian-like function
    def gaussian_function(m, mu, sigma, A):
        return A / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(m - mu) ** 2 / (2 * sigma ** 2))

    # Step 2: Define the constraints
    # Integral of f(m) from a to b should be 1
    def constraint_normalization(params, a, b):
        mu, sigma, A = params
        integral, _ = quad(gaussian_function, a, b, args=(mu, sigma, A))
        return integral - 1

    # Integral of f(m) from a to c should be k
    def constraint_partial_integral(params, a, c, k):
        mu, sigma, A = params
        integral, _ = quad(gaussian_function, a, c, args=(mu, sigma, A))
        return integral - k

    # Mean value constraint: <m> = c
    def constraint_mean_value(params, a, b, c):
        mu, sigma, A = params
        integral_fm, _ = quad(lambda m: gaussian_function(m, mu, sigma, A) * m, a, b)
        integral_f, _ = quad(gaussian_function, a, b, args=(mu, sigma, A))
        if integral_f<1e-20:
            integral_f = 1e-20
        mean_value = integral_fm / (integral_f)
        # mean_value = integral_fm / (integral_f+1e-3)
        return mean_value - c

    # Step 3: Objective function to minimize (sum of squared constraint violations)
    def objective(params, a, b, c, k):
        return (
            constraint_normalization(params, a, b) ** 2 +
            constraint_partial_integral(params, a, c, k) ** 2 +
            constraint_mean_value(params, a, b, c) ** 2
        )

    # Step 4: Set initial guesses and bounds for parameters
    m_mean = M/N
    a, b, c, k = 0.1*m_mean, 100.0*m_mean, m_mean, 0.8
    initial_guess = [-0.1, 0.1, 1.0]  # Initial guesses for [mu, sigma, A]
    bounds = [(-100, a), (0.01, 100.0), (1e-4, 1e4)]  # Bounds for [mu, sigma, A]

    # Step 5: Solve for the parameters
    result = minimize(objective, initial_guess, args=(a, b, c, k), bounds=bounds)
    mu_opt, sigma_opt, A_opt = result.x

    # Step 6: Define a custom distribution class to sample from the fitted Gaussian-like function
    class CustomGaussian(rv_continuous):
        def _pdf(self, m):
            return gaussian_function(m, mu_opt, sigma_opt, A_opt)
            # return gaussian_function(m, -0.01, sigma_opt, A_opt)
            # return gaussian_function(m, mu_opt, sigma_opt, A_opt)*(m/m_mean+1.)**-2

    # Step 7: Create an instance of the custom distribution
    custom_gaussian = CustomGaussian(a=a, b=b, name='custom_gaussian')

    # Step 8: Generate a sample of particle masses from the custom distribution
    # N = 10000
    mass_DF = custom_gaussian.rvs(size=N)
    mass_DF *= M/np.sum(mass_DF)

    # Step 9: Plot the distribution
    plt.figure()
    ads.DEBUG_PRINT_V(1, a, b, c, k, "abck")
    m1 = np.linspace(a,b,1000)
    fm1 = gaussian_function(m1, mu_opt, sigma_opt, A_opt)
    plt.plot(m1, fm1)
    plt.title('Particle Mass Distribution')
    plt.xlabel(r'$m$')
    plt.ylabel(r'$f_m(m)$')
    plt.legend()
    plt.grid(True)
    plt.savefig("../data/func_plot_truncated_Gaussian.png", format="png", bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    plt.hist(mass_DF, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(c, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Mass = {c:.4f}')
    plt.title('Particle Mass Distribution')
    plt.xlabel('Mass')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("../data/mass_DF_truncated_Gaussian.png", format="png", bbox_inches='tight')
    
    ads.DEBUG_PRINT_V(1, mu_opt, sigma_opt, A_opt, np.shape(mass_DF))
    return mass_DF

def data_mass_compare(data):
    m_mean_2 = np.mean(data)**2
    m_2_mean = np.mean(data**2)
    sig2 = m_2_mean-m_mean_2
    rate_m = m_2_mean/m_mean_2
    return m_mean_2, m_2_mean, sig2, rate_m

def display_geometry_IC_Perlin_cloud():
    # Step 1: Define grid for 2D sky
    sky_size = (500, 500)  # Define the resolution of the sky (500x500 pixels)
    x_vals = np.linspace(0, 5, sky_size[0])  # Horizontal coordinate values
    y_vals = np.linspace(0, 5, sky_size[1])  # Vertical coordinate values
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='ij')

    # Step 2: Generate multi-layer Perlin noise to represent clouds
    def generate_clouds_perlin(x_mesh, y_mesh, scale=1.0, octaves=4, persistence=0.5, lacunarity=2.0):
        clouds = np.zeros_like(x_mesh)
        
        for i in range(octaves):
            # Frequency and amplitude for each octave
            frequency = lacunarity ** i
            amplitude = persistence ** i
            
            for j in range(x_mesh.shape[0]):
                for k in range(x_mesh.shape[1]):
                    noise_value = pnoise2(
                        x_mesh[j, k] * scale * frequency, 
                        y_mesh[j, k] * scale * frequency
                    )
                    clouds[j, k] += noise_value * amplitude
        
        # Normalize clouds to range [0, 1]
        clouds = (clouds - np.min(clouds)) / (np.max(clouds) - np.min(clouds))
        return clouds

    # Step 3: Generate the initial condition for clouds
    clouds = generate_clouds_perlin(x_mesh, y_mesh, scale=0.5, octaves=5, persistence=0.5, lacunarity=2.0)

    # Step 4: Plot the generated clouds
    plt.figure(figsize=(10, 10))
    plt.imshow(clouds, cmap='gray', origin='upper')
    plt.title("Initial Condition of Clouds on 2D Sky")
    plt.axis('off')
    plt.colorbar(label="Cloud Density")
    # plt.show()
    plt.savefig("../data/Perlin_noise_geometry_IC.png", format="png", bbox_inches='tight')
    return 0

def display_geometry_IC_Perlin():
    # Step 1: Define grid for 2D space (e.g., terrain size)
    terrain_size = (500, 500)  # Define the resolution of the terrain (500x500 pixels)
    x_vals = np.linspace(0, 5, terrain_size[0])  # Horizontal coordinate values
    y_vals = np.linspace(0, 5, terrain_size[1])  # Vertical coordinate values
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='ij')

    # Step 2: Generate multi-layer Perlin noise to represent mountains
    def generate_mountains_perlin(x_mesh, y_mesh, scale=1.0, octaves=6, persistence=0.4, lacunarity=2.5):
        mountains = np.zeros_like(x_mesh)
        
        for i in range(octaves):
            # Frequency and amplitude for each octave
            frequency = lacunarity ** i
            amplitude = persistence ** i
            
            for j in range(x_mesh.shape[0]):
                for k in range(x_mesh.shape[1]):
                    noise_value = pnoise2(
                        x_mesh[j, k] * scale * frequency, 
                        y_mesh[j, k] * scale * frequency
                    )
                    mountains[j, k] += noise_value * amplitude
        
        # Normalize mountains to range [0, 1]
        mountains = (mountains - np.min(mountains)) / (np.max(mountains) - np.min(mountains))
        return mountains

    # Step 3: Generate the initial condition for mountains
    mountains = generate_mountains_perlin(x_mesh, y_mesh, scale=0.5, octaves=6, persistence=0.4, lacunarity=2.5)

    # Step 4: Plot the generated mountains
    plt.figure(figsize=(10, 10))
    plt.imshow(mountains, cmap='terrain', origin='upper')
    plt.title("Initial Condition of Mountains on 2D Space")
    plt.axis('off')
    plt.colorbar(label="Elevation")
    # plt.show()
    plt.savefig("../data/Perlin_noise_geometry_IC.png", format="png", bbox_inches='tight')
    return 0

def display_Perlin_noise(n0, N_grid_1d=100, k_std=0.1):
    import numpy as np
    import matplotlib.pyplot as plt

    # Step 1: Define large meshgrids for 3D cylindrical space to describe the original uniform DF
    R_vals = np.linspace(R_minus, R_plus, N_grid_1d)  # Radial values (kpc)
    phi_vals = np.linspace(0, 2 * np.pi, N_grid_1d)  # Azimuthal angle values (radians)
    z_vals = np.linspace(0, v0*t_during, N_grid_1d)  # Vertical values (kpc)

    R_mesh, phi_mesh, z_mesh = np.meshgrid(R_vals, phi_vals, z_vals, indexing='ij')
    # n0 = 1e11 / (4.0 * np.pi / 3.0 * (40.0)**3)  # Original uniform density function value
    n_original = np.full_like(R_mesh, n0)  # Original uniform DF n_original(R, phi, z) = n0

    # Step 2: Generate Perlin noise on a fixed grid in Cartesian coordinates and convert to cylindrical space
    def generate_noise_cartesian(R_mesh, phi_mesh, z_mesh):
        # Convert cylindrical to Cartesian coordinates
        x_mesh = R_mesh * np.cos(phi_mesh)
        y_mesh = R_mesh * np.sin(phi_mesh)

        # Generate Perlin noise on Cartesian grid
        noise = np.zeros_like(x_mesh)

        # scale = 0.5
        # for i in range(x_mesh.shape[0]):
        #     for j in range(x_mesh.shape[1]):
        #         for k in range(x_mesh.shape[2]):
        #             noise_value = pnoise3(x_mesh[i, j, k] * scale, y_mesh[i, j, k] * scale, z_mesh[i, j, k] * scale)
        #             noise[i, j, k] = noise_value
        
        # scale = 0.5
        # # scale = 5.
        # octaves = 4
        # persistence = 0.5
        # lacunarity = 2.0
        # for i in range(x_mesh.shape[0]):
        #     for j in range(x_mesh.shape[1]):
        #         for k in range(x_mesh.shape[2]):
        #             noise_value = pnoise3(
        #                 x_mesh[i, j, k] * scale, 
        #                 y_mesh[i, j, k] * scale, 
        #                 z_mesh[i, j, k] * scale,
        #                 octaves=octaves, 
        #                 persistence=persistence, 
        #                 lacunarity=lacunarity
        #             )
        #             noise[i, j, k] = noise_value
        
        # scale = 0.05
        scale = 0.5
        # scale = R_plus/5.
        # octaves = 4
        octaves = 6
        persistence = 0.5
        # persistence = 5.
        lacunarity = 2.0
        # lacunarity = 2.5
        for i in range(octaves):
            # Frequency and amplitude for each octave
            frequency = lacunarity ** i
            amplitude = persistence ** i
            # amplitude = persistence ** i * 0.1*n0
            for j in range(x_mesh.shape[0]):
                for k in range(x_mesh.shape[1]):
                    for l in range(x_mesh.shape[2]):
                        noise_value = pnoise3(
                            x_mesh[j, k, l] * scale * frequency, 
                            y_mesh[j, k, l] * scale * frequency, 
                            z_mesh[j, k, l] * scale * frequency
                        )
                        noise[j, k, l] += noise_value * amplitude

        # noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        # ads.DEBUG_PRINT_V(0, noise[10,10,10])
        return noise

    n_noise_cartesian = generate_noise_cartesian(R_mesh, phi_mesh, z_mesh)

    n_noise_cartesian -= np.mean(n_noise_cartesian)
    # n_total = n_original + n_noise_cartesian*n0 #rate 1.
    # n_total = n_original + n_noise_cartesian*n0*1.e2 #rate 1.
    # n_total = n_original + n_noise_cartesian*n0*1.e4 #rate 80.
    n_total = n_original + n_noise_cartesian*n0*1.e6 #rate 2.8e3.
    ads.DEBUG_PRINT_V(1, np.mean(n_total)/n0, np.std(n_total)/n0)
    # # Step 3: Constrain the noise arguments to meet the desired statistical properties
    # def normalize_noise(n_original, n_noise):
    #     # # mask signal #?? points and blanck
    #     # n_noise[n_noise<-n0] = 0.
    #     # n_noise[n_noise>2.*n0] = 10.*n0

    #     # Normalize the noise to have a mean of 0
    #     noise_mean = np.mean(n_noise)
    #     n_noise_normalized = n_noise - noise_mean
    #     # ads.DEBUG_PRINT_V(0, noise_mean)
        
    #     # Scale the noise to have a standard deviation of k_std * n0
    #     noise_std = np.std(n_noise_normalized)
    #     target_std = k_std * n0 #k_std ~ 0.1
    #     n_noise_scaled = n_noise_normalized * (target_std / noise_std)
    #     # ads.DEBUG_PRINT_V(1, n0, np.mean(n_noise), np.std(n_noise), "noise mean and std frac to n0")
    #     # ads.DEBUG_PRINT_V(1, 1., np.mean(n_noise)/n0, np.std(n_noise)/n0, "noise mean and std frac to n0")
    #     # ads.DEBUG_PRINT_V(1, np.mean(n_noise_normalized)/n0, np.mean(n_noise_scaled)/n0, "scaled noise mean and std frac to n0")

    #     # Add the scaled noise to the original uniform DF
    #     n_total = n_original + n_noise_scaled
    #     return n_total

    # n_total = normalize_noise(n_original, n_noise_cartesian)

    # Step 4: Summary statistics to verify the constraints
    def summary_statistics(n_total):
        mean_value = np.mean(n_total)
        std_value = np.std(n_total)
        return mean_value, std_value

    mean_n, std_n = summary_statistics(n_total)

    print(f"Mean of total DF: {mean_n}")
    print(f"Standard deviation of total DF: {std_n}")

    # Step 5: Plotting to display and compare n_original and n_total on Cartesian grids
    slice_idx = 50  # Choose a slice to visualize

    # Convert cylindrical to Cartesian coordinates for plotting
    x_vals = R_vals * np.cos(phi_vals[slice_idx])
    y_vals = R_vals * np.sin(phi_vals[slice_idx])

    # Plot original DF
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Density Function (n_original) on Cartesian Grid")
    plt.imshow(n_original[:, slice_idx, :], origin='lower', aspect='auto', extent=[x_vals[0], x_vals[-1], z_vals[0], z_vals[-1]])
    plt.xlabel("x (kpc)")
    plt.ylabel("z (kpc)")
    plt.colorbar(label="Density")

    # Plot total DF with noise
    plt.subplot(1, 2, 2)
    plt.title("Total Density Function (n_total) on Cartesian Grid")
    plt.imshow(n_total[:, slice_idx, :], origin='lower', aspect='auto', extent=[x_vals[0], x_vals[-1], z_vals[0], z_vals[-1]])
    plt.xlabel("x (kpc)")
    plt.ylabel("z (kpc)")
    plt.colorbar(label="Density")

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/Perlin_noise_compare.png", format="png", bbox_inches='tight')
    return n_original, n_total, R_vals, phi_vals, z_vals

# Vegas integration function for diffusion coefficient
def diffusion_pos_homog_vegas_meshgrid(n_total, R_vals, phi_vals, z_vals, num_samples=10000):
    # # Alternative 1: Use linear interpolation for the meshgrid
    # from scipy.interpolate import RegularGridInterpolator
    # n_total_interpolator = RegularGridInterpolator((R_vals, phi_vals, z_vals), n_total, bounds_error=False, fill_value=None)
    # # Function to be integrated
    # def integrand(x):
    #     R, phi, z = x[0], x[1], x[2]
    #     n_value = n_total_interpolator((R, phi, z))
    #     return n_value / R

    # Alternative 2: Function to be integrated
    def integrand(x):
        R, phi, z = x[0], x[1], x[2]
        # Find the closest indices in the meshgrid
        R_idx = np.searchsorted(R_vals, R) - 1
        phi_idx = np.searchsorted(phi_vals, phi) - 1
        z_idx = np.searchsorted(z_vals, z) - 1
        # Ensure indices are within bounds
        R_idx = np.clip(R_idx, 0, len(R_vals) - 1)
        phi_idx = np.clip(phi_idx, 0, len(phi_vals) - 1)
        z_idx = np.clip(z_idx, 0, len(z_vals) - 1)
        # Get the value from the meshgrid
        n_value = n_total[R_idx, phi_idx, z_idx]
        return n_value / R

    # Setting up the integration domain
    R_minus, R_plus = R_vals[0], R_vals[-1]
    z_max = z_vals[-1]
    integral_domain = [[R_minus, R_plus], [0, 2 * np.pi], [0, z_max]]

    # Vegas Monte Carlo integration
    integrator = vegas.Integrator(integral_domain)

    # Perform the integration
    result = integrator(integrand, nitn=10, neval=num_samples)
    # return result, result.mean, result.sdev
    D_diffu_value = (4 * G**2 * km * ma**2) / (v0**2 * t_during) * result.mean
    return D_diffu_value, result.mean, result.sdev

def mean_distance_each_other(a):
    N = len(a)
    d = 0
    idx = 0
    for i in np.arange(0,N):
        for j in np.arange(i,N):
            if i != j:
                d += np.linalg.norm(a[i]-a[j])
                idx += 1
    return d/idx

# Fitting pnoise by DF data
def fit_pnoise_by_DF_data():
    # Function to generate noise given the parameters
    def generate_noise_with_params(R_mesh, phi_mesh, z_mesh, scale, octaves, persistence, lacunarity):
        x_mesh = R_mesh * np.cos(phi_mesh)
        y_mesh = R_mesh * np.sin(phi_mesh)
        noise = np.zeros_like(x_mesh)
        
        for i in range(x_mesh.shape[0]):
            for j in range(x_mesh.shape[1]):
                for k in range(x_mesh.shape[2]):
                    noise_value = pnoise3(x_mesh[i, j, k] * scale, y_mesh[i, j, k] * scale, z_mesh[i, j, k] * scale,
                                        octaves=octaves, persistence=persistence, lacunarity=lacunarity)
                    noise[i, j, k] = noise_value
        return noise

    # # Objective function to minimize the difference between numerical DF and generated noise
    # def objective_function(params, R_mesh, phi_mesh, z_mesh, n_actual):
    #     scale, octaves, persistence, lacunarity = params
    #     n_noise = generate_noise_with_params(R_mesh, phi_mesh, z_mesh, scale, int(octaves), persistence, lacunarity)
    #     n_total = normalize_noise(n_original, n_noise)
    #     return np.sum((n_total - n_actual) ** 2)

    # # Initial guess for the parameters
    # initial_params = [0.5, 4, 0.5, 2.0]  # [scale, octaves, persistence, lacunarity]

    # # Perform the optimization
    # result = minimize(objective_function, initial_params, args=(R_mesh, phi_mesh, z_mesh, n_actual), bounds=[
    #     (0.1, 10),  # scale
    #     (1, 8),     # octaves (integer values)
    #     (0.1, 1),   # persistence
    #     (1.0, 5.0)  # lacunarity
    # ])

    # # Extract the optimal parameters
    # optimal_scale, optimal_octaves, optimal_persistence, optimal_lacunarity = result.x
    # print(f"Optimal Parameters - Scale: {optimal_scale}, Octaves: {optimal_octaves}, Persistence: {optimal_persistence}, Lacunarity: {optimal_lacunarity}")
    return 0

def generate_powerlaw_1d(M, N):
    # Step 1: Define parameters for the power-law distribution
    alpha = -1.5  # Power-law index (alpha < 0)
    xmin = 1.  # Minimum value for the power-law distribution
    r_typical = M / N # A typical value
    r_bound_down = 0.1 * r_typical
    r_bound_up = 100. * r_typical

    # Step 2: Generate a 1D sample with power-law distribution within bounds
    # Use the inverse transform sampling method
    sample = np.zeros(N)
    i_sampled = 0
    while i_sampled < N:
        r = np.random.uniform(0, 1)
        value = xmin * (1 - r) ** (1 / (alpha + 1))
        if r_bound_down <= value <= r_bound_up:
            sample[i_sampled] = value
            i_sampled += 1
    sample = np.array(sample)

    # Step 3: Normalize the total sum (1 order moment) to M
    sample *= M/np.sum(sample)

    # Step 4: Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(sample, bins=100, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title('1D Sample with Power-Law Distribution (alpha={})'.format(alpha))
    plt.grid(True)
    plt.savefig('../data/{}.png'.format('powerlaw_1d_sample'))

    return sample

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
        inside_counts[:, i] = np.array(counts)
    
    return radii, inside_counts

# Function to compute the two-point correlation function (TPCF)
from scipy.spatial import cKDTree
def TPCF_countneighbours(positions, r_max=50.0, num_bins=50, is_enable_normalization=True):
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
    rng = np.random.default_rng()
    points1 = rng.random((5, 2))
    points2 = rng.random((5, 2))
    kd_tree1 = cKDTree(points1)
    kd_tree2 = cKDTree(points2)
    N1 = kd_tree1.count_neighbors(kd_tree2, 0.2)

    indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)
    N2 = sum([len(i) for i in indexes])
    ads.DEBUG_PRINT_V(1, N1, N2)
    return 0

def calculate_tpcf(positions, bins):
    """
    Calculate the two-point correlation function (TPCF) of a 3D particle distribution.

    Parameters:
    - positions (ndarray): (N, 3) array containing the positions of the particles.
    - bins (ndarray): Array specifying the bin edges for the distance ranges.

    Returns:
    - tpcf (ndarray): Values of the two-point correlation function in each bin.
    """
    N = len(positions)
    tree = cKDTree(positions)

    # Count pairs within specified distance bins
    pair_counts = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        r_min, r_max = bins[i], bins[i + 1]
        count = tree.count_neighbors(tree, r_max) - tree.count_neighbors(tree, r_min)
        pair_counts[i] = count

    # Normalize pair counts to calculate TPCF
    pair_density = 2.0 * pair_counts / (N * (N - 1))
    tpcf = pair_density - 1.0

    return tpcf

def plot_tpcf(bins, tpcf_values, suffix="suffix"):
    """
    Plot the two-point correlation function (TPCF) and save the plot as a PNG file.

    Parameters:
    - bins (ndarray): Array specifying the bin edges for the distance ranges.
    - tpcf_values (ndarray): TPCF values corresponding to each distance bin.
    """
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, tpcf_values, 'o-', label='TPCF', color='b')
    plt.xlabel('Distance (r)')
    plt.ylabel('Two-Point Correlation Function (TPCF)')
    plt.title('Two-Point Correlation Function')
    plt.grid()
    plt.legend()
    plt.savefig("../data/TPCF_{}.png".format(suffix))
    plt.close()
    return 0

def plot_diffusion_histogram(Diffu_data, N_bins, Diffu0=None, is_mask_extreme=False, suffix="suffix"):
    pers = np.percentile(Diffu_data, q=[1., 10., 50., 90., 99.])
    mask = (Diffu_data>pers[0]) & (Diffu_data<pers[-1]) #1., 99.
    indices = np.where(mask)[0]
    if is_mask_extreme:
        Diffu_data = Diffu_data[indices]
    min_val = np.min(Diffu_data)
    max_val = np.max(Diffu_data)
    mean_val = np.mean(Diffu_data)
    median_val = np.median(Diffu_data)
    count_total = len(Diffu_data)
    ads.DEBUG_PRINT_V(1, pers, min_val, max_val, mean_val, median_val, count_total, "Diffu_data")
    
    fontsize = 40.
    pointsize = 3.2
    figsize = 20, 15 #for 3, 3
    dpi = 400
    plt.figure(figsize=figsize, dpi=dpi)
    bins = np.geomspace(min_val, max_val, N_bins)
    counts, bin_edges, _ = plt.hist(Diffu_data, bins=bins, density=False, alpha=0.5, edgecolor='black')
    bin_widths = np.diff(bin_edges)
    normalized_distribution = counts / (bin_widths * count_total)
    plt.clf()  # Clear the previous histogram
    plt.grid(True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    plt.plot(bin_centers, normalized_distribution, 'o-', label='Normalized Distribution', color='b', lw=pointsize)
    plt.plot([mean_val, mean_val], [0., np.max(normalized_distribution)], '--', label='Diffu_mean = {:.4f}'.format(mean_val), color='k', lw=pointsize)
    plt.plot([median_val, median_val], [0., np.max(normalized_distribution)], '-.', label='Diffu_median = {:.4f}'.format(median_val), color='k', lw=pointsize)
    if Diffu0 is not None:
        plt.plot([Diffu0, Diffu0], [0., np.max(normalized_distribution)], label='Diffu_0 = {:.4f}'.format(Diffu0), color='k', lw=pointsize)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(r"histogram of main diffusion coefficient of each particle", fontsize=fontsize)
    plt.xlabel(r"diffusion, $D_\mathrm{position, main}$ ($\mathrm{(km/s)^3/kpc}$)", fontsize=fontsize)
    plt.ylabel(r"distribution, $f$ ($\mathrm{kpc/(km/s)^3}$)", fontsize=fontsize)
    plt.legend(fontsize=fontsize*0.6, loc=0)
    plt.tick_params(which='major', length=0, labelsize=fontsize*0.8) #size of the number characters
    plt.tight_layout()
    plt.savefig("../data/examples_pos/Diffu_mainpart_histogram_"+suffix+".png", format="png", bbox_inches='tight')
    return 0

def plot_frac_dim_total(pos_info_store, suffix="suffix"):
    '''
    cols_name = ["mean_radius", "N_particles", "diffu_mean", "diffu_median", "diffu_0", "h_frac", "Dim_frac", 
        "eta_diffu", "eta_IR2", "eta_N", "pos_type", "name", ]
    pos_DFs = ["pos_obs", "pos_uniform", "pos_uniform_noise", "pos_simu", "pos_simu_noise", "pos_extra", ]
    '''
    fontsize = 40.
    pointsize = 3.2
    figsize = 20, 15 #for 3, 3
    dpi = 400
    N_points_plot = 100

    # Dim_frac_arr = np.linspace(0., 3., N_points_plot-1)
    # Dim_frac_arr = np.append(Dim_frac_arr, 2.)
    # N_particles_plot_arr = [1e4, 1e11]
    # eta_N_arr_dim = np.zeros((len(N_particles_plot_arr),N_points_plot))
    # xy_plot_dim_lb = []
    # for i in np.arange(len(N_particles_plot_arr)):
    #     for j in np.arange(N_points_plot):
    #         eta_N_arr_dim[i, j] = rate_t_relax_to_t_cross_count(N_particles_plot_arr[i], Dim_frac=Dim_frac_arr[j])
    #     xy_plot_dim_lb.append([Dim_frac_arr, np.log10(eta_N_arr_dim[i]), "N={}".format(N_particles_plot_arr[i])])
    # odp.plot_funcs_1d(
    #     xy_plot_dim_lb, 
    #     xlabel="Dim_frac", ylabel="eta_relax_log10", suffix="dim_1_log"
    # )
    # ads.DEBUG_PRINT_V(1, np.shape(eta_N_arr_dim), "eta_N_arr_dim")

    N_particles_arr = np.geomspace(1e3, 1e5, N_points_plot)
    # N_particles_arr = np.geomspace(1e0, 1e11, N_points_plot)
    # Dim_frac_lb = np.array([0.5, 1., 1.5, 2., 2.5, 3.])
    Dim_frac_lb = pos_info_store.loc[:, "Dim_frac"]
    N_dimlb_plot = len(Dim_frac_lb)
    eta_N_arr_count = np.zeros((N_dimlb_plot, N_points_plot))
    xy_plot_dim_lb = []
    for j in np.arange(N_dimlb_plot):
        for i in np.arange(N_points_plot):
            eta_N_arr_count[j, i] = rate_t_relax_to_t_cross_count(N_particles_arr[i], Dim_frac=Dim_frac_lb[j])
        xy_plot_dim_lb.append([np.log10(N_particles_arr), np.log10(eta_N_arr_count[j]), "Dim_frac={}".format(Dim_frac_lb[j])])
    # odp.plot_funcs_1d(xy_plot_dim_lb, xlabel="N_log10", ylabel="eta_relax_log10", suffix="count_log_log")
    
    N_particles_file = pos_info_store.loc[:, "N_particles"]
    eta_diffu = pos_info_store.loc[:, "eta_diffu"]
    eta_IR2 = pos_info_store.loc[:, "eta_IR2"]
    eta_N = pos_info_store.loc[:, "eta_N"]
    pos_type_name = pos_info_store.loc[:, "pos_type"]
    color = {
        "pos_obs": "k", "pos_uniform": "b", "pos_uniform_noise": "r", 
        "pos_simu": "g", "pos_simu_noise": "orange", "pos_extra": "purple", 
    }
    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True)
    for j in np.arange(N_dimlb_plot):
        plt.plot(N_particles_arr, eta_N_arr_count[j], label="eta_N, Dim_frac={:.2f}".format(Dim_frac_lb[j]), color=color[pos_type_name[j]], lw=pointsize)
        # plt.scatter(N_particles_file[j], eta_N[j], label="eta_IR2", color=color[pos_type_name[j]], s=pointsize*60., marker="*")
        # plt.scatter(N_particles_file[j], (eta_IR2[j]+eta_N[j])/2., label="eta_mean", color=color[pos_type_name[j]], s=pointsize*60., marker="x") #debug
        plt.scatter(N_particles_file[j], eta_diffu[j], label="eta_diffu of {}".format(pos_type_name[j]), color=color[pos_type_name[j]], s=pointsize*60., marker=".")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(r"eta_relax versus N", fontsize=fontsize)
    plt.xlabel(r"N_log10", fontsize=fontsize)
    plt.ylabel(r"eta_relax_log10", fontsize=fontsize)
    plt.legend(fontsize=fontsize*0.36, loc=0)
    plt.tick_params(which='major', length=0, labelsize=fontsize*0.8) #size of the number characters
    plt.tight_layout()
    plt.savefig("../data/examples_pos/pos_info_frac_dim_"+suffix+".png", format="png", bbox_inches='tight')
    return 0

def get_mean_center(sample):
    return np.mean(sample, axis=0)

def get_std_dispersion(sample):
    return np.std(sample, axis=0)

def get_mean_radius(sample, is_center=False):
    pos = sample*1.
    if is_center:
        pos -= np.mean(pos, axis=0)
    r = ads.norm_l(pos, axis=1)
    return np.mean(r)

def get_quadratic_mean(sample, is_center=False):
    smp = sample*1.
    if is_center:
        smp -= np.mean(smp, axis=0)
    qm = np.sqrt(np.mean(np.sum(smp**2, axis=1)))
    return qm

def get_percentile_radius(positions, is_center=False, pers=[50., 99., 100.]):
    pos = positions*1.
    if is_center:
        pos -= np.mean(pos, axis=0)
    r = ads.norm_l(pos, axis=1)
    return np.percentile(r, pers)

def get_mean_unit_kinetic_energy(velocities, is_center=False):
    vel = velocities*1.
    if is_center:
        vel -= np.mean(vel, axis=0)
    v_size = ads.norm_l(vel, axis=1)
    return 0.5*np.mean(v_size**2.)

def readjust_velocities(velocities, mean_3, dispersion_3):
    sample = velocities*1.
    old_mean = np.mean(sample, axis=0)
    sample -= old_mean
    sample += mean_3

    old_quadratic_mean = np.sqrt(np.mean(np.sum(sample**2, axis=1)))
    quadratic_mean = np.sqrt(np.sum(mean_3**2+dispersion_3**2))
    scaling_factor = quadratic_mean / old_quadratic_mean
    sample *= scaling_factor
    return sample

def print_sample_info(sample, sample_name=None):
    if len(np.shape(sample)) != 2:
        ads.DEBUG_PRINT_V(
            1, "Wrong shape of sample, please check.", 
            sample_name
        )
        return 1
    else:
        ads.DEBUG_PRINT_V(
            1, np.shape(sample), 
            get_mean_center(sample), get_std_dispersion(sample), 
            get_mean_radius(sample), get_quadratic_mean(sample), 
            sample_name
        )
        return 0

def screen_radius_samples(positions, radius_bound_down=-1e-20, radius_bound_up=np.inf):
    r = ads.norm_l(positions, axis=1)
    mask = (r>=radius_bound_down) & (r<=radius_bound_up)
    indices = np.where(mask)[0]
    indices_not = np.where(~mask)[0]
    return positions[indices], indices, indices_not

def screen_sample_by_min_radius_count(positions, N_select, mean_radius_setting=None):
    if len(positions) < N_select:
        return positions, None
    else:
        pos = positions*1.
        pos -= np.mean(pos, axis=0)
        radius = ads.norm_l(pos, axis=1)
        indices = np.argsort(radius)
        pos = (pos[indices])[0:N_select]
        # ads.DEBUG_PRINT_V(0, (radius[indices])[N_select], "original screen radius")

        r_mean_old = get_mean_radius(pos)
        if mean_radius_setting is not None:
            scaling_factor = mean_radius_setting/r_mean_old
            pos *= scaling_factor
        ads.DEBUG_PRINT_V(1, np.shape(pos), scaling_factor, "scaling_factor")
        return pos, r_mean_old

def screen_sample_by_random_select(positions, N_select, mean_radius_setting=None):
    if len(positions) < N_select:
        return positions
    else:
        pos = positions*1.
        np.random.shuffle(pos) #the outeast axis
        pos = pos[:N_select]
        if mean_radius_setting is not None:
            scaling_factor = mean_radius_setting/get_mean_radius(pos)
            pos *= scaling_factor
            ads.DEBUG_PRINT_V(1, np.shape(pos), scaling_factor, "scaling_factor")
        return pos

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
        counts[i] = (pairwise_distances < r).sum() #to fit count
        # counts[i] /= (4.*np.pi*r**3) #to fit count density
        
    log_r = np.log(r_values)  
    log_counts = np.log(counts)  
    
    # Fit a line to log(counts) vs log(radius)  
    fit = np.polyfit(log_r, log_counts, 1)  
    return fit[-1], fit[0]  # Slope of the line is the estimated fractal dimension  

def compute_correlation_dimension_volume(points, max_radius):  
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
        counts[i] = (pairwise_distances < r).sum() #to fit count
        counts[i] /= (4./3.*np.pi*r**3) #to fit count density
        
    log_r = np.log(r_values)  
    log_counts = np.log(counts)  
    
    # Fit a line to log(counts) vs log(radius)  
    fit = np.polyfit(log_r, log_counts, 1)
    h_frac = fit[-1]
    Dim_frac = 3.+fit[0] #alpha = 3.-Dim_frac
    return h_frac, Dim_frac  # Slope of the line is the estimated fractal dimension  

import warnings
def calculate_mean_neareast_count(pos, r_min=None, r_max=None, n_radii=12, suffix="suffix"):
    if r_max is None:
        mean_radius = get_mean_radius(pos, is_center=True)
        r_max = mean_radius*1.
    if r_min is None:
        r_min = r_max/10.
    elif r_min>=r_max:
        print("The current r_min = {}, r_max = {}".format(r_min, r_max))
        warnings.warn("The r_min is larger than r_max. Setting r_min to be default.")
        r_min = r_max/10.
    pos_targets = pos
    ads.DEBUG_PRINT_V(1, np.shape(pos_targets), "np.shape(pos_targets)")

    radii, inside_counts = inside_count_with_radius(pos, pos_targets, r_min=r_min, r_max=r_max, n_radii=n_radii)
    radii, inside_counts = radii, np.mean(inside_counts, axis=0)
    ads.DEBUG_PRINT_V(1, radii, inside_counts, "radii")
    popt = fit_fractal_dimension(radii, inside_counts, suffix=suffix)
    h_frac, Dim_frac = popt
    print("Optimal Parameters: h_frac = {:.2f}, Dim_frac = {:.2f}".format(h_frac, Dim_frac))
    return h_frac, Dim_frac

def Diffu_using_relative_velocity_descrete(vel_target, vel_sample, softenning=0.001):
    vel_relative = vel_target-vel_sample
    # vel_relative = vel_relative**2 #debug
    # vel_relative_size = ads.norm_l(vel_relative, axis=1)
    # vel_relative_size[vel_relative_size<softenning] = softenning #note: softenning
    vel_relative_size = np.sqrt(np.sum(vel_relative**2, axis=1)+softenning**2)
    I_11 = (vel_relative[:,1]**2 + vel_relative[:,2]**2) / vel_relative_size**3
    I_22 = (vel_relative[:,0]**2 + vel_relative[:,2]**2) / vel_relative_size**3
    I_33 = (vel_relative[:,0]**2 + vel_relative[:,1]**2) / vel_relative_size**3
    I_12 = (- vel_relative[:,0] * vel_relative[:,1]) / vel_relative_size**3
    I_13 = (- vel_relative[:,0] * vel_relative[:,2]) / vel_relative_size**3
    I_23 = (- vel_relative[:,1] * vel_relative[:,2]) / vel_relative_size**3
    D = np.array([
        [np.sum(I_11), np.sum(I_12), np.sum(I_13)], 
        [np.sum(I_12), np.sum(I_22), np.sum(I_23)], 
        [np.sum(I_13), np.sum(I_23), np.sum(I_33)]
    ]) # I_23 is N_sample shape
    return D

def diffusion_velocity_softenning_mean_descrete(
    vel_field, mass_field=None, M_total=None, softenning=None, n0=None, log_Alpha=None, 
    N_particles_pos=None, is_save=True, suffix="suffix"
):
    N_particles = len(vel_field)
    vel_target_size = np.zeros(N_particles) #a vector for velocity size
    min_distance_with_softenning = np.zeros(N_particles) #a vector for min distance to other particles
    mean_distance_with_softenning = np.zeros(N_particles) #a vector for mean distance to other particles
    Diffu_mainpart = np.zeros((N_particles, 3, 3)) #a vector for diffusion coef of each particle
    vel = vel_field-np.mean(vel_field, axis=0) #translate to mass center
    # v_relative_mean_0 = get_mean_radius(vel)
    v_relative_mean_0 = get_quadratic_mean(vel)

    if N_particles_pos is None:
        N_particles_pos = N_particles #usually set the count of samples in vel to be N_particles
    if M_total is None:
        M_total = 1.
    if n0 is None:
        n0 = 1. #N_particles_pos/(4.*np.pi*r_mean**3)
    if log_Alpha is None:
        log_Alpha = 1. #np.log(0.2*N_particles_pos)
    if softenning is None:
        # softenning = softenning
        # softenning = v_relative_mean_0*1e-4
        softenning = v_relative_mean_0/(0.2*N_particles)
    # m = 1. #debug diffu by particles count
    m = M_total/N_particles_pos
    coef_const = 4.*np.pi*G**2*m**2*n0*log_Alpha/N_particles
    # coef_const = 3. * G*G * M_total*M_total *log(lambda2 * N_particles) / (R0*R0*R0 * N_particles * N_samples) #C++
    ads.DEBUG_PRINT_V(1, M_total, N_particles, G, v_relative_mean_0, softenning, coef_const)

    # for i in np.arange(10): #debug
    for i in np.arange(N_particles):
        vel_target = vel[i]
        vel_sample = np.delete(vel, i, axis=0) #remove the index that the particle itself, i.e. i=j case
        # ads.DEBUG_PRINT_V(0, np.shape(vel), np.shape(vel_sample), "vel_sample")

        vts = ads.norm_l(vel_target)
        V = np.linalg.norm(vel_sample, axis=1) #its length is N_particles-1
        min_dist = np.min(V)
        mean_dist = np.mean(V)
        D = Diffu_using_relative_velocity_descrete(vel_target, vel_sample, softenning=softenning) #over 3d space
        
        vel_target_size[i] = vts
        min_distance_with_softenning[i] = min_dist
        mean_distance_with_softenning[i] = mean_dist
        Diffu_mainpart[i] = coef_const * D
        if i%1000 == 0: #sample to print
            print("ID = {}, vel_target_size = {}, Diffu_each[i=0] = {}".format(i, vts, Diffu_mainpart[i][0]))
    print("")

    savedata = np.hstack((
        np.array([ np.arange(N_particles) ]).T, #index of particles
        np.array([min_distance_with_softenning ]).T, 
        np.array([mean_distance_with_softenning ]).T, #each mean size of relative volocity
        np.array([Diffu_mainpart[:, 0, 0] ]).T, #each D_11
        np.array([Diffu_mainpart[:, 1, 1] ]).T, #each D_22
        np.array([Diffu_mainpart[:, 2, 2] ]).T, #each D_33
        np.array([Diffu_mainpart[:, 0, 1] ]).T, #each D_12
        np.array([Diffu_mainpart[:, 0, 2] ]).T, #each D_13
        np.array([Diffu_mainpart[:, 1, 2] ]).T  #each D_23
    ))
    if is_save:
        np.savetxt("../data/examples_vel/Diffu_sample_vel_"+suffix+".txt", savedata)
    # ads.DEBUG_PRINT_V(1, b90, r_mean, np.percentile(min_distance_with_softenning, q=odp.q_pers_see_more), 
    #     len(np.where(min_distance_with_softenning<b90)[0])*1./N_particles, "pers mindist")
    # return np.mean(Diffu_mainpart, axis=0) #(3,3)
    # return np.median(Diffu_mainpart, axis=0) #(3,3)
    # return Diffu_mainpart #(N,3,3)
    return np.sum(Diffu_mainpart, axis=(1,2)) #(N,)
    # return np.linalg.norm(Diffu_mainpart, axis=(1,2)) #(N,)

def Diffu_vel_component(vel_target, vel_sample, softenning=0.001):
    vel_relative = vel_target-vel_sample
    vel_relative_size = np.sqrt(np.sum(vel_relative**2, axis=1)+softenning**2)
    I_11 = (vel_relative[:,1]**2 + vel_relative[:,2]**2) / vel_relative_size**3
    I_22 = (vel_relative[:,0]**2 + vel_relative[:,2]**2) / vel_relative_size**3
    I_33 = (vel_relative[:,0]**2 + vel_relative[:,1]**2) / vel_relative_size**3
    I_12 = (- vel_relative[:,0] * vel_relative[:,1]) / vel_relative_size**3
    I_13 = (- vel_relative[:,0] * vel_relative[:,2]) / vel_relative_size**3
    I_23 = (- vel_relative[:,1] * vel_relative[:,2]) / vel_relative_size**3
    D = np.array([
        [np.sum(I_11), np.sum(I_12), np.sum(I_13)], 
        [np.sum(I_12), np.sum(I_22), np.sum(I_23)], 
        [np.sum(I_13), np.sum(I_23), np.sum(I_33)]
    ]) # I_23 is N_sample shape
    return D

def diffusion_velocity_to_compare(
    vel_field, mass_field=None, M_total=None, softenning=None, n0=None, log_Alpha=None, 
    N_particles=None, is_save=True, suffix="suffix"
):
    N_particles = len(vel_field)
    Diffu_tensor = np.zeros((N_particles, 3, 3)) #a vector for diffusion coef of each particle
    vel = vel_field-np.mean(vel_field, axis=0) #translate to mass center
    # v_relative_mean_0 = get_mean_radius(vel)
    v_relative_mean_0 = get_quadratic_mean(vel)

    softenning = v_relative_mean_0/(0.2*N_particles)
    m = M_total/N_particles
    coef_const = 4.*np.pi*G**2*m**2*n0*log_Alpha/N_particles
    # ads.DEBUG_PRINT_V(0, M_total, N_particles, G, v_relative_mean_0, softenning, coef_const)

    for i in np.arange(N_particles):
        vel_target = vel[i]
        vel_sample = np.delete(vel, i, axis=0) #remove the index that the particle itself, i.e. i=j case

        vts = ads.norm_l(vel_target)
        V = np.linalg.norm(vel_sample, axis=1) #its length is N_particles-1
        D = Diffu_vel_component(vel_target, vel_sample, softenning=softenning) #over 3d space
        
        Diffu_tensor[i] = coef_const * D
        if i%1000 == 0: #sample to print
            print("ID = {}, vel_target_size = {}, Diffu_each[i=0] = {}".format(i, vts, Diffu_tensor[i][0]))
    return Diffu_tensor #(N,3,3)

def generate_velocities_sample_Gaussian(N, mean_3, dispersion_3, is_readjust=True):
    """
    Generate a 3D velocity sample with Gaussian distribution and then readjust strictly.

    Parameters:
    - N (int): Number of samples.
    - mean_3 (array-like): Mean values for each dimension, shape (3,).
    - dispersion_3 (array-like): Dispersion (standard deviation) for each dimension, shape (3,).

    Returns:
    - sample (ndarray): The readjusted 3D velocity sample, shape (N, 3).
    """
    # Generate Gaussian-distributed samples for each dimension
    sample = np.random.normal(loc=mean_3, scale=dispersion_3, size=(N, 3))

    if is_readjust:
        # Translate the center
        old_mean = np.mean(sample, axis=0)
        sample -= old_mean
        sample += mean_3
        # ads.DEBUG_PRINT_V(1, old_mean, mean_3, "mean")

        # Rescale the sample to achieve the desired quadratic mean
        old_quadratic_mean = np.sqrt(np.mean(np.sum(sample**2, axis=1)))
        quadratic_mean = np.sqrt(np.sum(mean_3**2+dispersion_3**2))
        scaling_factor = quadratic_mean / old_quadratic_mean
        sample *= scaling_factor
        # ads.DEBUG_PRINT_V(1, old_quadratic_mean, quadratic_mean, "quadratic_mean")

    return sample

class DF_particles_6D:
    def __init__(self, 
        DF_name1, M_total1, N_particles1, 
        mean_radius1=None, mean_speed1=None, 
        v_mean_31=None, v_dispersions_31=None, N_samples_vel1=None, 
        N_particles_pos1=None, n0_pos1=None, log_Alpha1=None
    ):
        self.mean_radius = mean_radius1
        self.mean_speed = mean_speed1
        self.M_total = M_total1
        self.N_particles = N_particles1
        self.name = "{}_{}".format(DF_name1, N_particles1)
        self.state = 0 #default, do calculate

        self.pos = None
        self.pos_type = None
        self.pos_type_int = None
        self.diffu_all = None
        self.diffu_mean = None
        self.diffu_median = None
        self.diffu_0 = None
        self.h_frac = None
        self.Dim_frac = None
        self.eta_diffu = None
        self.eta_IR2 = None
        self.eta_N = None
        self.pos_info = None

        self.v_mean_3 = v_mean_31
        self.v_dispersions_3 = v_dispersions_31
        self.N_samples_vel = N_samples_vel1
        self.N_particles_pos = N_particles_pos1
        self.n0_pos = n0_pos1
        self.log_Alpha = log_Alpha1
        self.vel = None
        self.vel_type = None
        self.vel_type_int = None
        self.vel_info = None
        return None

    def generate_pos(self, 
        pos_DF_type, pos_original=None, is_using_extra=False, is_plot_sample=True,
        crystal_type=None, uniform_bound_shape="cylindrical", N_iter_Perlin=20
    ):
        '''
        The types are
        ["pos_obs", "pos_crystal", "pos_uniform", "pos_uniform_noise", 
        "pos_simu", "pos_simu_noise"].
        Each type has diffferent generating way.
        '''
        ads.DEBUG_PRINT_V(1, "generate_pos")
        self.pos_type = pos_DF_type

        if self.pos_type == "pos_crystal":
            self.pos_type_int = 0
            self.pos = None
            # self.pos = generate_initial_positions_crystal_sphere(self.mean_radius, self.N_particles)

        elif self.pos_type == "pos_obs":
            self.pos_type_int = 1
            if pos_original is None:
                raise ValueError("The pos_original is None before generating.")
            if len(pos_original) < self.N_particles:
                warnings.warn("len(pos_original) = {}, self.N_particles = {}\nThe length of original sample is less than the target length, please "
                    "check. Using the target length.".format(len(pos_original), self.N_particles))
                self.pos = pos_original
                self.state = 1 #do not calculate
            else:
                self.pos, r_mean_old = screen_sample_by_min_radius_count(pos_original, self.N_particles, mean_radius_setting=self.mean_radius)
                ads.DEBUG_PRINT_V(1, get_mean_radius(pos_original), r_mean_old, "obs")

        elif self.pos_type == "pos_uniform":
            self.pos_type_int = 2
            # ads.DEBUG_PRINT_V(0, self.mean_radius, self.N_particles, uniform_bound_shape)
            self.pos = generate_particle_positions(
                self.mean_radius, self.N_particles, bound_shape=uniform_bound_shape
            )

        elif self.pos_type == "pos_uniform_noise":
            self.pos_type_int = 3
            if pos_original is None:
                raise ValueError("The pos_original is None before generating.")
            if len(pos_original) < self.N_particles:
                warnings.warn("len(pos_original) = {}, self.N_particles = {}\nThe length of original sample is less than the target length, please "
                    "check. Using the target length.".format(len(pos_original), self.N_particles))
                self.pos = pos_original
                self.state = 1 #do not calculate
            else:
                self.pos = particles_3d_Perlin_motion_factor_scale(
                    pos_original, N_iter=N_iter_Perlin, scale=0.1, step_size=2., 
                    is_plot_witin_bound=False, suffix=self.name
                )

        elif self.pos_type == "pos_simu":
            self.pos_type_int = 4
            if pos_original is None:
                raise ValueError("The pos_original is None before generating.")
            if len(pos_original) < self.N_particles:
                warnings.warn("len(pos_original) = {}, self.N_particles = {}\nThe length of original sample is less than the target length, please "
                    "check. Using the target length.".format(len(pos_original), self.N_particles))
                self.pos = pos_original
                self.state = 1 #do not calculate
            else:
                # self.pos, r_mean_old = screen_sample_by_min_radius_count(pos_original, self.N_particles, mean_radius_setting=self.mean_radius)
                self.pos = screen_sample_by_random_select(pos_original, self.N_particles, mean_radius_setting=self.mean_radius)

        elif self.pos_type == "pos_simu_noise":
            self.pos_type_int = 5
            if pos_original is None:
                raise ValueError("The pos_original is None before generating.")
            if len(pos_original) < self.N_particles:
                warnings.warn("len(pos_original) = {}, self.N_particles = {}\nThe length of original sample is less than the target length, please "
                    "check. Using the target length.".format(len(pos_original), self.N_particles))
                self.pos = pos_original
                self.state = 1 #do not calculate
            else:
                self.pos = particles_3d_Perlin_motion_factor_scale(
                    pos_original, N_iter=N_iter_Perlin, scale=0.1, step_size=2., 
                    is_plot_witin_bound=False, suffix=self.name
                )

        elif self.pos_type=="pos_extra":
            if is_using_extra:
                self.pos = pos_original
                self.pos = ads.readjust_positions(self.pos, self.mean_radius)

        else:
            self.pos_type_int = None
            self.pos = None
            raise ValueError("No such type of DF can be generated, please check. Exit.")
        
        self.mean_radius = get_mean_radius(self.pos)
        self.N_particles = len(self.pos)
        if is_plot_sample:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.scatter(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2], s=0.1, alpha=0.6)
            ax.set_title("generated sample: Dim_frac_calculated={}".format(self.Dim_frac))
            plt.savefig("../data/examples_pos/generated_sample_{}.png".format(self.name), format="png", bbox_inches="tight")
            plt.close()
        ads.DEBUG_PRINT_V(1, self.mean_radius, "self.mean_radius")
        return self.pos

    def run_pos(self, is_fractal_dim=False, r_max_count=None):
        ads.DEBUG_PRINT_V(1, "run")
        is_3d_cross = True
        self.diffu_all = diffusion_descrete_cyl_softenning_mean(
            self.pos, suffix=self.name, 
            M_total=self.M_total, r_mean=self.mean_radius, v_mean=self.mean_speed, 
            # R_local_bound_count=128, #debug
            is_3d_cross=is_3d_cross
        )
        # if self.pos_type=="pos_uniform_noise":
        #     ads.DEBUG_PRINT_V(0, self.pos[0], self.diffu_all[0], "diffu_all")
        self.diffu_mean = np.mean(self.diffu_all)
        self.diffu_median = np.median(self.diffu_all)
        self.diffu_0 = diffusion_reference_value_cylinder(
            N_particles=self.N_particles, r_mean=self.mean_radius, v_mean=self.mean_speed, M_total=self.M_total
        )
        if is_fractal_dim:
            ads.DEBUG_PRINT_V(1, "run2")
            self.h_frac, self.Dim_frac = gfs.calculate_mean_neareast_count(self.pos, r_min=None, r_max=r_max_count, n_radii=12, suffix=self.name)
            # self.h_frac, self.Dim_frac = gfs.compute_correlation_dimension_volume(self.pos, max_radius=r_max_count)
            self.eta_diffu =  rate_t_relax_to_t_cross_diffu(self.diffu_median, r_mean=self.mean_radius, v_mean=self.mean_speed)
            self.eta_IR2 = rate_t_relax_to_t_cross_IR2(self.N_particles, h_frac=self.h_frac, Dim_frac=self.Dim_frac, R_p=self.mean_radius)
            self.eta_N = rate_t_relax_to_t_cross_count(self.N_particles, Dim_frac=self.Dim_frac)
            ads.DEBUG_PRINT_V(1, self.h_frac, self.Dim_frac, "Dim_frac")
        return 0

    def display_pos(self, N_bins=100):
        ads.DEBUG_PRINT_V(1, "display")
        Diffu0 = self.diffu_0
        Diffu_data = np.loadtxt("../data/examples_pos/Diffu_mainpart_each_particle_{}.txt".format(self.name))
        Diffu_data = Diffu_data[:, 3]
        # if self.pos_type=="pos_uniform_noise":
        #     ads.DEBUG_PRINT_V(0, self.pos[0], Diffu_data[0], "diffu_all")
        plot_diffusion_histogram(Diffu_data, N_bins, Diffu0=Diffu0, suffix=self.name)
        print("Plot {}, done.".format(self.name))
        self.pos_info = [
            self.mean_radius, self.N_particles, self.diffu_mean, self.diffu_median, self.diffu_0, 
            self.h_frac, self.Dim_frac, self.eta_diffu, self.eta_IR2, self.eta_N, 
            self.pos_type, self.name
        ]
        return self.pos_info

    def generate_vel(self, 
        vel_DF_type, vel_original=None, 
        v_dispersions_3_reset=None, N_iter_Perlin=30
    ):
        '''
        The types are
        ["vel_simu", "vel_iso", "vel_iso_noise", "vel_aniso", "vel_aniso_high", 
        "vel_speed_Gaussian", "vel_speed_tail", 
        "vel_obs", "vel_simu_noise", "vel_aniso_noise"].
        Each type has diffferent generating way.
        '''
        ads.DEBUG_PRINT_V(1, "generate_vel")
        self.vel_type = vel_DF_type

        if self.vel_type == "vel_obs": #?? by positions
            self.vel_type_int = 1
            if vel_original is None:
                raise ValueError("The vel_original is None before generating.")
            if len(vel_original) < self.N_samples_vel:
                warnings.warn("len(pos_original) = {}, self.N_samples_vel = {}\nThe length of original sample is less than the target length, please "
                    "check. Using the target length.".format(len(vel_original), self.N_samples_vel))
                self.vel = vel_original
                self.state = 1 #do not calculate
            else:
                self.vel = screen_sample_by_random_select(vel_original, self.N_samples_vel, mean_radius_setting=None)
                self.vel = readjust_velocities(self.vel, self.v_mean_3, self.v_dispersions_3)

        elif self.vel_type == "vel_iso":
            self.v_dispersions_3 = v_dispersions_3_reset
            if np.abs(self.v_dispersions_3[1]-self.v_dispersions_3[0])>self.v_dispersions_3[0]*0.01 or \
                np.abs(self.v_dispersions_3[2]-self.v_dispersions_3[0])>self.v_dispersions_3[0]*0.01:
                warnings.warn("The three value of self.v_dispersions_3 are not near. Reset it as the mean value.")
                self.v_dispersions_3 = np.ones(3)*np.sqrt(np.mean(self.v_dispersions_3**2))
            self.vel_type_int = 6
            self.vel = generate_velocities_sample_Gaussian(
                self.N_samples_vel, self.v_mean_3, self.v_dispersions_3
            )
            # tmp = np.loadtxt("../data/samples_vel/uniform.txt", dtype=float) #debug
            # self.vel = tmp[:,3:6]
            ads.DEBUG_PRINT_V(1, self.vel[0], "self.vel[0]")

        elif self.vel_type == "vel_iso_noise":
            self.vel_type_int = 7
            if vel_original is None:
                raise ValueError("The vel_original is None before generating.")
            if len(vel_original) < self.N_samples_vel:
                warnings.warn("len(pos_original) = {}, self.N_samples_vel = {}\nThe length of original sample is less than the target length, please "
                    "check. Using the target length.".format(len(vel_original), self.N_samples_vel))
                self.vel = vel_original
                self.state = 1 #do not calculate
            else:
                self.vel = particles_3d_Perlin_motion_factor_scale(
                    vel_original, N_iter=N_iter_Perlin, scale=0.02, step_size=20., 
                    is_plot=True, suffix=self.name
                    # is_plot=False, suffix=self.name
                )
                self.vel = readjust_velocities(self.vel, self.v_mean_3, self.v_dispersions_3)

        elif self.vel_type == "vel_aniso":
            self.v_dispersions_3 = v_dispersions_3_reset
            self.vel_type_int = 6
            self.vel = generate_velocities_sample_Gaussian(
                self.N_samples_vel, self.v_mean_3, self.v_dispersions_3
            )
            # self.vel = convert_to_rpt_vel(self.vel)

        elif self.vel_type == "vel_aniso_high":
            self.v_dispersions_3 = v_dispersions_3_reset
            self.vel_type_int = 6
            self.vel = generate_velocities_sample_Gaussian(
                self.N_samples_vel, self.v_mean_3, self.v_dispersions_3
            )
            # self.vel = convert_to_rpt_vel(self.vel)

        elif self.vel_type == "vel_simu":
            self.vel_type_int = 4
            if vel_original is None:
                raise ValueError("The vel_original is None before generating.")
            if len(vel_original) < self.N_samples_vel:
                warnings.warn("len(pos_original) = {}, self.N_samples_vel = {}\nThe length of original sample is less than the target length, please "
                    "check. Using the target length.".format(len(vel_original), self.N_samples_vel))
                self.vel = vel_original
                self.state = 1 #do not calculate
            else:
                self.vel = screen_sample_by_random_select(vel_original, self.N_samples_vel, mean_radius_setting=None)
                self.vel = readjust_velocities(self.vel, self.v_mean_3, self.v_dispersions_3)

        else:
            self.vel_type_int = None
            self.vel = None
            raise ValueError("No such type of DF can be generated, please check. Exit.")
        
        self.N_samples_vel = len(self.vel)
        print_sample_info(self.vel, self.name)
        return self.vel

    def run_vel(self):
        ads.DEBUG_PRINT_V(1, "run")
        is_3d_cross = True
        self.diffu_all = diffusion_velocity_softenning_mean_descrete(
            self.vel, M_total=self.M_total, softenning=None, n0=self.n0_pos, log_Alpha=self.log_Alpha, 
            N_particles_pos=self.N_particles_pos, is_save=True, suffix=self.name
        )
        self.diffu_mean = np.mean(self.diffu_all)
        self.diffu_median = np.median(self.diffu_all)
        vmean = np.sqrt(np.sum(self.v_mean_3**2+self.v_dispersions_3**2))
        rmean = G*self.M_total/vmean**2
        self.diffu_0 = diffusion_reference_value_cylinder(
            N_particles=self.N_particles_pos, r_mean=rmean, v_mean=vmean, M_total=self.M_total
        )
        # ads.DEBUG_PRINT_V(1, vmean, rmean, self.N_particles_pos, self.diffu_0, "diffu_0")
        # ads.DEBUG_PRINT_V(1, np.min(self.diffu_all), np.max(self.diffu_all), "diffu_all")
        return 0

    def display_vel(self, N_bins=100):
        ads.DEBUG_PRINT_V(1, "display")
        Diffu0 = self.diffu_0
        Diffu_data = np.loadtxt("../data/examples_vel/Diffu_sample_vel_{}.txt".format(self.name))
        Diffu_data = np.sum(Diffu_data[:, 3:6]+Diffu_data[:, 6:9]*2., axis=1)
        # plot_diffusion_histogram(Diffu_data, N_bins, Diffu0=Diffu0, suffix=self.name)
        plot_diffusion_histogram(Diffu_data, N_bins, Diffu0=None, suffix=self.name)
        print("Plot {}, done.".format(self.name))
        self.vel_info = [
            self.v_mean_3[0], self.v_mean_3[1], self.v_mean_3[2], 
            self.v_dispersions_3[0], self.v_dispersions_3[1], self.v_dispersions_3[2], 
            self.N_particles_pos, self.N_samples_vel, 
            self.diffu_mean, self.diffu_median, self.diffu_0, 
            self.vel_type, self.name
        ]
        return self.vel_info

def zeta_amplification_rate_outter_interpolation(N, D):
    return 0

def screen_sphere_samples(x, x_center, radius):
    r = ads.norm_l(x-x_center, axis=1)
    value_pers = np.percentile(r, [0.1, 1., 50., 99., 99.9])
    ads.DEBUG_PRINT_V(1, np.min(r), np.max(r), value_pers, "value_pers")
    mask = (r<=radius)
    indices = np.where(mask)[0]
    indices_not = np.where(~mask)[0]
    # ads.DEBUG_PRINT_V(1, np.shape(x), np.shape(indices), "np.shape(indices)")
    return x[indices], indices, indices_not

def choose_max_radius_to_fit(pos_targets, mean_radius_field, rate_bound_to_mean=2.):
    """
    To let the statistical spheres be within the system.
    """
    radius_targets = ads.norm_l(pos_targets, axis=1)
    r_max = mean_radius_field*rate_bound_to_mean - radius_targets
    r_max[r_max<0.] = -1. #mask the outter spheres
    return r_max

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

def fit_fractal_dimension(radii_list, inside_counts_list, suffix="suffix", is_plot=True):
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
        plt.title('Fractal Dimension Fit')
        plt.savefig('../data/examples_pos/fractal_dimension_fit_{}.png'.format(suffix))
        plt.show()
    return popt



#In[] main
## Main calculation
if __name__ == "__main__":

    # 1. load data
    # ads.DEBUG_PRINT_V(0, "1111")
    ## simulation data
    snapshot_ID = 10
    # snapshot_ID = 80
    data_path = "../data/snapshot_%03d.txt"%(snapshot_ID)
    # data_path = "../data/snapshot_%03d_big.txt"%(snapshot_ID)
    data = np.loadtxt(data_path, dtype=float)
    N_particles_data = len(data)
    pos_simu_original = data[:, 0:3]

    # data_path = "../diffu_r_simple_each/Diffu_each_of_snapshot_010.txt"
    # # data_path = "../diffu_r_simple_each/Diffu_each_of_snapshot_010_big.txt"
    # data = np.loadtxt(data_path, dtype=float)
    # pos = data[:, 1:4]
    # pos -= np.mean(pos, axis=0)
    # N_particles = len(pos)
    # M_total = M_total_gal_1e10MSun
    # r_mean = get_mean_radius(pos)
    # v_mean = np.sqrt(G*M_total/frac_mass/r_mean)
    # ads.DEBUG_PRINT_V(1, N_particles, r_mean, v_mean, M_total, "N_particles")
    # # r_mean = 40. #debug
    # Diffu_0_simu_big = diffusion_reference_value_cylinder(N_particles, r_mean, v_mean, M_total)
    # Diffu = data[:, 5]
    # Diffu_mean = np.mean(Diffu)
    # Diffu_median = np.median(Diffu)
    # ads.DEBUG_PRINT_V(0, Diffu_0_simu_big, Diffu_mean, Diffu_median, "pos_simu_big")
    

    
    ## Gaia data some range
    # file_path_pos = "../data/stellar_data_pos.csv"
    file_path_pos = odp.file_path_6D_Cartesian
    pos_obs_original = pd.read_csv(file_path_pos).to_numpy()[:,0:3]
    # pos_obs_read_center = np.mean(pos_obs, axis=0)
    # ads.DEBUG_PRINT_V(1, np.shape(pos_obs), pos_obs_read_center, "pos_obs original")



    # 2. process data
    ## total positions
    np.random.seed(39)
    mean_radius_setting = 50. #to 0.07581 kpc of mean radius of obs data 
    M_total = M_total_gal_1e10MSun
    r_mean = mean_radius_setting
    v_mean = np.sqrt(G*M_total/frac_mass/r_mean)
    ads.DEBUG_PRINT_V(1, M_total, r_mean, v_mean, "galaxy_info")
    Dim_frac_setting = 1.8
    # Dim_frac_setting = 2.8
    N_particles_list = [10000]
    # N_particles_list = [10000, 100000]
    # N_particles_list = [5000, 10000, 20000, 40000]
    pos_DFs = [
        # "pos_crystal", "pos_obs", 
        "pos_extra", 
        # "pos_uniform", "pos_uniform_noise", 
        # "pos_simu", "pos_simu_noise", 
        # "pos_obs", "pos_uniform", "pos_uniform_noise", "pos_simu", "pos_simu_noise", 
    ]
    r_max_count = r_mean*0.25
    # r_max_count = r_mean*0.5
    # r_max_count = r_mean*1.
    # r_max_count = r_mean*2.
    is_fractal_dim = True
    # is_fractal_dim = False

    pos_uniform = None
    pos_uniform_noise = None
    pos_obs = None
    pos_simu = None
    pos_simu_noise = None
    pos_extra = None
    pos_info_store_file = "../data/examples_pos/pos_info_store.csv"

    pos_info_store = [] #diffusion_descrete_cyl_softenning_mean
    for (j, N_particles_setting) in enumerate(N_particles_list):
        for (i, typenme) in enumerate(pos_DFs):
            pos_name = typenme
            pos_class = DF_particles_6D(pos_name, M_total, N_particles_setting, mean_radius1=r_mean, mean_speed1=v_mean)
            
            if pos_name=="pos_obs":
                pos_obs = pos_class.generate_pos(typenme, pos_original=pos_obs_original)
            elif pos_name=="pos_uniform":
                pos_uniform = pos_class.generate_pos(typenme)
            elif pos_name=="pos_uniform_noise": #must be after pos_uniform in the list
                # pos_uniform_noise = pos_class.generate_pos(typenme, pos_original=pos_uniform, N_iter_Perlin=10)
                pos_uniform_noise = pos_class.generate_pos(typenme, pos_original=pos_uniform, N_iter_Perlin=14)
                # pos_uniform_noise = pos_class.generate_pos(typenme, pos_original=pos_uniform, N_iter_Perlin=20)
            elif pos_name=="pos_simu":
                pos_simu = pos_class.generate_pos(typenme, pos_original=pos_simu_original)
            elif pos_name=="pos_simu_noise": #must be after pos_simu in the list
                pos_simu_noise = pos_class.generate_pos(typenme, pos_original=pos_simu, N_iter_Perlin=30)
            elif pos_name=="pos_extra":
                # pos_extra = gfs.generate_fractal_3D_vertex(N_particles_setting, Dim_frac_setting)
                # pos_extra = gfs.generate_fractal_3D_walk(N_particles_setting, Dim_frac_setting)
                pos_extra = gfs.load_samples_from_file("../diffu_r_simple_each/noised.txt", N_particles_setting)
                pos_extra = pos_class.generate_pos(typenme, pos_original=pos_extra, is_using_extra=True)
            else:
                raise ValueError("No such type of DF can be generated, please check. Exit.")
            ads.DEBUG_PRINT_V(1, pos_class.pos[0], "pos[0]")

            if pos_class.state != 0:
                warnings.warn("Unexpected state of pos_class. Skip this loop.")
                continue
            # if pos_name!="pos_uniform": #debug
            #     pos_class.run_pos(is_fractal_dim=is_fractal_dim, r_max_count=r_max_count)
            pos_class.run_pos(is_fractal_dim=is_fractal_dim, r_max_count=r_max_count)
            pos_info = pos_class.display_pos()
            pos_info_store.append(pos_info)
            ads.DEBUG_PRINT_V(1, pos_info, "pos_info")

    cols_name = ["mean_radius", "N_particles", "diffu_mean", "diffu_median", "diffu_0", "h_frac", "Dim_frac", 
        "eta_diffu", "eta_IR2", "eta_N", "pos_type", "name", ]
    pos_info_store = pd.DataFrame(pos_info_store, columns=cols_name)
    pos_info_store.to_csv(pos_info_store_file, index=False)

    pos_info_store = pd.read_csv(pos_info_store_file)
    if is_fractal_dim:
        plot_frac_dim_total(pos_info_store, suffix="type_5_N_40000")
    ads.DEBUG_PRINT_V(0, len(pos_info_store), "pos_info_store")



    # ## old2
    # obs data
    # # r_max_obs_screen = 0.05 #kpc #Gaia #nothing
    # # r_max_obs_screen = 0.08 #kpc #Gaia #nothing #5e3
    # r_max_obs_screen = 0.1 #kpc #Gaia #nothing #1e4
    # # r_max_obs_screen = 0.12 #kpc #Gaia #1.7e4 (1.7/1. < 9.9/4.1)
    # # r_max_obs_screen = 0.2 #kpc #Gaia #usual #4e4
    # # r_max_obs_screen = 0.3 #kpc #Gaia
    # # r_max_obs_screen = 0.5 #kpc #Gaia
    # # r_max_obs_screen = 1. #kpc #Gaia
    # # r_max_obs_screen = 0.16 #kpc #Angus22
    # pos_obs_screen_center = pos_obs_read_center*1. #note: change each time after downloading
    # # pos_obs_screen_center = pos_obs[10] #note: change each time after downloading
    # # pos_obs_screen_center = np.array(odp.pos_Sun_in_GC_Cartesian) #note: change each time after downloading
    # # pos_obs_screen_center = np.array([-8.39, 2.2, -3.5]) #note: change each time after downloading
    # # pos_obs_screen = pos_obs
    # pos_obs_screen, indices_pos_obs_screen, __ = screen_sphere_samples(pos_obs, pos_obs_screen_center, r_max_obs_screen)
    # if len(pos_obs_screen)==0: raise ValueError("Empty screen data, please check. Exit.")
    
    # pos_obs = pos_obs_screen #note: changed length
    # pos_obs = pos_obs #note: this do not use screen data
    # # vel_obs = data_obs[indices_pos_obs_screen, 3:6]
    # N_particles_data = len(pos_obs)
    # # N_particles_sample = N_particles_data*1 #default
    # # N_particles_sample = 5000
    # N_particles_sample = 10000
    # # N_particles_sample = 20000
    # # N_particles_sample = 40000
    # #Calculate integration $I = \int_{R_-}^{R_+} \mathrm{d}r r^(D-3) r^4 / (r^2+R_\epsilon)^2$, where $R_-, R_+, R_\epsilon, D$ is real positive, and $D-3<0$.

    # pos_obs_center = np.mean(pos_obs, axis=0)
    # pos_obs -= pos_obs_center #note: transate to the reference point
    # mean_radius_obs_setting = 50.
    # mean_radius_obs_current = np.mean(np.linalg.norm(pos_obs, axis=1)) #old current
    # pos_obs *= (mean_radius_obs_setting/mean_radius_obs_current)
    # mean_radius_obs = get_mean_radius(pos_obs) #new current
    # ads.DEBUG_PRINT_V(1, np.shape(pos_obs), pos_obs_center, "pos_obs screen")
    # ads.DEBUG_PRINT_V(1, get_mean_radius(pos_obs), "get_mean_radius pos_obs")



    # ## sample data
    # # N_iter_0 = 20 #smaller flucation
    # N_iter_0 = 30 #larger flucation
    # # N_iter = 5 #crystal
    # N_iter = N_iter_0

    # # pos_crystal = generate_initial_positions_crystal_sphere(mean_radius_obs, N_particles_sample)
    # # # pos_crystal_noise = particles_3d_Perlin_motion_factor(pos_crystal, N_iter=N_iter, is_plot_witin_bound=False, suffix="poscrystal")
    # # ads.DEBUG_PRINT_V(1, get_mean_radius(pos_crystal), "get_mean_radius pos_crystal")
    
    # # # N_iter = 50 #uniform
    # # pos_uniform = generate_particle_positions(mean_radius_obs, N_particles_sample, bound_shape="cylindrical")
    # # # pos_uniform_noise = particles_3d_Perlin_motion_factor(pos_uniform, N_iter=N_iter, is_plot_witin_bound=False, suffix="posuniform")
    # # # ads.DEBUG_PRINT_V(1, pos_uniform[0], "pos_uniform[0]")
    # # # ads.DEBUG_PRINT_V(1, get_mean_radius(pos_uniform), get_mean_radius(pos_uniform_noise), "get_mean_radius pos_uniform")
    
    # # # N_iter = 30 #data
    # # pos_simu = pos_simu #note: the data length is not N_particles_sample
    # # # pos_simu_noise = particles_3d_Perlin_motion_factor(pos_simu, N_iter=N_iter, is_plot_witin_bound=True, suffix="posdata")
    # # # ads.DEBUG_PRINT_V(0, get_mean_radius(pos_simu), get_mean_radius(pos_simu_noise), "get_mean_radius pos_simu")

    # N_iter = 10
    # np.random.seed(39)
    # # np.random.seed(65065)
    # # np.random.seed(114514)
    # # pos_uniform = generate_initial_positions_crystal_sphere(mean_radius_obs, N_particles_sample, lattice_constant=1./300)
    # # pos_uniform = generate_particle_positions(mean_radius_obs, N_particles_sample, bound_shape="spherical")
    # pos_uniform = generate_particle_positions(
    #     mean_radius_obs_setting, N_particles_sample, bound_shape="cylindrical"
    # ) #debug diffu by particles count
    # # pos_uniform = generate_vel_DF_Gaussian(
    # #     10000, sigma_xyz=30., velmean_xyz=np.array([0., 0., 0.]), mean_radius=50., vel_bound=None
    # # ) #debug diffu by particles count
    # # pos_uniform = generate_particle_positions(mean_radius_obs, N_particles_sample, bound_shape="cylindrical")
    # ads.DEBUG_PRINT_V(1, get_mean_radius(pos_uniform), "get_mean_radius pos_obs")
    # pos_uniform_noise = particles_3d_Perlin_motion_factor_scale(
    #     pos_uniform, N_iter=20, scale=0.1, step_size=2., #smaller noise
    #     # pos_uniform, N_iter=50, scale=0.1, step_size=2.,  #larger noise
    #     # pos_uniform, N_iter=50, scale=0.21, step_size=2., 
    #     is_plot_witin_bound=False, suffix="posuniform"
    # ) #debug diffu by particles count
    # # pos_uniform_noise = particles_3d_Perlin_motion_factor_scale(pos_uniform, N_iter=N_iter, is_plot_witin_bound=False, suffix="posuniform")
    # ads.DEBUG_PRINT_V(1, get_mean_radius(pos_uniform_noise), "get_mean_radius pos_obs")
    


    # ## plot
    # # odp.plot_data_of_observed_stars(positions_SkyCoor, proj=[0,1], labels_pos=["ra", "dec"], suffix="pos_obs_radec")
    # odp.plot_data_of_observed_stars(pos_obs+pos_obs_center, proj=[0,1], suffix="pos_obs_not_move_xy")
    # odp.plot_data_of_observed_stars(pos_obs+pos_obs_center, proj=[0,2], suffix="pos_obs_not_move_xz")
    # # odp.plot_data_of_observed_stars(vel_obs, proj=[0,1], suffix="vel_obs_xy")
    # # odp.plot_data_of_observed_stars(vel_obs, proj=[0,2], suffix="vel_obs_xz")
    # # odp.plot_data_of_observed_stars(pos_crystal, proj=[0,1], suffix="pos_crystal")
    # odp.plot_data_of_observed_stars(pos_uniform, proj=[0,1], suffix="pos_uniform")
    # odp.plot_data_of_observed_stars(pos_uniform_noise, proj=[0,1], suffix="Angus22_xy")
    # # ads.DEBUG_PRINT_V(0, pos_obs_center, "pos_obs_center")

    # m_input = M_total_gal_1e10MSun*1e10/N_total_MW
    # dens_mean_grid, dens_fluc_grid = plot_mass_contour_postions_or_velocities(
    #     x_input=pos_uniform, m_input=m_input, is_pos=False, 
    #     savename="../data/mass_contour_pos_simu"
    # )
    # dens_mean_grid_noise, dens_fluc_grid_noise = plot_mass_contour_postions_or_velocities(
    #     x_input=pos_uniform_noise, m_input=m_input, is_pos=False, 
    #     savename="../data/mass_contour_pos_simu_noise"
    # )
    # ads.DEBUG_PRINT_V(1, 
    #     dens_mean_grid, dens_fluc_grid, 
    #     dens_mean_grid_noise, dens_fluc_grid_noise, 
    #     "dens_fluc_grid"
    # )
    # ads.DEBUG_PRINT_V(0, np.shape(pos_obs), "pos_simu_noise")



    # ## fractal dim
    # # pos = pos_crystal
    # # pos = pos_crystal_noise
    # # pos = pos_uniform
    # # pos = pos_uniform_noise
    # # pos = pos_simu
    # # pos = pos_simu_noise
    # # pos = pos_uniform
    # pos = pos_uniform_noise
    # # pos = pos_obs

    # # suffix = "pos_crystal"
    # # suffix = "pos_crystal_noise"
    # # suffix = "pos_uniform"
    # # suffix = "pos_uniform_noise"
    # # suffix = "pos_simu"
    # # suffix = "pos_simu_noise"
    # # suffix = "pos_uniform"
    # suffix = "pos_uniform_noise"
    # # suffix = "pos_obs"

    # # r_max = 5. #kpc
    # r_max = 10. #kpc
    # # r_max = 20. #kpc
    # # r_max = mean_radius/10.
    # # r_max = mean_radius/1.
    # r_min = r_max/10. #kpc
    # # radius_bound_down = mean_radius*0.5
    # # radius_bound_up = mean_radius*1.5-r_max
    # # pos_screen, _, __ = screen_radius_samples(pos, radius_bound_up=radius_bound_up)
    # # radius_bound_down = mean_radius*0.
    # # radius_bound_up = mean_radius*0.4
    # # radius_bound_down = mean_radius*0.8
    # # radius_bound_up = mean_radius*1.2
    # radius_bound_down = mean_radius*1.6-r_max
    # radius_bound_up = mean_radius*2.-r_max
    # pos_screen, _, __ = screen_radius_samples(pos, radius_bound_down=radius_bound_down, radius_bound_up=radius_bound_up)
    # pos_targets = pos
    # # pos_targets = pos_screen

    # # r_max = r_max_obs_screen*0.05
    # # r_max = r_max_obs_screen*0.2 #usual
    # r_max = r_max_obs_screen*1. #largest count sphere
    # # r_max = 10.*1. #debug diffu by particles count
    # # # r_max = 50.*1.
    # r_min = r_max/10.
    # # r_max = r_max_obs_screen*0.5
    # # r_min = 0.02
    # # radius_bound_down = 0.
    # # radius_bound_up = r_max_obs_screen-r_max
    # # pos_screen, _, __ = screen_radius_samples(pos, radius_bound_down=radius_bound_down, radius_bound_up=radius_bound_up)
    # # # pos_targets = pos_screen
    # pos_targets = pos

    # # TPCF = TPCF_countneighbours(pos_targets, r_max=r_max, num_bins=50)
    # bins = np.geomspace(r_min, r_max, 12)  # Define distance bins
    # tpcf_values = calculate_tpcf(pos, bins)
    # plot_tpcf(bins, tpcf_values, suffix)
    # ads.DEBUG_PRINT_V(0, tpcf_values, "tpcf_values")

    # n_radii = 12 #usual
    # # n_radii = 20
    # radii, inside_counts = inside_count_with_radius(pos, pos_targets, r_min=r_min, r_max=r_max, n_radii=n_radii)
    # radii, inside_counts = radii, np.mean(inside_counts, axis=0)
    # ads.DEBUG_PRINT_V(1, radii, inside_counts, "radii")
    # popt = fit_fractal_dimension(radii, inside_counts, suffix=suffix)
    # h_frac, Dim_frac = popt
    # print("Optimal Parameters: h_frac = {:.2f}, Dim_frac = {:.2f}".format(h_frac, Dim_frac))
    # ads.DEBUG_PRINT_V(1, np.shape(pos_targets), "np.shape(pos_targets)")

    # # pos_target = np.array([ [5., 5., 5.] ])
    # # pos_target = np.array([ [10., 10., 10.] ])
    # # pos_target = np.array([ [20., 20., 20.] ])
    # # pos_target = np.array([ pos[0] ])
    # # pos_target = np.array([ pos[5] ])
    # pos_target = np.array([ pos[10] ])
    # r_min = 0.5 #kpc
    # r_max = choose_max_radius_to_fit(pos_target, mean_radius, rate_bound_to_mean=2.)[0]
    # ads.DEBUG_PRINT_V(1, r_max, "r_max")
    # if r_max<0.: raise ValueError("Too far targets.")
    # n_radii = 20
    # radii, inside_counts = inside_count_with_radius(pos, pos_target, r_max=r_max, n_radii=n_radii)
    # radii, inside_counts = radii[0], inside_counts[0]
    # popt = fit_fractal_dimension(radii, inside_counts, suffix=suffix)
    # print("Optimal Parameters: h_D = {:.2f}, D = {:.2f}".format(*popt))

    # pos_targets = pos[0:1000]
    # # pos_targets = pos
    # N_targets = len(pos_targets)
    # r_min = 0.5 #kpc
    # # r_max = 10. #kpc
    # r_max = 20. #kpc
    # # r_max = 50. #kpc
    # # r_max = mean_radius/10.
    # # r_max = mean_radius/1.
    # ads.DEBUG_PRINT_V(1, r_max, "r_max")
    # n_radii = 20
    # radii, inside_counts = inside_count_with_radius(pos, pos_targets, r_max=r_max, n_radii=n_radii)
    # ads.DEBUG_PRINT_V(1, np.shape(radii), np.shape(inside_counts), "np.shape(inside_counts)")
    # h_frac_many, Dim_frac_many = np.zeros(N_targets), np.zeros(N_targets)
    # for i in np.arange(N_targets):
    #     popt = fit_fractal_dimension(radii, inside_counts[i], suffix=suffix, is_plot=False)
    #     h_frac_many[i], Dim_frac_many[i] = popt
    # ads.DEBUG_PRINT_V(1, np.min(Dim_frac_many), np.max(Dim_frac_many), "Dim_frac_many min and max")
    # h_frac, Dim_frac = np.mean(h_frac_many), np.mean(Dim_frac_many)
    # print("Optimal Parameters: h_frac = {:.2f}, Dim_frac = {:.2f}".format(h_frac, Dim_frac))



    # ## diffusion coef
    # M_total = M_total_gal_1e10MSun #convert unit to 1e10 MSun
    # r_mean = mean_radius_obs
    # # r_mean = 50. #debug diffu by particles count
    # v_mean = np.sqrt(G*M_total/frac_mass/r_mean)
    # is_3d_cross = True
    # # is_3d_cross = False

    # Diffu1 = diffusion_descrete_cyl_softenning_mean(pos_uniform, suffix="pos_uniform", M_total=M_total, r_mean=r_mean, v_mean=v_mean, is_3d_cross=is_3d_cross)
    # # Diffu2 = diffusion_descrete_cyl_softenning_mean(pos_uniform_noise, suffix="pos_uniform_noise", M_total=M_total, r_mean=r_mean, v_mean=v_mean, is_3d_cross=is_3d_cross)
    # Diffu3 = diffusion_descrete_cyl_softenning_mean(pos_simu, suffix="pos_simu", M_total=M_total, r_mean=r_mean, v_mean=v_mean, is_3d_cross=is_3d_cross)
    # # Diffu4 = diffusion_descrete_cyl_softenning_mean(pos_simu_noise, suffix="pos_simu_noise", M_total=M_total, r_mean=r_mean, v_mean=v_mean, is_3d_cross=is_3d_cross)
    # ads.DEBUG_PRINT_V(1, Diffu1c, Diffu1, Diffu3, "Diffu1c, Diffu1, Diffu3")

    # Diffu1u = diffusion_distant_encounters_uniform_truncated(len(pos_obs), M_total=M_total, r_mean=r_mean, v_mean=v_mean) #note: the rate is by mo means
    # Diffu1c = diffusion_descrete_cyl_softenning_mean(pos_crystal, suffix="pos_crystal", M_total=M_total, r_mean=r_mean, v_mean=v_mean, is_3d_cross=is_3d_cross)
    # # Diffu2c = diffusion_descrete_cyl_softenning_mean(pos_crystal_noise, suffix="pos_crystal_noise", M_total=M_total, r_mean=r_mean, v_mean=v_mean, is_3d_cross=is_3d_cross)
    # ads.DEBUG_PRINT_V(1, Diffu1u, Diffu1c, "Diffu1u, Diffu1c")

    # Diffu1o = diffusion_descrete_cyl_softenning_mean(pos_uniform, suffix="pos_uniform", M_total=M_total, r_mean=r_mean, v_mean=v_mean, is_3d_cross=is_3d_cross)
    # ads.DEBUG_PRINT_V(1, Diffu1o, "Diffu1o")
    # # ads.DEBUG_PRINT_V(1, np.log10(Diffu1o), "log10 Diffu1o")
    # Diffu1on = diffusion_descrete_cyl_softenning_mean(pos_uniform_noise, suffix="pos_uniform_noise", M_total=M_total, r_mean=r_mean, v_mean=v_mean, is_3d_cross=is_3d_cross)
    # print("len = {}, Diffu1on = {:.2e}".format(len(pos_uniform_noise), Diffu1on)) #debug diffu by particles count
    # ads.DEBUG_PRINT_V(1, Diffu1on, "Diffu1on")

    # Diffu3o = diffusion_descrete_cyl_softenning_mean(pos_obs, suffix="pos_obs", M_total=M_total, r_mean=r_mean, v_mean=v_mean, is_3d_cross=is_3d_cross)
    # ads.DEBUG_PRINT_V(1, Diffu3o, "Diffu3o")
    # ads.DEBUG_PRINT_V(1, np.log10(Diffu3o), "log10 Diffu3o")
    # # amplification_rate = Diffu3o/Diffu1o
    # # print("amplification_rate: {}".format(amplification_rate))
    
    # N_bins = 100
    # Diffu0 = v_mean**3/r_mean * 8.*np.log(N_particles_sample)/N_particles_sample * 3./128
    # suffix_tmp = "pos_uniform"
    # Diffu_data = np.loadtxt("../data/Diffu_mainpart_each_particle_"+suffix_tmp+".txt")
    # Diffu_data = Diffu_data[:, 3]
    # plot_diffusion_histogram(Diffu_data, N_bins, Diffu0=Diffu0, suffix=suffix_tmp)
    # suffix_tmp = "pos_uniform_noise"
    # Diffu_data = np.loadtxt("../data/Diffu_mainpart_each_particle_"+suffix_tmp+".txt")
    # Diffu_data = Diffu_data[:, 3]
    # plot_diffusion_histogram(Diffu_data, N_bins, Diffu0=Diffu0, suffix=suffix_tmp)
    # # suffix_tmp = "pos_obs"
    # # Diffu_data = np.loadtxt("../data/Diffu_mainpart_each_particle_"+suffix_tmp+".txt")
    # # plot_diffusion_histogram(Diffu_data, N_bins, suffix=suffix_tmp)

    # ## particles count
    # R_m = None
    # # R_m = 1e-3
    # R_p = r_mean*2.
    # # N_particles_sample = 20000
    # # h_frac = 839807.44
    # # Dim_frac = 2.25
    # N_particles_sample = 40000
    # h_frac = 4602002.33
    # Dim_frac = 2.48
    # eta_D = -1.
    # eta_N = -1.
    # eta_D = rate_t_relax_to_t_cross_diffu(Diffu1o, r_mean=r_mean, v_mean=v_mean)
    # # eta_N = rate_t_relax_to_t_cross_IR2(N_particles_sample, h_frac=h_frac, Dim_frac=Dim_frac, R_p=R_p, R_m=None)
    # ads.DEBUG_PRINT_V(1, R_p, N_particles_sample, eta_D, eta_N, "eta_relax")

    # ## plot some funcs
    # h_frac = 1e4
    # R_p = 10.
    # N_points_plot = 100
    # N_particles_typ = 1e6
    # N_particles_arr = np.geomspace(1e0, 1e11, N_points_plot)
    # Dim_frac_lb = np.array([0.5, 1., 1.5, 2., 2.5, 3.])
    # N_dimlb_plot = len(Dim_frac_lb)
    # Dim_frac_arr = np.linspace(0., 3., N_points_plot-1)
    # Dim_frac_arr = np.append(Dim_frac_arr, 2.)
    # eta_N_arr_count = np.zeros((N_dimlb_plot, N_points_plot))
    # eta_N_arr_dim = np.zeros(N_points_plot)
    # xy_plot_dim_lb = []
    # for j in np.arange(N_dimlb_plot):
    #     for i in np.arange(N_points_plot):
    #         eta_N_arr_count[j, i] = rate_t_relax_to_t_cross_IR2(N_particles_arr[i], h_frac=h_frac, Dim_frac=Dim_frac_lb[j], R_p=R_p, R_m=None)
    #     xy_plot_dim_lb.append([np.log10(N_particles_arr), np.log10(eta_N_arr_count[j]), "Dim_frac={}".format(Dim_frac_lb[j])])
    # odp.plot_funcs_1d(xy_plot_dim_lb, xlabel="N_log10", ylabel="eta_relax_log10", suffix="count_log_log")
    # for i in np.arange(N_points_plot):
    #     eta_N_arr_dim[i] = rate_t_relax_to_t_cross_IR2(N_particles_typ, h_frac=h_frac, Dim_frac=Dim_frac_arr[i], R_p=R_p, R_m=None)
    # odp.plot_funcs_1d([[Dim_frac_arr, np.log10(eta_N_arr_dim), "N={}".format(N_particles_typ)]], xlabel="Dim_frac", ylabel="eta_relax_log10", suffix="dim_1_log")



    # ## old1
    # N_particles = 10000
    # M_total = 100.
    
    # ## Kepler data Angus22
    # data_obs_path = "../data/table1_Angus22_xv_Cartesian.txt" #obs Angus22
    # # data_obs_path = "../data/table1_Gaia_some.txt" #obs Gaia some
    # data_obs = np.loadtxt(data_obs_path, dtype=float)
    # data_obs = ads.remain_only_finite_index(data_obs)
    # pos_obs = data_obs[:, 0:3]
    # odp.plot_data_of_observed_stars(data_obs, proj=[0,1], suffix="Angus22_xy")
    # odp.plot_data_of_observed_stars(data_obs, proj=[1,2], suffix="Angus22_yz")
    # pos_obs_center = np.mean(pos_obs, axis=0)

    # mass_field_uniform = np.ones(N_particles)*M_total/N_particles #uniform DF
    # mass_field = mass_field_uniform*1.
    # # # mass_field = generate_mass_distribution(M_total, N_particles) #truncated Gaussian DF
    # # mass_field = generate_powerlaw_1d(M_total, N_particles)
    # # # ads.DEBUG_PRINT_V(0, mass_field, np.shape(mass_field))
    # # mass_mean = np.mean(mass_field)
    # # perc_less = len(mass_field[mass_field<mass_mean])/N_particles
    # # ads.DEBUG_PRINT_V(1, np.min(mass_field), np.max(mass_field), np.mean(mass_field), perc_less, "mass_DF")
    # # # meandist_init = mean_distance_each_other(initial_positions) #37.
    # # # meandist_new = mean_distance_each_other(new_positions) #38.
    # # # ads.DEBUG_PRINT_V(0, meandist_init, meandist_new)
    # # ads.DEBUG_PRINT_V(1, np.sum(mass_field_uniform**2), np.sum(mass_field**2))
    # # # data_m = None
    # # # mmmm = data_mass_compare(data_m)
    # # # ads.DEBUG_PRINT_V(0, mmmm, "mmmm")

    # pos_target_to_center = [6., 0., 0.]
    # # pos_target_to_center = [0., 0., 0.]
    # # R_size = R_plus

    # # initial_positions, new_positions = particles_3d_Perlin_motion(N_particles, R_size)
    # # new_positions = positions

    # # initial_positions_ = positions
    # # initial_positions, new_positions = particles_3d_Perlin_motion(N_particles, R_size)
    # # initial_positions1, new_positions = particles_3d_Perlin_motion(N_particles, R_size, initial_positions_)

    # initial_positions = positions
    # initial_positions, new_positions = particles_3d_Perlin_motion(N_particles, R_size, initial_positions)

    # Diffu1 = diffusion_descrete_cyl_softenning(initial_positions, initial_positions, mass_field_uniform, None, pos_target_to_center)
    # # Diffu2 = diffusion_descrete_cyl_softenning(initial_positions, initial_positions, mass_field, None, pos_target_to_center)
    # Diffu2 = diffusion_descrete_cyl_softenning(initial_positions, new_positions, mass_field_uniform, None, pos_target_to_center)
    # # Diffu2 = diffusion_descrete_cyl_softenning(initial_positions, new_positions, mass_field, None, pos_target_to_center)
    # # Diffu3 = 1e-10 #?? short distance as 2.0 times of mean distance
    # # Diffu4 = 1e-10 #?? n varies and mean value
    # ads.DEBUG_PRINT_V(1, Diffu1, Diffu2)
    # amplification_rate = Diffu2/Diffu1
    # print("amplification_rate: {}".format(amplification_rate))



    # # ## 1. Perlin noise DF and diffusion
    # # ## data
    # # R_bin, z_bin, v_R_bin, v_phi_bin, v_z_bin, sigma_R_bin, sigma_phi_bin, sigma_z_bin = load_obs_Zhu()
    # # # idx_bin = 0
    # # # sigma_R_bin, sigma_phi_bin, sigma_z_bin = sigma_R_bin[idx_bin], sigma_phi_bin[idx_bin], sigma_z_bin[idx_bin]
    # # ads.DEBUG_PRINT_V(1, np.shape(sigma_R_bin), np.mean(sigma_R_bin), np.std(sigma_R_bin), "sigma_R_bin")
    # # ads.DEBUG_PRINT_V(1, np.shape(sigma_phi_bin), np.mean(sigma_phi_bin), np.std(sigma_phi_bin), "sigma_phi_bin")
    # # ads.DEBUG_PRINT_V(1, np.shape(sigma_z_bin), np.mean(sigma_z_bin), np.std(sigma_z_bin), "sigma_z_bin")
    # # beta_veldisp = beta_velocity_dispersion_deviation(sigma_R_bin, sigma_phi_bin, sigma_z_bin)
    # # ads.DEBUG_PRINT_V(0, np.mean(beta_veldisp), np.min(beta_veldisp), np.max(beta_veldisp), np.std(beta_veldisp))

    # n0 = N/Vol0       # Uniform distribution value (arbitrary units)
    # ads.DEBUG_PRINT_V(1, N, ma, Vol0, n0)

    # ## display Perlin noise
    # N_grid_1d = 100
    # # N_grid_1d = 200
    # k_std = 0.1 #rate 1.003
    # # k_std = 0.5 #rate 1.02
    # # k_std = 10. #rate 1.5
    # # display_geometry_IC_Perlin()
    # n_original, n_total, R_vals, phi_vals, z_vals = display_Perlin_noise(n0, N_grid_1d, k_std)

    # # ## diffusion coef
    # num_samples = 2000
    # # num_samples = 10000
    # D0 = diffusion_distant_encounters_formula(n0)
    # D1 = diffusion_pos_homog_vegas_meshgrid(n_original, R_vals, phi_vals, z_vals, num_samples=num_samples)
    # D2 = diffusion_pos_homog_vegas_meshgrid(n_total, R_vals, phi_vals, z_vals, num_samples=num_samples)
    # ads.DEBUG_PRINT_V(1, D0, D1, D2)
    # amplification_rate = D2[0]/D1[0]
    # print("amplification_rate: {}".format(amplification_rate))

    # ## X. noise diffusion
    # # n_noise_func = n_noise_sin
    # # # n_noise_func = n_noise_sinrandom
    # # D_diffu_value_uniform, integral, error = diffusion_pos_homog_quad(n0, n_noise_nothing)
    # # D_diffu_value, integral, error = diffusion_pos_homog_quad(n0, n_noise_func)
    
    # # n_noise_func = n_noise_sin
    # # n_noise_func = n_noise_sinrandom
    # n_noise_func = n_noise_perlin
    # num_samples = 10000
    # # num_samples = 100000
    # # D_diffu_value_uniform = diffusion_pos_homog_monte_carlo(n0, n_noise_nothing, num_samples=num_samples)
    # # D_diffu_value = diffusion_pos_homog_monte_carlo(n0, n_noise_func, num_samples=num_samples)
    # N_scale_value_uniform = normalization_pos_homog_vegas(n0, n_noise_nothing, num_samples=num_samples)
    # N_scale_value = normalization_pos_homog_vegas(n0, n_noise_func, num_samples=num_samples)
    # ads.DEBUG_PRINT_V(1, N_scale_value_uniform, N_scale_value, N_scale_value/N_scale_value_uniform, "N_scale")
    # D_diffu_value_uniform = diffusion_pos_homog_vegas(n0, n_noise_nothing, num_samples=num_samples)
    # D_diffu_value = diffusion_pos_homog_vegas(n0, n_noise_func, num_samples=num_samples)

    # print("Diffusion Coefficient D_diffu: {}, {}, {}".format(
    #     D_diffu_value_distant_encounters, D_diffu_value_uniform, D_diffu_value
    # ))
    # print("amplification rate: {}".format(D_diffu_value/D_diffu_value_uniform))
