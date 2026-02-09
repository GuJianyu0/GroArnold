#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from noise import pnoise3


def generate_uniform_samples(N, bound_shape="spherical"):
    samples = None # The samples
    rs = 1. # The scale length

    if bound_shape == "spherical": #spherical shape
        r_vals = None
        theta_vals = None
        phi_vals = None
        # but I dnt know how to write ...

    elif bound_shape == "cylindrical": #cylindrical shape
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

if __name__ == "__main__":

    N_particles = 10000
    rs = 1.
    new_positions = np.random.uniform(0, rs*2., (N_particles, 3))
    
    N_iter = 14
    scale = 0.1
    step_size = 2.
    for i in range(N_iter):
        new_positions = Perlin_motion(new_positions, scale=scale, octaves=4, step_size=step_size)
    print("np.shape(new_positions): {}".format(np.shape(new_positions)))
