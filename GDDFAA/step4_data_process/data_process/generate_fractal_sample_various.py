#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Ensure save directory exists
save_path = "../data/small/"
os.makedirs(save_path, exist_ok=True)

# 1. Random Walk
def generate_random_walk_from_uniform(N_samples, Dim_frac):
    """
    Generate a 3D fractal sample using the Random Walk method.
    """
    positions = np.random.uniform(0., 1., (N_samples, 3))
    # N_iter = 1
    N_iter = 30
    # step_size = 1.
    step_size = 10.
    # for i in np.arange(1, N_samples):
    for i in np.arange(0, N_samples):
        walk = np.zeros(3)
        for it in np.arange(N_iter):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            direction = np.array([
                np.sin(phi) * np.cos(theta), 
                np.sin(phi) * np.sin(theta), 
                np.cos(phi)
            ])
            step = (np.random.rand() ** (1 / Dim_frac)) # Scale step size
            walk += step*direction
        # positions[i] = (positions[i-0]+positions[i])/2. + walk*step_size
        positions[i] = positions[i-0] + walk*step_size
    return positions

def generate_random_walk(N_samples, Dim_frac):
    """
    Generate a 3D fractal sample using the Random Walk method.
    """
    step_size = 1.0
    positions = np.zeros((N_samples, 3))
    for i in range(1, N_samples):
        direction = np.random.randn(3) # Random direction in 3D
        direction /= np.linalg.norm(direction) # Normalize to unit vector
        step = step_size * (np.random.rand() ** (1 / Dim_frac)) # Scale step size
        positions[i] = positions[i - 1] + step * direction
    return positions

def generate_biased_random_walk(N_samples, Dim_frac):
    """
    Generate a 3D fractal sample using a biased random walk method.
    """
    step_size = 1.0
    positions = np.zeros((N_samples, 3))
    # positions = np.random.uniform(0., 1., (N_samples, 3))
    for i in range(1, N_samples):
        # direction = np.random.randn(3) # Random direction in 3D
        # direction /= np.linalg.norm(direction) # Normalize to unit vector
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        direction = np.array([
            np.sin(phi) * np.cos(theta), 
            np.sin(phi) * np.sin(theta), 
            np.cos(phi)
        ])
        step = step_size * (np.random.rand() ** (1 / Dim_frac)) # Scale step size
        # step = step_size * (np.random.rand() ** (1 / Dim_frac**2)) # Scale step size
        # bias = 0.  # Bias toward center
        # bias = positions[i-1]*-5e-4  # Bias toward center
        bias = positions[i-1]*-1e-5  # Bias toward center
        # positions[i] = positions[i-1] + step * direction
        positions[i] = positions[i-1] + step * direction + bias
        # positions[i] = (positions[i-1]+positions[int( (i-1)*0.99 )])/2. + step * direction
        # positions[i] = positions[i] + step * direction
    return positions


# 2. L-System
def generate_l_system(N_samples, Dim_frac):
    """
    Generate a 3D fractal sample using an L-System method.
    """
    def apply_rule(axiom, rules, iterations):
        result = axiom
        for _ in range(iterations):
            result = ''.join(rules.get(char, char) for char in result)
        return result

    axiom = "F"
    rules = {"F": "F+F-F-F+F"} # Simple 2D rule
    iterations = int(np.log(N_samples) / np.log(4)) # Scale iterations to approximate N_samples
    step_size = 1.0 / (4 ** iterations)
    path = apply_rule(axiom, rules, iterations)

    # Convert 2D L-System to 3D
    positions = [(0, 0, 0)]
    direction = np.array([1, 0, 0])
    rotation_matrix = lambda theta: np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    for char in path:
        if char == "F":
            positions.append(positions[-1] + step_size * direction)
        elif char == "+":
            direction = rotation_matrix(np.pi / 2) @ direction
        elif char == "-":
            direction = rotation_matrix(-np.pi / 2) @ direction

    positions = np.array(positions)
    if len(positions) > N_samples:
        positions = positions[:N_samples] # Truncate to N_samples
    elif len(positions) < N_samples:
        additional_points = N_samples - len(positions)
        noise = np.random.randn(additional_points, 3) * step_size * 0.1
        positions = np.vstack([positions, positions[-1] + noise])
    return positions


# 3. Diffusion-Limited Aggregation (DLA)
def generate_dla(N_samples, Dim_frac):
    """
    Generate a 3D fractal sample using the Diffusion-Limited Aggregation (DLA) method.
    """
    grid_size = int(N_samples ** (1/3)) + 10
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    center = grid_size // 2
    grid[center, center, center] = True # Seed the cluster
    positions = [np.array([center, center, center])]

    def is_neighboring_cluster(point):
        x, y, z = point
        neighbors = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1)
        ]
        return any(0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size and grid[nx, ny, nz]
            for nx, ny, nz in neighbors)

    while len(positions) < N_samples:
        # Start a random particle far away
        particle = np.random.randint(0, grid_size, size=3)
        while True:
            # Random walk the particle
            direction = np.random.choice([-1, 1], size=3)
            particle = (particle + direction) % grid_size

            # Check if it sticks to the cluster
            if is_neighboring_cluster(particle):
                x, y, z = particle
                grid[x, y, z] = True
                positions.append(np.array([x, y, z]))
                break
    positions = np.array(positions) - center # Center the cluster
    positions = positions / grid_size * N_samples ** (1 / 3) # Scale
    return positions



# Visualization and comparison
if __name__ == "__main__":

    # Parameters
    N_samples = 10000
    Dim_frac = 1.8

    # Generate samples
    methods = {
        "random_walk": generate_random_walk, 
        "random_walk_biased": generate_biased_random_walk, 
        "l_system": generate_l_system, 
        # "dla": generate_dla, 
    }

    for name, method in methods.items():
        samples = method(N_samples, Dim_frac)

        # Plot the 3D positions
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=1)
        ax.set_title(f"{name} (N_samples={N_samples}, Dim_frac={Dim_frac})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()

        # Save the figure
        save_filename = os.path.join(save_path, f"{name}_fractal.png")
        plt.savefig(save_filename)
        plt.close()
        print(f"Saved {save_filename}")
