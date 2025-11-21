#fractal
'''
Fractal Dimension and Density of States
Fractal Dimension (
�
D):
The fractal dimension 
�
D describes how the mass or number of particles (
�
N) scales with the length scale (
�
L) in the medium:

�
(
�
)
∼
�
�
N(L)∼L 
D
 
Here, 
�
(
�
)
N(L) is the number of particles within a region of size 
�
L.

Density of States (DoS) in a Fractal Medium:
In a fractal medium, the density of states 
�
(
�
)
g(E) (number of states per unit energy interval) can be influenced by the fractal dimension. If 
�
0
(
�
)
g 
0
​
 (E) represents the density of states in a regular medium, the fractal density of states can be modified to reflect the fractal dimension 
�
D:

�
(
�
)
∼
�
0
(
�
)
⋅
�
�
−
�
2
g(E)∼g 
0
​
 (E)⋅E 
2
D−d
​
 
 
where 
�
d is the topological dimension of the space (e.g., 
�
=
3
d=3 in a 3D space). This modification indicates how the available states are redistributed due to the fractal nature of the medium.

Interaction Strengths and Resonance
Interaction Potential in a Fractal Medium:
The interaction potential 
�
(
�
)
V(r) between particles at a distance 
�
r in a fractal medium can also be affected by the fractal dimension. For example, if the potential in a uniform medium follows 
�
0
(
�
)
V 
0
​
 (r), the potential in a fractal medium might be scaled by a factor related to the fractal dimension:

�
(
�
)
∼
�
0
(
�
)
⋅
�
�
−
�
V(r)∼V 
0
​
 (r)⋅r 
D−d
 
This scaling captures the varying interaction strengths due to the fractal structure.

Resonance Condition:
The resonance condition in a fractal medium, where particles interact with collective modes, can be influenced by the modified density of states and interaction strengths. If the resonance condition in a regular medium is given by:

�
⋅
�
=
�
k⋅v=ω
where 
�
k is the wave vector, 
�
v is the velocity, and 
�
ω is the frequency, in a fractal medium, this condition can be modified to include the fractal scaling:

�
⋅
�
∼
�
⋅
�
�
−
�
2
k⋅v∼ω⋅r 
2
D−d
​
 
 
Relaxation Time in a Fractal Medium
Fractional Relaxation Time:
The relaxation time 
�
relax
t 
relax
​
  in a fractal medium can be estimated by considering the modified diffusion coefficient and interaction strengths. If the relaxation time in a regular medium is 
�
0
t 
0
​
 , in a fractal medium, it can be scaled as:
�
relax
∼
�
0
⋅
�
�
−
�
2
t 
relax
​
 ∼t 
0
​
 ⋅N 
2
d−D
​
 
 

where 
�
N is the number of particles. This formula indicates that the relaxation time depends on the fractal dimension and the number of particles, reflecting how the fractal structure accelerates or decelerates the relaxation process.
'''

import numpy as np
from scipy.integrate import quad

# Define the function to integrate
def integrand(x):
    return np.exp(-x)

# Define the integration limits
a = 0
b = np.inf

# Perform the integration
result, error = quad(integrand, a, b)

print("Integral result:", result)
print("Estimated error:", error)



import numpy as np
import matplotlib.pyplot as plt

def correlation_dimension(data, num_bins=50):
    """
    Calculate the correlation dimension of a set of points.
    
    Parameters:
    data (numpy array): A 3xN array where N is the number of points.
    num_bins (int): Number of bins to use for the histogram of distances.
    
    Returns:
    float: The correlation dimension.
    """
    N = data.shape[1]
    
    # Calculate pairwise distances
    distances = []
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(data[:, i] - data[:, j])
            distances.append(dist)
    distances = np.array(distances)
    
    # Compute the correlation sum C(r)
    max_distance = np.max(distances)
    bins = np.logspace(np.log10(np.min(distances)), np.log10(max_distance), num_bins)
    counts, _ = np.histogram(distances, bins=bins)
    cumulative_counts = np.cumsum(counts)
    correlation_sum = cumulative_counts / cumulative_counts[-1]
    
    # Perform linear regression on log-log scale
    log_r = np.log10(bins[:-1])
    log_Cr = np.log10(correlation_sum)
    slope, intercept = np.polyfit(log_r, log_Cr, 1)
    
    # Plot the results
    plt.figure()
    plt.plot(log_r, log_Cr, 'o', label='Data')
    plt.plot(log_r, slope * log_r + intercept, '-', label=f'Fit: slope={slope:.3f}')
    plt.xlabel('log10(r)')
    plt.ylabel('log10(C(r))')
    plt.legend()
    plt.title('Correlation Dimension Calculation')
    plt.show()
    
    return slope

# Example usage with random data
N = 1000  # Number of points
data = np.random.rand(3, N)  # 3xN array of random points in [0, 1]

D = correlation_dimension(data)
print(f"Estimated Correlation Dimension: {D}")

def count_particles(data, radius):
    """
    Count the number of particles within a given radius.
    
    Parameters:
    data (numpy array): A 3xN array where N is the number of points.
    radius (float): Radius of the spherical volume.
    
    Returns:
    int: Number of particles within the radius.
    """
    center = np.mean(data, axis=1)
    distances = np.linalg.norm(data - center[:, np.newaxis], axis=0)
    count = np.sum(distances <= radius)
    return count

# Example usage
R = 1.0
N_3 = count_particles(data, R)
print(f"Particle Count in 3D Sphere: {N_3}")

def eta(D):
    """
    Calculate the factor η(D) for fractal volume integration.
    
    Parameters:
    D (float): Fractal dimension.
    
    Returns:
    float: The factor η(D).
    """
    from scipy.special import gamma
    return (2**(5-D) * np.pi * gamma(3/2)) / (D * gamma(D/2))

def fractal_number_density(N_3, R, D):
    """
    Calculate the fractal number density.
    
    Parameters:
    N_3 (int): Actual particle count in 3D.
    R (float): Radius of the spherical volume.
    D (float): Fractal dimension.
    
    Returns:
    float: Fractal number density.
    """
    return N_3 / (R**D * eta(D))

# Example usage
n_D0 = fractal_number_density(N_3, R, D)
print(f"Fractal Number Density: {n_D0}")
