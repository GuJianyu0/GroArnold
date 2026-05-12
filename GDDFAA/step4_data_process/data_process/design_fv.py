import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

# Constants for escape speed
G = 43007.1  # Gravitational constant
M = 137.0    # Mass
R_b = 75.0   # Radius

# Escape speed
v_e = np.sqrt(2 * G * M / R_b)
print(f"Escape speed v_e = {v_e:.4f}")

# Speed distribution function
def f_speed(v, sigma, v_m=0.):
    """Maxwell-Boltzmann speed distribution function."""
    normalization = 4 * np.pi * v**2 / (2 * np.pi * sigma**2)**(3/2)
    return normalization * np.exp(-v**2 / (2 * sigma**2))

# def f_speed(v, sigma, v_m): #?? wrong
#     """Normalized speed distribution derived from isotropic velocity DF."""
#     normalization = (4 * np.pi * v) / ((2 * np.pi * sigma**2)**(3/2) * v_m)
#     exp_factor = np.exp(-(v**2 + v_m**2) / (2 * sigma**2))
#     sinh_factor = np.sinh(v * v_m / sigma**2)
#     return normalization * exp_factor * sinh_factor

# Cumulative distribution function
def CDF(sigma, v_m, v_e):
    """CDF of the speed distribution up to v_e."""
    result, _ = quad(lambda v: f_speed(v, sigma, v_m), 0, v_e)
    return result

# Solve for sigma and v_m
def solve_sigma_and_vm():
    """Find sigma and v_m such that 90% of particles have speed < v_e."""
    target = 0.9 #particles below v_e
    # target = 0.0001 #particles below v_e
    # target = 0.99 #particles below v_e
    # rate_vm = 1.
    rate_vm = 0.
    # rate_vm = 1e2

    # Constraint: v_m = 1.2 * sigma
    def root_function(sigma):
        v_m = rate_vm * sigma
        return CDF(sigma, v_m, v_e) - target

    # Use root_scalar to solve for sigma
    sol = root_scalar(root_function, bracket=[1, 1000], method='brentq')  # Bracket ensures stability
    sigma = sol.root
    v_m = rate_vm * sigma
    return sigma, v_m

# Main execution
if __name__ == "__main__":

    sigma, v_m = solve_sigma_and_vm()
    print(f"Calculated sigma = {sigma:.4f}")
    print(f"Calculated v_m = {v_m:.4f}")

    # Plot the distribution
    import matplotlib.pyplot as plt

    v_values = np.linspace(0, 1.5 * v_e, 500)
    f_values = [f_speed(v, sigma, v_m) for v in v_values]

    plt.figure(figsize=(8, 5))
    plt.plot(v_values, f_values, label=f"Speed DF (sigma={sigma:.4f}, v_m={v_m:.4f})")
    plt.axvline(v_e, color="red", linestyle="--", label=f"Escape Speed (v_e={v_e:.4f})")
    plt.xlabel("Speed (v)")
    plt.ylabel("f(v)")
    plt.title("Speed Distribution")
    plt.savefig("../data/small/Speed_Distribution.png", format="png", bbox_inches='tight')
    plt.legend()
    plt.grid()
    plt.show()
