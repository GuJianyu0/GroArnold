


import sys
import schwimmbad
from multiprocessing import cpu_count

ncpu = cpu_count()
count_paral = int(ncpu/2)
# count_paral = 1
print("{0} CPUs used".format(count_paral))

mpi_time = !mpiexec -n {ncpu} python3 fit_galaxy_distribution_function.py 2 70 nothing 
mpi_time = float(mpi_time[0])
print("MPI took {0:.1f} seconds".format(mpi_time))



# import numpy as np
# import time
# import emcee
# # import os
# # os.environ["OMP_NUM_THREADS"] = "1"

# from multiprocessing import Pool
# from multiprocessing import cpu_count
# ncpu = cpu_count()
# print("{0} CPUs".format(ncpu))
# count_paral = int(ncpu/2)
# # count_paral = 1
# print("{0} CPUs used".format(count_paral))



# def log_prob(theta):
#     t = time.time() + np.random.uniform(0.005, 0.008)
#     while True:
#         if time.time() >= t:
#             break
#     return -0.5 * np.sum(theta**2)



# if __name__ == '__main__':

#     np.random.seed(42)
#     initial = np.random.randn(32, 5)
#     nwalkers, ndim = initial.shape
#     nsteps = 100

#     serial_time = 21.4
#     print("Serial took {0:.1f} seconds".format(serial_time))

#     with Pool(processes=count_paral) as pool:
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
#         start = time.time()
#         sampler.run_mcmc(initial, nsteps, progress=True)
#         end = time.time()
#         multi_time = end - start
#         print("Multiprocessing took {0:.1f} seconds".format(multi_time))
#         print("{0:.1f} times faster than serial".format(serial_time / multi_time))

#     sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
#     start = time.time()
#     sampler.run_mcmc(initial, nsteps, progress=True)
#     end = time.time()
#     serial_time = end - start
#     print("Serial took {0:.1f} seconds".format(serial_time))
