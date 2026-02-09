#README of Nbody-TACT.
# Nbody-TACT

Nbody-TACT: https://github.com/starlifting1/Nbody-TACT/

- Upstream TACT (GPL-3.0): actions/angles library; please cite Sanders & Binney (2016) and Triaxial Stackel Fudge (2015).

- This fork: adds snapshot I/O, SCF/direct potentials, and batch per-particle actions in the triaxial Stackel Fudge method.

Nbody-TACT is a sub-project of GroArnold. 
GroArnold: https://github.com/starlifting1/GroArnold  
In-tree path: `GDDFAA/step3_actions/step2_Nbody_TACT/`

## Author

Jianyu Gu et al.

If you use Nbody-TACT, please cite this repository and Sanders & Binney (2016).

## Requirements

Build tools: `g++` (C++11), `make`

Libraries: `gsl`, `fftw3`, `eigen3`, `lapack`, `hdf5`, `mpich`, `python3`

## Installation

```bash
cd Nbody-TACT/aa/
bash step1_1_compile_all.bat
```

## Running

```bash
cd Nbody-TACT/aa/
bash step1_2_compile_SCF.bat
bash step1_3_prepare_foci.bat
bash step2_run.bat
```

## Changing from TACT (Sanders & Binney, 2016)

What we extended in Nbody-TACT from original TACT:

Most of the extensions are about snapshot data processing, potential computation (direct summation and SCF) and some robustness extensions to Nbody SCF potential.

The modifications to original TACT are marked as "//gjy change" or "gjy add".

Most of files step3_actions/step2_Nbody_TACT/pot/Inc/potential.h (and cpp), step2_Nbody_TACT/aa/inc/aa.h (and cpp), step2_Nbody_TACT/aa/Inc/stackel_aa.h (and cpp), step2_Nbody_TACT/aa/Inc/lmn_orb.h (and cpp).

We also add some other files like step2_Nbody_TACT/DataInterface.h (and cpp) for snapshot data processing (triaxiality alignment by position and velocity center, total angular moment, total moment of inertia and total angular frequency), and step2_Nbody_TACT/aa/mains/data.cpp for the main function to batch per-particle actions in MPI.




#README of original TACT (Sanders & Binney, 2016).
# tact

[![Build Status](https://travis-ci.org/jls713/tact.svg?branch=master)](https://travis-ci.org/jls713/tact)

Code for calculating actions, angles and frequencies in various ways

## Author

Jason Sanders -- jls at ast dot cam dot ac dot uk

Please cite the accompanying paper Sanders & Binney (2016) if you find the code useful.

## Requirements

1. [gsl](http://www.gnu.org/software/gsl/)

## Installation

* Make sure environment variable $(CXX) gives c++ compiler or specify compiler path in Makefile.inc (need a C++-11 compatible compiler, it compiles with clang 3.4 and g++ 4.8 or 4.9)
* Specify path to gsl in Makefile.inc
* Run make

This will install the basic package. To access all the features one should also install Torus and LAPACK:

* Some code uses [Torus](https://github.com/PaulMcMillan-Astro/Torus). To use this install Torus (use 'make CPP="$CXX -fPIC"' to ensure libraries are compiled with fPIC flag) , add path to Torus to Makefile.inc and run 'make TORUS=1'. Currently there are problems compiling with Torus using clang. This appears to be due to different compiler flags in Torus and tact. If users wish to use clang then they should make sure the compiler flags are the same for both.
* Some code uses [LAPACK](http://www.netlib.org/lapack/). To use this install LAPACK, add path to LAPACK to Makefile.inc and run 'make LAPACK=1'
* To do both run 'make TORUS=1 LAPACK=1'

One can also compile the code into a python module. This requires the boost library and the paths in Makefile.inc to python and boost to be correctly set. With these set either run 'make python' or use the setup.py in /aa like 'TORUS=1 LAPACK=1 python setup.py install'.

There is also test code that runs using googletest. However, there is a quick test to run detailed below.

## Methods

1. Analytic potentials (Isochrone and Harmonic oscillator)
2. General spherical potentials
3. [Cylindrical Adiabatic Approximation (CAA)](http://arxiv.org/abs/1109.4417), Schoenrich & Binney (2012)
4. Spheroidal Adiabatic Approximation (SAA) (unpublished, in my thesis)
5. [Stackel fitting](http://arxiv.org/abs/1208.2813), Sanders (2012)
6. [Axisymmetric Stackel fudge](http://arxiv.org/abs/1207.4910), Binney (2012)
7. [Interpolation using Axisymmetric Stackel fudge](http://arxiv.org/abs/1207.4910), Binney (2012)
8. [Triaxial Stackel fudge](http://arxiv.org/abs/1412.2093), Sanders & Binney (2015)
9. [Generating function from orbit (axisymmetric and triaxial, O2GF)](http://arxiv.org/abs/1401.3600), Sanders & Binney (2014)
10. [Average generating function from orbit (AvGF)](http://arxiv.org/abs/1401.2985), Bovy (2014), [Fox (2014)](http://arxiv.org/abs/1407.1688)
11. [Iterative Torus Machine (ItTC)](http://arxiv.org/abs/1412.2093), Sanders & Binney (2015)

## Docs

There is some documentation that can be produced by doxygen by running
```
make docs
```
The python library also contains doc strings.

## Test case

After successful compilation the command
```
cd aa; ./mains/test_actions.exe 8. 1. 0.2 40. 200. 50. acts.dat
```
should integrate an orbit with initial conditions X=(8. 1. 0.2) kpc and V = (40. 200. 50.)km/s in the potential

Phi(x)=Vc^2/2 log(R^2+(z/q)^2)

with Vc=220km/s and q=0.9 (or in the Piffl 2014 potential if Torus is installed) and compute the actions for each point using a variety of methods. The results are output in acts.dat with two columns per method (JR and Jz).

## Paper 

The accompanying paper is Sanders & Binney (2016). [action_comp_plots.py](aa/action_comp_plots.py) reproduces all but one of the plots in the paper. To produce the data for these plots the following scripts are available. 

1. Fig. 2 data is produced by the command 
```
./orbits.sh
```
2. Fig. 3, 4, 5, 6 data are produced by the command
```
./mains/./many_tori.exe many_tori_output.dat
```
3. Fig. 7 data are produced by
```
./orbits_converg.sh
```

