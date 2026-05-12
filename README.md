#README of GroArnold

# Introduction

GroArnold is a computational framework designed to construct stable action-based distribution functions (DF) -- specifically, the Number Density Distribution Function of Actions for axisymmetric and triaxial galaxies and dark matter halos.

GroArnold: https://github.com/starlifting1/GroArnold

Pipeline: initial conditions (DICE) -> N-body (GADGET-2) -> triaxiality alignment -> actions per particle (TACT-derived Nbody-TACT) -> DF fit & plots.

Environment: Ubuntu >= 20.04

Dependencies: `gcc g++ gfortran cmake gsl fftw3 eigen3 lapack hdf5 mpich python3`

License: GPL-3.0

## Author

Jian-Yu Gu et al.

If you use GroArnold, please cite this repository and Sanders & Binney (2016).

## Quickstart

```bash
# clone and rename
git clone https://github.com/starlifting1/GroArnold.git
mv GroArnold-master/ GroArnold_framework/
cd GroArnold_framework/

#read GroArnold_framework/install_and_run/installation_guide.md and install Dependencies.

# run the full pipeline on the provided example model
cd GroArnold_framework/install_and_run/
python3 workflow_wrapper.py ./initial_conditions/settings_Ein_multicomp_spinL_axisLH/ all
```

## Settings you will edit most

install_and_run/initial_conditions/settings_{MODEL}/unified_settings.yaml — unified high-level config.

IC_DICE_manucraft.params — DICE IC details (masses, scale lengths, counts).

run.param — GADGET-2 runtime options.

More details are in below sections of this file if you need.



# Usage:

## (1) Installation
Environment: Ubuntu 20.04 system or higher.

Dependencies: `gcc`, `g++`, `gfortran`, `cmake`, `gsl`, `fftw3`, `eigen3`, `lapack`, `hdf5`, `mpich`, `python3`

Please download, compile and install packages in the instructions of "GroArnold_framework/install_and_run/installation_guide.md".

Note that the GroArnold framework depends on too much niche packages, so the installation and subsequent development may be inconvenient.

## (2) Settings
File "GroArnold_framework/install_and_run/initial_conditions/settings_{your_galaxy_model}/unified_settings.yaml" is the Unified settings for some manners of the entire GroArnold program; 

Other files in the same folder: "IC_DICE_manucraft.params" for DICE prog about galaxy initial condition (you may need to reset a lot for another galaxy), these files has some important parameters of initial condition, e.g. mass fraction, scale length, particle count of each component, while some parameters would not strict after generating and simulation; 

"run.param" for gadget about Nbody simulation (you may not reset too much).

## (3) Running
### running all
#Suppose your_galaxy_model is Ein_multicomp_spinL_axisLH and your YAML settings, IDCE settings and Gadget2 settings file lives in:

GroArnold_framework/initial_conditions/settings_Ein_multicomp_spinL_axisLH/

```bash
#Run the full pipeline for all models in terminal

python3 workflow_wrapper.py ./initial_conditions/settings_Ein_multicomp_spinL_axisLH/ all

#Or one can leave terminal by nohup

nohup python3 workflow_wrapper.py ./initial_conditions/settings_Ein_multicomp_spinL_axisLH/ all &
```

### resume
#To resume at a specific point; enable --resume_point would not backup galaxy_general/ or galaxy_general_XXX/ folders.

#resume_point 1. initial condition and simulation only (module 1; stops after simulate)

#resume_point 2. triaxial alignment only (module 2)

#resume_point 3. actions only (module 3)

#resume_point 4. DF fit and plots only (module 4)

#resume_point 5. rename current galaxy folder and continue with the next model(s)

#resume_point 6. compare only (post-run compare step)

```bash
#run from example resume point 2 in modelnumber 0 till run to end for all galaxy modelnumers in whole prog (recommanded): run point 2 in modelnumber 1, run point 3 in modelnumber 1, ..., run point 5 in modelnumber 1 (rename galaxy folder (galaxy_general/ as the current) into folder about modelnumber 1); make galaxy folder for modelmuber 3, run point 1 in model number 3, ... (suppose modelnumber 3 is the max number); run point 6 for all modelnumbers (compare models)

python3 workflow_wrapper.py ./initial_conditions/settings_Ein_multicomp_spinL_axisLH/ --resume_point 2 1 --modelnumber 1

#run from example resume point 2 in modelnumber 0 and then exit immediately (debug mode): run point 2 in modelnumber 0, exit the whole prog without any other running

python3 workflow_wrapper.py ./initial_conditions/settings_Ein_multicomp_spinL_axisLH/ --resume_point 2 0 --modelnumber 1
```

### Checking running
#Check whether running (optional).

```bash
jobs -l

ps -aux|egrep 'dice|mpirun|Gadget2|out.exe|data.exe|fit_galaxy_distribution_function.py|plot_action_figs.py'

#kill the controller if detached

kill -9 [the about workflow_wrapper.py list]
```

## (4) Result
suppose your_galaxy_model is Ein_multicomp_spinL_axisLH. The result data is at GroArnold_framework/GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/Ein_multicomp_spinL_axisLH/fit/.



# Modules

This research code integrates several independent modules and modifies upstream projects.

Module 1: Initial condition and N-body simulation (DICE).

Extanded from DICE (DICE is a prog to generate initial conditions of galaxies or streams for gadget simulation, 
https://bitbucket.org/vperret/dice/wiki/browse/, by V. Perret in C code).

Contains code from Gadget2 (https://wwwmpa.mpa-garching.mpg.de/galform/gadget/, in C code). 

Module 2: Triaxiality alignment for galaxy snapshot data.

Module 3: Nbody-TACT. Angle-action variables computation for all particles in a snapshot for triaixial galaxy.

Extanded from TACT (https://github.com/jls713/tact, in C++ code).

- Upstream TACT (GPL-3.0): actions/angles library; please cite Sanders & Binney (2016).

- This fork: adds snapshot I/O, SCF/direct potentials, and batch per-particle actions in the triaxial Stackel Fudge method.

Module 4: Actions-based DF fitting and plotting.

Compute DF by kernel density estiamtion from particle data and fit by mpfit (https://millenia.cars.aps.anl.gov/software/python/mpfit.html).

It is recommanded to learn about the dependencies e.g. DICE, gadget, TACT for more details.

## How to cite
References you may cite if using GroArnold.

#@ARTICLE{} Gu et al. ()

@ARTICLE{P14,
	author = {{Perret}, V. and {Renaud}, F. and {Epinat}, B. and {Amram}, P. and {Bournaud}, F. and {Contini}, T. and {Teyssier}, R. and {Lambert}, J. -C.},
	title = "{Evolution of the mass, size, and star formation rate in high redshift merging galaxies. MIRAGE - A new sample of simulations with detailed stellar feedback}",
	journal = {\aap},
	keywords = {galaxies: evolution, galaxies: formation, galaxies: high-redshift, galaxies: star formation, galaxies: interactions, methods: numerical, Astrophysics - Cosmology and Nongalactic Astrophysics},
	year = 2014,
	month = feb,
	volume = {562},
	eid = {A1},
	pages = {A1},
	doi = {10.1051/0004-6361/201322395},
	archivePrefix = {arXiv},
	eprint = {1307.7130},
	primaryClass = {astro-ph.CO},
	adsurl = {https://ui.adsabs.harvard.edu/abs/2014A&A...562A...1P},
	adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Gadget2,
       author = {{Springel}, Volker},
        title = "{The cosmological simulation code GADGET-2}",
      journal = {\mnras},
     keywords = {methods: numerical, galaxies: interactions, dark matter, Astrophysics},
         year = 2005,
        month = dec,
       volume = {364},
       number = {4},
        pages = {1105-1134},
          doi = {10.1111/j.1365-2966.2005.09655.x},
archivePrefix = {arXiv},
       eprint = {astro-ph/0505010},
 primaryClass = {astro-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2005MNRAS.364.1105S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{H92,
	author = {{Hernquist}, Lars and {Ostriker}, Jeremiah P.},
	title = "{A Self-consistent Field Method for Galactic Dynamics}",
	journal = {\apj},
	keywords = {Celestial Mechanics, Computational Astrophysics, Galaxies, Stellar Motions, Algorithms, Astronomical Models, Dynamical Systems, Numerical Analysis, Astrophysics, CELESTIAL MECHANICS, STELLAR DYNAMICS, METHODS: NUMERICAL},
	year = 1992,
	month = feb,
	volume = {386},
	pages = {375},
	doi = {10.1086/171025},
	adsurl = {https://ui.adsabs.harvard.edu/abs/1992ApJ...386..375H},
	adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{S15,
	author = {{Sanders}, Jason L. and {Binney}, James},
	title = "{A fast algorithm for estimating actions in triaxial potentials}",
	journal = {\mnras},
	keywords = {methods: numerical, Galaxy: kinematics and dynamics, galaxies: kinematics and dynamics, Astrophysics - Astrophysics of Galaxies},
	year = 2015,
	month = mar,
	volume = {447},
	number = {3},
	pages = {2479-2496},
	doi = {10.1093/mnras/stu2598},
	archivePrefix = {arXiv},
	eprint = {1412.2093},
	primaryClass = {astro-ph.GA},
	adsurl = {https://ui.adsabs.harvard.edu/abs/2015MNRAS.447.2479S},
	adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{S16,
	author = {{Sanders}, Jason L. and {Binney}, James},
	title = "{A review of action estimation methods for galactic dynamics}",
	journal = {\mnras},
	keywords = {methods: numerical, galaxies: kinematics and dynamics, Astrophysics - Astrophysics of Galaxies},
	year = 2016,
	month = apr,
	volume = {457},
	number = {2},
	pages = {2107-2121},
	doi = {10.1093/mnras/stw106},
	archivePrefix = {arXiv},
	eprint = {1511.08213},
	primaryClass = {astro-ph.GA},
	adsurl = {https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.2107S},
	adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}



# Structures

## Files dictionary
GroArnold_framework/install_and_run/: the path about installation and running.

GroArnold_framework/GDDFAA/: GDDFAA/ (Galaxy Gynamics Distribution Function based on Angle-Actions) is the path contains most source code files and running result data.

GroArnold_framework/GDDFAA/dependencies/: the path about dependencies.

GroArnold_framework/GDDFAA/step1_galaxy_IC_preprocess/: generating initial condition, about module 1.

GroArnold_framework/GDDFAA/step2_Nbody_simulation/: Nbody simulation, about module 1.

GroArnold_framework/GDDFAA/step3_actions/: triaxility alignment and actions computing, about module 2 and module 3.

GroArnold_framework/GDDFAA/step1_galaxy_IC_preprocess/: fit and plot, about module 4.

## Path to result data
GroArnold_framework/GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general_{modelname}/: galaxy result data; 

where txt/snapshot_%03d.txt%(snapshot_ID) are files about snapshot data like position-velocity before triaxility alignment, 

aa/snapshot_%03d.action.method_all%(snapshot_ID) are files about snapshot data like position-velocity and angle-action after triaxility alignment, 

and the fit/ contains figures.

## Some important code files
GroArnold_framework/install_and_run/workflow_wrapper.py: the workflow controller.

GroArnold_framework/GDDFAA/step1_galaxy_IC_preprocess/step2_select_generate_IC_DICE/src/dice_structure.c: the DICE source file contains profile models of galaxy for user setting before running DICE; we have modified the code for more profile models than original code.

GroArnold_framework/GDDFAA/step3_actions/step2_Nbody_TACT/DataInterface/DataInterface.h: the interface for snapshot data processing (triaxiality alignment by position and velocity center, total angular moment, total moment of inertia and total angular frequency).

GroArnold_framework/GDDFAA/step3_actions/step2_Nbody_TACT/aa/mains/data.cpp: the file we added into Nbody_TACT for the main function to batch per-particle actions in MPI.

GroArnold_framework/GDDFAA/step4_data_process/data_process/fit_galaxy_distribution_function.py: the file to fit DF based on actions.
