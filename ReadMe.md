######################################################################################
# Instructions
GroArnold is a computational framework designed to construct stable action-based distribution functions (DF) -- specifically, the Number Density Distribution Function of Actions for axisymmetric and triaxial galaxies and dark matter halos.
######################################################################################

## 1. Introduction
GroArnold integrates several independent modules: an initial conditions (IC) module, an N-body simulation module, a data preprocessing module, 
an actions estimation module (the aforementioned Nbody-TACT), and a data analysis module.

Module 1: Initial condition and N-body simulation (DICE).
Extansions from DICE (DICE is a prog to generate initial conditions of galaxies or streams for gadget simulation, 
https://bitbucket.org/vperret/dice/wiki/browse/, by V. Perret in C code).

Module 2: Triaxiality alignment for galaxy snapshot data.
Extansions from Gadget2 (https://wwwmpa.mpa-garching.mpg.de/galform/gadget/, in C code). 

Module 3: Nbody-TACT. Angle-action variables computation for all particles in a snapshot for triaixial galaxy.
Extansions from TACT (https://github.com/jls713/tact, in C++ code).

Module 4: Actions-based DF fitting and plotting
Compute DF by kernel density estiamtion from particle data and fit by mpfit (https://millenia.cars.aps.anl.gov/software/python/mpfit.html).

## 2. How to use:
### (1) Installation
Environment: Ubuntu 20.04 system or higher.
Dependencies: gcc g++ gfortran cmake mpich libgsl-dev libfftw3-dev 
libeigen3-dev liblapack-dev libhdf5-devlibgtest-dev 
libopencv-dev python3.
Please download, compile and install packages in "./install_and_run/installation_guide.md".
Note that the GroArnold framework depends on too much niche packages, so the installation may be inconvenient.

### (2) Settings
File "./install_and_run/initial_conditions/settings_{your_galaxy_model}/unified_settings.yaml" is the Unified settings 
for some manners of the entire GroArnold program; 
Other files in the same folder: "IC_DICE_manucraft.params" for DICE prog about galaxy initial condition (you may need to reset a lot for another galaxy), 
these files has some important parameters of initial condition, e.g. mass fraction, scale length, particle count of each component, 
while some parameters would not strict after generating and simulation; 
"run.param" for gadget about Nbody simulation (you may not reset too much).

### (3) Running
#### running all
#Suppose your_galaxy_model is Ein_multicomp_spinL_axisLH and your YAML settings, IDCE settings and Gadget2 settings file lives in:
./initial_conditions/settings_Ein_multicomp_spinL_axisLH/

#run the full pipeline for all models in YAML
#one need open terminal
python3 workflow_wrapper.py ./initial_conditions/settings_Ein_multicomp_spinL_axisLH/ all

#one can leave terminal by nohup
nohup python3 workflow_wrapper.py ./initial_conditions/settings_Ein_multicomp_spinL_axisLH/ all &

#### resume
#To resume at a specific point; enable --resume_point would not backup galaxy_general/ or galaxy_general_XXX/ folders.
#resume_point 1. initial condition and simulation only (module 1; stops after simulate)
#resume_point 2. triaxial alignment only (module 2)
#resume_point 3. actions only (module 3)
#resume_point 4. DF fit and plots only (module 4)
#resume_point 5. rename current galaxy folder and continue with the next model(s)
#resume_point 6. compare only (post-run compare step)

#run from example resume point 2 in modelnumber 0 till run to end for all galaxy modelnumers in whole prog (recommanded): run point 2 in modelnumber 1, run point 3 in modelnumber 1, ..., run point 5 in modelnumber 1 (rename galaxy folder (galaxy_general/ as the current) into folder about modelnumber 1); make galaxy folder for modelmuber 3, run point 1 in model number 3, ... (suppose modelnumber 3 is the max number); run point 6 for all modelnumbers (compare models)
python3 workflow_wrapper.py ./initial_conditions/settings_Ein_multicomp_spinL_axisLH/ --resume_point 2 1 --modelnumber 1

#run from example resume point 2 in modelnumber 0 and then exit immediately (debug mode): run point 2 in modelnumber 0, exit the whole prog without any other running
python3 workflow_wrapper.py ./initial_conditions/settings_Ein_multicomp_spinL_axisLH/ --resume_point 2 0 --modelnumber 1

#### checking running
#check whether running
jobs -l
ps -aux|egrep 'dice|mpirun|Gadget2|out.exe|data.exe|fit_galaxy_distribution_function.py|plot_action_figs.py'

#kill the controller if detached
kill [the id about workflow_wrapper.py list]

### (4) Result
suppose your_galaxy_model is Ein_multicomp_spinL_axisLH. The result data is at ./GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/Ein_multicomp_spinL_axisLH/fit/.