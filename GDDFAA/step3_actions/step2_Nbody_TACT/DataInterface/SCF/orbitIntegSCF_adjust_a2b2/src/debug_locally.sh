#!/usr/bin/env bash
set -e -u
echo -e "Begin to run."
## To debug for prog in this folder

#: 1. modify code in ./
#: modification
echo -e "step1 end"

#: 2. To calculate coef and copy
galaxy_general_name="galaxy_general"
# galaxy_general_name="galaxy_general_Ein_spinL_axisLH0"

# cd ../../SCF_coeff_pot/
# #: make current in SCF_coeff_pot folder
# gfortran -O3 CoefSCF.f -o CoefSCF.exe #-ffree-form
# cp ../../../../../step2_Nbody_simulation/gadget/Gadget-2.0.7/${galaxy_general_name}/aa/galaxy_general.SCF.txt data.inp #the last snapshot
# ./CoefSCF.exe #run in SCF_coeff_pot folder
# cp scfocoef scficoef

# #: donot need to update to galaxy_general file here
# # cp scfmod scfpar pot_scf.dat force_scf.dat scficoef ../../../../../step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/intermediate/
# # cp scfmod scfpar pot_scf.dat force_scf.dat scficoef scfocoef ../../../DataInterface/ #to debug
# # cp scfmod scfpar pot_scf.dat force_scf.dat scficoef scfocoef ../../../aa/ #to aa/ ??
# echo -e "step2 end"

# #: 3. clean, make, copy coef files and run
# #\ note that this will change the currunt foci file some_lmn_foci_Pot.txt
# cd ../orbitIntegSCF_adjust_a2b2/src/
# make default
# # make run
# mpirun -np 4 ./out.exe



# i=10
# cd ./orbit/
# echo -e "#now at folder: ${PWD}"
# # bash TACT_actions_foci.bat #using fit NFW potential in TACT #not used
# if [ -d orbit_${i} ]; then
#     mv ./orbit_${i} ./orbit_${i}_bak
# fi
# mkdir orbit_${i}/
# mv orbit_*.dat orbit_${i}/ #from ./ to ./orbit_${i}, here ./ has many files before this step

# cd ../../../../../../../step2_Nbody_simulation/gadget/Gadget-2.0.7/${galaxy_general_name}/intermediate/
# echo -e "#now at folder: ${PWD}"
# if [ -d orbit_${i} ]; then
#     mv ./orbit_${i} ./orbit_${i}_bak
# fi

# cd ${folder_process}
# python3 recalculate_foci_table.py 10

#: Then see ./some_lmn_foci_Pot.txt and ./orbit/orbit_debug/ check
echo -e "step3 end"

set +e +u
echo -e "End to run."
