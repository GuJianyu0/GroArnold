
# snapshot_dealwith=${1}
# # copy snapshot coef here

cd ../DataInterface/SCF/orbitIntegSCF_adjust_a2b2/src/
make default
# make run
mpirun -np 6 ./out.exe
cd ../../../../aa/

# cp ../DataInterface/SCF/orbitIntegSCF_adjust_a2b2/src/some_lmn_foci_Pot.txt ../../../step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/intermediate/snapshot_${snapshot_dealwith}_lmn_foci_Pot.txt

