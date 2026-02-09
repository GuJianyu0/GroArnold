
rm *.g1

cp ../step1_from_ascii_to_g1_and_run/IC_before_run.g1 ./

#cp some/path/to/run.param ./

echo -e "Running Gadget for simulation the galaxy ..."
mpirun -np 2 ~/workroom/0prog/gadget/Gadget-2.0.7/Gadget2/Gadget2 run.param
# mpirun -np 4 ~/workroom/0prog/gadget/Gadget-2.0.7/Gadget2/Gadget2 run.param > record &
