
echo -e "Running Gadget for simulation the galaxy ..."
mpirun -np ${1} Gadget2 run.param
# mpirun -np 4 ~/workroom/0prog/gadget/Gadget-2.0.7/Gadget2/Gadget2 run.param > record &
