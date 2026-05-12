## fortran SCF function and C calling

default: scfcoef

current: scfcoef runcoef

example: scfcoef runcoef_example

all: default scfpot tt

clean:
	rm *.o *.exe



scfcoef:
	gfortran -O3 CoefSCF.f -o CoefSCF.exe #-ffree-form

runcoef:
	cp ../../../../../step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/galaxy_general.SCF.txt data.inp #the current triaxialized snapshot
	./CoefSCF.exe
	cp scfocoef scficoef
	cp scfmod scfpar pot_scf.dat force_scf.dat scficoef ../../../../../step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/intermediate/
	cp scfmod scfpar pot_scf.dat force_scf.dat scficoef scfocoef ../../../DataInterface/ #to debug
	cp scfmod scfpar pot_scf.dat force_scf.dat scficoef scfocoef ../../../aa/ #to aa/ ??

runcoef_example:
	cp example.SCF  data.inp
	./CoefSCF.exe
	cp scfocoef scficoef
	# cp scfmod scfpar pot_scf.dat force_scf.dat ../../ #to DataInterface/
	# cp scficoef ../../scficoef
	# cp scfmod scfpar pot_scf.dat force_scf.dat ../../../aa/ #to aa/
	# cp scficoef ../../../aa/scficoef



scfpot:
	gfortran -c SCF-code.f force_SCF.f -fPIC
	# gcc -c inc.c -fPIC
	g++ -c inc.cpp -fPIC

tt:
	g++ -c out.cpp -fPIC
	g++ -fPIC SCF-code.o force_SCF.o inc.o out.o -lm -lgfortran -o out.exe
	./out.exe
