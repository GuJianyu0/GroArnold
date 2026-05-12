
#now we are in GDDFAA/step3_actions/step2_Nbody_TACT/aa/

#make some obj/ folders
mkdir -p ../../step1_preprocess/
mkdir -p ../general/coordtransforms/obj/
mkdir -p ../pot/obj/
mkdir -p ../aa/obj/
mkdir -p ../aa/lib/
mkdir -p ../aa/lib/
# cp step3_actions/makefile.TACT_Torus ../../Torus-master/
# mkdir -p ../../Torus-master/lib/

#compile
make clean; #clean .o and .exe in ./aa/ and ./aa/mains/
cd ..; make LAPACK=1 TORUS=1; #make all, here not clean all .o before
cd aa;

