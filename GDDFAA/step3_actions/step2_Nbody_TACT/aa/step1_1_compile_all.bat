
make clean; #clean .o and .exe in ./aa/ and ./aa/mains/
cd ..; make LAPACK=1 TORUS=1; #make all, here not clean all .o before
cd aa;
#?? there is SCF_coef and need coordtrans .o before
