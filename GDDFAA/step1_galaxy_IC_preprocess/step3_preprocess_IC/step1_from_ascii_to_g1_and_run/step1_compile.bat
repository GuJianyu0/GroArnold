
rm *.exe
# cd ../../../step3_actions/step2_Nbody_TACT/DataInterface/
# make default
# cd -
g++ read_snapshot.cpp ../../../step3_actions/step2_Nbody_TACT/DataInterface/Gadget2FormatData_io.cpp -o read_snapshot.exe
