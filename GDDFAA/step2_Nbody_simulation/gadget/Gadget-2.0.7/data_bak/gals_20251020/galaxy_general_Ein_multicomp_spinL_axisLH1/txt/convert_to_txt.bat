
set -e -u #no use

echo -e "#now at folder: ${PWD}"
cd ../../../../../step1_galaxy_IC_preprocess/step3_preprocess_IC/step1_from_ascii_to_g1_and_run/
bash step1_compile.bat

cd -
cp ../snapshot/${1} ./
cp ../../../../../step1_galaxy_IC_preprocess/step3_preprocess_IC/step1_from_ascii_to_g1_and_run/read_snapshot.exe ./
echo -e "convert ${1} to txt:"
./read_snapshot.exe 2 ${1}

set +e +u
