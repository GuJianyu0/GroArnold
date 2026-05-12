if [ -d ./dice_local/ ]; then
    rm -rf ./dice_local/ #[note] rm
fi
if [ -d ./build/ ]; then
    rm -rf ./build/ #[note] rm
fi
mkdir dice_local/
mkdir -p build/
mkdir -p build/bin/
cd build/
make clean
CXX=gcc cmake .. -DENABLE_THREADS=ON -DCMAKE_INSTALL_PREFIX=${folder_make_DICE}dice_local/
make && make install

cd ../
cp build/bin/dice run/
echo -e "#now at folder: ${PWD}"
