#!/usr/bin/env bash
# Installation
#This file is an install guide for GroArnold (GDDFAA) prog -- a framework for compution DF based on angle-actions of triaxial galaxies.

#[note] !!!! This is just a instruction, and do not run this directly.

```bash
#prepare and install
echo -e "Begin to install ... \n"
set -e -u
```



## step0. Preparing

```bash
#: [note] one should replace folder_main into your actual path to GroArnold_framework/
# folder_main=path/to/GroArnold_framework/
folder_main=${HOME}/workroom/0prog/GroArnold_framework/
folder_packages=${folder_main}packages/
folder_dependencies=${folder_main}GDDFAA/dependencies/
folder_thisFile=${folder_main}install_and_run/ #it is the path of this the file
folder_make_DICE=${folder_main}GDDFAA/step1_galaxy_IC_preprocess/step2_select_generate_IC_DICE/
folder_simulation=${folder_main}GDDFAA/step2_Nbody_simulation/gadget/
folder_dependencies=${folder_main}GDDFAA/dependencies/
folder_actions=${folder_main}GDDFAA/step3_actions/

cd ${folder_main}
echo -e "#now at folder: ${PWD}"
```



## step1. Dependencies

```bash
# configure enviroment in Ubuntu 20.04 or higher
#[note]: If you have configured them but the version is wrong, please install them in a local folder.

#: basic denpendencies
sudo apt update
sudo apt install gcc g++ gfortran mpich
sudo apt install python3 python3-pip
sudo apt install libgsl-dev
sudo apt install libfftw3-dev
sudo apt install liblapack-dev
sudo apt install libeigen3-dev

#: optional
# sudo apt install git wget csh cmake
# sudo apt install libhdf5-dev
# sudo apt install libopencv-dev
# sudo apt install libboost-dev

# git packages_XXX
```



## Install sub-progs by existing source files in denpendencies (quickly but not standard)
#GroArnold_framework/GDDFAA/dependencies/ has copy the source files)

### use source files of modified-DICE, gadget2, etc. in GroArnold
```bash
# do like "#note: compile" in workflow_wrapper.py

cd ${folder_main}GDDFAA/step1_galaxy_IC_preprocess/step2_select_generate_IC_DICE/
bash step1_compile.bat

cd ${folder_main}GDDFAA/step1_galaxy_IC_preprocess/step3_preprocess_IC/step1_from_ascii_to_g1_and_run/
bash step1_compile.bat

cd ${folder_main}GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/
bash step1_compile.bat
```

### install ebf, Torus, Nbody-TACT
```bash
cd ${folder_main}
mkdir packages/; cd packages/

# 1. install ebf
# download ebf like https://sourceforge.net/projects/ebfformat/files/libebf/idl/libebf_idl-0.0.3.tar.gz/download

# move libebf_c_cpp-0.0.3.tar to packages/
cd ${folder_packages}
tar -zxvf libebf_c_cpp-0.0.3.tar

cd libebf_c_cpp-0.0.3/
./configure --prefix=${folder_dependencies} #use your full path name
make && make install

# 2. install Torus
# download Torus from https://github.com/jls713/Torus

# install Torus-master next to step2_Nbody_TACT/
cd ${folder_packages}
tar -zxvf Torus-master.tar.gz
cp -rfap Torus-master/ ${folder_actions}
cd ${folder_actions}Torus-master/

# replace certain makefile of Torus for GroArnold to the original makefile, and this is for link to ${folder_dependencies}
mv makefile makefile.original
cp ../step2_Nbody_TACT/makefile_for_TACT_Torus makefile

make clean && make

# 3. compile Nbody_TACT in GroArnold
cd ${folder_main}GDDFAA/step3_actions/step_Nbody_TACT/aa/
bash step1_1_compile_all.bat
```



## Install sub-progs step by step (alternative)

```bash
### 1. Install programs for initial conditions and other functions
is_install_IC=1
if [ ${is_install_IC} -eq 1 ]; then
    echo -e "install DICE:\n"
    #: move (only at the first time)
    # tar -zxvf DICE_galaxy_initial_conditions.tar.gz
    # mv DICE_galaxy_initial_conditions/ ${folder_main}GDDFAA/step1_galaxy_IC_preprocess/
    # cd ${folder_main}GDDFAA/step1_galaxy_IC_preprocess/
    # mv DICE_galaxy_initial_conditions/ step2_select_generate_IC_DICE/
    # cd step2_select_generate_IC_DICE/
    cd ${folder_main}GDDFAA/step1_galaxy_IC_preprocess/step2_select_generate_IC_DICE/
    if [ -d ./dice_local/ ]; then
        rm -rf ./dice_local/ #[note] rm
    fi
    if [ -d ./build/ ]; then
        rm -rf ./build/ #[note] rm
    fi
    mkdir dice_local/
    mkdir build/ && cd build/
    # make clean
    CXX=gcc cmake .. -DENABLE_THREADS=ON -DCMAKE_INSTALL_PREFIX=${folder_make_DICE}dice_local/
    make && make install
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "install AGAMA:\n"
    cd ${folder_main}GDDFAA/step1_galaxy_IC_preprocess/step2_select_generate_IC_and_get_DFA_AGAMA_not_install/
    #() install
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "install cold_python:\n"
    #nothing
    echo -e "#now at folder: ${PWD}"
fi



### 2. install packages for simulation: 
is_install_simulation=0
if [ ${is_install_simulation} -eq 1 ]; then
    #\ mpi, gsl, fftw2, hdf5 -> Gadget2.0.7
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "install gsl (old version):\n"
    #: download old version gsl-1.16 at http://www.gnu.org/software/gsl/
    tar -zxvf gsl-1.16.tar.gz
    cd gsl-1.16/
    ./configure --prefix=${folder_dependencies} #PWD #/usr/local/
    make && make install
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "install fftw2 (with float):\n"
    tar -zxvf fftw-2.1.5.tar.gz
    # mv fftw-2.1.5/ ../ && cd ../
    cd fftw-2.1.5/
    ./configure --enable-mpi --enable-type-prefix --enable-float --prefix=${folder_dependencies} #PWD #/usr/local/
    make && make install
    ./configure --enable-mpi --enable-type-prefix --prefix=${folder_dependencies} #PWD #/usr/local/
    make && make install
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "install hdf5 (optional):\n"
    # sudo apt
    #[select]:
    tar -xzf hdf5-1.6.9.tar.gz
    # mv hdf5-1.6.9/ ../ && cd ../
    cd hdf5-1.6.9/
    ./configure --prefix=${folder_dependencies} #PWD #/usr/local/
    make && make install
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "install Gadget2:\n"
    #[note]: Some settings of original package has been changed for this work: 
    #\ folder structure and  makefile contents.
    #\ The link of original package is above.
    #\ at http://wwwmpa.mpa-garching.mpg.de/gadget/gadget-2.0.7.tar.gz
    tar -zxvf Gadget-2.0.7.tar.gz
    cp -rfap Gadget-2.0.7/* ${folder_simulation}Gadget-2.0.7/ #[note] rm
    cd ${folder_simulation}Gadget-2.0.7/Gadget2/
    echo -e "#now at folder: ${PWD}"
    cp ${folder_packages}makefiles/Makefile_Gadget2 ${folder_simulation}Gadget-2.0.7/Gadget2/Makefile
    make clean; make
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"
fi



is_install_actions=0
if [ ${is_install_actions} -eq 1 ]; then
    ### 3. install packages for actions
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "make for dependencies:\n"
    #() install gsl-1.16 gtest ebf lapack blas (boost) Torus Nbody_TACT

    echo -e "install gsl (old version):\n"
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "install gtest:\n"
    # sudo apt install libgtest-dev
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "install ebf:\n"
    tar -zxvf libebf_c_cpp-0.0.3.tar
    cd libebf_c_cpp-0.0.3/
    ./configure --prefix=${folder_dependencies}
    make && make install
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "install lapack and blas:\n"
    # http://www.netlib.org/lapack/
    # sudo apt install liblapack-dev

    # #: or
    #: change the content in Makefile
    # lib: lapacklib tmglib
    # #lib: blaslib variants lapacklib tmglib
    #: into
    # #lib: lapacklib tmglib
    # lib: blaslib variants lapacklib tmglib

    #[note]: The make file of lapack has been changed.
    # tar -zxvf lapack-3.4.2.tgz
    # cd lapack-3.4.2/
    # cp make.inc.example make.inc
    # make
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "install boost (optional if changed to python):\n"
    # https://www.boost.org
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "install Nbody_TACT and Torus (with SCF knn other):\n"
    #[note]: The make file of Torus has been changed.
    #\ original TACT: https://github.com/jls713/tact
    #\ TACT has been changed to Nbody_TACT.
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"
    tar -zxvf Torus-master.tar.gz
    cp -rfap Torus-master/ ${folder_actions}
    cd ${folder_actions}Torus-master/
    echo -e "#now at folder: ${PWD}"
    # ./configure --prefix=${PWD} #nothing
    make clean
    make
    # about Nbody_TACT ...
    cd ${folder_packages}
    echo -e "#now at folder: ${PWD}"

    echo -e "make Nbody_TACT and Torus (with SCF knn other):\n"
    cd ${folder_actions}
    echo -e "#now at folder: ${PWD}"
    cd ${folder_actions}step2_Nbody_TACT/aa/;
    make clean && cd ../;
    make LAPACK=1 TORUS=1;
    cd aa;
    cd ${folder_actions}
    echo -e "#now at folder: ${PWD}"
fi
```

## Install python packages

```bash
pip3 install numpy matplotlib scipy
pip3 install sklearn pandas
pip3 install opencv-python
pip3 install emcee corner

# pip3 install pdb tqdm
# pip3 install astropy galpy GALA 
```

```bash
set +e +u
echo -e "End to install.\n"
```
