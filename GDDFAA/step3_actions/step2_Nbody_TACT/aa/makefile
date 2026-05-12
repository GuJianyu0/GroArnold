
# Makefile for /aa

#------------------------------------------------------
include ../Makefile.inc
#------------------------------------------------------

SRC = src/
POT = pot/
OBJ = obj/
OBJ_POT = ../pot/obj/
#gjy add: it do have a path of DataInterface.h, potential.h and cpp, and #include absolute path; no use
FALCONOBJ=$(OBJ)falcONobj/
#these are /aa/inc and so on

#------------------------------------------------------
# Set paths to other code

GENERALPATH = ../general/
CTPATH = $(GENERALPATH)coordtransforms/
NEMOPATH = /data/jls/nemo/
FALCONPATH = $(NEMOPATH)usr/dehnen/falcON.P/
POTPATH = ../pot/

GNUPLOTLIBPATH = $(GENERALPATH)gnuplot
NUMRECLIBPATH = $(GENERALPATH)jamestools/numrec/
JAMESLIBPATH = $(GENERALPATH)jamestools/jamestools/
POTLIBPATH = ../pot/
CUBALIBPATH = -L$(GENERALPATH)cuba
CUBA = $(CUBALIBPATH) -lcuba

#compile option: make TORUS=1
ifeq ($(TORUS), 1)
	TORUSINCPATH = -I$(TORUSPATH)/src/ -I$(TORUSPATH)/src/utils/ -I$(TORUSPATH)/src/pot
	TORUSLIBPATH = -L$(TORUSPATH)/obj/ -L$(TORUSPATH)/WDlib/obj/ -L$(EBFPATH)/lib/
	TORUSLIBS = -lTorus -lebf_cpp
else
	TORUSINCPATH =
	TORUSLIBPATH =
	TORUSLIBS =
endif

# make LAPACK=1
ifeq ($(LAPACK), 1)
	LAPACKPATH=$(LAPACKPATH)
	LAPACKLIBS=-llapack -lblas
else
	LAPACKLIBPATH=
	LAPACKLIBS=
endif
#------------------------------------------------------
# Set include and library paths

INC = -I../../ -I../ -Iinc/ $(TORUSINCPATH) -I$(GENERALPATH) -I$(GSLPATH)include/ -I$(POTPATH)inc/ -I$(CTPATH)inc/ -I/usr/include/eigen3/

INCPY = -I$(PYTHONINCPATH) -I$(BOOSTINCPATH)

LIBPATHS =
# -Wl,-rpath=$(POTLIBPATH)
# -Wl,-R$(GENERALPATH) -Wl,-R$(NUMRECLIBPATH) -Wl,-R$(JAMESLIBPATH)\
		   # -Wl,-R$(CUBALIBPATH) -Wl,-R$(GNUPLOTLIBPATH) -Wl,-R$(TORUSLIBPATH)\
		   # -Wl,-R$(GSLPATH)lib/ -Wl,-R$(CTPATH)

LIB = -Wl,-rpath,$(TORUSPATH)/obj/ -Wl,-rpath,$(TORUSPATH)/WDlib/obj/  -Wl,-rpath,$(EBFPATH)/lib/ -L$(GENERALPATH) -L$(NUMRECLIBPATH) -L$(JAMESLIBPATH) \
	$(CUBALIBPATH)\
	  -L$(GNUPLOTLIBPATH) $(TORUSLIBPATH) -L$(GSLPATH)lib/ -L$(POTLIBPATH)\
	  -L$(CTPATH) $(TORUSLIBS) -lpot_js -lcoords \
	  -lpress_cp -lplot -lgsl -lgslcblas -lm $(EXTRACLIB) $(LAPACKLIBPATH) -ljamestools $(LAPACKLIBS) $(TORUSFLAG) $(LAPACKFLAG) $(LIBPATHS)\

BOOSTLIB = -lboost_python -lpython2.7
LIBPY = $(PYTHONLIBPATH) $(BOOSTLIBPATH) $(BOOSTLIB)

#gjy add: DataInterface
LIBDATASO = -L../DataInterface/ -lDataInterface -lgfortran
ADDDATALIBPATH = -Wl,--rpath=../DataInterface/

#------------------------------------------------------
# Additional stuff for falcON programs

FALCONINC = -I$(FALCONPATH)inc/ -I$(FALCONPATH)inc/public/ -I$(FALCONPATH)utils/inc/ -I$(NEMOPATH)inc/ -I$(NEMOPATH)inc/max/ -I$(NEMOPATH)lib/ -DfalcON_SINGLE -DfalcON_NEMO -DfalcON_PROPER

FALCONLIB = -L$(FALCONPATH)lib/ -lfalcON.P -L$(FALCONPATH)utils/lib/ -lWDutils -L$(NEMOPATH)lib/ -lnemo++ -lnemo -DfalcON_SINGLE -DfalcON_NEMO -DfalcON_PROPER

#------------------------------------------------------
# Set files

LIBRARY = libaa.so

OBJECTS = $(OBJ)DataInterface.o $(OBJ)utils.o $(OBJ)aa.o $(OBJ)debug.o $(OBJ)spherical_aa.o $(OBJ)stackel_aa.o  $(OBJ)get_closed_Rz.o $(OBJ)uv_orb.o $(OBJ)lmn_orb.o $(OBJ)stackel_fit.o $(OBJ)analytic_aa.o $(OBJ)genfunc_aa.o $(OBJ)adiabatic_aa.o $(OBJ)tables_aa.o
#$(OBJ_POT)potential.o $(OBJ_POT)orbit.o $(OBJ_POT)Multipole.o 
OBJECTSALL = $(OBJ)DataInterface.o $(OBJ)utils.o $(OBJ)aa.o $(OBJ)debug.o $(OBJ)spherical_aa.o $(OBJ)stackel_aa.o  $(OBJ)get_closed_Rz.o $(OBJ)uv_orb.o $(OBJ)lmn_orb.o $(OBJ)stackel_fit.o $(OBJ)analytic_aa.o $(OBJ)genfunc_aa.o $(OBJ)adiabatic_aa.o $(OBJ)tables_aa.o $(OBJ_POT)potential.o $(OBJ_POT)orbit.o $(OBJ_POT)Multipole.o
OBJECTSDATAINTERFACE = $(OBJ)DataInterface.o
# scf.o inc.o Gd.o
# $(OBJ_POT)potential.o $(OBJ_POT)orbit.o $(OBJ_POT)Multipole.o
#gjy add

ifeq ($(TORUS), 1)
	OBJECTS += $(OBJ)it_torus.o
endif
HEADERS = $(wildcard inc/*.h)

#------------------------------------------------------
#compile options:

default: clean all

#gjy add
all: lib/libaa.so mains/data.exe mains/funcs.exe
# all: lib/libaa.so mains/test_actions.exe mains/test_test.exe
# all: lib/libaa.so mains/test_actions.exe mains/test_angles.exe mains/many_tori.exe mains/genfunc_converg.exe mains/many_tori_james.exe

python: lib/aa_py.so

forceall: clean all

#------------------------------------------------------

$(OBJ)aa_py.o : $(SRC)aa_py.cpp ${HEADERS}
	$(CCOMPILER) $(OPTS) -c $< -o $@ $(INC) $(INCPY) $(TORUSFLAG) $(LAPACKFLAG)

$(OBJ)utils.o : ../general/utils.cpp ${HEADERS}
	$(CCOMPILER) $(OPTS) -c $< -o $@ $(INC) $(TORUSFLAG) $(LAPACKFLAG)

$(OBJ)DataInterface.o : ../DataInterface/DataInterface.cpp ${HEADERS}
	$(CCOMPILER) $(OPTS) -c $< -o $@ $(INC) $(TORUSFLAG) $(LAPACKFLAG)
#gjy add: DataInterface

$(OBJ)%.o : $(SRC)%.cpp ${HEADERS}
	$(CCOMPILER) $(OPTS) -c $< -o $@ $(INC) $(TORUSFLAG) $(LAPACKFLAG)
$(OBJ)%.o : mains/%.cpp ${HEADERS}
	$(CCOMPILER) $(OPTS) -c $< -o $@ $(INC) $(TORUSFLAG) $(LAPACKFLAG)

$(FALCONOBJ)%.o : $(SRC)%.cpp
	$(CCOMPILER) $(OPTS) -c $< -o $@ $(INC) $(FALCONINC) $(TORUSFLAG)

#------------------------------------------------------

## TESTS
analytic_test.exe: $(OBJECTS)
	$(CCOMPILER) $(OPTS) -o analytic_test.exe $(OBJECTS) $(LIB)

genfunc_test.exe: $(OBJECTS)
	$(CCOMPILER) $(OPTS) -o genfunc_test.exe $(OBJECTS) $(LIB) $(CUBA)

test.exe: $(OBJECTS) $(OBJ)test.o
	$(CCOMPILER) $(OPTS) -o test.exe $(OBJECTS) $(OBJ)test.o $(INC) $(LIB) $(CUBA) $(TORUSFLAG) $(LAPACKFLAG)



#gjy add
#:: here only remake data.cpp
EXE_clean:
	rm -rf obj/data.o; rm -rf obj/funcs.o; rm -rf mains/*.exe
EXE_data: $(OBJ)data.o
	$(CCOMPILER) $(OPTS) -o mains/data.exe $(OBJECTS) $(OBJ)data.o $(LIB) $(CUBA) $(ADDDATALIBPATH) $(LIBDATASO)
# -static-libstdc++
EXE_funcs: $(OBJ)funcs.o
	$(CCOMPILER) $(OPTS) -o mains/funcs.exe $(OBJECTS) $(OBJ)funcs.o $(LIB) $(CUBA) $(ADDDATALIBPATH) $(LIBDATASO)
# -static-libstdc++
EXE_default: EXE_clean EXE_data EXE_funcs

# DATAEXE_L_T: ENABLE_L_T DATAEXE #??
# ENABLE_L_T:
#	$(TORUSFLAG) $(LAPACKFLAG) 
# 	LAPACK=1 TORUS=1

#:: how to make in terminal
#() make EXE_default LAPACK=1 TORUS=1
#:: lib pot_js .so should be remake with LAPACK=1 TORUS=1 at other loactions, lib DataInterface .so not need without TORUS
# -rm -rf ../pot/obj/*.o; -rm -rf ../pot/*.so;



#gjy add
mains/data.exe: $(OBJECTS) $(OBJ)data.o
	$(CCOMPILER) $(OPTS) -o mains/data.exe $(OBJECTS) $(OBJ)data.o $(LIB) $(CUBA) $(ADDDATALIBPATH) $(LIBDATASO) -static-libstdc++
# $(CCOMPILER) $(OPTS) -o mains/data.exe $(OBJECTS) $(OBJ)data.o $(LIB) $(CUBA) -static-libstdc++

#gjy add
mains/funcs.exe: $(OBJECTS) $(OBJ)funcs.o
	$(CCOMPILER) $(OPTS) -o mains/funcs.exe $(OBJECTS) $(OBJ)funcs.o $(LIB) $(CUBA) $(ADDDATALIBPATH) $(LIBDATASO) -static-libstdc++

# #gjy add
# mains/test_test.exe: $(OBJECTS) $(OBJ)test_test.o
# 	$(CCOMPILER) $(OPTS) -o mains/test_test.exe $(OBJECTS) $(OBJ)test_test.o $(LIB) $(CUBA) -static-libstdc++



mains/test_actions.exe: $(OBJECTS) $(OBJ)test_actions.o
	$(CCOMPILER) $(OPTS) -o mains/test_actions.exe $(OBJECTS) $(OBJ)test_actions.o $(LIB) $(CUBA) -static-libstdc++

mains/test_angles.exe: $(OBJECTS) $(OBJ)test_angles.o
	$(CCOMPILER) $(OPTS) -o mains/test_angles.exe $(OBJECTS) $(OBJ)test_angles.o $(LIB) $(CUBA) -static-libstdc++

mains/many_tori.exe: $(OBJECTS) $(OBJ)many_tori.o
	$(CCOMPILER) $(OPTS) -o mains/many_tori.exe $(OBJECTS) $(OBJ)many_tori.o $(LIB) $(CUBA)

mains/many_tori_james.exe: $(OBJECTS) $(OBJ)many_tori_james.o
	$(CCOMPILER) $(OPTS) -o mains/many_tori_james.exe $(OBJECTS) $(OBJ)many_tori_james.o $(LIB) $(CUBA)

mains/genfunc_converg.exe: $(OBJECTS) $(OBJ)genfunc_converg.o
	$(CCOMPILER) $(OPTS) -o mains/genfunc_converg.exe $(OBJECTS) $(OBJ)genfunc_converg.o $(LIB) $(CUBA)

spherical_test.exe: $(OBJECTS)
	$(CCOMPILER) $(OPTS) -o spherical_test.exe $(OBJECTS) $(LIB)

stackel_fit_test.exe: $(OBJECTS)
	$(CCOMPILER) $(OPTS) -o stackel_fit_test.exe $(OBJECTS) $(LIB) $(CUBA)

lmnnew_test.exe: $(OBJECTS)
	$(CCOMPILER) $(OPTS) -o lmnnew_test.exe $(OBJECTS) $(LIB) $(CUBA)



## LIBRARY
lib/libaa.so: $(OBJECTS)
	$(CCOMPILER) -fPIC $(LIBFLAG) $(CFL) $(OBJECTS) $(INC) -Wl,-$(LIBNAME),$(CURDIR)/lib/libaa.so  $(LIB) $(CUBA) -o lib/libaa.so

triax_py.so: $(OBJECTS) $(OBJ)triax_py.o
	$(CCOMPILER) -fPIC $(LIBFLAG) $(CFL) $(OBJECTS) $(OBJ)triax_py.o $(INC) $(LIB)  -o triax_py.so

lib/aa_py.so: $(OBJECTS) $(OBJ)aa_py.o
	$(CCOMPILER) -fPIC $(LIBFLAG) $(OPTS) $(OBJECTS) $(OBJ)aa_py.o $(INC) $(LIB) $(LIBPY) $(CUBA) -o lib/aa_py.so

## OTHER

falcON_aa.exe: $(OBJECTS) $(FALCONOBJ)falcON_aa.o
	$(CCOMPILER) $(OPTS) -o falcON_aa.exe $(OBJECTS) $(FALCONOBJ)falcON_aa.o $(FALCONLIB) $(LIB) $(LIBPY) $(CUBA)

.PHONY: clean
clean:
	rm -rf obj/*.o; rm -rf *.so; rm -rf mains/*.exe
