default: clean all

all:  test
forceall:	clean all

HOST=$(shell hostname)
ifeq ($(HOST), calx019.ast.cam.ac.uk)
	CPP = g++
	CFLAGS = -std=c++0x -Wall -O3 -ffast-math -fno-strict-aliasing
	GSLPATH = /opt/ioa/software/gsl/2.1/
else
	CPP =  g++
	CFLAGS = -std=c++11 -Wall -O3 -ffast-math -fno-strict-aliasing -Wno-deprecated-register
	EXTRACLIB =  -lc++
	GSLPATH = ../../../gsl-1.16/
endif
#gjy changed

test.o: test.cpp
	$(CPP) $(CFLAGS) -c test.cpp -o test.o

test.exe: test.o
	$(CPP) $(CFLAGS) -o test.exe test.o -L$(GSLPATH)lib/ -lgsl -lgslcblas -lm

#gjy add:
test7.exe:
	g++ $(CFLAGS) -c test.cpp -o test7.o
	g++ $(CFLAGS) -o test7.exe test7.o -L$(GSLPATH)lib/ -lgsl -lgslcblas -lm

clean:
	rm -rf *.o; rm -rf *.so; rm -rf *.exe libGSLInterface.a
