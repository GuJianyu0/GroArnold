////==================================================
/// An interface to functions of fortran prog SCF.
////==================================================

#ifndef _SCF_
#define _SCF_

#include <stdio.h>
#include <time.h>

extern void get_parameter_();

extern void get_pot_(double *x, double *y, double *z, double *pot);
extern void get_acc_(double *x, double *y, double *z, double *accx, double* accy, double* accz);
extern void get_force_(double xi[3], double vi[3], double ai[3], double adoti[3]); //paramaters??

// extern void read_data_fortran_(double* x, double* y, double* z, double* mass, int* n);

#endif