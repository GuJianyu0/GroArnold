////==================================================
//// An interface to functions of some Fortran variables.
////==================================================

#ifndef _SCF_
#define _SCF_

#include<stdio.h>
#include<time.h>

// extern void get_parameter_();
// extern void get_pot_xyz_(double *x, double *y, double *z, double *pot);
// extern void main_for_orbit_integrating_();
// extern void cdafkvnsdlkfdcs();

/*  The main subroutine for reading particle data and calculate coef.
*/
namespace SCFORB{
// extern void main_for_coef_();

/*  To calculte coef file (scficoef) from particle data (data.inp). 
*/
extern void get_parameter_();

/*  To interplate.
*/
extern void interpolate_3o_spline_1d_(
    double *x, double *y, int *n, 
    double *sx, double *f, double *f1, int *m
);

/*  To calculte potential and forces in a given Cartesian point.
*/
extern void get_pot_xyz_(double *x, double *y, double *z, double *pot);
extern void get_acc_xyz_(double *x, double *y, double *z, double *accx, double* accy, double* accz);
extern void get_pot_(double *xi, double *poti);
extern void get_force_(double *xi, double *vi, double *ai, double *adoti);

/*  The main subroutine for orbit integrating.
    Two steps.
*/
// extern void main_for_orbit_integrating_();
// namespace orbintegz_;
extern void __orbintegz_MOD_main_for_orbit_integrating(double *yy, char *fn);
extern void __orbintegx_MOD_main_for_orbit_integrating(double *yy, char *fn);

extern void writetofile_release_index_(int *idx); //to release the occupication from Fortran

}

#endif