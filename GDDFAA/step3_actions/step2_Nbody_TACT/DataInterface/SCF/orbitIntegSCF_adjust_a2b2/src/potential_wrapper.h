#ifndef _POTENTIAL_WRAPPER_
#define _POTENTIAL_WRAPPER_

extern "C"{
#include"F_to_C.h"
}
#include<math.h>
#include<vector>
using namespace SCFORB;

//: in DatInterface.h
const double Err1 = 1.e-2;
//\ Local configuration-space smoothing for SCF potential
// const bool is_enable_smoothing_potential_SCF = true; //stronger level to control whether enable smoothing
const bool is_enable_smoothing_potential_SCF = false; //stronger level to control whether enable smoothing
const double scf_smoothing_h = 4.; //smoothing scale in code units (kpc)
// const double scf_smoothing_h = 10.; //smoothing scale in code units (kpc)
// const double scf_smoothing_h = -1.; //the value would changed by compute_scf_smoothing_scale() if it <= 0.

/*	Potential: Self-consist field (SCF) method. This is not smoothed potential. 
    Low-level SCF potential evaluation with only the tiny-axis guard. 
    Now only potential of a fixed snapshot is provided.
*/
// double potential_SCF(const std::vector<double>& x_tgt){
double potential_SCF_raw(const std::vector<double>& x_tgt){
    double x = x_tgt[0], y = x_tgt[1], z = x_tgt[2];
    double pot;

    if( !(x_tgt[0] > Err1 && x_tgt[1] > Err1) ){
        double p = 0.0;

        if(!(std::fabs(x_tgt[0]) > Err1)){
            x =  Err1;
        }
        if(!(std::fabs(x_tgt[1]) > Err1)){
            y =  Err1;
        }
        get_pot_xyz_(&x, &y, &z, &pot);
        p += pot;

        if(!(std::fabs(x_tgt[0]) > Err1)){
            x = -Err1;
        }
        if(!(std::fabs(x_tgt[1]) > Err1)){
            y = -Err1;
        }
        get_pot_xyz_(&x, &y, &z, &pot);
        p += pot;

        pot = 0.5 * p;
    }else{
        get_pot_xyz_(&x, &y, &z, &pot);
    }

    // printf("potential_SCF(): %e; ", pot);
    return pot;
}

/*  SCF potential with brute-force local spatial averaging. 
    We evaluate the SCF potential at the target point and at \pm h along 
    each Cartesian axis and do a small Gaussian-weighted average.
*/
// double potential_SCF_smoothed(const std::vector<double>& x_tgt){
double potential_SCF(const std::vector<double>& x_tgt){
    // Determine (and cache) a global smoothing length h in kpc
    double h = scf_smoothing_h;

    // If for some reason h is still non-positive, fall back to raw SCF
    if((!is_enable_smoothing_potential_SCF) || (h<=0.0)){
        return potential_SCF_raw(x_tgt);
    }

    // 7-point stencil: center + 6 neighbors at |Δx| = h
    // We use simple Gaussian weights: w0 at center, w1 at neighbors
    const double w0   = 1.0;
    // const double w1   = std::exp(-0.5);     // r = h => exp(-h^2 / (2 h^2)) = exp(-1/2)
    const double w1   = 1.0;     // r = h => exp(-h^2 / (2 h^2)) = exp(-1/2)
    const double wsum = w0 + 6.0 * w1;

    double pot_sum = 0.0;

    // center
    pot_sum += w0 * potential_SCF_raw(x_tgt);

    std::vector<double> x_shift(3);

    // +x
    x_shift = x_tgt;
    x_shift[0] += h;
    pot_sum += w1 * potential_SCF_raw(x_shift);

    // -x
    x_shift = x_tgt;
    x_shift[0] -= h;
    pot_sum += w1 * potential_SCF_raw(x_shift);

    // +y
    x_shift = x_tgt;
    x_shift[1] += h;
    pot_sum += w1 * potential_SCF_raw(x_shift);

    // -y
    x_shift = x_tgt;
    x_shift[1] -= h;
    pot_sum += w1 * potential_SCF_raw(x_shift);

    // +z
    x_shift = x_tgt;
    x_shift[2] += h;
    pot_sum += w1 * potential_SCF_raw(x_shift);

    // -z
    x_shift = x_tgt;
    x_shift[2] -= h;
    pot_sum += w1 * potential_SCF_raw(x_shift);

    double pot_smoothed = pot_sum / wsum;
    return pot_smoothed;
}

#endif