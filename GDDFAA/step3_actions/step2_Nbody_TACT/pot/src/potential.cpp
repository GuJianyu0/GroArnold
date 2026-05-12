// ============================================================================
/// \file inc/potential.h
// ============================================================================
/// \author Jason Sanders
/// \date 2014-2015
/// Institute of Astronomy, University of Cambridge (and University of Oxford)
// ============================================================================

// ============================================================================
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// ============================================================================
/// \brief Potential classes
// ============================================================================

/*======================================*/
/* 			    Potential_JS 			*/
/*======================================*/

// #include <Python.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "coordsys.h"
#include "coordtransforms.h"
#include "potential.h"
using namespace std;

// ============================================================================
struct RE_st{ //结构体RE_st下的Potential_JS类, 下面有再下的Forces函数, 用...->...->...
	double E; Potential_JS *Pot;
	RE_st(double E, Potential_JS *Pot):E(E), Pot(Pot){}
};

double RE_fn(double r, void* params){
	RE_st *RE = (RE_st*)params; //gjy note:为了把参数传入到RE_st结构体(的RE对象)的目标函数函数里作为参数
	double refn = 0.5*r*abs(RE->Pot->Forces({r,0.,0.})[0])+RE->Pot->Phi({r,0.,0.})-RE->E; //gjy changed: abs(F[0])
	// double refn = -0.5*r*RE->Pot->Forces({r,0.,0.},1)[0]+RE->Pot->Phi({r,0.,0.},1)-RE->E; //gjy note: 令函为0, 所用为向心力公式 //gjy add: coor=1
	// std::cout<<"Potential_JS::RE_fn() is called:\n r = "<<r<<", Forces({r,0.,0.},1)[0] = "<<RE->Pot->Forces({r,0.,0.},1)[0]
	// <<", RE->Pot->Phi({r,0.,0.},1) = "<<RE->Pot->Phi({r,0.,0.},1)<<", RE->E = "<<RE->E<<", self ="<<refn<<std::endl; //gjy add
	return refn;
}
struct RL_st{
	double Lz2; Potential_JS *Pot;
	RL_st(double Lz, Potential_JS *Pot):Lz2(Lz*Lz), Pot(Pot){}
};

double RL_fn(double R, void*params){
	RL_st *RL = (RL_st*)params;
	double rlfn = RL->Pot->Forces({R,0.,0.},2)[0]+RL->Lz2/pow(R,3); //gjy note: 令函为0, 即得轴对称势下的导心半径R_g; 先给定Lz, 在计算此z-角动量下的R=Rg //gjy add: coor
	// printf("rlfn = %f\n", rlfn);
	return rlfn;
}

double Potential_JS::R_E(double E,double r){ //gjy add, note: Er in, xyz out //r defualt -1
	// VecDoub v1 = {x[3],x[4],x[5]}; //gjy changed
	// double v12 = lpnorm_real_with_distance_and_coef(&v1);
	// return conv::G*Mass_vir/v12;
	// return -0.5*conv::G*Mass_vir/E; //as point mass

	// Can pass initial guess for speed
	if(E>Phi_max()){ //gjy add
		std::cout<<"R_E(): E>Phi_max(), now set is as RadiusMaxConsidered.\n";
		DEBUG_PRINT_V1d(10, (VecDoub){E, Phi_max()}, "E and Phi_max()");
		return RadiusMaxConsidered;
	}
	RE_st RES(E,this); //gjy note: this: potential_JS or its inherited class
	// root_find RF(1e-6,200);
	root_find RF(1e-1,200); //gjy changed
	double up = 1e5,down=1e-5;
	if(r>0.){up=10.*r;down=0.1*r;}
    double rr = RF.findroot(&RE_fn,down,up,&RES); //gjy note: 找函RE_fn关于其内之那个r在范围[r_down, r_up]的跟
	rr = selectMax(rr, RadiusMinConsidered);
	return rr;
}

double Potential_JS::R_E(double E, double r_scale, double M_scale, double r) //gjy add
{
	return 0.;
}

double Potential_JS::R_L(double Lz,double r){
	// Can pass initial guess for speed
	// printf("Potential_JS::R_L() called.\n");
	RL_st RLS(Lz,this);
	root_find RF(1e-6,200);
	double up = 1e5,down=1e-5;
	if(r>0.){up=10.*r;down=0.1*r;}
	// printf("Potential_JS::R_L() here2\n");
    return RF.findroot(&RL_fn,down,up,&RLS);
}

double Lzmax_fn(double r, void*params){
	RE_st *RE = (RE_st*)params;
	return -2.*r*r*(RE->E-RE->Pot->Phi({r,0.,0.},1)); //gjy add: coor=1
}

double Potential_JS::Lzmax(double E,double RC){
	RE_st RES(E,this);int status;
	minimiser1D min(&Lzmax_fn,RC/2.,SMALL,RC,1e-4,0.,&status,&RES);
    double r = min.minimise(100);
    return -Lzmax_fn(r,&RES);
}

double Potential_JS::R_L(const VecDoub &x){
	assert(x.size()==6);
	return R_L(Lz(x),::norm<double>({x[0],x[1],x[2]}));
}

double Potential_JS::R_E(const VecDoub &x){
	// std::cout<<"R_E(xv) is called"<<std::endl; //gjy add
	assert(x.size()==6);
	// std::cout<<"norm r is "<<::norm<double>({x[0],x[1],x[2]})<<std::endl; //gjy add
	return R_E(H(x),::norm<double>({x[0],x[1],x[2]}));
	// return R_E_estimete(H(x),::norm<double>({x[0],x[1],x[2]})); //gjy changed
}

double Potential_JS::L_E(double E){
	double R_e = R_E(E);
	return sqrt(pow(R_e,3.)*-Forces({R_e,0.,0.},2)[0]); //gjy add: coor
}

double Potential_JS::L_E(const VecDoub &x){
	double R_e = R_E(x);
	return sqrt(pow(R_e,3.)*-Forces({R_e,0.,0.},2)[0]); //gjy add: coor
}

double Potential_JS::torb(const VecDoub &x){
	// std::cout<<"Potential_JS::torb() is called"<<std::endl; //gjy add
	double R = R_E(x);
	std::cout<<"Potential_JS::torb() R = "<<R<<std::endl; //gjy add
	return 2.*PI*sqrt(R/abs(Forces({R,0.,0.})[0])); //gjy changed: abs(F[0])
	// return 2.*PI*sqrt(R/-Forces({R,0.,0.},2)[0]); //用的向心力表达式, 此F1(R)为赤道上(R=r)?的向心力 //gjy add: coor
}

VecDoub Potential_JS::dPhidRdz(const VecDoub& Rz){
	// std::cout<<"dPhi_dRdz_partial() is called"<<std::endl; //gjy add
	double Delta = 0.005;

	VecDoub Rplus =  Forces({Rz[0]+Delta,0.,Rz[1]},2); //gjy add: coor
	VecDoub Rminus = Forces({Rz[0]-Delta,0.,Rz[1]},2);
	VecDoub zplus =  Forces({Rz[0],0.,Rz[1]+Delta},2);
	VecDoub zminus = Forces({Rz[0],0.,Rz[1]-Delta},2); //gjy add: coor
	return {0.5*(Rminus[0]-Rplus[0])/Delta,
			0.5*(Rminus[2]-Rplus[2])/Delta,
			0.5*(zminus[2]-zplus[2])/Delta};
}

double Potential_JS::DeltaGuess(const VecDoub& x){
	// Returns a guess of Gamma-Alpha = Delta^2 assuming dldv((l-v)V)=0
	double R = ::norm<double>({x[0],x[1]});
	double z = x[2];
	if(x[2]==0.)z+=TINY; // d2P/dRdz identically zero for up-down sym potential at z=0
	VecDoub F = Forces({R,0.,z},2); //gjy add: coor
	VecDoub d2P = dPhidRdz({R,z});
	return z*z-R*R+(-3.0*z*F[0]+3.0*R*F[2]+R*z*(d2P[0]-d2P[2]))/d2P[1];
}



VecDoub Potential_JS::Phi_secondderive_diag_diff(const VecDoub &x, double dx, int coor){ //gjy add //the accuracy cannot be guaranteed...
	if(coor == 0){ //Cartesian
		double Phi__xx = ( Phi({x[0]+dx, x[1], x[2]}) + Phi({x[0]-dx, x[1], x[2]}) - 2*Phi({x[0], x[1], x[2]}) ) / (dx*dx);
		double Phi__yy = ( Phi({x[0], x[1]+dx, x[2]}) + Phi({x[0], x[1]-dx, x[2]}) - 2*Phi({x[0], x[1], x[2]}) ) / (dx*dx);
		double Phi__zz = ( Phi({x[0], x[1], x[2]+dx}) + Phi({x[0], x[1], x[2]-dx}) - 2*Phi({x[0], x[1], x[2]}) ) / (dx*dx);
		return {Phi__xx, Phi__yy, Phi__zz};
	}
	if(coor == 2){ //Polar
		VecDoub xx = x; //conv::CartesianToPolar(x); //{x,y,z} -> {R,p,z}
		VecDoub ds = {dx, dx*xx[0], dx}; //square line element ds^2 = [dx,dy,dz] g_{ij} transpose[dx,dy,dz]; in Polar coor, metric g_{ij} = diag{1,R^2,1}

		//3 points 2-derivs
		double Phi__RR0 = ( Phi(conv::PolarToCartesian({xx[0]+ds[0], xx[1], xx[2]})) + Phi(conv::PolarToCartesian({xx[0]-ds[0], xx[1], xx[2]})) - 2*Phi({x[0], x[1], x[2]}) ) / (ds[0]*ds[0]);
		double Phi__pp0 = ( Phi(conv::PolarToCartesian({xx[0], xx[1]+ds[1], xx[2]})) + Phi(conv::PolarToCartesian({xx[0], xx[1]-ds[1], xx[2]})) - 2*Phi({x[0], x[1], x[2]}) ) / (ds[1]*ds[1]);
		double Phi__zz0 = ( Phi({x[0], x[1], x[2]+dx}) + Phi({x[0], x[1], x[2]-dx}) - 2*Phi({x[0], x[1], x[2]}) ) / (dx*dx);

		// //5 points 2-derivs
		// double Phi__RR = ( -Phi(conv::PolarToCartesian({xx[0] -2*ds[0], xx[1], xx[2]})) + 16*Phi(conv::PolarToCartesian({xx[0] -ds[0], xx[1], xx[2]})) - 30*Phi(conv::PolarToCartesian({xx[0], xx[1], xx[2]})) 
		// 					+ 16*Phi(conv::PolarToCartesian({xx[0] +ds[0], xx[1], xx[2]})) - Phi(conv::PolarToCartesian({xx[0] +2*ds[0], xx[1], xx[2]})) ) / (12*ds[0]*ds[0]);
		// double Phi__pp = ( -Phi(conv::PolarToCartesian({xx[0], xx[1] -2*ds[1], xx[2]})) + 16*Phi(conv::PolarToCartesian({xx[0], xx[1] -ds[1], xx[2]})) - 30*Phi(conv::PolarToCartesian({xx[0], xx[1], xx[2]})) 
		// 					+ 16*Phi(conv::PolarToCartesian({xx[0], xx[1] +ds[1], xx[2]})) - Phi(conv::PolarToCartesian({xx[0], xx[1] +2*ds[1], xx[2]})) ) / (12*ds[1]*ds[1]);
		// double Phi__zz = ( -Phi({x[0], x[1], x[2] -2*dx}) + 16*Phi({x[0], x[1], x[2] -dx}) - 30*Phi({x[0], x[1], x[2]}) 
		// 					+16*Phi({x[0], x[1], x[2] +dx}) - Phi({x[0], x[1], x[2] +2*dx}) ) / (12*dx*dx);

		// printf("Phi__RR: %f %f %f    %f\n", Phi(conv::PolarToCartesian({xx[0]+ds[0], xx[1], xx[2]})), Phi(conv::PolarToCartesian({xx[0]-ds[0], xx[1], xx[2]})), 
		// - 2*Phi({x[0], x[1], x[2]}), (ds[0]*ds[0]) );
		// printf("Phi__zz: %f %f %f    %f\n", Phi({x[0], x[1], x[2]+dx}), Phi({x[0], x[1], x[2]-dx}),  
		// - 2*Phi({x[0], x[1], x[2]}), (dx*dx) );
		// printf("5p: %f %f %f\n", Phi__RR, Phi__pp, Phi__zz);
		// printf("3p: %f %f %f\n", Phi__RR0, Phi__pp0, Phi__zz0);
		// return {Phi__RR, Phi__pp, Phi__zz};
		return {Phi__RR0, Phi__pp0, Phi__zz0};
	}
	else{
		printf("No result in such coordinate provided now!\n");
		return {0.,0.,0.};
	}
}

VecDoub Potential_JS::f_epicycle_axisympot_diff(double R, double Lz1, double dx){ //gjy add
	double Rg = R_L(Lz1);
	VecDoub dd = Phi_secondderive_diag_diff({Rg, 0., 0.},dx,2); //it is same for axisym sys
	double frequency_circular = Lz1/(R*R); //Omega({R,0,0})
	printf("epicycal: %f %f %f\n", dd[0], 3*pow(Lz1,2)/pow(R,4), dd[2]);
	double frequency_radical  = sqrt(dd[0] + 3*pow(Lz1,2)/pow(Rg,4)); //kappa({Rg,0,0})
	double frequency_verticle = sqrt(dd[2]); //nu({Rg,0,0})
	return {frequency_radical, frequency_circular, frequency_verticle,    Rg};
}

VecDoub Potential_JS::f_epicycle_axisympot_sum(double R, double Lz1){ //gjy add
	double Rg = R_L(Lz1);
	VecDoub dd = Phi_secondderive_diag_sum({Rg, 0., 0.},2); //it is same for axisym sys
	double frequency_circular = Lz1/(R*R); //Omega({R,0,0})
	printf("epicycal: %f %f %f\n", dd[0], 3*pow(Lz1,2)/pow(R,4), dd[2]);
	double frequency_radical  = sqrt(dd[0] + 3*pow(Lz1,2)/pow(Rg,4)); //kappa({Rg,0,0})
	double frequency_verticle = sqrt(dd[2]); //nu({Rg,0,0})
	return {frequency_radical, frequency_circular, frequency_verticle,    Rg};
}
//Potential_JS end.



struct pot_int_struc{
	Potential_JS *Pot;
	double intercept;
	int coeff;
	pot_int_struc(Potential_JS *Pot,double intercept,int coeff):Pot(Pot), intercept(intercept),coeff(coeff){}
};

static double find_potential_int(double x, void *params){
	pot_int_struc *RS = (pot_int_struc *) params;
	VecDoub Phi(3,1e-5);
	Phi[RS->coeff]=x;
	return RS->Pot->Phi(Phi)-RS->intercept;
}

double Potential_JS::find_potential_intercept(double Phi0, int direction,double xmin,double xmax){
	root_find RF(1e-4,100);
	pot_int_struc PP(this,Phi0,direction);
	return RF.findroot(&find_potential_int,xmin,xmax,&PP);
}

// ============================================================================
// Oblate Stackel Perfect Ellipsoid Potential
// ============================================================================

double StackelOblate_PerfectEllipsoid::G(double tau){
	/* de Zeeuw's G(tau) function */
	std::cout<<"G(tau) is called..."<<std::endl; //gjy add
	double Gamma = CS->gamma();
	if(tau+Gamma<SMALL) return Const*(1.-(tau+Gamma)/3.);
	double sqG =sqrt(-Gamma/(tau+Gamma));
	return Const*sqG*atan(1./sqG);
}
double StackelOblate_PerfectEllipsoid::GPrime(double tau){
	/* derivative of G wrt tau */
	double Gamma = CS->gamma();
	if(tau+Gamma<SMALL) return 0.5*Const*(-2./3.+4.*(tau+Gamma)/5.);
	double sqG =sqrt(-Gamma/(tau+Gamma));
	return 0.5*Const*sqG*sqG*(sqG*atan(1./sqG)/Gamma+1./tau);
}

VecDoub StackelOblate_PerfectEllipsoid::Vderivs(const VecDoub& tau){
	/* Calculates the derivatives of the Potential_JS wrt tau given tau=(l,v) */
        double Gl = G(tau[0]), Gv = G(tau[1]), Gamma = CS->gamma();
	double dVdl = 	(-GPrime(tau[0])*(tau[0]+Gamma)-Gl
			+(Gl*(tau[0]+Gamma)-Gv*(tau[1]+Gamma))/(tau[0]-tau[1]))
			/(tau[0]-tau[1]);
	double dVdv = 	(GPrime(tau[1])*(tau[1]+Gamma)+Gv
			-(Gl*(tau[0]+Gamma)-Gv*(tau[1]+Gamma))/(tau[0]-tau[1]))
			/(tau[0]-tau[1]);
        VecDoub derivs = {dVdl, dVdv};
	return derivs;
}


VecDoub StackelOblate_PerfectEllipsoid::Forces(const VecDoub& x, int coor){ //gjy changed
	/* forces at Cartesian x */
	VecDoub derivs = CS->derivs(x);
	VecDoub tau = {derivs[0],derivs[1]};
	VecDoub Vderiv = Vderivs(tau);
	double dvdR = -Vderiv[0]*derivs[2]-Vderiv[1]*derivs[4];
	double R = ::norm<double>({x[0],x[1]});
	VecDoub result ={ 	x[0]*dvdR/R, x[1]*dvdR/R,
			  			-Vderiv[0]*derivs[3]-Vderiv[1]*derivs[5]};
	return result;
}

double StackelOblate_PerfectEllipsoid::Phi_tau(const VecDoub& tau){
	/* Potential at tau */
	double Gamma = CS->gamma();
	return -((tau[0]+Gamma)*G(tau[0])-(tau[2]+Gamma)*G(tau[2]))/(tau[0]-tau[2]);
}

double StackelOblate_PerfectEllipsoid::Phi(const VecDoub& x, int coor){ //gjy changed
	/* potential at Cartesian x */
	VecDoub tau = CS->x2tau(x);
	return Phi_tau(tau);
}

VecDoub StackelOblate_PerfectEllipsoid::x2ints(const VecDoub& x, VecDoub *tau){
	VecDoub Ints = {H(x), 0.5*pow(Lz(x),2.)};
	if(!tau) (*tau) = CS->xv2tau(x);
	Ints.push_back(
	 ((*tau)[0]+CS->gamma())*
	 	(Ints[0]-(Ints[1]/((*tau)[0]+CS->alpha()))+G((*tau)[0]))
	 -(pow(((*tau)[3]*((*tau)[0]-(*tau)[2])),2.0))
	 	/(8.0*((*tau)[0]+CS->alpha())*((*tau)[0]+CS->gamma())));
	Ints.push_back(
	 ((*tau)[2]+CS->gamma())*
	 	(Ints[0]-(Ints[1]/((*tau)[2]+CS->alpha()))+G((*tau)[2]))
	 -(pow(((*tau)[5]*((*tau)[0]-(*tau)[2])),2.0))
	 	/(8.0*((*tau)[2]+CS->alpha())*((*tau)[2]+CS->gamma())));
	// Ints[3]=Ints[2];
	return Ints;
}

// ============================================================================
// Triaxial Stackel Perfect Ellipsoid Potential
// ============================================================================

struct TriaxialStackel_GIntegrand_struct{
	double taugl, acgl, bcgl, c2gl;
	TriaxialStackel_GIntegrand_struct(double taugl, double acgl, double bcgl, double c2gl):taugl(taugl),acgl(acgl),bcgl(bcgl),c2gl(c2gl){};
};

static double G_integrand(double s,void* params){
	TriaxialStackel_GIntegrand_struct *TG =
		(TriaxialStackel_GIntegrand_struct* )params;
	return sqrt(TG->c2gl+TG->bcgl*s*s)/sqrt(TG->c2gl+TG->acgl*s*s)/(TG->c2gl+(TG->taugl-TG->c2gl)*s*s);
}

double StackelTriaxial::G(double tau){
	/* de Zeeuw's perfect ellipsoid G function 			*/
	/* calculated using GL integration of eq B9 of dZ85 */
	TriaxialStackel_GIntegrand_struct P(tau,a*a-c*c,b*b-c*c,c*c);
	integrator Int(1e6);
	return Const*Int.integrate(&G_integrand,0.,1.,1e-5,&P);
}

static double GP_integrand(double s,void* params){
	TriaxialStackel_GIntegrand_struct *TG =
		(TriaxialStackel_GIntegrand_struct* )params;
	double p = TG->c2gl+(TG->taugl-TG->c2gl)*s*s;
	return -sqrt(TG->c2gl+TG->bcgl*s*s)/sqrt(TG->c2gl+TG->acgl*s*s)*s*s/p/p;
}
double StackelTriaxial::GPrime(double tau){
	/* de Zeeuw's perfect ellipsoid GPrime function 	*/
	/* calculated using GL integration of eq B9 of dZ85 */
	TriaxialStackel_GIntegrand_struct P(tau,a*a-c*c,b*b-c*c,c*c);
	integrator Int(1e6);
	return Const*Int.integrate(&GP_integrand,0.,1.,1e-5,&P);
}

/*
// Attempts at analytic expressions for above -- doesn't work

double StackelTriaxial::G(double tau){
	// std::cout<<l<<" "<<sinm<<" "<<-(tau+CS->alpha())/(CS->alpha()-CS->gamma())<<std::endl;
	// std::cout<<ellint_third(l,sinm,-(tau+CS->alpha())/(CS->alpha()-CS->gamma()))<<std::endl;
	std::cout<<ellint_third(l,sinm,0.)<<" "<<Flm<<" "<<Elm<<std::endl;
	return Const/(tau+CS->alpha())*
		   ((tau+CS->beta())*ellint_third(l,sinm,-(tau+CS->alpha())/(CS->alpha()-CS->gamma()))
		   +(CS->alpha()-CS->beta())*Flm);
}

double StackelTriaxial::GPrime(double tau){
	return 0.5*Const/(tau+CS->alpha())/(tau+CS->gamma())*
		   (((tau+CS->alpha())*(tau+CS->gamma())-(tau+CS->alpha())*(tau+CS->beta())-(tau+CS->gamma())*(tau+CS->beta()))
		   	*ellint_third(l,sinm,-(tau+CS->alpha())/(CS->alpha()-CS->gamma()))/(tau+CS->alpha())
		   +((tau+CS->alpha())*(CS->beta()-CS->gamma())+(tau+CS->gamma())*(CS->beta()-CS->alpha()))
		    *Flm/(tau+CS->alpha())
		   -(tau+CS->alpha())*b*c*sin(l)/CS->alpha()+(CS->gamma()-CS->alpha())*Elm);
}
*/

VecDoub StackelTriaxial::Vderivs(const VecDoub& tau){
	/* Calculates the derivatives of the Potential_JS wrt tau given tau=(l,m,v) */
	double Gl = G(tau[0]), Gm = G(tau[1]), Gn = G(tau[2]), Alpha = CS->alpha(), Gamma = CS->gamma();
	double  F_l = (tau[0]+Alpha)*(tau[0]+Gamma)*Gl,
			F_m = (tau[1]+Alpha)*(tau[1]+Gamma)*Gm,
			F_n = (tau[2]+Alpha)*(tau[2]+Gamma)*Gn;
	double dlm = tau[0]-tau[1], dln = tau[0]-tau[2];
	double dml = tau[1]-tau[0], dmn = tau[1]-tau[2];
	double dnl = tau[2]-tau[0], dnm = tau[2]-tau[1];

	double dVdl = -F_l*GPrime(tau[0])/Gl/dlm/dln-(2.*tau[0]+Alpha+Gamma)*Gl/dlm/dln+F_l/dlm/dln*(1./dln+1./dlm)
					-F_m/dmn/dml/dml-F_n/dnl/dnl/dnm;
	double dVdm = -F_m*GPrime(tau[1])/Gm/dml/dmn-(2.*tau[1]+Alpha+Gamma)*Gm/dmn/dml+F_m/dml/dmn*(1./dmn+1./dml)
					-F_l/dlm/dlm/dln-F_n/dnl/dnm/dnm;
	double dVdn = -F_n*GPrime(tau[2])/Gn/dnl/dnm-(2.*tau[2]+Alpha+Gamma)*Gn/dnm/dnl+F_n/dnl/dnm*(1./dnm+1./dnl)
					-F_l/dlm/dln/dln-F_m/dmn/dmn/dml;
	VecDoub derivs = {dVdl, dVdm, dVdn};
	return derivs;
}

VecDoub StackelTriaxial::Forces(const VecDoub& x){
	/* forces at Cartesian x */
	VecDoub derivs = CS->derivs(x);
	VecDoub tau = {derivs[0],derivs[1],derivs[2]};
	VecDoub Vderiv = Vderivs(tau);
	VecDoub result ={ 	-Vderiv[0]*derivs[3]-Vderiv[1]*derivs[4]-Vderiv[2]*derivs[5],
						-Vderiv[0]*derivs[6]-Vderiv[1]*derivs[7]-Vderiv[2]*derivs[8],
			  			-Vderiv[0]*derivs[9]-Vderiv[1]*derivs[10]-Vderiv[2]*derivs[11]};
	return result;
}

double StackelTriaxial::Phi_tau(const VecDoub& tau){
	/* Potential at tau */
	double  F_l = (tau[0]+CS->alpha())*(tau[0]+CS->gamma())*G(tau[0]),
			F_m = (tau[1]+CS->alpha())*(tau[1]+CS->gamma())*G(tau[1]),
			F_n = (tau[2]+CS->alpha())*(tau[2]+CS->gamma())*G(tau[2]);
	return -F_l/(tau[0]-tau[1])/(tau[0]-tau[2])-F_m/(tau[1]-tau[2])/(tau[1]-tau[0])-F_n/(tau[2]-tau[0])/(tau[2]-tau[1]);
}

double StackelTriaxial::Phi(const VecDoub& x){
	/* potential at Cartesian x */
	VecDoub tau = CS->x2tau(x);
	return Phi_tau(tau);
}

VecDoub StackelTriaxial::tau2ints(const VecDoub& tau){
	VecDoub pp = CS->tau2p(tau);
	double X = 0.5*pp[0]-(tau[0]+CS->alpha())*(tau[0]+CS->gamma())*G(tau[0])/(tau[0]-tau[1])/(tau[0]-tau[2]);
	double Y = 0.5*pp[1]-(tau[1]+CS->alpha())*(tau[1]+CS->gamma())*G(tau[1])/(tau[1]-tau[0])/(tau[1]-tau[2]);
	double Z = 0.5*pp[2]-(tau[2]+CS->alpha())*(tau[2]+CS->gamma())*G(tau[2])/(tau[2]-tau[1])/(tau[2]-tau[0]);
	VecDoub Ints = {X+Y+Z};
	double J =(tau[1]+tau[2])*X+(tau[2]+tau[0])*Y+(tau[0]+tau[1])*Z;
	double K = tau[1]*tau[2]*X+tau[2]*tau[0]*Y+tau[0]*tau[1]*Z;
	Ints.push_back((CS->alpha()*(CS->alpha()*Ints[0]+J)+K)/(CS->alpha()-CS->gamma()));
	Ints.push_back((CS->gamma()*(CS->gamma()*Ints[0]+J)+K)/(CS->gamma()-CS->alpha()));
	return Ints;
}
// ============================================================================
// PowerLaw Potential
// ============================================================================
double PowerLaw::Phi(const VecDoub& x){
	/* potential at Cartesian x */
	double r = x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2;
	return -GM*pow(r,-.5*k);
}
VecDoub PowerLaw::Forces(const VecDoub& x){
	/* Forces at Cartesian x */
	double r = x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2;
	r = -k*GM*pow(r,-.5*k-1);
	return {x[0]*r,x[1]*r/q1,x[2]*r/q2};
}
double PowerLaw::density_spherical(double r){
	if(q1!=1. or q2!=1.)
		std::cerr<<"Spherical density called but q1 or q2!=1"<<std::endl;
	return GM/conv::FPG*k*(1-k)*pow(r,-k-2.);
}
// ============================================================================
// Isochrone Potential
// ============================================================================
double Isochrone::Phi(const VecDoub& x, int coor){
	/* potential at Cartesian x */
	double r = x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2;
	return -GM/(b+sqrt(r+b*b));
}
VecDoub Isochrone::Forces(const VecDoub& x, int coor){
	/* Forces at Cartesian x */
	double r = x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2;
	r = GM/(b+sqrt(r+b*b))/(b+sqrt(r+b*b))/sqrt(r+b*b);
	return {-x[0]*r,-x[1]*r/q1,-x[2]*r/q2};
}
double Isochrone::density_spherical(double r){
	if(q1!=1. or q2!=1.)
		std::cerr<<"Spherical density called but q1 or q2!=1"<<std::endl;
	double a = sqrt(b*b+r*r);
	return GM/conv::FPG*(3.*(b+a)*a*a-r*r*(b+3.*a))/pow(a*(b+a),3.);
}

// ============================================================================
// Logarithmic Potential
// ============================================================================
double Logarithmic::Phi(const VecDoub& x, int coor){ //gjy changed
	/* potential at Cartesian x */
	// Phi0 is to make potential negative so E<0 for orbits
	// printf("Logarithmic::Phi() called.\n"); //gjy add
	return Vc2/2.*log(x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2)-Phi0;
}
VecDoub Logarithmic::Forces(const VecDoub& x, int coor){ //gjy changed
	/* Forces at Cartesian x */
	// std::cout<<"Logarithmic::Forces() called"<<"\n"; //gjy add
	double r = Vc2/(x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2);
	VecDoub Forces = {x[0],x[1]/q1,x[2]/q2};
	Forces = Forces*-r;
	return Forces;
}


// ============================================================================
// HarmonicOscillator Potential
// ============================================================================
double HarmonicOscillator::Phi(const VecDoub &x){
	/* potential at Cartesian x */
	double P = 0.;
	for(int i=0;i<3;i++) P+=Om[i]*Om[i]*x[i]*x[i];
	return P*.5;
}
VecDoub HarmonicOscillator::Forces(const VecDoub &x){
	/* Forces at Cartesian x */
	VecDoub F(3,0);
	for(int i=0;i<3;i++) F[i]=-Om[i]*Om[i]*x[i];
	return F;
}

// ============================================================================
// Dehnen Potential
// ============================================================================
double Dehnen::Phi(const VecDoub &x){
	/* potential at Cartesian x */
	assert(x.size()==3);
	double r = ::norm<double>(x);
	double chi = pow(r/rs,1./alpha);
	chi=chi/(1+chi);
	return -conv::FPG*rhoS*rs*rs*alpha*
	(rs/r*incomplete_beta(alpha*(3-gamma),alpha*(beta-3),chi)
	+incomplete_beta(alpha*(beta-2),alpha*(2-gamma),1-chi));
}
VecDoub Dehnen::Forces(const VecDoub &x){
 	/* Forces at Cartesian x */
 	assert(x.size()==3);
 	double r = ::norm<double>(x);
	double chi = pow(r/rs,1./alpha);
	double dchi = chi/r/alpha/(1+chi)*(1.-chi/(1+chi));
	chi = chi/(1+chi);
	r = -conv::FPG*rhoS*rs*rs*alpha*
	(-rs/r/r*incomplete_beta(alpha*(3-gamma),alpha*(beta-3),chi)
	+rs/r*pow(chi,alpha*(3-gamma)-1)*pow(1-chi,alpha*(beta-3)-1)*dchi
	-pow(1-chi,alpha*(beta-2)-1)*pow(chi,alpha*(2-gamma)-1)*dchi);
 	VecDoub F = x*-r;
 	return F;
}
double Dehnen::Density(const VecDoub& x){
	assert(x.size()==3);
	double r = ::norm<double>(x)/rs;
	return rhoS*pow(r,-gamma)*pow(1.+pow(r,1./alpha),(gamma-beta)*alpha);

}
double Dehnen::TotalMass(){
	return 4.*PI*rhoS*rs*rs*rs*alpha
			*complete_beta(alpha*(3-gamma),alpha*(beta-3));
}

// ============================================================================
// Miyamoto-Nagai Potential
// ============================================================================
double MiyamotoNagai_JS::Phi(const VecDoub& x){
	/* potential at Cartesian x */
	double AZB=A+sqrt(x[2]*x[2]+Bq);
	return -GM/sqrt(x[0]*x[0]+x[1]*x[1]+AZB*AZB);
}

double MiyamotoNagai_JS::Vc(double R){
	double t = A+sqrt(Bq);
	double F = 1./(R*R+t*t);
	return sqrt(GM*R*R*F*sqrt(F));
}

VecDoub MiyamotoNagai_JS::Forces(const VecDoub& x){
	/* Forces at Cartesian x */
	double  ZB=sqrt(x[2]*x[2]+Bq), AZB = A + ZB;
	double f = 1./(x[0]*x[0]+x[1]*x[1]+AZB*AZB),rtF=sqrt(f);
	VecDoub Force = {-GM*f*rtF*x[0], -GM*x[1]*f*rtF, ZB? -GM*x[2]*f*rtF*AZB/ZB : 0.};
	return Force;
}
// ============================================================================
// Jaffe Bulge Potential
// ============================================================================

double Bulge::Phi(const VecDoub& x){
	/* potential at Cartesian x */
	double r = sqrt(x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2);
	return GM/b_bulge*log(r/(r+b_bulge));
}
VecDoub Bulge::Forces(const VecDoub& x){
	// Forces at Cartesian x
	double r = sqrt(x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2);
	double dpdr = GM/b_bulge*(1./r-1./(r+b_bulge))/r;
	VecDoub Force = {x[0], x[1]/q1, x[2]/q2};
	Force = Force*-dpdr;
	return Force;
}

// ============================================================================
// NFW Potential
// ============================================================================

double NFW::Phi(const VecDoub& x, int coor){ //gjy changed
	/* potential at Cartesian x */
	double r = sqrt(x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2);
	return -GM*log(1.+r/rs)/r;
}
VecDoub NFW::Forces(const VecDoub& x, int coor){ //gjy changed
	/* Forces at Cartesian x */
	double r = sqrt(x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2);
	double dpdr = GM*(log(1.+r/rs)/r-1./rs/(1.+r/rs))/r;
	VecDoub Force = {x[0], x[1]/q1, x[2]/q2};
	Force = Force*(-dpdr/r);
	return Force;
}
double NFW::density(const VecDoub& x){
	double Delta = 0.005;
	VecDoub xtmp=x;
	xtmp[0]+=Delta;		VecDoub plus = Forces(xtmp);
	xtmp[0]-=2.*Delta; 	VecDoub minus = Forces(xtmp);
	double d2p = (minus[0]-plus[0])/2./Delta;
	xtmp[0]=x[0];
	xtmp[1]+=Delta;		plus = Forces(xtmp);
	xtmp[1]-=2.*Delta; 	minus = Forces(xtmp);
	d2p += (minus[1]-plus[1])/2./Delta;
	xtmp[1]=x[1];
	xtmp[2]+=Delta;		plus = Forces(xtmp);
	xtmp[2]-=2.*Delta; 	minus = Forces(xtmp);
	d2p += (minus[2]-plus[2])/2./Delta;
	xtmp[2]=x[2];
	return d2p/4./PI;
}

// ============================================================================
// Burkert Potential
// ============================================================================

double Burkert::Phi(const VecDoub& x, int coor){ //gjy changed
	/* potential at Cartesian x */
	double r = sqrt(x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2);
	return -GM*log(1.+r/rs)/r;
}
VecDoub Burkert::Forces(const VecDoub& x, int coor){ //gjy changed
	/* Forces at Cartesian x */
	double r = sqrt(x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2);
	double dpdr = GM*(log(1.+r/rs)/r-1./rs/(1.+r/rs))/r;
	VecDoub Force = {x[0], x[1]/q1, x[2]/q2};
	Force = Force*(-dpdr/r);
	return Force;
}
double Burkert::density(const VecDoub& x){
	double Delta = 0.005;
	VecDoub xtmp=x;
	xtmp[0]+=Delta;		VecDoub plus = Forces(xtmp);
	xtmp[0]-=2.*Delta; 	VecDoub minus = Forces(xtmp);
	double d2p = (minus[0]-plus[0])/2./Delta;
	xtmp[0]=x[0];
	xtmp[1]+=Delta;		plus = Forces(xtmp);
	xtmp[1]-=2.*Delta; 	minus = Forces(xtmp);
	d2p += (minus[1]-plus[1])/2./Delta;
	xtmp[1]=x[1];
	xtmp[2]+=Delta;		plus = Forces(xtmp);
	xtmp[2]-=2.*Delta; 	minus = Forces(xtmp);
	d2p += (minus[2]-plus[2])/2./Delta;
	xtmp[2]=x[2];
	return d2p/4./PI;
}

// ============================================================================
// Hernquist Potential
// ============================================================================

double Hernquist::Phi(const VecDoub& x, int coor){ //gjy changed
	/* potential at Cartesian x */
	double r = sqrt(x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2);
	// printf("Hernquist ");
	return -GM/(rs+r);
}
VecDoub Hernquist::Forces(const VecDoub& x, int coor){ //gjy changed
	/* Forces at Cartesian x */
	double r = sqrt(x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2);
	double dpdr = GM/(rs+r)/(rs+r);
	VecDoub Force = {x[0], x[1]/q1, x[2]/q2};
	Force = Force*(-dpdr/r);
	return Force;
}
// ============================================================================
// Plummer Potential
// ============================================================================

double Plummer::Phi(const VecDoub& x, int coor){ //gjy changed
	/* potential at Cartesian x */
	double r2 = x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2;
	return -GM/sqrt(rs*rs+r2);
}
VecDoub Plummer::Forces(const VecDoub& x, int coor){ //gjy changed
	/* Forces at Cartesian x */
	double r2 = x[0]*x[0]+x[1]*x[1]/q1+x[2]*x[2]/q2;
	double dpdr = GM/(rs*rs+r2)/sqrt(rs*rs+r2);
	VecDoub Force = {x[0], x[1]/q1, x[2]/q2};
	Force = Force*(-dpdr);
	return Force;
}

#ifdef TORUS
// ============================================================================
// GalPot Potential for interface with Walter Dehnen's code
// ============================================================================
GalPot::GalPot(std::string TpotFile){
	if(TpotFile.substr(TpotFile.find_last_of(".") + 1)!="Tpot")
		std::cerr<<"No functionality implemented for passing anything but Tpot file to Galpot."<<std::endl;
	std::ifstream file;
	file.open(TpotFile);
	if(!file.is_open())
		std::cerr<<TpotFile<<" cannot be opened."<<std::endl;
	PhiWD=new GalaxyPotential(file);
	file.close();
}

double GalPot::Phi(const VecDoub& x, int coor){ //gjy changed
	/* potential at Cartesian x */
	double R = ::norm<double>({x[0],x[1]});
	return conv::kpcMyr2kmsSq*(*PhiWD)(R,x[2]);
}
VecDoub GalPot::Forces(const VecDoub& x, int coor){ //gjy changed
	/* Forces at Cartesian x */
	double R = ::norm<double>({x[0],x[1]});
	double dR, dz;
	(*PhiWD)(R,x[2],dR,dz);
	VecDoub f = {x[0]*dR/R,x[1]*dR/R, dz};
	f = f*-conv::kpcMyr2kmsSq;
	return f;
}

VecDoub GalPot::freqs(double R){
	/* frequencies: kappa, omega_c and nu at polar R */
	Frequencies freqs=(*PhiWD).KapNuOm(R);
	VecDoub freqs_v = {freqs(0),freqs(1),freqs(2)};
	freqs_v = freqs_v*conv::kpcMyr2kms;
	return freqs_v;
}

double GalPot::Vc(double R){
	return sqrt(R*-Forces({R,0.,0.})[0]);
}
#endif

double PowerLawSphericalExpCut::Phi_r(double r){
	double rrc2 = r/rc; rrc2*=rrc2;
    return amp/r*(r/rc*sgk2*gamma_incP_fn(mk2,rrc2)-sg15k2*gamma_incP_fn(m5k2,rrc2));
}
double PowerLawSphericalExpCut::dPhi_r(double r){
	double rrc2 = r/rc; rrc2*=rrc2;
	return amp*gamma_incP_fn(m5k2,rrc2)*sg15k2/r/r;
}
// ============================================================================
// Bowden NFW Potential
// ============================================================================

double BowdenNFW::Phi(const VecDoub &x){
	double r = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
	double ct = x[2]/r;ct*=ct;
	double pot = -rho0*log(1+r/rs)/r;
	pot+=rho1*r/pow(r+r1,2.)*(3*ct-1.);
	double R2 = (x[0]*x[0]+x[1]*x[1]);
	double iR2 = 1./R2;
	double c2p = (R2>0.?(x[0]*x[0]-x[1]*x[1])*iR2:0.); // cos(2\phi)
	ct=1-ct;// sin^2(theta)
	pot-=rho2*r/pow(r+r2,2.)*ct*c2p;
	return pot*conv::FPG;
}


VecDoub BowdenNFW::Forces(const VecDoub &x){
	double r = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
	double R2 = (x[0]*x[0]+x[1]*x[1]);
	double iR2 = 1./R2;
	double ct = x[2]/r, st = sqrt(1-ct*ct);
	double dPdr = rho0/r*(-log(1+r/rs)/r+1./(rs+r));
	dPdr+=rho1*(2.*r/(r+r1)-1)/pow(r+r1,2.)*(3*ct*ct-1.);
	double dPdt=rho1*r/pow(r+r1,2)*6.*st*ct;
	double c2p = (R2>0.?(x[0]*x[0]-x[1]*x[1])*iR2:0.); // cos(2\phi)
	double s2p = (R2>0.?2.*x[0]*x[1]*iR2:0.); // sin(2\phi)
	dPdr-=rho2*(2.*r/(r+r2)-1)/pow(r+r2,2.)*st*st*c2p;
	dPdt+=rho2*r/pow(r+r2,2.)*2.*st*ct*c2p;
	double dPdp=-rho2*r/pow(r+r2,2)*st*st*2.*s2p;
	VecDoub f = {0.,0.,0.};
    for(int i=0;i<3;i++) f[i] = dPdr*x[i]/r;
    double ist=1./st;
    f[0]+= (R2>0.?-x[1]*iR2*dPdp:0.)+(st!=0.?x[0]*x[2]/pow(r,3.)*dPdt*ist:0.);
    f[1]+= (R2>0.?x[0]*iR2*dPdp:0.) +(st!=0.?x[1]*x[2]/pow(r,3.)*dPdt*ist:0.);
    f[2]+=-(st!=0.?(1.-ct*ct)/r*dPdt*ist:0.);
	return f*(conv::FPG);
}
// ============================================================================

#ifdef TORUS
VecDoub torusPSPT2cartvec(PSPT FF){
	VecDoub X = {FF[0]*cos(FF[2]),FF[0]*sin(FF[2]),FF[1],FF[3]*cos(FF[2])-FF[5]*sin(FF[2]),FF[3]*sin(FF[2])+FF[5]*cos(FF[2]),FF[4]};
	for(int i=3;i<6;++i)X[i]*=conv::kpcMyr2kms;
	return X;
}

double WrapperTorusPotential::operator()(const double R, const double z) const{
	return Pot->Phi({R,0.,z})/conv::kpcMyr2kmsSq;
}

double WrapperTorusPotential::operator()(const double R, const double z, double& dPdR, double& dPdz) const{
	VecDoub F = Pot->Forces({R,0.,z});
	dPdR = -F[0]/conv::kpcMyr2kmsSq;
	dPdz = -F[2]/conv::kpcMyr2kmsSq;
	return Pot->Phi({R,0.,z})/conv::kpcMyr2kmsSq;
}
double WrapperTorusPotential::RfromLc(const double L_in, double* dR) const
{


  bool more=false;
  double R,lR=0.,dlR=0.001,dPR,dPz,LcR,oldL,L=fabs(L_in);
  R=exp(lR);
  (*this)(R,0.,dPR,dPz);
  LcR=sqrt(R*R*R*dPR);
  if(LcR == L) return R;
  if(L>LcR) more=true;
  oldL=LcR;

  for( ; ; ) {
    lR += (more)? dlR : -dlR;
    R=exp(lR);
    (*this)(R,0.,dPR,dPz);
    LcR=sqrt(R*R*R*dPR);
    if(LcR == L) return R;
    if((L< LcR && L>oldL) ||(L>LcR && L<oldL)){
  R=(more)? exp(lR-0.5*dlR) : exp(lR+0.5*dlR);
  return R;}
    oldL=LcR;
  }
}
double WrapperTorusPotential::LfromRc(const double R, double* dR) const
{
  double dPR,dPz;
  (*this)(R,0.,dPR,dPz);
  return sqrt(R*R*R*dPR);
}
Frequencies WrapperTorusPotential::KapNuOm(            // returns kappa,nu,Om
            const double R) const  // given R at z=0
{
  Frequencies epi;
  double dPR,dPz, tmp, dPR2, dPz2;
  (*this)(R,0.,dPR,dPz);
  tmp = dPR/R;
  double delz = 2e-3, delR = 0.01*R;
  (*this)(R+delR,0.,dPR,dPz2);
  (*this)(R-delR,0.,dPR2,dPz2);
  (*this)(R,delz,dPR,dPz2);
  epi[2] = sqrt(tmp);
  epi[1] = sqrt((dPz2-dPz)/delz);
  epi[0] = sqrt(.5*(dPR-dPR2)/delR+3*tmp);
  return epi;
}
#endif

#ifdef GALPY
// Implementation
void  galpyPotential_JS::error(const char* msgs) const
{
  cerr << " Error in class galpyPotential_JS: " << msgs << '\n';
  exit(1);
}

double galpyPotential_JS::Phi (const VecDoub&x)
{
	double R = sqrt(x[0]*x[0]+x[1]*x[1]);
	return evaluatePotentials(R,x[2],nargs,potentialArgs);
}

VecDoub galpyPotential_JS::Forces (const VecDoub&x)
{
	double R = sqrt(x[0]*x[0]+x[1]*x[1]);
	dPdR= calcRforce(R,x[2],0.,0.,nargs,potentialArgs);
	dPdz= calczforce(R,x[2],0.,0.,nargs,potentialArgs);
	return {dPdR*x/R,dPdR*y/R,dPdz};
}
// ============================================================================
#endif
// potential.cpp
