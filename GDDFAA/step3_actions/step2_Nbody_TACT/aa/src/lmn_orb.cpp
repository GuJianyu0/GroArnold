// ============================================================================
/// \file src/lmn_orb.cpp
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
/// \brief Axisymmetric Staeckel fudge using Delta estimation from shells
///
/// lmn_orb: Wraps triaxial Staeckel fudge using the alpha, beta (or Delta_1
/// and Delta_2) estimation routine from Sanders & Binney (2014).
/// We find the closed loop orbits in the (x,y) and (y,z) planes and fit
/// ellipses to these orbits to find alpha and beta respectively.
///
//============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "GSLInterface/GSLInterface.h"
#include "gnuplot/gnuplot_i.h"
#include <gsl/gsl_poly.h>
#include "utils.h"
#include "coordsys.h"
#include "coordtransforms.h"
#include "potential.h"
#include "orbit.h"
#include "stackel_aa.h"
#include "lmn_orb.h"
#include "debug.h"
// What is gsl_set_error_handler_off() doing? l125
// ============================================================================

static double min_distance(double y, void *params){ //gjy comment: called by findclosed()

	// gsl_set_error_handler_off(); //gjy add
	root_struct_mindist *RS = (root_struct_mindist *) params;
	// Orbit orbit(RS->Pot);
	SymplecticOrbit orbit(RS->Pot); //gjy changed
	VecDoub X = {0.,y,0.,0.,0.,0.};
	double p = sqrt(2.*(RS->E-RS->Pot->Phi(X)));
	if(RS->swit==0) // Short axis
		X[3]=p;
	else if(RS->swit==1) // Long axis
		X[5]=p;
	VecDoub QQ = X; VecDoub QQ2;
	// Choose appropriate time-step
	double torb = RS->Pot->torb(X);
	DEBUG_PRINT_V1d(1, (VecDoub){y, torb}, "torb in lmn_orb min_distance");
	// double step = 1e-3*torb; //gjy note: original
	// int i=0, maxsteps = 10000;
	int maxsteps = 1000; //gjy changed
	double step = torb*10./maxsteps;
DEBUG_PRINT_I(2221);
	int i=0;
	while(i<maxsteps){
		QQ2=orbit.integrate(QQ, step,step);
		DEBUG_PRINT_V1d(1, QQ2, "results_QQ2");
		if(RS->swit==0 and QQ[0]*QQ2[0]<0. and QQ2[1]<0. and QQ2[3]<0.) break;
		if(RS->swit==1 and QQ[2]*QQ2[2]<0. and QQ2[1]<0. and QQ2[5]<0.) break;
		QQ = QQ2;
		i++;
	}
DEBUG_PRINT_I(2222);
	double min = 4.*PI*PI/torb/torb*(-QQ[1]-X[1])*(-QQ2[1]-X[1]);
	if(RS->swit==0)min+=(-QQ[3]-X[3])*(-QQ[3]-X[3]);
	else if(RS->swit==1)min+=(-QQ[5]-X[5])*(-QQ[5]-X[5]);
	// double min = abs( 4.*PI*PI/torb/torb*(-QQ[1]-X[1])*(-QQ2[1]-X[1]) ); //gjy changed
	// if(RS->swit==0) min += abs( (-QQ[3]-X[3])*(-QQ[3]-X[3]) );
	// else if(RS->swit==1) min += abs( (-QQ[5]-X[5])*(-QQ[5]-X[5]) );
	return min;
}

std::vector<int> lmn_orb::angular_momentum(const VecDoub &x){ //gjy note: not called by functions in this .cpp file
	VecDoub xx = {x[0],x[1],x[2]}, vv = {x[3],x[4],x[5]};
	VecDoub ll = cross_product<double>(xx,vv);
	return {(int)sign(ll[0]),(int)sign(ll[1]),(int)sign(ll[2])};
}

int lmn_orb::check_ang_mom(double y, double E, int swit){ //gjy note: not called by functions in this .cpp file
	double P = Pot->Phi({0.,y,0.});
	double p = sqrt(2.*(E-P));
	// Orbit orbit(Pot);
	SymplecticOrbit orbit(Pot); //gjy changed
	VecDoub X = {0.,y,0.,0.,0.,0.};
	int index = 0; if(swit==0) index = 2;
	if(swit==0) X[3]=p;
	else if(swit==1) X[5]=p;
	// double step = 1e-2*Pot->torb(X);
	int maxsteps = 1000; //gjy changed
	double step = Pot->torb(X)*10./maxsteps;
	// VecDoub QQ=orbit.integrate(X, 10000*step, step); //gjy note: original
	VecDoub QQ=orbit.integrate(X, 1000*step, step);
	//orbit.plot(0,1);
	int l=1, l0=angular_momentum(orbit.results()[0])[index];
    int result = 1;
	for(auto it = std::begin(orbit.results())+1;
	    	 it!= std::end(orbit.results());
	    	 ++it)
	{
		l = angular_momentum(*it)[index];
		if(l!=l0){ result = 0; break;}
	}
	return result;
}

VecDoub lmn_orb::check_orbit(double y, double E, int swit, int plot){ //gjy note: called by findalpha()
	// Plots closed orbit to check if it is closed

	double p = sqrt(2.*(E-Pot->Phi({0.,y,0.})));
	// Orbit orbit(Pot);
	SymplecticOrbit orbit(Pot); //gjy changed
	VecDoub X = {0.,y,0.,0.,0.,0.};
	if(swit==0) X[3]=p;
	else if(swit==1) X[5]=p;
DEBUG_PRINT_I(130); //gjy add
	// double step = 1e-4*Pot->torb(X); //gjy note: original
	// VecDoub QQ=orbit.integrate(X, 100000*step, step);
	int maxsteps = 10000; //gjy changed
	double step = Pot->torb(X)*10./maxsteps;
	VecDoub QQ=orbit.integrate(X, maxsteps*step, step);
// DEBUG_PRINT_I(131); //gjy add
	// std::cout<<"first several points of orbit: "<<std::endl; //gjy add
	printVector(orbit.results()[0]); //gjy add
	// std::cout<<"end. "<<std::endl; //gjy add
	double ymax = Max<double>(transpose(orbit.results())[swit]);
	double zmax = Max<double>(transpose(orbit.results())[swit+1]);
// DEBUG_PRINT_I(132); //gjy add
	// if(debug_find_Delta){
	if(1){
		orbit.output2file("orbit.tmp");
		// std::cin.ignore(); //gjy note: input enter to go on (reset record file)
		// if(swit==0 and plot==1)orbit.plot(0,1);
		// if(swit==1) orbit.plot(1,2);
	}
DEBUG_PRINT_I(133); //gjy add
	return {ymax,zmax};
}

static double EminusPot(double x, void *p){  //gjy note: called by findclosed()
	root_struct_mindist *RS = (root_struct_mindist *) p;
	return RS->E - RS->Pot->Phi({0.,x,0.}); //gjy note: to let target E be near Phi({0.,y_init,0.})
}



double lmn_orb::find_closed(double E, int swit){ //gjy note: called by findalpha()
	return Pot->R_E(E); //gjy add

	// At a fixed energy finds the short (swit=0) or long (swit=1)
	// axis closed loop orbit
	root_struct_mindist RS(Pot,E,swit);
	double yMax=ymax,yMin=1e-3, mid=0.;
	root_find RF(1e-4,100);
DEBUG_PRINT_I(101);
	yMax = RF.findroot(&EminusPot,1e-5,ymax,&RS)*.99;
DEBUG_PRINT_I(102);
	mid = yMax*0.7;
	double Dmax=min_distance(yMax, &RS),Dmin=min_distance(yMin, &RS),Dmid=min_distance(mid, &RS);
DEBUG_PRINT_I(103);

	// // Check if only boxes
	// if(fabs((Dmax-Dmin)/(yMax-yMin))-fabs((Dmid-Dmin)/(mid-yMin))<.5)
	//    return -1.;

	bracketer bra;
	int brr = bra.bracket(&yMax,&mid,&yMin,&Dmax,&Dmid,&Dmin,&min_distance,&RS);
	if(yMin<0. or yMax<0. or mid<0. or brr==0)
		return -1.;
	// if(debug_find_Delta)
	// 	std::cout<<yMin<<" "<<mid<<" "<<yMax<<" "<<Dmin<<" "<<Dmid<<" "<<Dmax<<std::endl;
	std::cout<<yMin<<" "<<mid<<" "<<yMax<<" "<<Dmin<<" "<<Dmid<<" "<<Dmax<<std::endl;
DEBUG_PRINT_I(104);
	int status=0;
	// minimiser1D min(&min_distance,mid,yMin,yMax,1e-4,0.,&status,&RS);
	minimiser1D min(&min_distance,mid,yMin,yMax,1e-1,0.,&status,&RS); //gjy changed
	if(status!=0) return -1.;
	double y = min.minimise(100);
	// if(swit==1) check_orbit(y,E,swit);
	return y;
}

static double min_action(double b, void *params){ //gjy note: called by findalpha()
	root_struct_action *RS = (root_struct_action *) params;
	if(RS->swit==0)RS->ATSF->CS->newalpha(b);
	else if(RS->swit==1)RS->ATSF->CS->newbeta(b);
	double L = (double)RS->orbit->results().size();
	double means=0.,vars=0.;int index = RS->swit+1;
	for(auto Y: RS->orbit->results()){
		VecDoub i = RS->ATSF->actions(Y);
		if(i[index]==i[index]){means+=i[index];	vars+=i[index]*i[index];}
	}
	double f = sqrt(MAX(0.,vars/L-means*means/L/L));
	return f;
}

static double min_lambda(double b, void *params){ //gjy comment: called by find_beta()
	root_struct_action *RS = (root_struct_action *) params;
	if(RS->swit==0)RS->ATSF->CS->newalpha(b);
	else if(RS->swit==1)RS->ATSF->CS->newbeta(b); //gjy comment: swit is for selection to alpha or beta
	double L = (double)RS->orbit->results().size();
	double means=0.,vars=0.;int index = 0;
	for(auto Y: RS->orbit->results()){
		VecDoub i = RS->ATSF->CS->xv2tau(Y); //gjy comment: meaning the ellipsoidal coordinates of the orbital point in the new case of beta=b
		i[index]+=RS->ATSF->CS->alpha();
		if(i[index]==i[index]){means+=i[index];	vars+=i[index]*i[index];}
	}
	double f = sqrt(MAX(0.,vars/L-means*means/L/L)); //gjy comment: standard deviation
	return f;
}

void lmn_orb::plot_Delta2(double E){
	double y = find_closed(E,1);
	double P = Pot->Phi({0.,y,0.});
	double p = sqrt(2.*(E-P));
	// Orbit orbit(Pot);
	SymplecticOrbit orbit(Pot); //gjy changed
	VecDoub X = {0.0,y,0.0,0.0,0.0,p};
	VecDoub QQ=orbit.integrate(X, 2.5, 0.005);
	orbit.output2file("orbit_Delta2_loweralpha.dat");
	Actions_TriaxialStackel_Fudge ATSF(Pot,-80.,-20.);
	root_struct_action RS(&ATSF,&orbit,1);
	std::ofstream outFile; outFile.open("Delta2_loweralpha.dat");
	for(double b = ATSF.CS->alpha()+0.01;b<ATSF.CS->gamma()-0.01;b+=1.)
		outFile<<sqrt(ATSF.CS->gamma()-b)<<" "<<min_action(b,&RS)<<std::endl;
	outFile.close();
	return;
}

void lmn_orb::plot_Delta1(double E){
	double y = find_closed(E,0);
	double P = Pot->Phi({0.,y,0.});
	double p = sqrt(2.*(E-P));
	// Orbit orbit(Pot);
	SymplecticOrbit orbit(Pot); //gjy changed
	VecDoub X = {0.0,y,0.0,p,0.0,0.0};
	VecDoub QQ=orbit.integrate(X, 2.5, 0.005);
	orbit.output2file("orbit_Delta1.dat");
	Actions_TriaxialStackel_Fudge ATSF(Pot,-80.,-20.);
	root_struct_action RS(&ATSF,&orbit,0);
	std::ofstream outFile; outFile.open("Delta1.dat");
	for(double a = ATSF.CS->alpha()+1.;a<ATSF.CS->beta()-1.;a+=0.5)
		outFile<<sqrt(ATSF.CS->beta()-a)<<" "<<min_action(a,&RS)<<std::endl;
	outFile.close();
	return;
}

double lmn_orb::find_beta(double E, double a_init, double b_init){ //gjy comment: called by fillGrads(), to find beta
	if(E<E0){
		std::cerr<<"find_beta: Energy too low\n"<<std::endl;
		return 0.;
	}
	double y = find_closed(E,1);
	if(y<0.){
		std::cerr<<"find_beta: (y<0.)\n"; //gjy add
		// exit(1); //gjy add
		return 10.; // We have failed to find loop orbit at this energy
	}

	// Use orbit shape -- assume elliptical -- default
	if(!use_acts){
DEBUG_PRINT_I(1001); //gjy add
		VecDoub orbit_shape = check_orbit(y,E,1,1);
DEBUG_PRINT_I(1002); //gjy add
		if(orbit_shape[1]<orbit_shape[0])
			std::cout<<"Orbit wrong shape!!!: "
					 <<orbit_shape[0]<<" "<<orbit_shape[1]<<"; "
					 <<Pot->Phi({0.,orbit_shape[0],0.})<<" "
					 <<Pot->Phi({0.,0.,orbit_shape[1]})<<std::endl;
DEBUG_PRINT_I(1003); //gjy add
		return -1-(orbit_shape[1]*orbit_shape[1]-orbit_shape[0]*orbit_shape[0]); //gjy comment: original
		// return -1-abs(orbit_shape[1]*orbit_shape[1]-orbit_shape[0]*orbit_shape[0]); //gjy changed
	} //gjy comment: minimize actions
	else{
		double P = Pot->Phi({0.,y,0.});
		double p = sqrt(2.*(E-P));
		// Orbit orbit(Pot);
		SymplecticOrbit orbit(Pot); //gjy changed
		VecDoub X = {1e-8,y,1e-8,0.,0.,p};
		double torb = Pot->torb(X);
		DEBUG_PRINT_V0d(10, torb, "torb in find_beta else"); //gjy add
		// double step = 1e-4*torb; //gjy comment: original
		// int maxsteps = 100000;
		int maxsteps = 1000; //gjy changed
		double step = torb*10./maxsteps;
		VecDoub QQ=orbit.integrate(X, maxsteps*step, step);
		Actions_TriaxialStackel_Fudge ATSF(Pot, a_init, b_init); //gjy comment: Stackel Fudge, but this is to compute alpha()
		root_struct_action RS(&ATSF,&orbit,1);
		double a = -1.-y*y;
		// a = ATSF.CS->alpha()+0.0001;
		double g = ATSF.CS->gamma()-1e-6;

		auto min_func = /*&min_action;//*/&min_lambda;

		double down = min_func(a,&RS), up = min_func(g,&RS);
		double av = (a+g)/2.;
		double avM = min_func(av,&RS);

		if(debug_find_Delta){
			std::cerr<<"Beta finder at energy "<<E<<": ";
			std::cerr<<"beta = "<<a<<", spread = "<<down;
			std::cerr<<", beta = "<<av<<", spread = "<<avM;
			std::cerr<<", beta = "<<g<<", spread = "<<up<<std::endl;
		}
		while(avM>down){
			av = (av+a)/2.; avM = min_func(av,&RS);
		}
		while(avM>up){
			av = (av+g)/2.; avM = min_func(av,&RS);
		}
		if(avM==up) return -1.001;
		else{
			int status = 0;
			minimiser1D min(min_func,av,a,g,1e-3,0.,&status,&RS); //gjy comment: finally, one still need to use the minimization function to find the minimum focal length
			if(status!=0) std::cerr<<"Minimum not encompassed in find_beta\n";
			return min.minimise(100);
		}
	}
}

double lmn_orb::find_alpha(double E, double alpha_i, double beta){ //gjy comment: called by fillGrads()
	if(E<E0){
		std::cerr<<"find_alpha: Energy too low"<<std::endl;
		return 0.;
	}

	double y = find_closed(E,0);
	if(y<0.){
		std::cerr<<"find_alpha: (y<0.)\n"; //gjy add
		// exit(1); //gjy add
		return 10.; // We have failed to find loop orbit at this energy
	}

	// We either use the shape of the orbit -- assume elliptical --default
	if(!use_acts){
DEBUG_PRINT_I(11113); //gjy add
		VecDoub orbit_shape = check_orbit(y,E,0,1);
DEBUG_PRINT_I(11114); //gjy add
		if(orbit_shape[1]<orbit_shape[0])
			std::cout<<"Orbit wrong shape!!!: "
					 <<orbit_shape[0]<<" "<<orbit_shape[1]<<" "
					 <<Pot->Phi({orbit_shape[0],0.,0.})<<" "
					 <<Pot->Phi({0.,orbit_shape[1],0.})<<std::endl;
DEBUG_PRINT_I(11115); //gjy add
DEBUG_PRINT_V1d(1, (VecDoub){beta, orbit_shape[0], orbit_shape[1], beta-abs(orbit_shape[1]*orbit_shape[1]-orbit_shape[0]*orbit_shape[0])}, "why");
		// return beta-(orbit_shape[1]*orbit_shape[1]-orbit_shape[0]*orbit_shape[0]);
		return beta-abs(orbit_shape[1]*orbit_shape[1]-orbit_shape[0]*orbit_shape[0]); //gjy changed
	}else{// or minimising the actions
		double P = Pot->Phi({0.,y,0.});
		double p = sqrt(2.*(E-P));
		// Orbit orbit(Pot);
		SymplecticOrbit orbit(Pot); //gjy changed
		VecDoub X = {1e-8,y,1e-8,p,0.,0.};
		double torb = Pot->torb(X);
		// double step = 1e-4*torb; //gjy note: original
		// int maxsteps = 100000;
		int maxsteps = 1000; //gjy changed
		double step = torb*10./maxsteps;
		VecDoub QQ=orbit.integrate(X, maxsteps*step, step);
		Actions_TriaxialStackel_Fudge ATSF(Pot, alpha_i, beta);
		root_struct_action RS(&ATSF,&orbit,0);
		double b = ATSF.CS->beta()-1e-5;
		double a1 = 10.*b;
		double av = b-1.;

		auto min_func = /*&min_action;//*/&min_lambda;

		double down = min_func(a1,&RS), up = min_func(b,&RS), avM = min_func(av,&RS);
		while(avM>down){
			av = (av+a1)/2.;avM = min_func(av,&RS);
		}
		while(avM>up){
			av = (av+b)/2.;avM = min_func(av,&RS);
		}
		if(avM==down or avM==up) return -1.002;
		else{
			int status = 0;
			minimiser1D min(min_func,av,a1,b,5e-2,0.,&status,&RS);
			if(status!=0) std::cerr<<"Minimum not encompassed in find_alpha\n";
			return min.minimise(100);
		}
	}
}

VecDoub lmn_orb::find_ab_from_box(double E){ //gjy comment: not called by functions in this .cpp file
	root_struct_mindist RS(Pot,E,0);
	root_find RF(1e-3,100);
	double y = RF.findroot(&EminusPot,1e-5,ymax,&RS);
	double p = sqrt(2.*(E-Pot->Phi({0.,y*.7,0.})));
	return find_best_alphabeta({0.,y*.7,0.,p/sqrt(2.),0.,p/sqrt(2.)});
}

static VecDoub actionSD(Actions_TriaxialStackel_Fudge *ATSF, const std::vector<VecDoub> & results, bool with_freq=false){

	if(with_freq) ATSF->set_freq(true);
	std::vector<VecDoub> Results;
	for(auto Y: results) Results.push_back(ATSF->actions(Y));

	return concatVectors<double>(columnMean<double>(Results),
	                             columnSD<double>(Results),
	                             columnMedian<double>(Results));
}

static double min_actions(const gsl_vector *v, void *params){
	root_struct_actions *RS = (root_struct_actions *) params;
	if(gsl_vector_get(v,1)<gsl_vector_get(v,0) or gsl_vector_get(v,1)>-1. or gsl_vector_get(v,0)>-1.)return 1e10;
	Actions_TriaxialStackel_Fudge ATSF(RS->Pot,gsl_vector_get(v,0),gsl_vector_get(v,1));
	VecDoub f = actionSD(&ATSF,RS->orbit->results());
	return sqrt(f[3]*f[3]+f[4]*f[4]+f[5]*f[5]);
}

void lmn_orb::alphabeta_grid(const VecDoub& X){
	std::ofstream outfile; outfile.open("grid.grid");
	double En = Pot->H(X);
	findDelta_interp(En);
	// Orbit orbit(Pot); //gjy comment: original
	SymplecticOrbit orbit(Pot); //gjy changed
	VecDoub QQ=orbit.integrate(X, 5., 0.1);
	root_struct_actions RS(Pot,&orbit);
	gsl_vector *v; v = gsl_vector_alloc(2);
	for(double alpha = -20.;alpha<-1.1;alpha+=0.2){
		gsl_vector_set(v,0,alpha);
		for(double beta = alpha+0.001;beta<-1.01;beta+=0.2){
			gsl_vector_set(v,1,beta);
			outfile<<sqrt(beta-alpha)<<" "<<sqrt(-1.-beta)<<" "<<min_actions(v,&RS)<<std::endl;
		}
	}
	outfile.close();
	gsl_vector_free(v);
	return;
}

VecDoub lmn_orb::find_best_alphabeta(const VecDoub& X){
	double torb = Pot->torb(X);
	// double step = 5e-1*torb;
	int maxsteps = 1000; //gjy changed
	double step = torb*10./maxsteps;
	// Orbit orbit(Pot);
	SymplecticOrbit orbit(Pot); //gjy changed
	VecDoub QQ=orbit.integrate(X, 100*step, step);
	//orbit.plot(1,2);
	root_struct_actions RS(Pot,&orbit);
	VecDoub a1 = {-1.5,-1.1};
	VecDoub sizes ={.5,.1};
	minimiser min(&min_actions,a1,sizes,1e-5,&RS);
	VecDoub results;
	min.minimise(&results,100,0);
	return results;
}



void lmn_orb::readDeltagrids(const std::string& file){
	DEBUG_PRINT_V0d(1, file, "filename_lmn");
	std::ifstream infile; infile.open(file);
	if(!infile.is_open())std::cerr<<"Problem: "<<file<<" doesn't exist."<<std::endl;
// DEBUG_PRINT_I(99011); //gjy add
	double tmp,tmp2,tmp3, tmp9;
	int idx = 0;
	// while(infile>>tmp>>tmp2>>tmp3){ //gjy comment: original
	while(infile>>tmp>>tmp2>>tmp3>>tmp9>>tmp9>>tmp9>>tmp9>>tmp9){ //gjy changed
		E.push_back(tmp);
		Beta.push_back(tmp2);
		Alpha.push_back(tmp3);
		if(idx==0){E0 = tmp;} //gjy add: reset {Emin, Emax} from the first value to avoid bad outter interpolation
		idx++;
	}
	Emax = tmp; //gjy add: reset {Emin, Emax} from the last value
// DEBUG_PRINT_I(99012); //gjy add
	infile.close();
}

void lmn_orb::fillDeltagrids(const std::string& file){
DEBUG_PRINT_I(12051); //gjy add
	// readDeltagrids(name!=""?name+".Delta_lmn":"Delta_lmn.tmp"); //gjy add
	//gjy comment: the file is to compute Delta(ab) which is called in initialization
	E = VecDoub(NE,0.); Beta = E; Alpha = E; //gjy note: Alpha, Beta在这里开始的!

	if(use_log_grid)
		E = create_log_range(E0,Emax,NE);
	else
		E = create_range(E0,Emax,NE);
DEBUG_PRINT_I(12052); //gjy add

	// for(int i=0;i<NE;i++){ //gky add
	// 	double e = 0., yi = 0.;
	// 	std::cout<<yi<<e<<"\n";
	// }

	#pragma omp parallel for schedule(dynamic)
	for(int i=0;i<NE;i++){
DEBUG_PRINT_I(12053); //gjy add
		double alpha=-1.2, beta=-1.05;
		std::cout<<"fillDeltagrids() for E_grid_"<<i<<std::endl; //gjy add
		beta = find_beta(E[i],alpha,beta);
DEBUG_PRINT_V1d(1, (VecDoub){(double)i, beta}, "beta");
DEBUG_PRINT_I(12054); //gjy add
		if(beta>1.){
DEBUG_PRINT_I(120541); //gjy add
			// These are energies where there are no loop orbits
			beta = 1.;alpha = 1.;
		}else{
DEBUG_PRINT_I(120542); //gjy add
			alpha = find_alpha(E[i],beta-0.001, beta);
		}
DEBUG_PRINT_I(12055); //gjy add
		// just check they are not the same
		if(beta>-1.){
DEBUG_PRINT_I(120551); //gjy add
			// double xin = find_closed(E[i],1);
			beta=-1.05;
			// while(sqrt(-1.-beta)<.5*xin) beta=-1.-(-1.-beta)*2.;
		}
		if(alpha>=beta){
DEBUG_PRINT_I(120552); //gjy add
			// double xin = find_closed(E[i],0);
			alpha=beta-0.1;//-0.05;
			// while(sqrt(beta-alpha)<.5*xin) alpha=beta-(beta-alpha)*2.;
		}
		Alpha[i] = alpha;
		Beta[i] = beta;
		std::cerr<<E[i]<<" "<<beta<<" "<<alpha<<std::endl;
	}
DEBUG_PRINT_I(12056); //gjy add
	// Extrapolate back for those with no loop orbits
	int n_a=-1, n_b=-1;
	for(unsigned i=0;i<E.size();i++)
		if(Beta[i]<0.){ n_b = i; break; }
	for(unsigned i=0;i<E.size();i++)
		if(Alpha[i]<0.){ n_a = i; break; }
	if(n_a<0 and n_b<0)
		std::cerr<<"ALL ENERGIES DISALLOW LOOP ORBITS!!!!"<<std::endl;

	for(int i=0;i<n_a;i++)
		Alpha[i] = (Alpha[n_a+1]-Alpha[n_a])/(E[n_a+1]-E[n_a])*(E[i]-E[n_a])+Alpha[n_a];
	for(int i=0;i<n_b;i++)
		Beta[i] = (Beta[n_b+1]-Beta[n_b])/(E[n_b+1]-E[n_b])*(E[i]-E[n_b])+Beta[n_b];
DEBUG_PRINT_I(12057); //gjy add

	std::ofstream outfile; outfile.open(file);
	double tmp = 0.;
DEBUG_PRINT_V0d(1, E.size(), "E.size()");
	for(unsigned i=0;i<E.size();i++)
		outfile<<E[i]<<" "<<Beta[i]<<" "<<Alpha[i]
		<<" "<<tmp<<" "<<tmp<<" "<<tmp<<" "<<tmp<<" "<<tmp
		<<" \n";
	outfile.close();
DEBUG_PRINT_I(12058); //gjy add
}

VecDoub lmn_orb::findDelta_interp(double En){ //gjy comment: this is called
	double a, b;
	// if(En>Emax){
	// 	double interp = (log(-Alpha[NE-1])-log(-Alpha[NE-2]))/(log(-E[NE-1])-log(-E[NE-2]));
	// 	a = pow(En/E[NE-1],interp)*Alpha[NE-1];
	// 	interp = (log(-Beta[NE-1])-log(-Beta[NE-2]))/(log(-E[NE-1])-log(-E[NE-2]));
	// 	b = pow(En/E[NE-1],interp)*Beta[NE-1];
	// }
	if(En>Emax){
		a = Alpha[NE-1];
		b = Beta[NE-1];
	}
	else if(En<E0){
		a = Alpha[0];
		b = Beta[0];
	}
	else{
		int i=0;
		while(E[i]<En and i<NE-1)	i++; //gjy comment: to find E[i] near En
		a = (Alpha[i]-Alpha[i-1])/(E[i]-E[i-1])*(En-E[i-1])+Alpha[i-1]; //gjy comment: linear
		b = (Beta[i]-Beta[i-1])/(E[i]-E[i-1])*(En-E[i-1])+Beta[i-1];
	}
	return {a,b};
}

VecDoub lmn_orb::actions(const VecDoub& x, void* params){
	
	//one must have loaded Pot with particle data
	double En = Pot->H(x);
	if(En>0){ //gjy add
		std::cout<<"You have passed an unbound orbit:"<<std::endl;
		printVector(x);
		VecDoub actions(4,0);
		return actions;
	}

	if(params!=nullptr){
		double *alphabeta = (double*)params;
		ABC_presentCalculation = {alphabeta[0], alphabeta[1]};
// DEBUG_PRINT_V1d(1, ABC_presentCalculation, "alphabeta inputed");
		Actions_TriaxialStackel_Fudge ATSF(Pot,alphabeta[0],alphabeta[1]); 
		return ATSF.actions(x); //gjy comment: use designated focus
	}
	else{
		VecDoub ab = findDelta_interp(En);
// DEBUG_PRINT_V1d(1, x, "x");
// DEBUG_PRINT_V0d(1, En, "En");
// DEBUG_PRINT_V1d(1, ab, "alphabeta interpolation");
		ABC_presentCalculation = ab;

		// // if alpha>beta or beta>gamma then the axes are not in the right order
		// // so we must flip the axes and change alpha and beta
		// if(ab[0]>ab[1]){
		// 	if(ab[0]<-1.){
		// 		// This is the case when z is minor, x inter, y major
		// 		VecDoub X = x;
		// 		double tmp=ab[1];ab[1]=ab[0];ab[0]=tmp;
		// 		tmp=X[1];X[1]=X[0];X[0]=tmp;
		// 		Actions_TriaxialStackel_Fudge ATSF(Pot,ab[0],ab[1]);
		// 		return ATSF.actions(x);
		// 	}
		// 	else if(ab[1]>-1.)
		// 		// This is the case when x is minor, y inter, z major
		// 		VecDoub X = x;
		// 		double tmp=ab[2];ab[2]=ab[0];ab[0]=tmp;
		// 		tmp=X[2];X[2]=X[0];X[0]=tmp;
		// 	}
		// 	else{
		// 		// This is the case when x is minor, z inter, y major
		// 		VecDoub X = x;
		// 		double tmp=ab[1];ab[1]=ab[0];ab[0]=tmp;
		// 		ab[1]=-2.-ab[1];
		// 		tmp=X[1];X[1]=X[2];X[2]=tmp;
		// 		tmp=X[1];X[1]=X[0];X[0]=tmp;
		// 		Actions_TriaxialStackel_Fudge ATSF(Pot,ab[0],ab[1]);
		// 		return ATSF.actions(x);
		// 	}
		// }
		// else if(ab[1]>-1.){
		// 	if(ab[0]<-1.){
		// 		// This is the case when y is minor, z inter, x major
		// 		VecDoub X = x;
		// 		ab[1]=-2.-ab[1];
		// 		double tmp=X[2];X[2]=X[1];X[1]=tmp;
		// 		Actions_TriaxialStackel_Fudge ATSF(Pot,ab[0],ab[1]);
		// 		return ATSF.actions(x);
		// 	}
		// 	else{
		// 		// This is the case when y is minor, x inter, z major
		// 		VecDoub X = x;
		// 	}
		// }
		Actions_TriaxialStackel_Fudge ATSF(Pot,ab[0],ab[1]); //gjy note: use estimated focus by method in this class
		return ATSF.actions(x);
	}
}

VecDoub lmn_orb::actions(const int& ID, const double& t, const VecDoub& x_foci, void* params){

	//one must have loaded Pot with particle data
// DEBUG_PRINT_V0d(1, this->Pot->pSTAGE->dt_load, "this->Pot->pSTAGE->dt_load");
// 	int s = t_to_loadsnapshot(t, this->Pot->pSTAGE->dt_load, this->Pot->pSTAGE->t_init, 0.);
// DEBUG_PRINT_V0d(1, s, "s");
// DEBUG_PRINT_V0d(1, this->Pot->pSTAGE->SS[s]->P[ID].Pos[0], "this->Pot->pSTAGE->SS[s]->P[ID].Pos[0]");
// DEBUG_PRINT_V0d(0, t, "t");
// 	VecDoub x = {this->Pot->pSTAGE->SS[s]->P[ID].Pos[0], this->Pot->pSTAGE->SS[s]->P[ID].Pos[1], this->Pot->pSTAGE->SS[s]->P[ID].Pos[2], 
// 		this->Pot->pSTAGE->SS[s]->P[ID].Vel[0], this->Pot->pSTAGE->SS[s]->P[ID].Vel[1], this->Pot->pSTAGE->SS[s]->P[ID].Vel[2]};
	
	VecDoub x = x_foci;
	double En = Pot->H(x);
	if(En>0){ //gjy add
		std::cout<<"You have passed an unbound orbit:"<<std::endl;
		printVector(x);
		VecDoub actions(4,0);
		return actions;
	}

	if(params!=nullptr){
		double *alphabeta = (double*)params;
		ABC_presentCalculation = {alphabeta[0], alphabeta[1]};
		Actions_TriaxialStackel_Fudge ATSF(Pot,alphabeta[0],alphabeta[1]); 
		return ATSF.actions(ID, t);
	}
	else{
		VecDoub ab = findDelta_interp(En);
		ABC_presentCalculation = ab;
		Actions_TriaxialStackel_Fudge ATSF(Pot,ab[0],ab[1]);
		return ATSF.actions(ID, t);
	}
}

VecDoub lmn_orb::angles(const VecDoub& x,void *params){
	double En = Pot->H(x);
	if(params!=nullptr){
		double *alphabeta = (double*)params;
		Actions_TriaxialStackel_Fudge ATSF(Pot,alphabeta[0],alphabeta[1]);
		return ATSF.angles(x);
	}
	else{
		VecDoub ab = findDelta_interp(En);
		Actions_TriaxialStackel_Fudge ATSF(Pot,ab[0],ab[1]);
		return ATSF.angles(x);
	}
}

double lmn_orb::sos(int comp, const VecDoub& x, const std::string& outfile){
	double En = Pot->H(x);
	VecDoub alphabeta = findDelta_interp(En);
	Actions_TriaxialStackel_Fudge ATSF(Pot,alphabeta[0],alphabeta[1]);
	return ATSF.sos(x,comp,outfile);
}
// ============================================================================
