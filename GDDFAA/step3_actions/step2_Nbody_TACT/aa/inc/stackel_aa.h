// ============================================================================
/// \file inc/stackel_aa.h
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
/// \brief Action finding in Staeckel potentials and Staeckel fudges
///
/// Four classes are implemented:
/// 1. Actions_AxisymmetricStackel: Action finding in axisymmetric Staeckel
///    potentials (currently accepts perfect ellipsoid potential)
/// 2. Actions_TriaxialStackel: Action finding in triaxial Staeckel potential
/// 3. Actions_AxisymmetricStackel_Fudge: Action estimation in axisymmetric
///    potential using Staeckel fudge (as in Binney (2012))
/// 4. Actions_TriaxialStackel_Fudge :Action estimation in triaxial potential
///    using Staeckel fudge (as in Sanders & Binney (2014))
///
//============================================================================

#ifndef STACKEL_AA_H
#define STACKEL_AA_H

#include "potential.h"
#include "aa.h"

//============================================================================
/*! Action finding in axisymmetric Staeckel potential */
class Actions_AxisymmetricStackel : public Action_Finder{
	private:
		StackelOblate_PerfectEllipsoid *Pot; /*< Staeckel potential */
		std::vector<VecDoub> dtau01dint;

		VecDoub find_limits(const VecDoub& x, const VecDoub& ints);
	public:
		//! Actions_AxisymmetricStackel constructor.
	    /*!
	      \param pot StackelOblate_PerfectEllipsoid potential in which to
	      compute the actions
	    */
		Actions_AxisymmetricStackel(StackelOblate_PerfectEllipsoid *pot): Pot(pot){}
		//! Finds actions
	    /*!
	      \param x phase-space point (x,v)
	      \param params -- doesn't do anything

		  \return actions -- 3D vector J=(J_R,J_phi,J_z)
	    */
		VecDoub actions(const VecDoub& x, void *params=nullptr);
		//! Finds angles
	    /*!
	      \param x phase-space point (x,v)
	      \param with_hess -- option to calculate hessian

		  \return angles and frequencies --
		      	6D vector (theta_R,theta_phi,theta_z,Omega_R,Omega_phi,Omega_z)
	    */
		VecDoub angles(const VecDoub& x, void *params=nullptr);

		// Unimportant functions that need to be accessed by integration routines
		void dtaudint(const VecDoub& limits, const VecDoub& ints);
		double dtaudint(const VecDoub& limits, int i, int j, double theta);
		double dDeltaGLdint(const VecDoub& limits, int i, int j);
		double dp2dtau(double tau, const VecDoub& ints);
		double BigFPrime(double t){return Pot->BigFPrime(t);}
};

/*! Helper structure for finding limits of action integrals for axisymmetric
	Stackel */
struct root_struct_axi{
	StackelOblate_PerfectEllipsoid *P;
	VecDoub Ints; //gjy comment: three integrals of motion
	root_struct_axi(StackelOblate_PerfectEllipsoid *PP, VecDoub ints)
		:P(PP),Ints(ints){}
};

/*! Helper structure for integration of actions for axisymmetric Stackel */
struct action_struct_axi{
	root_struct_axi RS;
	double taubargl, Deltagl, tiny_number;
	action_struct_axi(StackelOblate_PerfectEllipsoid *PP, VecDoub ints, double tb, double Dl, double tn)
		:RS(PP,ints),taubargl(tb),Deltagl(Dl), tiny_number(tn){}
};

/*! Helper structure for finding Hessian for axisymmetric Stackel */
struct hess_struct_axi{
	Actions_AxisymmetricStackel *ASS;
	VecDoub ints,limits;
	double taubargl, Deltagl;
	root_struct_axi RS;
	hess_struct_axi(Actions_AxisymmetricStackel *ASS, StackelOblate_PerfectEllipsoid *pot, VecDoub ints, VecDoub limits,double tb, double Dl)
		:ASS(ASS),ints(ints),limits(limits),taubargl(tb),Deltagl(Dl),RS(pot,ints){}
};

//============================================================================
/*! Action estimation in general axisymmetric potential using Staeckel fudge */
class Actions_AxisymmetricStackel_Fudge : public Action_Finder{
	private:
		Potential_JS *Pot;
		const double tiny_number = 1e-10; /*!< Tolerance for \int 1/p_tau  */
		double E, I2; 					/*!< Energy, I_2 = 0.5 L_z^2       */
		VecDoub Kt;	  					/*!< Third integrals */
		VecDoub find_limits(const VecDoub& x); /*!<Find tau limits 		   */
		void integrals(const VecDoub& tau);	/*!< Find E, I2 and Kt 		   */
	public:
		std::unique_ptr<ProlateSpheroidCoordSys> CS; /*!< Coordinate system*/
		//! Actions_AxisymmetricStackel_Fudge constructor.
	    /*!
	      \param pot Potential (axisymmetric) in which to compute the actions
	      \param a   alpha value to use for coordinate system (will be
	      overridden by actions and angles if required)
	    */
		Actions_AxisymmetricStackel_Fudge(Potential_JS *pot, double a): Pot(pot){
			CS = std::unique_ptr<ProlateSpheroidCoordSys>(new ProlateSpheroidCoordSys(a));
			Kt.resize(2,0);
			// printf("\naxislength a=%f\n\n\n\n\n\n", a); //gjy add
		}
		//! Actions_AxisymmetricStackel_Fudge copy constructor.
		Actions_AxisymmetricStackel_Fudge(const Actions_AxisymmetricStackel_Fudge& a):Pot(a.Pot),CS(new ProlateSpheroidCoordSys(*a.CS)){
			Kt.resize(2,0);
		}
		inline void reset(Potential_JS *pot){Pot = pot;}
		//! Compute potential at tau coordinate
		inline double Phi_tau(const VecDoub& tau){
			VecDoub v = CS->tau2x(tau);
			// double p = Pot->Phi(CS->tau2x(tau)); //gjy add
			// printf("CS->tau2x(tau): ");
			// print_vec(tau);
			// print_vec(v);
			// std::cout<<p<<"\n"; //gjy add
			return Pot->Phi(CS->tau2x(tau));
		}
		//! Compute potential at tau =(l,0,n)
		inline double Phi_tau(double l, double n){
			return Pot->Phi(CS->tau2x({l,0.,n}));
		}
		inline double chi_lam(const VecDoub& tau){
			// printf(" [chi_lam(): tau = %f %f %f, Phi_tau = %f] ", tau[0], tau[1], tau[2], Phi_tau(tau)); //gjy add
			return -(tau[0]-tau[2])*Phi_tau(tau);
		}
		inline double chi_nu(const VecDoub& tau){
			return -(tau[2]-tau[0])*Phi_tau(tau);
		}
		inline double delta(void){return sqrt(CS->gamma()-CS->alpha());}
		//! Finds actions
	    /*!
	      \param x phase-space point (x,v)
	      \param params -- if null, use alpha specified in constructor
	      				-- if >0, estimate using derivatives of potential eq(8) Sanders (2012)
	      				-- if <0, use value passed as new alpha

		  \return actions -- 3D vector J=(J_R,J_phi,J_z)
	    */
		VecDoub actions(const VecDoub& x, void *params=nullptr);
		//! Finds angles
	    /*!
	      \param x phase-space point (x,v)
	      \param params -- if null, use alpha specified in constructor
	      				-- if >0, estimate using derivatives of potential eq(8) Sanders (2012)
	      				-- if <0, use value passed as new alpha

		  \return angles and frequencies --
		      	6D vector (theta_R,theta_phi,theta_z,Omega_R,Omega_phi,Omega_z)
	    */
		VecDoub angles(const VecDoub& x, void *params=nullptr);
};

/*! Helper structure for root-finding for axisymmetric Stackel fudge */
struct root_struct_axi_fudge{
	Actions_AxisymmetricStackel_Fudge *ASF;
	VecDoub Ints;
	VecDoub tau_i;
	int swit;
	root_struct_axi_fudge(Actions_AxisymmetricStackel_Fudge *ASF, VecDoub ints, VecDoub tau_i, int swit)
		:ASF(ASF),Ints(ints),tau_i(tau_i),swit(swit){}
};

/*! Helper structure for action integrals for axisymmetric Stackel fudge */
struct action_struct_axi_fudge{
	Actions_AxisymmetricStackel_Fudge *ASF;
	VecDoub Ints;
	VecDoub tau_i;
	double taubargl, Deltagl;
	int swit;
	double tiny_number;
	action_struct_axi_fudge(Actions_AxisymmetricStackel_Fudge *ASF, VecDoub ints, VecDoub tau_i,double tb, double Dl,int swit, double tn)
		:ASF(ASF),Ints(ints),tau_i(tau_i),taubargl(tb),Deltagl(Dl),swit(swit), tiny_number(tn){}
};

//============================================================================
/*! Action finding in triaxial Staeckel potential */
class Actions_TriaxialStackel : public Action_Finder{
	private:
		StackelTriaxial *Pot; /*< Staeckel potential */
		VecDoub find_limits(const VecDoub& x,const VecDoub& ints);
	public:
		//! Actions_TriaxialStackel constructor.
	    /*!
	      \param pot Triaxial Stackel Potential (axisymmetric)
	    */
		Actions_TriaxialStackel(StackelTriaxial *pot): Pot(pot){}
		//! Finds actions
	    /*!
	      \param x phase-space point (x,v)
	      \param params -- if not null, returns frequencies

		  \return actions -- 3D vector J=(J_R,J_phi,J_z)
	    */
		VecDoub actions(const VecDoub& x0, void *params=nullptr);
};

/*! Helper structure for root-finding for triaxial Staeckel actions */
struct root_struct_triax{
	StackelTriaxial *P;
	VecDoub Ints;
	root_struct_triax(StackelTriaxial *PP, VecDoub ints)
		:P(PP),Ints(ints){}
};

/*! Helper structure for action integrals for triaxial Staeckel actions */
struct action_struct_triax{
	StackelTriaxial *P;
	VecDoub Ints;
	double taubargl, Deltagl;
	action_struct_triax(StackelTriaxial *PP, VecDoub ints, double tb, double Dl)
		:P(PP),Ints(ints),taubargl(tb),Deltagl(Dl){}
};

//============================================================================
/*! Action estimation in general triaxial potentials using Staeckel fudge */
class Actions_TriaxialStackel_Fudge : public Action_Finder{
	private:
		const double tiny_number = 1e-6;	/*!< Tolerance for \int 1/p_tau */
		double E; 							/*!< Energy 				    */
		VecDoub Jt, Kt;						/*!< Second and third integrals */
		VecDoub find_limits(const VecDoub& x); /*!<Find tau limits 		    */
		void integrals(const VecDoub& tau);	/*!< Find E, Jt and Kt 		    */
		bool freq_yes;						/*!< Returns freq from actions  */
	public:
		Potential_JS *Pot; //gjy changed: private->public
		vector<vector<AA_integrating_data>> PAD; //gjy add
		bool is_record_action_integrand = 0; //gjy add: not record, only record when run actions(): J_integrand() when calculate J.
		bool is_recording = 0; //gjy add: during recording, do not change
		int push_tau_ptau_range(double tau, int swit, const VecDoub& x, bool is_update_integrals=0){ //gjy add
			double a = CS->alpha(), b = CS->beta(), c = CS->gamma();
			auto tau_i = CS->xv2tau(x); //abc
			if(is_update_integrals){
				set_integrals(x);
			}
			double phi=0.;
			if(swit==0)			phi = chi_lam({tau,tau_i[1],tau_i[2]});
			else if(swit==1)	phi = chi_mu ({tau_i[0],tau,tau_i[2]});
			else if(swit==2)	phi = chi_nu ({tau_i[0],tau_i[1],tau});
			double p = (E*tau*tau -Jt[swit]*tau +Kt[swit] +phi);
			double pt = p / (2.*(tau+a)*(tau+b)*(tau+c));
			if(1){
				AA_integrating_data AD;
				AD.swit = swit;
				AD.theta = 0.;
				AD.alpha = a;
				AD.beta = b;
				AD.gamma = c;
				AD.tau = tau;
				AD.phichi = phi;
				AD.ptau_root = p;
				AD.ptau = pt;
				AD.ptau_return = 0.;
				this->PAD[swit].push_back(AD);
			}
			return 1;
		}
		// int write_PAD(int ptc_ID){}
		inline double Phi_xyz(const VecDoub x){ //gjy add
			return Pot->Phi(x);
		}
		inline void set_integrals(const VecDoub x){ //gjy add
			this->E = Pot->H(x);
			double a = CS->alpha(), b = CS->beta(), c = CS->gamma();
			auto tau_i = CS->xv2tau(x); //abc
			this->integrals(tau_i);
		}
		inline VecDoub return_integrals(const VecDoub x_only_for_E){ //gjy add
			return {this->Pot->H(x_only_for_E), Jt[0], Jt[1], Jt[2], Kt[0], Kt[1], Kt[2]};
		}
		inline int reset_AA_integrating_data(void){ //gjy add
			vector<vector<AA_integrating_data>>().swap(this->PAD);
			PAD.resize(3);
			is_record_action_integrand = 0;
			is_recording = 0;
			return 1;
		}
		AA_motion_data AM; //gjy add
		void record_AA_motion_data(const VecDoub x, void *params=nullptr){ //gjy add
			for(int i=0;i<6;i++){AM.xv[i] = x[i];}
			AM.ABC[0] = CS->alpha(), AM.ABC[1] = CS->beta(), AM.ABC[2] = CS->gamma();
			AM.phixyz = Pot->Phi(x);
			auto at = CS->xv2tau(x);
			for(int i=0;i<6;i++){AM.tau[i] = at[i];}
			integrals(at); //no +=, running many times is OK
			AM.Ints[0] = E, AM.Ints[1] = E, AM.Ints[2] = E;
			AM.Ints[0+3] = Jt[0], AM.Ints[1+3] = Jt[1], AM.Ints[2+3] = Jt[2];
			AM.Ints[0+3+3] = Kt[0], AM.Ints[1+3+3] = Kt[1], AM.Ints[2+3+3] = Kt[2];
			auto af = find_limits(at);
			for(int i=0;i<6;i++){AM.limits[i] = af[i];}
			for(int i=0;i<4;i++){AM.actions[i] = 0.;} //out
			for(int i=0;i<11;i++){AM.angles[i] = 0.;}
		}

		std::unique_ptr<ConfocalEllipsoidalCoordSys> CS;/*!<coordinate system*/
		//! Actions_TriaxialStackel_Fudge constructor.
	    /*!
	      \param pot Potential_JS (triaxial)
	      \param a   alpha for coordinate system
	      \param b   beta for coordinate system
	    */
		Actions_TriaxialStackel_Fudge(Potential_JS *pot,double a,double b): Pot(pot){ //gjy note: initialization
			if(b>-1.)
				throw std::invalid_argument("Beta must be less than Gamma=-1 in Actions_TriaxialStackel_Fudge");
			if(a>b)
				throw std::invalid_argument("Alpha must be less than Beta in Actions_TriaxialStackel_Fudge");
			CS = std::unique_ptr<ConfocalEllipsoidalCoordSys>
				(new ConfocalEllipsoidalCoordSys(a,b));
			Jt.resize(3,0);Kt.resize(3,0);
			freq_yes=false;
		}
		//! Actions_TriaxialStackel_Fudge copy constructor.
		Actions_TriaxialStackel_Fudge(const Actions_TriaxialStackel_Fudge& s):Pot(s.Pot),
			CS(new ConfocalEllipsoidalCoordSys(*s.CS)){
			Jt.resize(3,0);Kt.resize(3,0);
			freq_yes=false;
		}
		//! reset -- change potential
		inline void reset(Potential_JS *pot){Pot = pot;}
		//! return freq from actions or not
		inline void set_freq(bool s){freq_yes=s;}
		//! computes potential at ellipsoidal coordinates tau
		inline double Phi_tau(const VecDoub& tau){
			// DEBUG_PRINT_V1d(1, CS->tau2x(tau), "CSX");
			// DEBUG_PRINT_V0d(1, Pot->algorithm, "ALG");
			// DEBUG_PRINT_V0d(0, Pot->Phi(CS->tau2x(tau)), "PPT");
			// return Pot->Phi(CS->tau2x(tau)); //gjy comment: orginal
			auto xyz = CS->tau2x(tau); //gjy change: make average because of insymmetry of Nbody density profile
			return ( 
					Pot->Phi({xyz[0],xyz[1],xyz[2]}) 	+Pot->Phi({xyz[0],xyz[1],-xyz[2]})
					+Pot->Phi({xyz[0],-xyz[1],xyz[2]}) 	+Pot->Phi({xyz[0],-xyz[1],-xyz[2]})
					+Pot->Phi({-xyz[0],xyz[1],xyz[2]}) 	+Pot->Phi({-xyz[0],xyz[1],-xyz[2]})
					+Pot->Phi({-xyz[0],-xyz[1],xyz[2]}) +Pot->Phi({-xyz[0],-xyz[1],-xyz[2]})
				)/8;
		}
		//! computes potential at ellipsoidal coordinates tau=(l,m,n)
		inline double Phi_tau(double l, double m, double n){
			return Pot->Phi(CS->tau2x({l,m,n}));
		}
		inline double chi_lam(const VecDoub& tau){ //gjy comment: potential computing for certain target point called by later Fudge
			return (tau[0]-tau[1])*(tau[2]-tau[0])*Phi_tau(tau);
		}
		inline double chi_mu(const VecDoub& tau){
			return (tau[1]-tau[2])*(tau[0]-tau[1])*Phi_tau(tau);
		}
		inline double chi_nu(const VecDoub& tau){
			return (tau[2]-tau[0])*(tau[1]-tau[2])*Phi_tau(tau);
		}
		//! Finds actions
	    /*!
	      \param x phase-space point (x,v)
	      \param params -- does nothing

		  \return actions -- 3D vector J=(J_R,J_phi,J_z)
	    */
		VecDoub actions(const VecDoub& x, void *params=nullptr);
		VecDoub actions(const int& ID, const double& t0, void *params=nullptr);
		//! Finds angles
	    /*!
	      \param x phase-space point (x,v)
	      \param params -- does nothing

		  \return angles and frequencies --
		      	6D vector (theta_R,theta_phi,theta_z,Omega_R,Omega_phi,Omega_z)
	    */
		VecDoub angles(const VecDoub& x, void *params=nullptr);
		VecDoub angles(const double& t0, const int& ID, void *params=nullptr);
		//! computes surface of section
	    /*!
	      \param x phase-space point (x,v)
	      \param comp -- if 0 use x, if 1 use y, if 2 use z
	      \param outfile -- output file

		  \return nothing
	    */
		double sos(const VecDoub& x, int comp,const std::string& outfile);
};

/*! Helper structure for root-finding for triaxial Staeckel fudge */
struct root_struct_triax_fudge{
	Actions_TriaxialStackel_Fudge *ATSF;
	VecDoub Ints;
	VecDoub tau_i;
	int swit;
	root_struct_triax_fudge(Actions_TriaxialStackel_Fudge *ATSF, VecDoub ints, VecDoub tau_i, int swit)
		:ATSF(ATSF),Ints(ints),tau_i(tau_i),swit(swit){}
};
/*! Helper structure for action integrals for triaxial Staeckel fudge */
struct action_struct_triax_fudge{
	Actions_TriaxialStackel_Fudge *ATSF;
	VecDoub Ints;
	VecDoub tau_i;
	double taubargl, Deltagl;
	int swit;
	action_struct_triax_fudge(Actions_TriaxialStackel_Fudge *ATSF, VecDoub ints, VecDoub tau_i, double tb, double Dl, int swit)
		:ATSF(ATSF),Ints(ints),tau_i(tau_i),taubargl(tb),Deltagl(Dl),swit(swit){}
};

// struct orbit_data_intepolation_tau{
// 	Actions_TriaxialStackel_Fudge *ATSF;
// 	VecDoub Ints;
// 	VecDoub tau_i;
// 	int swit;
// 	root_struct_triax_fudge(Actions_TriaxialStackel_Fudge *ATSF, VecDoub ints, VecDoub tau_i, int swit)
// 		:ATSF(ATSF),Ints(ints),tau_i(tau_i),swit(swit){}
// };
struct action_struct_triax_fudge_orbdata{ //gjy add
	Actions_TriaxialStackel_Fudge *ATSF;
	// VecDoub Ints;
	VecDoub tau_i; //or1 initial, tau[6]
	double taubargl, Deltagl;
	int ID; //or1, -1
	int swit, leaf;
	action_struct_triax_fudge_orbdata(Actions_TriaxialStackel_Fudge *ATSF, VecDoub tau_i, double tb, double Dl, int ID, int swit, int leaf)
		:ATSF(ATSF),tau_i(tau_i),taubargl(tb),Deltagl(Dl),ID(ID),swit(swit),leaf(leaf){}
};
#endif
// ============================================================================
