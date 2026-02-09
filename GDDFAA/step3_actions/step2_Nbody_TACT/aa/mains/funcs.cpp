//fundamental
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <cmath>
#include <limits>
#include <functional>
#include <algorithm>
#include <string>
using namespace std;

//about TACT
#include "gnuplot/gnuplot_i.h"
#include <gsl/gsl_poly.h>
#include <mpi.h>
#include "GSLInterface/GSLInterface.h"
#include "utils.h"
#include "coordsys.h"
#include "coordtransforms.h"
#include "potential.h" //where DataInterface.h was included
#include "orbit.h"
#include "Multipole.h" //?? ME
#include "aa.h"
#include "stackel_aa.h"
#include "spherical_aa.h"
#include "genfunc_aa.h"
#include "adiabatic_aa.h"
#include "uv_orb.h"
#include "lmn_orb.h"
#include "stackel_fit.h"
// #ifdef TORUS
// #include "falPot.h"
// #include "it_torus.h"
// #include "PJM_cline.h"
// #endif
#include "gtest/gtest.h"



//// main()
int main(int argc, char* argv[]){

	return 0;
}

// 	////1. input
// 	//:: in linux shell: 
// 	// # cd ~/workroom/0prog/Stackel/tact-master/aa; make clean && cd .. && make LAPACK=1 TORUS=1 && cd aa;  
// 	// # mpirun -np 4 mains/./data.exe  
// 	// # <input arguments below>
// 	//:: snapshots range
// 	double t_init 				= atof(argv[1]); 	//0.
//     double t_final 				= atof(argv[2]); 	//9.
// 	double dt_load				= atof(argv[3]); 	//0.01
// 	double dt_step 				= atof(argv[4]);	//0.001
// 	double t_aligment			= atof(argv[5]);	//0.

// 	//:: what to run
// 	int is_writeSnapshot 		= atoi(argv[6]);	//write snapshots //0
// 	int is_affineTransformation	= atoi(argv[7]); 	//0: not <1>; 1: write and centerize //0
// 	int onlyRun_Actionmethod	= atoi(argv[8]);	//7 = 111(2)

// 	int AlgorithmPot 			= atoi(argv[9]); 	//0: direct summation; 5: direct summation 1
// 	int WhatPotential 			= atoi(argv[10]); 	//0: formula potential; 1: data potential //2
// 	int WhatSymmetry			= atoi(argv[11]); 	//0: spherical; 1: axisymmetric; 2: triaxial //2
// 	int WhatActionmethod		= atoi(argv[12]); 	//0: SF; 1: PPOD all; 2: {Jl} by <1> and {Jm,Jn} by <2> //3

// 	//:: time and particle_ID range to run
// 	double t_start_run			= atof(argv[13]); 	//5
// 	double t_end_run			= atof(argv[14]); 	//5.01
// 	double dt_run 				= atof(argv[15]); 	//1.

// 	int ID_start 				= atoi(argv[16]);	//1
// 	int N_ptcs 					= atoi(argv[17]); 	//the max index of particles //10000: N_all

// 	//:: other
// 	// double changedParticle_mass	= atof(argv[18]);	//1. //0.1



// 	////2. MPI settings
// 	int my_rank, comm_sz;
//     MPI_Status status;
// 	MPI_Init(&argc, &argv);
// 	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
// 	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
// DEBUG_PRINT_I(1);



// 	////3. snapshot settings
// 	string path_work = getworkpath();
// 	string path_base = path_work+"0prog/gadget/Gadget-2.0.7/galaxy_general/";
// 	string path_IC = "IC_param.txt";
// 	Stage STAGE(path_base+path_IC);
// 	Stage* pSTAGE = &STAGE;

// 	////calculate
// 	pSTAGE->load_multi_snapshots(t_init, t_final, dt_load, dt_step, 0, 0);
// 	double Alpha = TACT_semiaxis_Alpha, Beta = TACT_semiaxis_Beta;
// 	MPI_Barrier(MPI_COMM_WORLD);
// DEBUG_PRINT_I(2);

// 	////potentials presetting
// 	Potential_JS DPot(&STAGE);
// 	Potential_JS* pDPot = &DPot;
// 	//pDPot->Phi(..., Potential_other*=nullptr) //??
// 	pDPot->set_algorithm(AlgorithmPot);

// 	// int N_calculate_load = (int)((t_end_run-t_start_run)/dt_run);
// 	vector<Potential_JS> vFPot; //??
// DEBUG_PRINT_I(3);

// 	for(double t0=t_start_run;t0<t_end_run;t0+=dt_run){ //time
// 		int s = t_to_loadsnapshot(t0, dt_load, t_init, 0.); //load
// 		int snapshot = pSTAGE->SS[s]->snap; //snapshot
// 		int calculateIndex = (int)((t0-t_start_run)/dt_run); //calculate
// 		DEBUG_PRINT_V0d(1, t0, "t0");
// 		DEBUG_PRINT_V0d(1, s, "s");
// 		DEBUG_PRINT_V0d(1, snapshot, "snapshot");
// 		DEBUG_PRINT_V0d(1, calculateIndex, "s");

// 		////(1) data potential
// 		string disctrp = "_processed";
// 		// pDPot->pSTAGE->SS[s]->adjust_center_rewrite();
// 		pDPot->pSTAGE->SS[s]->triaxialize();
// 		pDPot->pSTAGE->SS[s]->write_PD_txt(disctrp);
// DEBUG_PRINT_I(4);

// 		////trees
// 		////{x1,x2,x3}-tree
// 		DEBUG_PRINT_V0d(10, s, "before xtree");
// 		std::array<std::array<double, 3>, N_total> xdata; //dynamic tree?? //x-tree and J-tree
// 		for(int i=0;i<N_total;i++){
// 			int iP = i+1;
// 			xdata[i][0] = pSTAGE->SS[s]->P[iP].Pos[0];
// 			xdata[i][1] = pSTAGE->SS[s]->P[iP].Pos[1];
// 			xdata[i][2] = pSTAGE->SS[s]->P[iP].Pos[2];
// 		}
// DEBUG_PRINT_I(5);
// 		KDtree<double, N_total, Dimension> kdt(&xdata);
// 		pSTAGE->SS[s]->loadtree(&kdt);
// 		DEBUG_PRINT_V0d(10, s, "after xtree");



// 		////(2) formula potential
// 		Galaxy_components GA; //GA();
// 		GA.read_fit(path_base, snapshot);
// 		//the expected galaxies values are not used
// 		//TORUS is not used
// DEBUG_PRINT_I(6);

// 		////main examples for potential
// 		// for(int icmp=0;icmp<N_comp;icmp++){ //multi components??
// 		// 	//each potential summation of each componemts
// 		// }
// 		int icmp = 1;
// 		// NFW FPot1(conv::G*GA.M_scale_comp[icmp], GA.scaled_length_fit_comp[icmp], GA.axis_ratio_y_fit_comp[icmp], GA.axis_ratio_z_fit_comp[icmp]);
// 		// Hernquist
// 		// Burkert
// 		// Einasto
// 		// Plummer
// 		// IsothermalTruncated
// 		// Density_DoublePowerLaw rho(1.,10.,{1.,0.6,0.3});
// 		Density_Einasto rho(GA.scaled_density_fit_comp[icmp], GA.scaled_length_fit_comp[icmp], 
// 			{GA.axis_ratio_x_fit_comp[icmp], GA.axis_ratio_y_fit_comp[icmp], 
// 			GA.axis_ratio_z_fit_comp[icmp]}, GA.power1_fit_comp[icmp]
// 		);
// 		MultipoleExpansion FPot1(&rho,5000,12,12,-1,1.,0.0001,10000.,false,true,true);
// 		// TriaxialPotential FP(&rho,1e6);
// 		// MultipoleExpansion_Triaxial FP(&rho,5000,12,12,-1,1.,0.0001,10000.,false,true,true);
		
// 		////pointers
// 		// Potential_JS* pFPot = nullptr;
// 		void* pFPot = nullptr;
// 		// switch(modelId){ //??
// 		// 	case 100:	{
// 		// 		pFPot = &FPot1; //model1
// 		// 	    break;
// 		// 	}
// 		//     case 101: {
// 		// 		pFPot = &FPot2; //model2
// 		//         break;
// 		//     }
// 		// 	// case ...
// 		//     default: { //other
// 		//         printf("No such model! We use sperical NFW formula potential model.\n");
// 		// 		pFPot = &FPot1; //model1
// 		//         break;
// 		//     }
// 		// }
// 		vFPot.push_back(FPot1); //when need FormulaPotential_time
// 		pFPot = &FPot1; //*??
// 		printf("Potentials has been set. Then estimate actions...\n\n\n\n\n\n");
// DEBUG_PRINT_I(7);



// 		////4. actions
// 		double Alpha = TACT_semiaxis_Alpha, Beta = TACT_semiaxis_Beta;
// 		Write_aa = (struct write_angleaction *) malloc(sizeof(struct write_angleaction)*(N_ptcs));
// 		// Write_aa = (struct write_angleaction *) malloc(sizeof(struct write_angleaction)*(N_ptcs+comm_sz));
// 		VecDoub xv0(6);

// 		//:: class of actions/frequencies/angles methods
// 		Actions_Spherical AA_SS_FP(pFPot);
// 		Actions_AxisymmetricStackel_Fudge AA_AF_FP(pFPot,Alpha);
// 		Actions_TriaxialStackel_Fudge AA_TF_FP(pFPot,Alpha,Beta);
// 		// AA_TEPPOD_FP in AA_TF_FP
// 		Actions_Genfunc AA_GF_FP(pFPot,"triaxial");

// 		Actions_Spherical_DataPotential AA_SS_DP(pDPot);
// 		Actions_AxisymmetricStackel_Fudge AA_AF_DP(pDPot,Alpha);
// 		Actions_TriaxialStackel_Fudge AA_TF_DP(pDPot, Alpha, Beta);
// 		// AA_TEPPOD_DP in AA_TF_DP
// 		// Actions_Genfunc AA_GF_DP(pDPot, "triaxial");

// 		//:: VecDoub of actions/frequencies/angles by formula potential or data potential
// 		VecDoub Value_Actions_SS_FP, Value_AnglesFrequencies_SS_FP;
// 		VecDoub Value_Actions_AF_FP, Value_AnglesFrequencies_AF_FP;
// 		VecDoub Value_Actions_TF_FP, Value_AnglesFrequencies_TF_FP;
// 		VecDoub Value_Actions_TEPPOD_FP, Value_AnglesFrequencies_TEPPOD_FP; //not provided, need orbit in formula potential
// 		VecDoub Value_Actions_GF_FP, Value_AnglesFrequencies_GF_FP;

// 		VecDoub Value_Actions_SS_DP, Value_AnglesFrequencies_SS_DP;
// 		VecDoub Value_Actions_AF_DP, Value_AnglesFrequencies_AF_DP;
// 		VecDoub Value_Actions_TF_DP, Value_AnglesFrequencies_TF_DP;
// 		VecDoub Value_Actions_TEPPOD_DP, Value_AnglesFrequencies_TEPPOD_DP;
// 		VecDoub Value_Actions_GF_DP, Value_AnglesFrequencies_GF_DP; //not provided, need data potential in O2GF
		


// 		////5. MPI: calculation distribution
// 		int remainder = N_ptcs%(comm_sz);
// 		// int interval = N_ptcs/comm_sz+(int)(N_ptcs*0.001/comm_sz);
// 		int interval = N_ptcs/comm_sz;
// 		int local_from, local_count, total_count = 0;

// 		////[main start
// 		if(my_rank==0){
			
// 			int my_from = interval*(comm_sz-1);
// 			int my_count = N_ptcs - my_from;
// 			int my_endout = my_from + my_count;
// 			if(my_count>interval*2){
// 				std::cerr<<"Bad task distribution because of too many ranks, which lead "
// 					"the count of tasks in rank 0 is much more than the count of tasks in rank other.\n";
// 			}
// 			printf("MPI tasks distribution: total number of tasks to do is %d, common size is %d, task interval is %d, "
// 				"the remainder is %d, rest = %d.\n", 
// 				N_ptcs, comm_sz, interval, remainder, my_count);
// 			printf("my_rank is %d: zero my_count = %d, my_from = %d, my_endout = %d.\n", 0, my_count, my_from, my_endout);
// 			printf("Now actions calculation ...\n\n");
// 			for(int i=my_from;i<my_endout;i++)
// 			{
// 				//:: calculate
// 				int ID = i+ID_start;
// 				pDPot->set_partical_ID(ID);
// 				pDPot->set_time(t0);

// 				xv0 = {pDPot->pSTAGE->SS[s]->P[ID].Pos[0], pDPot->pSTAGE->SS[s]->P[ID].Pos[1], pDPot->pSTAGE->SS[s]->P[ID].Pos[2], 
// 					pDPot->pSTAGE->SS[s]->P[ID].Vel[0], pDPot->pSTAGE->SS[s]->P[ID].Vel[1], pDPot->pSTAGE->SS[s]->P[ID].Vel[2] };
// 				// for(int ix=0;ix<6;ix++){ //something uncalculatable when zero??
// 				// 	if(abs(xv0[ix])<err) xv0[ix] = err;
// 				// }

// 				int B0 = onlyRun_Actionmethod % 2; //(bool)Is run Sperical and Fudge in FPot
// 				int B1 = (onlyRun_Actionmethod - B0)/2 % 2; //(bool)Is run Sperical and Fudge in FPot
// 				int B2 = (onlyRun_Actionmethod - B0 - B1*2)/2/2 % 2; //(bool)Is run TEPPOD(none) and O2GF in FPot
// 				int B3 = (onlyRun_Actionmethod - B0 - B1*2 - B2*2*2)/2/2/2 % 2; //(bool)Is run TEPPOD and O2GF(none) in DPot
// 				if((bool)B0){
// 					Value_Actions_SS_FP = AA_SS_FP.actions(xv0);
// 					Value_AnglesFrequencies_SS_FP = AA_SS_FP.angles_and_freqs(xv0);

// 					Value_Actions_AF_FP = AA_AF_FP.actions(xv0);
// 					Value_AnglesFrequencies_AF_FP = AA_AF_FP.angles(xv0);

// 					Value_Actions_TF_FP = AA_TF_FP.actions(xv0);
// 					Value_AnglesFrequencies_TF_FP = AA_TF_FP.angles(xv0);
// 				}else{
// 					Value_Actions_SS_FP.resize(3, 0.);
// 					Value_AnglesFrequencies_SS_FP.resize(6, 0.);
// 					Value_Actions_AF_FP.resize(3, 0.);
// 					Value_AnglesFrequencies_AF_FP.resize(6, 0.);
// 					Value_Actions_TF_FP.resize(4, 0.);
// 					Value_AnglesFrequencies_TF_FP.resize(11, 0.);
// 				}
// 				if((bool)B1){
// 					Value_Actions_SS_DP = AA_SS_DP.actions(xv0);
// 					Value_AnglesFrequencies_SS_DP = AA_SS_DP.angles_and_freqs(xv0);

// 					Value_Actions_AF_DP = AA_AF_DP.actions(xv0);
// 					Value_AnglesFrequencies_AF_DP = AA_AF_DP.angles(xv0);

// 					Value_Actions_TF_DP = AA_TF_DP.actions(xv0);
// 					Value_AnglesFrequencies_TF_DP = AA_TF_DP.angles(xv0);
// 				}else{
// 					Value_Actions_SS_DP.resize(3, 0.);
// 					Value_AnglesFrequencies_SS_DP.resize(6, 0.);
// 					Value_Actions_AF_DP.resize(3, 0.);
// 					Value_AnglesFrequencies_AF_DP.resize(6, 0.);
// 					Value_Actions_TF_DP.resize(4, 0.);
// 					Value_AnglesFrequencies_TF_DP.resize(11, 0.);
// 				}
// 				if((bool)B2){
// 					Value_Actions_TEPPOD_FP.resize(LENGTH_TEPPOD, 0.);
// 					Value_AnglesFrequencies_TEPPOD_FP.resize(6, 0.);

// 					Value_Actions_GF_FP = AA_GF_FP.actions(xv0);
// 					Value_AnglesFrequencies_GF_FP = AA_GF_FP.angles(xv0);
// 				}else{
// 					Value_Actions_TEPPOD_FP.resize(LENGTH_TEPPOD, 0.);
// 					Value_AnglesFrequencies_TEPPOD_FP.resize(6, 0.);
// 					Value_Actions_GF_FP.resize(3, 0.);
// 					Value_AnglesFrequencies_GF_FP.resize(6, 0.);
// 				}
// 				if((bool)B3){
// 					Value_Actions_TEPPOD_DP = AA_TF_DP.actions(ID, t0);
// 					Value_AnglesFrequencies_TEPPOD_DP.resize(6, 0.); //??
// 					pDPot->pSTAGE->write_orbitApproxPeriod();
// 					pDPot->pSTAGE->reset_orbitdata();

// 					Value_Actions_GF_DP.resize(3, 0.);
// 					Value_AnglesFrequencies_GF_DP.resize(6, 0.);
// 				}else{
// 					Value_Actions_TEPPOD_DP.resize(LENGTH_TEPPOD, 0.);
// 					Value_AnglesFrequencies_TEPPOD_DP.resize(6, 0.);
// 					Value_Actions_GF_DP.resize(3, 0.);
// 					Value_AnglesFrequencies_GF_DP.resize(6, 0.);
// 				}
// 				Value_Actions_SS_FP.push_back(0); //to let the length be at least 4, similarly hereinafter
// 				Value_Actions_AF_FP.push_back(0);
// 				Value_Actions_TF_FP.push_back(0);
// 				Value_Actions_TEPPOD_FP.push_back(0);
// 				Value_Actions_GF_FP.push_back(0);
// 				Value_Actions_SS_DP.push_back(0);
// 				Value_Actions_AF_DP.push_back(0);
// 				Value_Actions_TF_DP.push_back(0);
// 				Value_Actions_TEPPOD_DP.push_back(0);
// 				Value_Actions_GF_DP.push_back(0);
				
// 				//:: gather
// 				//gather basic info, from index 0, counts 6+1+3+2
// 				for(int ii=0;ii<6;ii++) Write_aa[i].particle_xv0[ii] = xv0[ii]; //init phase position
// 				Write_aa[i].particle_ID = ID;
// 				Write_aa[i].particle_otherInfo[0] = pDPot->pSTAGE->SS[s]->P[ID].Type;
// 				Write_aa[i].particle_otherInfo[1] = pDPot->pSTAGE->SS[s]->P[ID].Mass;
// 				Write_aa[i].particle_otherInfo[2] = pDPot->pSTAGE->SS[s]->P[ID].dAdt; //or other
// 				Write_aa[i].particle_otherInfo[3] = pFPot->Phi(xv0);
// 				Write_aa[i].particle_otherInfo[4] = pDPot->Phi(xv0);

// 				//gather acitions, angles and frequencies
// 				//B0:
// 				// VecDoub Value_Actions_SS_FP, Value_AnglesFrequencies_SS_FP;
// 				// VecDoub Value_Actions_AF_FP, Value_AnglesFrequencies_AF_FP;
// 				// VecDoub Value_Actions_TF_FP, Value_AnglesFrequencies_TF_FP;
// 				//B1:
// 				// VecDoub Value_Actions_SS_DP, Value_AnglesFrequencies_SS_DP;
// 				// VecDoub Value_Actions_AF_DP, Value_AnglesFrequencies_AF_DP;
// 				// VecDoub Value_Actions_TF_DP, Value_AnglesFrequencies_TF_DP;
// 				//B2:
// 				// VecDoub Value_Actions_TEPPOD_FP, Value_AnglesFrequencies_TEPPOD_FP;
// 				// VecDoub Value_Actions_GF_FP, Value_AnglesFrequencies_GF_FP;
// 				//B3:
// 				// VecDoub Value_Actions_TEPPOD_DP, Value_AnglesFrequencies_TEPPOD_DP;
// 				// VecDoub Value_Actions_GF_DP, Value_AnglesFrequencies_GF_DP;
				
// 				//AA about B0, from index 12, counts (4+6)*3
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_SS_FP[ii] = Value_Actions_SS_FP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_SS_FP[ii] = Value_AnglesFrequencies_SS_FP[ii];
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_AF_FP[ii] = Value_Actions_AF_FP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_AF_FP[ii] = Value_AnglesFrequencies_AF_FP[ii];
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TF_FP[ii] = Value_Actions_TF_FP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TF_FP[ii] = Value_AnglesFrequencies_TF_FP[ii];
				
// 				//AA about B1, from index 42, counts (4+6)*3
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_SS_DP[ii] = Value_Actions_SS_DP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_SS_DP[ii] = Value_AnglesFrequencies_SS_DP[ii];
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_AF_DP[ii] = Value_Actions_AF_DP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_AF_DP[ii] = Value_AnglesFrequencies_AF_DP[ii];
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TF_DP[ii] = Value_Actions_TF_DP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TF_DP[ii] = Value_AnglesFrequencies_TF_DP[ii];
				
// 				//AA about B2, from index 72, counts (4+6)*2
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TEPPOD_FP[ii] = Value_Actions_TEPPOD_FP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TEPPOD_FP[ii] = Value_AnglesFrequencies_TEPPOD_FP[ii];
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_GF_FP[ii] = Value_Actions_GF_FP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_GF_FP[ii] = Value_AnglesFrequencies_GF_FP[ii];

// 				//AA about B3, from index 92, counts (4+6)*2
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TEPPOD_DP[ii] = Value_Actions_TEPPOD_DP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TEPPOD_DP[ii] = Value_AnglesFrequencies_TEPPOD_DP[ii];
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_GF_DP[ii] = Value_Actions_GF_DP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_GF_DP[ii] = Value_AnglesFrequencies_GF_DP[ii];
// 				//:: end, totally counts 112 (the most datatype is double).

// 				//:: Tell AA about particle_ID has been calculated
// 				printf("ID_%d(%d, %f) ", ID, my_rank, (double)(i-my_from)/interval);
// 			}
// 			printf("\n\n");
// 			// memcpy(&Write_aa[my_from], &Write_aa[my_from], sizeof(struct write_angleaction)*my_count);
// 			// memcpy(&Write_aa[my_from], &Write_aa0[my_from], sizeof(struct write_angleaction)*my_count);
// 			total_count += my_count;
// // DEBUG_PRINT_I(75);

// 			//recive from other rank:
// 			for(int source=1;source<comm_sz;source++){
// 				/* comparations:
// 				at rank other (send):	~		at rank 0 (recive):
// 				my_rank							source
// 				my_from							local_from
// 				my_count						local_count
// 				my_sum							local_sum
// 				&S[my_from]						&SS[local_from]
// 				: the struct vector is recollectived */
// 				local_from = interval*(source-1);
// 				local_count = interval;
// 				MPI_Recv(&Write_aa[local_from],sizeof(struct write_angleaction)*local_count,MPI_BYTE, source, 0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
// 				total_count += local_count;
// 			}
// // DEBUG_PRINT_I(76);
// 		}else{

// 			int my_from = interval*(my_rank-1);
// 			int my_count = interval;
// 			int my_endout = my_from + my_count;
// 			printf("my_rank is %d: here my_count = %d, my_from = %d, my_endout = %d.\n", 
// 				my_rank, my_count, my_from, my_endout);
// 			for(int i=my_from;i<my_endout;i++)
// 			{
// 				//:: calculate
// 				int ID = i+ID_start;
// 				pDPot->set_partical_ID(ID);
// 				pDPot->set_time(t0);

// 				xv0 = {pDPot->pSTAGE->SS[s]->P[ID].Pos[0], pDPot->pSTAGE->SS[s]->P[ID].Pos[1], pDPot->pSTAGE->SS[s]->P[ID].Pos[2], 
// 					pDPot->pSTAGE->SS[s]->P[ID].Vel[0], pDPot->pSTAGE->SS[s]->P[ID].Vel[1], pDPot->pSTAGE->SS[s]->P[ID].Vel[2] };
// 				// for(int ix=0;ix<6;ix++){ //something uncalculatable when zero??
// 				// 	if(abs(xv0[ix])<err) xv0[ix] = err;
// 				// }

// 				int B0 = onlyRun_Actionmethod % 2; //(bool)Is run Sperical and Fudge in FPot
// 				int B1 = (onlyRun_Actionmethod - B0)/2 % 2; //(bool)Is run Sperical and Fudge in FPot
// 				int B2 = (onlyRun_Actionmethod - B0 - B1*2)/2/2 % 2; //(bool)Is run TEPPOD(none) and O2GF in FPot
// 				int B3 = (onlyRun_Actionmethod - B0 - B1*2 - B2*2*2)/2/2/2 % 2; //(bool)Is run TEPPOD and O2GF(none) in DPot
// 				if((bool)B0){
// 					Value_Actions_SS_FP = AA_SS_FP.actions(xv0);
// 					Value_AnglesFrequencies_SS_FP = AA_SS_FP.angles_and_freqs(xv0);

// 					Value_Actions_AF_FP = AA_AF_FP.actions(xv0);
// 					Value_AnglesFrequencies_AF_FP = AA_AF_FP.angles(xv0);

// 					Value_Actions_TF_FP = AA_TF_FP.actions(xv0);
// 					Value_AnglesFrequencies_TF_FP = AA_TF_FP.angles(xv0);
// 				}else{
// 					Value_Actions_SS_FP.resize(3, 0.);
// 					Value_AnglesFrequencies_SS_FP.resize(6, 0.);
// 					Value_Actions_AF_FP.resize(3, 0.);
// 					Value_AnglesFrequencies_AF_FP.resize(6, 0.);
// 					Value_Actions_TF_FP.resize(4, 0.);
// 					Value_AnglesFrequencies_TF_FP.resize(11, 0.);
// 				}
// 				if((bool)B1){
// 					Value_Actions_SS_DP = AA_SS_DP.actions(xv0);
// 					Value_AnglesFrequencies_SS_DP = AA_SS_DP.angles_and_freqs(xv0);

// 					Value_Actions_AF_DP = AA_AF_DP.actions(xv0);
// 					Value_AnglesFrequencies_AF_DP = AA_AF_DP.angles(xv0);

// 					Value_Actions_TF_DP = AA_TF_DP.actions(xv0);
// 					Value_AnglesFrequencies_TF_DP = AA_TF_DP.angles(xv0);
// 				}else{
// 					Value_Actions_SS_DP.resize(3, 0.);
// 					Value_AnglesFrequencies_SS_DP.resize(6, 0.);
// 					Value_Actions_AF_DP.resize(3, 0.);
// 					Value_AnglesFrequencies_AF_DP.resize(6, 0.);
// 					Value_Actions_TF_DP.resize(4, 0.);
// 					Value_AnglesFrequencies_TF_DP.resize(11, 0.);
// 				}
// 				if((bool)B2){
// 					Value_Actions_TEPPOD_FP.resize(LENGTH_TEPPOD, 0.);
// 					Value_AnglesFrequencies_TEPPOD_FP.resize(6, 0.);

// 					Value_Actions_GF_FP = AA_GF_FP.actions(xv0);
// 					Value_AnglesFrequencies_GF_FP = AA_GF_FP.angles(xv0);
// 				}else{
// 					Value_Actions_TEPPOD_FP.resize(LENGTH_TEPPOD, 0.);
// 					Value_AnglesFrequencies_TEPPOD_FP.resize(6, 0.);
// 					Value_Actions_GF_FP.resize(3, 0.);
// 					Value_AnglesFrequencies_GF_FP.resize(6, 0.);
// 				}
// 				if((bool)B3){
// 					Value_Actions_TEPPOD_DP = AA_TF_DP.actions(ID, t0);
// 					Value_AnglesFrequencies_TEPPOD_DP.resize(6, 0.); //??
// 					pDPot->pSTAGE->write_orbitApproxPeriod();
// 					pDPot->pSTAGE->reset_orbitdata();

// 					Value_Actions_GF_DP.resize(3, 0.);
// 					Value_AnglesFrequencies_GF_DP.resize(6, 0.);
// 				}else{
// 					Value_Actions_TEPPOD_DP.resize(LENGTH_TEPPOD, 0.);
// 					Value_AnglesFrequencies_TEPPOD_DP.resize(6, 0.);
// 					Value_Actions_GF_DP.resize(3, 0.);
// 					Value_AnglesFrequencies_GF_DP.resize(6, 0.);
// 				}
// 				Value_Actions_SS_FP.push_back(0); //to let the length be at least 4, similarly hereinafter
// 				Value_Actions_AF_FP.push_back(0);
// 				Value_Actions_TF_FP.push_back(0);
// 				Value_Actions_TEPPOD_FP.push_back(0);
// 				Value_Actions_GF_FP.push_back(0);
// 				Value_Actions_SS_DP.push_back(0);
// 				Value_Actions_AF_DP.push_back(0);
// 				Value_Actions_TF_DP.push_back(0);
// 				Value_Actions_TEPPOD_DP.push_back(0);
// 				Value_Actions_GF_DP.push_back(0);
				
// 				//:: gather
// 				//gather basic info, from index 0, counts 6+1+3+2
// 				for(int ii=0;ii<6;ii++) Write_aa[i].particle_xv0[ii] = xv0[ii]; //init phase position
// 				Write_aa[i].particle_ID = ID;
// 				Write_aa[i].particle_otherInfo[0] = pDPot->pSTAGE->SS[s]->P[ID].Type;
// 				Write_aa[i].particle_otherInfo[1] = pDPot->pSTAGE->SS[s]->P[ID].Mass;
// 				Write_aa[i].particle_otherInfo[2] = pDPot->pSTAGE->SS[s]->P[ID].dAdt; //or other
// 				Write_aa[i].particle_otherInfo[3] = pFPot->Phi(xv0);
// 				Write_aa[i].particle_otherInfo[4] = pDPot->Phi(xv0);

// 				//gather acitions, angles and frequencies
// 				//B0:
// 				// VecDoub Value_Actions_SS_FP, Value_AnglesFrequencies_SS_FP;
// 				// VecDoub Value_Actions_AF_FP, Value_AnglesFrequencies_AF_FP;
// 				// VecDoub Value_Actions_TF_FP, Value_AnglesFrequencies_TF_FP;
// 				//B1:
// 				// VecDoub Value_Actions_SS_DP, Value_AnglesFrequencies_SS_DP;
// 				// VecDoub Value_Actions_AF_DP, Value_AnglesFrequencies_AF_DP;
// 				// VecDoub Value_Actions_TF_DP, Value_AnglesFrequencies_TF_DP;
// 				//B2:
// 				// VecDoub Value_Actions_TEPPOD_FP, Value_AnglesFrequencies_TEPPOD_FP;
// 				// VecDoub Value_Actions_GF_FP, Value_AnglesFrequencies_GF_FP;
// 				//B3:
// 				// VecDoub Value_Actions_TEPPOD_DP, Value_AnglesFrequencies_TEPPOD_DP;
// 				// VecDoub Value_Actions_GF_DP, Value_AnglesFrequencies_GF_DP;
				
// 				//AA about B0, from index 12, counts (4+6)*3
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_SS_FP[ii] = Value_Actions_SS_FP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_SS_FP[ii] = Value_AnglesFrequencies_SS_FP[ii];
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_AF_FP[ii] = Value_Actions_AF_FP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_AF_FP[ii] = Value_AnglesFrequencies_AF_FP[ii];
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TF_FP[ii] = Value_Actions_TF_FP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TF_FP[ii] = Value_AnglesFrequencies_TF_FP[ii];
				
// 				//AA about B1, from index 42, counts (4+6)*3
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_SS_DP[ii] = Value_Actions_SS_DP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_SS_DP[ii] = Value_AnglesFrequencies_SS_DP[ii];
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_AF_DP[ii] = Value_Actions_AF_DP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_AF_DP[ii] = Value_AnglesFrequencies_AF_DP[ii];
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TF_DP[ii] = Value_Actions_TF_DP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TF_DP[ii] = Value_AnglesFrequencies_TF_DP[ii];
				
// 				//AA about B2, from index 72, counts (4+6)*2
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TEPPOD_FP[ii] = Value_Actions_TEPPOD_FP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TEPPOD_FP[ii] = Value_AnglesFrequencies_TEPPOD_FP[ii];
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_GF_FP[ii] = Value_Actions_GF_FP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_GF_FP[ii] = Value_AnglesFrequencies_GF_FP[ii];

// 				//AA about B3, from index 92, counts (4+6)*2
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TEPPOD_DP[ii] = Value_Actions_TEPPOD_DP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TEPPOD_DP[ii] = Value_AnglesFrequencies_TEPPOD_DP[ii];
// 				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_GF_DP[ii] = Value_Actions_GF_DP[ii];
// 				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_GF_DP[ii] = Value_AnglesFrequencies_GF_DP[ii];
// 				//:: end, totally counts 112 (the most datatype is double).

// 				//:: Tell AA about particle_ID has been calculated
// 				printf("ID_%d(%d, %f) ", ID, my_rank, (double)(i-my_from)/interval);
// 			}
// 			printf("\n\n");

// 			// MPI_Barrier(MPI_COMM_WORLD);
// 			printf("before MPI_Send()\n");
// 			MPI_Send(&Write_aa[my_from],sizeof(struct write_angleaction)*my_count,MPI_BYTE, 0, 0,MPI_COMM_WORLD);
// 			printf("after MPI_Send()\n");
// 		}
// 		MPI_Barrier(MPI_COMM_WORLD);
// 		printf("\nmy_rank (%d): Calculation actions ... done.\n\n", my_rank);
// 		////main end]



// 		////6. samples and gJ, space xv density ~ space OJ density
// 		//another file MPI //??
// 		//read samples 1e6 //??
// 		//each SFFP, TEPPOD(should orbit, slow) O2GF(slow) //??
// 		//write AA 1e6 //??



// 		////write
// 		if(my_rank==0){
// 			if(total_count!=N_ptcs){
// 				std::cerr<<"Wrong total count of particles: total_count("<<total_count<<") != N_ptcs("<<N_ptcs<<"). Please check!\n";
// 				exit(0);
// 			}
// 			printf("\ntotal_count = %d.\n", total_count);

// 			////7. write actions
// DEBUG_PRINT_I(8);
// 			char wt_fname[MaxCharactersInString];
// 			sprintf(wt_fname, "%saa/snapshot_%d.action.samples.txt", path_base.data(), snapshot);
// 			string file_info = "##write: #xv(double[6]);     ID(int)     "
// 				"particle_type(int->double) particle_mass particle_other_info(double; no use) "
// 				"potential_formula_here(double) potential_data_here(double).                 "
// 				// "actions_spherical(double[3]; these are about formula potential, begin)     "
// 				// "angles_and_frequencies_spherical(double[3], double[3])         "
// 				// "actions_axisymmetricFudge(double[3])      "
// 				// "angles_and_frequencies_axisymmetricFudge(double[3], double[3])         "
// 				// "actions_triaxialFudge(double[3]) orbit_type(int->double; by fudge)     "
// 				// "angles_and_frequencies_triaxialFudge(double[3], double[3])     "
// 				// "angles_and_frequencies_otherinfo(double[5]; no use; "
// 				// "these are about formula potential, end)                 "
// 				// "actions...(the rest of the line are actions and angles_and_frequencies "
// 				// "about data potential) "
// 				"AA(actions[3+1], frequecies[3] and angles[3]) by Sperical, Axisymmetric, Triaxial Fudge in FPot; "
// 				"AA by Sperical, Axisymmetric, Triaxial Fudge in DPot; "
// 				"AA by TEPPOD(not provided) and O2GF in FPot; "
// 				"AA by TEPPOD and O2GF(not provided) in DPot. "
// 				"#The main index: ([0] ..., [12] ..., [42] ..., [72] ..., [92], ...); "
// 				"The totally count is 112.";
// 			std::ofstream outfile;
// 			outfile.open(wt_fname);
// 			outfile<<file_info.data()<<"\n";
// 			for(int i=0;i<N_ptcs;i++){
// 				//xv
// 				for(auto aa : Write_aa[i].particle_xv0) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				//other info
// 				outfile<<Write_aa[i].particle_ID<<" ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].particle_otherInfo) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				outfile<<"    ";

// 				//AA B0
// 				for(auto aa : Write_aa[i].Value_Actions_SS_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_AnglesFrequencies_SS_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_Actions_AF_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_AnglesFrequencies_AF_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_Actions_TF_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_AnglesFrequencies_TF_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				outfile<<"    ";

// 				//AA B1
// 				for(auto aa : Write_aa[i].Value_Actions_SS_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_AnglesFrequencies_SS_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_Actions_AF_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_AnglesFrequencies_AF_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_Actions_TF_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_AnglesFrequencies_TF_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				outfile<<"    ";

// 				//AA B2
// 				for(auto aa : Write_aa[i].Value_Actions_TEPPOD_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_AnglesFrequencies_TEPPOD_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_Actions_GF_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_AnglesFrequencies_GF_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				outfile<<"    ";

// 				//AA B3
// 				for(auto aa : Write_aa[i].Value_Actions_TEPPOD_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_AnglesFrequencies_TEPPOD_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_Actions_GF_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				for(auto aa : Write_aa[i].Value_AnglesFrequencies_GF_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				outfile<<"    ";
// 				outfile<<"#endl\n";
// 			}
// 			outfile.close();
// 			printf("\nWrite actions to \"%s\", done.\n\n\n", s, wt_fname);
// DEBUG_PRINT_I(9);



// 			/*
// 			////8. write kernel density of mass and actions
// 			//another file, or in python instead of this slow-debug-snale CC++ //??
// 			std::array<std::array<double, 3>, N_total> xdata0, xdata1; //dynamic tree?? //x-tree and J-tree
// 			wtfh = (struct write_firsthand *) malloc(sizeof(struct write_firsthand)*(N_total+64));
// 			wtsh = (struct write_secondhand *) malloc(sizeof(struct write_secondhand)*(N_total+64));
// 			int WhatCannonical;

// 			WhatCannonical = 2;
// 			pSTAGE->SS[s]->load_to_firsthand_from_PD(); //??
// 			for(int i=0;i<N_total;i++){
// 				int iP = i+1;
// 				xdata0[i][0] = pSTAGE->SS[s]->P[iP].Pos[0];
// 				xdata0[i][1] = pSTAGE->SS[s]->P[iP].Pos[1];
// 				xdata0[i][2] = pSTAGE->SS[s]->P[iP].Pos[2];
// 			}
// 			KDtree<double, N_total, Dimension> kdt0(&xdata0);
// 			pSTAGE->SS[s]->loadtree(&kdt0);
// 			pSTAGE->SS[s]->write_secondhand_all_txt(WhatCannonical, 0, 0, 0);
// 			// pSTAGE->SS[s]->remove_tree(); //then after {}, the array will be release
// 			free(wtfh);
// 			free(wtsh);
// DEBUG_PRINT_I(10);

// 			WhatCannonical = 5; //WhatCannonical
// 			for(int WP=0;WP<2;WP++){ //WhatPotential
// 				for(int WS=0;WS<3;WS++){ //WhatSymmetry
// 					for(int WA=0;WA<3;WA++){ //WhatActionmethod
// 						wtfh = (struct write_firsthand *) malloc(sizeof(struct write_firsthand)*(N_total+64));
// 						wtsh = (struct write_secondhand *) malloc(sizeof(struct write_secondhand)*(N_total+64));
// 						int isExistFile = pSTAGE->SS[s]->read_firsthand_all_txt(WhatCannonical, WP, WS, WA);
// 						if(isExistFile!=0){
// 							continue;
// 						}
// 						for(int i=0;i<N_total;i++){
// 							xdata1[i][0] = wtfh[i].QP[3]; xdata1[i][1] = wtfh[i].QP[4]; xdata1[i][2] = wtfh[i].QP[5]; //J-tree
// 						}
// 						KDtree<double, N_total, Dimension> kdt1(&xdata1);
// 						pSTAGE->SS[s]->loadtree1(&kdt1);
// 						pSTAGE->SS[s]->write_secondhand_all_txt(WhatCannonical, WP, WS, WA); //char* info
// 						// pSTAGE->SS[s]->remove_tree();
// 						free(wtfh);
// 						free(wtsh);
// 					}
// 				}
// 			}
// 			*/

// DEBUG_PRINT_I(11);
// 		}
// 		total_count = 0;

// 		free(Write_aa);
// 		MPI_Barrier(MPI_COMM_WORLD);
// DEBUG_PRINT_I(12);
// 	}

// 	////9. end
// 	MPI_Barrier(MPI_COMM_WORLD);
// 	// if(my_rank==0){
// 	if(1){
// 		printf("\nDone.\n");
// 	}
//     MPI_Finalize();
// 	return 0;
// }



/*
	////compare potentials and actions
	// VecDoub xv0_debug = {1., 10., 1., 220., 10., 1.};
	// for(){}

	////gJ
	// read samples, simply senn it as dd of dV1/dV2

	////fJ
	// is_correct, gJ3 and fJ select //in python

	////collision, diffusion, resonance, phase mixing scales
	// [...]

	////stream
	// [...]

	////laws of self-adaptive multi-agents evolution/game
	// [...]
	
	////ABCDEFG
	// [...]
*/



/*

	////others //from Sanders TACT/test/
	// TEST(MultipoleT,Stackel){
	// 	TestDensity_Stackel rho(1.,-30.,-10.);
	// 	MultipoleExpansion ME(&rho,5000,12,12,-1,1.,0.0001,10000.,false,true,true);

	// 	for(auto qq: {"No","general"}){
	// 	for(int p=0;p<3;++p){
	// 	VecDoub X = {1.,1.,1.};
	// 	double centre  = rho.Phi(X);
	// 	double centre3 = ME.Phi(X);

	// 	int NMAX = 100;

	// 	#pragma omp parallel for schedule(dynamic)
	// 	for(int xn = 0; xn<NMAX; xn++){
	// 		double exact,multipole;
	// 		double x = (double)xn+.1;
	// 		VecDoub X2 = X;
	// 		X2[p]=x;
	// 		if(qq=="general"){
	// 			X[0]=x/2.;X[1]=x/2.;X[2]=x/2.;
	// 		}
	// 		exact = (rho.Phi(X2)-centre);
	// 		multipole = (ME.Phi(X2)-centre3);
	// 		EXPECT_NEAR(exact,multipole,5e-3*fabs(exact));
	// 	}
	// }}
	// }


	// int a = 0;
		////write data
		if(1){
			//:: AM
			AA_TF.record_AA_motion_data(xv0);
			PAMF[i]= AA_TF.AM;
			for(int j=0;j<sizeof(Actions_TFudge_FPot)/sizeof(double);j++){PAMF[i].actions[j] = Actions_TFudge_FPot[j];}
			for(int j=0;j<sizeof(AnglesFrequencies_TFudge_FPot)/sizeof(double);j++){PAMF[i].angles[j] = AnglesFrequencies_TFudge_FPot[j];}

			AA_TD.record_AA_motion_data(xv0);
			PAMD[i] = AA_TD.AM;
			for(int j=0;j<sizeof(Actions_TFudge_DPot)/sizeof(double);j++){PAMD[i].actions[j] = Actions_TFudge_DPot[j];}
			for(int j=0;j<sizeof(AnglesFrequencies_TFudge_DPot)/sizeof(double);j++){PAMD[i].angles[j] = AnglesFrequencies_TFudge_DPot[j];}

			//:: calculations
			printf("ID_%d ", ID);

			//basic info:
			//counts 6+1+3+2
			for(int ii=0;ii<6;ii++) Write_aa[i].particle_xv0[ii] = xv0[ii]; //init phase position
			Write_aa[i].particle_ID = ID;
			Write_aa[i].particle_otherInfo[0] = pDPot->Datas->SS[0]->P[ID].Type;
			Write_aa[i].particle_otherInfo[1] = pDPot->Datas->SS[0]->P[ID].Mass;
			Write_aa[i].particle_otherInfo[2] = pDPot->Datas->SS[0]->P[ID].dAdt; //or other
			Write_aa[i].particle_otherInfo[3] = pFPot->Phi(xv0);
			Write_aa[i].particle_otherInfo[4] = pDPot->Phi(xv0);
			//formula potential:
			//counts 3+6
			for(int ii=0;ii<3;ii++) Write_aa[i].Actions_Spherical_FPot[ii] = Actions_Spherical_FPot[ii];
			for(int ii=0;ii<6;ii++) Write_aa[i].AnglesFrequencies_Spherical_FPot[ii] = AnglesFrequencies_Spherical_FPot[ii];
			//counts 3+6
			for(int ii=0;ii<3;ii++) Write_aa[i].Actions_AFudge_FPot[ii] = Actions_AFudge_FPot[ii];
			for(int ii=0;ii<6;ii++) Write_aa[i].AnglesFrequencies_AFudge_FPot[ii] = AnglesFrequencies_AFudge_FPot[ii];
			//counts 4+11
			for(int ii=0;ii<4;ii++) Write_aa[i].Actions_TFudge_FPot[ii] = Actions_TFudge_FPot[ii];
			for(int ii=0;ii<11;ii++) Write_aa[i].AnglesFrequencies_TFudge_FPot[ii] = AnglesFrequencies_TFudge_FPot[ii];
			//data potential:
			//counts 3+6
			for(int ii=0;ii<3;ii++) Write_aa[i].Actions_Spherical_DPot[ii] = Actions_Spherical_DPot[ii];
			for(int ii=0;ii<6;ii++) Write_aa[i].AnglesFrequencies_Spherical_DPot[ii] = AnglesFrequencies_Spherical_DPot[ii];
			//counts 3+6
			for(int ii=0;ii<3;ii++) Write_aa[i].Actions_AFudge_DPot[ii] = Actions_AFudge_DPot[ii];
			for(int ii=0;ii<6;ii++) Write_aa[i].AnglesFrequencies_AFudge_DPot[ii] = AnglesFrequencies_AFudge_DPot[ii];
			//counts 4+11
			for(int ii=0;ii<4;ii++) Write_aa[i].Actions_TFudge_DPot[ii] = Actions_TFudge_DPot[ii];
			for(int ii=0;ii<11;ii++) Write_aa[i].AnglesFrequencies_TFudge_DPot[ii] = AnglesFrequencies_TFudge_DPot[ii];
			//totally counts 78 (the most datatype is double).
			
			char wt_fname_dbg[200];
			sprintf(wt_fname_dbg, "%s0prog/gadget/Gadget-2.0.7/%saa/action_abnormal_TF.debug.txt", (char*)getworkpath().data(), modelPath);
			FILE* fp1 = fopen(wt_fname_dbg, "w");
			if(fp1==NULL){
				printf("Cannot open file. Donot write.\n");
				return 0;
			}
			int swit = 0; //lambda
			fprintf(fp1, "##info: xv[6]     id swit     ABC[3]     tau[3]     potxyz ints[9]_lam[3]     limits[6]_lam[2]     actions[4]\n");
			for(int i=0;i<N_ptcs;i++){
				fprintf(fp1, "%e %e %e %e %e %e     %d %d     %e %e %e     %e %e %e %e %e %e     %e %e %e %e     %e %e     %e %e %e %d\n", 
					PAMF[i].xv[0], PAMF[i].xv[1], PAMF[i].xv[2], PAMF[i].xv[3], PAMF[i].xv[4], PAMF[i].xv[5], 
					i, swit, 
					PAMF[i].ABC[0], PAMF[i].ABC[1], PAMF[i].ABC[2], 
					PAMF[i].tau[0], PAMF[i].tau[1], PAMF[i].tau[2], PAMF[i].tau[3], PAMF[i].tau[4], PAMF[i].tau[5], 
					PAMF[i].phixyz, PAMF[i].Ints[0*3+swit], PAMF[i].Ints[1*3+swit], PAMF[i].Ints[2*3+swit], 
					PAMF[i].limits[0], PAMF[i].limits[1], 
					PAMF[i].actions[0], PAMF[i].actions[1], PAMF[i].actions[2], PAMF[i].actions[3]
				);
			}
			printf("Write file %s ... done.\n", wt_fname_dbg);
			fclose(fp1);

			sprintf(wt_fname_dbg, "%s0prog/gadget/Gadget-2.0.7/%saa/action_abnormal_TD.debug.txt", 
				(char*)getworkpath().data(), modelPath);
			// fp1 = fopen(wt_fname_dbg, "w");
			FILE* fp2 = fopen(wt_fname_dbg, "w");
			if(fp2==NULL){
				printf("Cannot open file. Donot write.\n");
				return 0;
			}
			swit = 0; //lambda
			fprintf(fp2, "##info: xv[6]     id swit     ABC[3]     tau[3]     "
				"potxyz ints[9]_lam[3]     limits[6]_lam[2]     actions[4]\n");
			for(int i=0;i<N_ptcs;i++){
				fprintf(fp2, "%e %e %e %e %e %e     %d %d     %e %e %e     %e %e %e %e %e %e     %e %e %e %e     %e %e     %e %e %e %d\n", 
					PAMD[i].xv[0], PAMD[i].xv[1], PAMD[i].xv[2], PAMD[i].xv[3], PAMD[i].xv[4], PAMD[i].xv[5], 
					i, swit, 
					PAMD[i].ABC[0], PAMD[i].ABC[1], PAMD[i].ABC[2], 
					PAMD[i].tau[0], PAMD[i].tau[1], PAMD[i].tau[2], PAMD[i].tau[3], PAMD[i].tau[4], PAMD[i].tau[5], 
					PAMD[i].phixyz, PAMD[i].Ints[0*3+swit], PAMD[i].Ints[1*3+swit], PAMD[i].Ints[2*3+swit], 
					PAMD[i].limits[0], PAMD[i].limits[1], 
					PAMD[i].actions[0], PAMD[i].actions[1], PAMD[i].actions[2], PAMD[i].actions[3]
				);
			}
			printf("Write file %s ... done.\n", wt_fname_dbg);
			fclose(fp2);
		}

		free(PAMF);
		free(PAMD);
		free(Write_aa);

		if(1){
			// char wt_fname_dbg[200];
			// FILE* fp = fopen(wt_fname_dbg, "w");
			// sprintf(wt_fname_dbg, "%s0prog/gadget/Gadget-2.0.7/%saa/action_range_TF_tau%d.debug.txt", 
			//		(char*)getworkpath().data(), modelPath, 0);
			// fp = fopen(wt_fname_dbg, "w");
			// if(fp==NULL){
			// 	printf("Cannot open file. Donot write.\n");
			// 	exit(0);
			// }
			// fprintf(fp, "##aaaa\n");
			// fclose(fp);
			// printf("Write file %s ... done.\n", wt_fname_dbg);
		}
*/