//fundamental
#include <iostream>
#include <fstream>
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
// #include "gtest/gtest.h"
// #ifdef TORUS
// #include "falPot.h"
// #include "it_torus.h"
// #include "PJM_cline.h"
// #endif



//// main()
int main(int argc, char* argv[]){

DEBUG_PRINT_I(11);
	////1. input
	//:: in linux shell: 
	// # cd path/of/aa/; make clean && cd .. && make LAPACK=1 TORUS=1 && cd aa;  
	// # mpirun -np 4 mains/./data.exe  
	// # <input arguments below>

	//:: or by reading input file
	bool is_read_input_file = false;
	// bool is_read_input_file = true;
	if(is_read_input_file){
		string file_input_argv = "../step2_run.bat";
		// string file_input_argv = "../step2_run_debug.bat";
		// read_pure_number(file_input_argv, 28);
	}

	//:: snapshots range
	double t_init 				= atof(argv[1]); 	//0.
    double t_final 				= atof(argv[2]); 	//9.
	double dt_load				= atof(argv[3]); 	//0.01
	double dt_step 				= atof(argv[4]);	//0.001
	double t_aligment			= atof(argv[5]);	//0.

	//:: what to run
	int is_writeSnapshot 		= atoi(argv[6]);	//write snapshots //0
	int is_preprocessed			= atoi(argv[7]); 	//0: not <1>; 1: write and centerize //0
	int onlyRun_Actionmethod	= atoi(argv[8]);	//7 = 111(2)

	int AlgorithmPot 			= atoi(argv[9]); 	//0: direct summation; 5: direct summation 1
	int WhatPotential 			= atoi(argv[10]); 	//0: formula potential; 1: data potential //2
	int WhatSymmetry			= atoi(argv[11]); 	//0: spherical; 1: axisymmetric; 2: triaxial //2
	int WhatActionmethod		= atoi(argv[12]); 	//0: SF; 1: PPOD all; 2: {Jl} by <1> and {Jm,Jn} by <2> //3

	//:: time and particle_ID range to run
	double t_start_run			= atof(argv[13]); 	//5
	double t_end_run			= atof(argv[14]); 	//5.01
	double dt_run 				= atof(argv[15]); 	//1.

	int ID_start 				= atoi(argv[16]);	//1
	int N_ptcs 					= atoi(argv[17]); 	//the max index of particles, value -1 is as NumPart or N_allPtcs
	
	int is_run_samples 			= atoi(argv[18]); 	//samples for action state density
	int N_action_samples		= atoi(argv[19]);

	int is_DEBUG				= atoi(argv[20]);

	//:: other
	double input_other_1	= atof(argv[21]);
	double input_other_2	= atof(argv[22]);

	double input_other_3	= atof(argv[23]);
	double input_other_4	= atof(argv[24]);
	double input_other_5	= atof(argv[25]);
	double input_other_6	= atof(argv[26]);
	double input_other_7	= atof(argv[27]);
	double input_other_8	= atof(argv[28]);
DEBUG_PRINT_I(12);

	// DEBUG_PRINT_V0d(1, t_init, "t_init");
	// DEBUG_PRINT_V0d(1, AlgorithmPot, "AlgorithmPot");
	DEBUG_PRINT_V0d(1, t_start_run, "t_start_run");
	DEBUG_PRINT_V0d(10, t_end_run, "t_end_run");



	////2. MPI settings
	int my_rank, comm_sz;
    MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
DEBUG_PRINT_I(1);



	////3. snapshot settings
	string path_IC = "../../../../"
		"GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/"
		"IC_param.txt";
	Stage STAGE(path_IC);
	Stage* pSTAGE = &STAGE;
DEBUG_PRINT_I(13);

	////calculate
	string path_gm_1 = pSTAGE->path_gm;
	printf("path_gm: %s\n", path_gm_1.data());
	pSTAGE->load_multi_snapshots(t_init, t_final, dt_load, dt_step, 0, 0);
DEBUG_PRINT_I(14);
	double Alpha = TACT_semiaxis_Alpha, Beta = TACT_semiaxis_Beta;
	MPI_Barrier(MPI_COMM_WORLD);
DEBUG_PRINT_I(2);

	////potentials presetting
	Potential_JS DPot(&STAGE);
	Potential_JS* pDPot = &DPot;
	//pDPot->Phi(..., Potential_other*=nullptr) //??
	pDPot->set_algorithm(AlgorithmPot);
	DEBUG_PRINT_V0d(1, AlgorithmPot, "the main AlgorithmPot");

	// int N_calculate_load = (int)((t_end_run-t_start_run)/dt_run);
	vector<Potential_JS> vFPot; //??
DEBUG_PRINT_I(3);

	for(double t0=t_start_run;t0<t_end_run;t0+=dt_run){ //time
		DEBUG_PRINT_V0d(1, t0, "t0 for");
		DEBUG_PRINT_V0d(1, t_init, "t_init");
		int s = t_to_loadsnapshot(t0, dt_load, t_init, 0.); //load
		DEBUG_PRINT_V0d(1, s, "s");
		int snapshot = pSTAGE->SS[s]->snap; //snapshot
		pDPot->set_time(t0); //Note: One should set time before calculate potential to use the right snapshot of time
		DEBUG_PRINT_V0d(10, snapshot, "snapshot");
		if(N_ptcs==-1){
			N_ptcs = pSTAGE->SS[s]->NumPart;
		}
		if(N_action_samples==-1){
			N_action_samples = N_ptcs;
		}

		////(1) data potential
		//!!?? debug preprocess: false when multiload, true now
		// #ifdef DEBUG_GJY
		pSTAGE->is_preprocess_rotation = true;
		DEBUG_PRINT_V0d(10, pSTAGE->is_preprocess_rotation, "pSTAGE->is_preprocess_rotation");
		pSTAGE->SS[s]->preprocess(false); //to get triaxialize matix and rotate frame xv of this s~snapshot
		// pSTAGE->SS[s]->preprocess(true); //debug
		// #endif
		
		// DEBUG_PRINT_V0d(10, s, "before xtree");
		// std::array<std::array<double, Dim>, N_total> xdata; //dynamic tree?? //x-tree and J-tree
		// for(int i=0;i<N_total;i++){
		// 	int iP = i+1;
		// 	xdata[i][0] = pSTAGE->SS[s]->P[iP].Pos[0];
		// 	xdata[i][1] = pSTAGE->SS[s]->P[iP].Pos[1];
		// 	xdata[i][2] = pSTAGE->SS[s]->P[iP].Pos[2];
		// }
		// KDtree<double, N_total, Dimension> kdt(&xdata);
		// pSTAGE->SS[s]->loadtree(&kdt);
		// DEBUG_PRINT_V0d(10, s, "after xtree");
DEBUG_PRINT_I(4);

		if(0){
		// if(is_DEBUG){
	// 		VecDoub ar = {1., 0.99, 0.98};
	// 		double rs0 = 19.6;
	// DEBUG_PRINT_I(41);
	// 		pDPot->set_partical_ID(-1);
	// 		pDPot->set_time(t0);

	// 		TRSP3Potential TRSP3Pot(pSTAGE);
	// DEBUG_PRINT_I(42);
	// 		Potential_JS* pFPot_1 = &TRSP3Pot;
	// DEBUG_PRINT_I(43);
	// 		pFPot_1->RebuildPotentialSP3_loadsnapsnot(s, ar, rs0); //virual to call derived class function ??
	// DEBUG_PRINT_I(44);

	// 		VecDoub xsp3 = {0.01, 2., 30.};
	// 		auto pot2 = pDPot->pSTAGE->SS[s]->potential_SCF(xsp3);
	// DEBUG_PRINT_V0d(10, pot2, "pot2");
	// DEBUG_PRINT_I(is_DEBUG);
	// 		auto pot1 = pDPot->Phi(xsp3);
	// DEBUG_PRINT_V0d(10, pot1, "pot1");
	// 		auto pot = pFPot_1->Phi(xsp3);
	// DEBUG_PRINT_V0d(10, pot, "pot");
	// 		auto forces = pFPot_1->Forces(xsp3);
	// DEBUG_PRINT_I(442);

			Plummer FPot_1(conv::G*Mass_vir, scale_length_comp[0], 0.9, 0.8);
			Potential_JS* pFPot_1 = &FPot_1;

			vector<double> XX = {input_other_3, input_other_4, input_other_5, 
				input_other_6, input_other_7, input_other_8};
			// XX = pSTAGE->SS[s]->TC.new_coordinate(XX, true, true);
			XX = XX;
			vector<double> AA00, AA0, AA1, AA2;
			clock_t start,end;

			Actions_TriaxialStackel_Fudge AA_TF1(pFPot_1, input_other_1, input_other_2);
			AA00 = AA_TF1.actions(XX);
			Actions_TriaxialStackel_Fudge AA_TF2(pDPot, input_other_1, input_other_2);
			AA0 = AA_TF2.actions(XX);

			start = clock();
			string Delta_lmn_name1 = pSTAGE->path_gm+"intermediate/snapshot_"+to_string(snapshot)+"_lmn_foci_Pot.txt";
			cout<<"Delta_lmn_name1 debug: "<<Delta_lmn_name1<<"\n";
			// lmn_orb AA_lmn_1(pFPot_1, 0.05, 400., 32, true, false, Delta_lmn_name1);
			// lmn_orb AA_lmn_1(pDPot, 2., 300., 32, true, false, Delta_lmn_name1);
			lmn_orb AA_lmn_1(pDPot, 0.5, 300., 32, true, false, Delta_lmn_name1);
			// lmn_orb AA_lmn_1(pDPot, 0.05, 400., 32, true, true, Delta_lmn_name1);
			AA_lmn_1.fillDeltagrids(Delta_lmn_name1);
			end = clock();
			printf("time load_lmn = %f\n", (double)(end-start)/1e6);
			AA1 = AA_lmn_1.actions(XX);

			print_vec(AA00);
			print_vec(AA0);
			print_vec(AA1);
		}
DEBUG_PRINT_I(4441);

		// exit(0);
		// return 1;



		////(2) formula potential
		Galaxy_components GA; //GA();
		GA.read_fit_DF_x_mass(path_gm_1, snapshot); //?? //note: now the FPot has only halo component
		//the expected galaxies values are not used
		//TORUS is not used
DEBUG_PRINT_I(5);



		////main examples for potential
		// for(int icmp=0;icmp<N_comp;icmp++){ //multi components??
		// 	//each potential summation of each componemts
		// }
		int icmp = 0;
		DEBUG_PRINT_V0d(1, GA.scaled_density_fit_comp[icmp], "GA.scaled_density_fit_comp");
		DEBUG_PRINT_V0d(1, GA.scaled_length_fit_comp[icmp], "GA.scaled_length");
		DEBUG_PRINT_V0d(1, GA.axis_ratio_y_fit_comp[icmp], "GA.axis_ratio_y_fit_comp");
		DEBUG_PRINT_V0d(1, GA.axis_ratio_z_fit_comp[icmp], "about z");
		DEBUG_PRINT_V0d(10, GA.powerA_fit_comp[icmp], "GA.powerA_fit_comp[icmp]");
		DEBUG_PRINT_V0d(10, GA.powerC_fit_comp[icmp], "GA.powerC_fit_comp[icmp]");

		// Density_MDPLEP rho(); //??
		// TestDensity_NFW rho(GA.M_scale_comp[icmp], GA.scaled_length_fit_comp[icmp], 
		//   {GA.axis_ratio_x_fit_comp[icmp], GA.axis_ratio_y_fit_comp[icmp], GA.axis_ratio_z_fit_comp[icmp]}
		// ); //wrong energy
		Density_DoublePowerLaw rho(GA.scaled_density_fit_comp[icmp], GA.scaled_length_fit_comp[icmp], 
			{GA.axis_ratio_x_fit_comp[icmp], GA.axis_ratio_y_fit_comp[icmp], 
			GA.axis_ratio_z_fit_comp[icmp]}, GA.powerA_fit_comp[icmp], GA.powerB_fit_comp[icmp]
		);
		// Density_Einasto rho(GA.scaled_density_fit_comp[icmp], GA.scaled_length_fit_comp[icmp], 
		// 	{GA.axis_ratio_x_fit_comp[icmp], GA.axis_ratio_y_fit_comp[icmp], 
		// 	GA.axis_ratio_z_fit_comp[icmp]}, GA.powerA_fit_comp[icmp]
		// );
		string input_ME_name = path_gm_1+"input_ME.txt";
		VecDoub input_ME = read_pure_number(input_ME_name, 10);
DEBUG_PRINT_I(51);

		/*
		////DEBUG //begin DEBUG
		if(0){
			double GM = conv::G*GA.M_scale_comp[icmp];
			double rhos = GA.scaled_density_fit_comp[icmp];
			double rs = GA.scaled_length_fit_comp[icmp];
			double qx = GA.axis_ratio_x_fit_comp[icmp], 
				qy = GA.axis_ratio_y_fit_comp[icmp], 
				qz = GA.axis_ratio_z_fit_comp[icmp];

			NFW Pot_NFW_Exc(GM, rs, qy, qz);
			TestDensity_NFW rho_NFW(GM, rs, {qx, qy, qz});
			TriaxialPotential Pot_NFW_TS(&rho_NFW, (double)input_ME[6]);
			MultipoleExpansion Pot_NFW_ME(&rho_NFW, (int)input_ME[0], (int)input_ME[1], (int)input_ME[2], 
				(int)input_ME[3], (double)input_ME[4], (double)input_ME[5], (double)input_ME[6], 
				(bool)input_ME[7], (bool)input_ME[8], (bool)input_ME[9]
			);
			VecDoub a(9, 0.);
			a[0] = rho_NFW.density_m2(0.1);
			a[1] = rho_NFW.density_m2(1.);
			a[2] = rho_NFW.density_m2(10.);

			Density_DoublePowerLaw rho_DPL(GA.scaled_density_fit_comp[icmp], GA.scaled_length_fit_comp[icmp], 
				{GA.axis_ratio_x_fit_comp[icmp], GA.axis_ratio_y_fit_comp[icmp], GA.axis_ratio_z_fit_comp[icmp]}, 
				1., 2. //not fitted powers instead of NFW powers
				// GA.powerA_fit_comp[icmp], GA.powerB_fit_comp[icmp]
			);
			MultipoleExpansion Pot_DPL_ME(&rho_DPL, (int)input_ME[0], (int)input_ME[1], (int)input_ME[2], 
				(int)input_ME[3], (double)input_ME[4], (double)input_ME[5], (double)input_ME[6], 
				(bool)input_ME[7], (bool)input_ME[8], (bool)input_ME[9]
			);
			a[3] = rho_DPL.density_m2(0.1);
			a[4] = rho_DPL.density_m2(1.);
			a[5] = rho_DPL.density_m2(10.);

			Density_Einasto rho_SER(GA.scaled_density_fit_comp[icmp], GA.scaled_length_fit_comp[icmp], 
				{GA.axis_ratio_x_fit_comp[icmp], GA.axis_ratio_y_fit_comp[icmp], GA.axis_ratio_z_fit_comp[icmp]}, 
				GA.powerA_fit_comp[icmp]
			);
			MultipoleExpansion Pot_Sersic_ME(&rho_SER, (int)input_ME[0], (int)input_ME[1], (int)input_ME[2], 
				(int)input_ME[3], (double)input_ME[4], (double)input_ME[5], (double)input_ME[6], 
				(bool)input_ME[7], (bool)input_ME[8], (bool)input_ME[9]
			);
			a[6] = rho_SER.density_m2(0.1);
			a[7] = rho_SER.density_m2(1.);
			a[8] = rho_SER.density_m2(10.);
			DEBUG_PRINT_V1d(1, a, "rho at rq2 = 1. or 10.");
			//result: DEBUG_PRINT: 0.0264736 0.00669819 0.0264736 0.00119085 0.614154 0.00761373 (rho at rq2 = 1. or 10.)

			//for particle or for line
			int x_count = 100;
			double x_min = 1e-3, x_max = 1e3;
			VecDoub xlist = GJY::gen_logspace1D(x_min, x_max, x_count);
			double P;
			VecDoub F(3, 0.);
			VecDoub X(3, 0.);
			vector<struct data_debug> vdd;
			vdd.resize(x_count);
			for(int i=0;i<x_count;i++){
				X = {xlist[i]/1., xlist[i]/1., xlist[i]/1., 0., 0., 0.};
				DEBUG_PRINT_V1d(1, X, "X");
				vdd[i].xv = X;

				//potential 0, Pot_NFW_Exc
				P = Pot_NFW_Exc.Phi(X);
				F = Pot_NFW_Exc.Forces(X);
				vdd[i].value_double.push_back(P);
				vdd[i].value_double.push_back(F[0]);
				vdd[i].value_double.push_back(F[1]);
				vdd[i].value_double.push_back(F[2]);

				// //potential 1, Pot_NFW_TS
				// P = Pot_NFW_TS.Phi(X);
				// F = Pot_NFW_TS.Forces(X);
				// vdd[i].value_double.push_back(P);
				// vdd[i].value_double.push_back(F[0]);
				// vdd[i].value_double.push_back(F[1]);
				// vdd[i].value_double.push_back(F[2]);

				//potential 1, Pot_NFW_ME
				P = Pot_NFW_ME.Phi(X);
				F = Pot_NFW_ME.Forces(X);
				vdd[i].value_double.push_back(P);
				vdd[i].value_double.push_back(F[0]);
				vdd[i].value_double.push_back(F[1]);
				vdd[i].value_double.push_back(F[2]);

				//potential 2, Pot_DPL_ME
				P = Pot_DPL_ME.Phi(X);
				F = Pot_DPL_ME.Forces(X);
				vdd[i].value_double.push_back(P);
				vdd[i].value_double.push_back(F[0]);
				vdd[i].value_double.push_back(F[1]);
				vdd[i].value_double.push_back(F[2]);

				//potential 3, Pot_Sersic_ME
				P = Pot_Sersic_ME.Phi(X);
				F = Pot_Sersic_ME.Forces(X);
				vdd[i].value_double.push_back(P);
				vdd[i].value_double.push_back(F[0]);
				vdd[i].value_double.push_back(F[1]);
				vdd[i].value_double.push_back(F[2]);

				//potential 4, Pot_Nbody_SCF
				DPot.set_algorithm(4); //SCF
				// F = {0., 0., 0.}; //resize default 0??
				P = DPot.Phi(X);
				F = DPot.Forces(X);
				vdd[i].value_double.push_back(P);
				vdd[i].value_double.push_back(F[0]);
				vdd[i].value_double.push_back(F[1]);
				vdd[i].value_double.push_back(F[2]);
				DPot.set_algorithm(AlgorithmPot); //DS

				//potential 5, Pot_Nbody_DS
				P = DPot.Phi(X);
				F = DPot.Forces(X);
				vdd[i].value_double.push_back(P);
				vdd[i].value_double.push_back(F[0]);
				vdd[i].value_double.push_back(F[1]);
				vdd[i].value_double.push_back(F[2]);
			}
			string filename_debug = path_gm_1+"debug/data_debug.txt";
			write_data_debug(vdd, filename_debug);
			exit(0);
		} //end DEBUG
		*/

		//go on
		//[] expect: 5000, 12, 12, -1, 1., 0.001, 1000., false, true, true
		MultipoleExpansion FPot1(&rho, (int)input_ME[0], (int)input_ME[1], (int)input_ME[2], 
			(int)input_ME[3], (double)input_ME[4], (double)input_ME[5], (double)input_ME[6], 
			(bool)input_ME[7], (bool)input_ME[8], (bool)input_ME[9]
		);

		VecDoub vecNFW = pDPot->pSTAGE->SS[s]->similar_NFW_param();
		DEBUG_PRINT_V0d(1, pDPot->Phi({1.,1.,1., 1.,1.,1.}), "pDPot->Phi({1.,1.,1., 1.,1.,1.})");
		DEBUG_PRINT_V1d(1, vecNFW, "vecSimilar");
		// NFW FPot2(conv::G*vecNFW[0], vecNFW[1]/3., vecNFW[2], vecNFW[3]);
		Isochrone FPot2(conv::G*vecNFW[0], vecNFW[1], vecNFW[2], vecNFW[3]);
		


		////pointers
		Potential_JS* pFPot = nullptr;
		// void* pFPot = nullptr; //[learn code] cannot void* to <other>* in C++, can do in C
		// switch(modelId){ //??
		// 	case 100:	{
		// 		pFPot = &FPot1; //model1
		// 	    break;
		// 	}
		//     case 101: {
		// 		pFPot = &FPot2; //model2
		//         break;
		//     }
		// 	// case ...
		//     default: { //other
		//         printf("No such model! We use sperical NFW formula potential model.\n");
		// 		pFPot = &FPot1; //model1
		//         break;
		//     }
		// }
		vFPot.push_back(FPot1); //when need FormulaPotential_time //??
		// pFPot = &FPot1;
		pFPot = &FPot2;
		printf("Potentials has been set. Then estimate actions...\n\n\n\n\n\n");



		////4. TASK1: actions
		////[TASK1 start
		////(1)

		// [foci]
		double Alpha = TACT_semiaxis_Alpha, Beta = TACT_semiaxis_Beta, Gamma = TACT_semiaxis_Gamma;
		DEBUG_PRINT_V1d(1, (VecDoub){Alpha, Beta, Gamma}, "default semiaxises");
		double* AlphaBeta = new double[2]; AlphaBeta[0] = input_other_1, AlphaBeta[1] = input_other_2;
		string Delta_lmn_name = pSTAGE->path_gm+"intermediate/snapshot_"+to_string(snapshot)+"_lmn_foci_Pot.txt";
		// string Delta_lmn_name = pSTAGE->path_gm+"intermediate/lmn_foci_Pot_fixed_example.txt";
		
		/*
		if(0){
		// if(is_DEBUG){
			vector<double> XX = {input_other_3, input_other_4, input_other_5, 
				input_other_6, input_other_7, input_other_8};
			XX = pSTAGE->SS[s]->TC.new_coordinate(XX, true, true);
			double p1 = pFPot->Phi(XX);
			DEBUG_PRINT_V0d(10, p1, "FP");
			double E1 = pFPot->H(XX);
			DEBUG_PRINT_V0d(10, E1, "FP");
			double p2 = pDPot->Phi(XX);
			DEBUG_PRINT_V0d(1, p2, "DP");
			double E2 = pDPot->H(XX);
			DEBUG_PRINT_V0d(1, E2, "DP");
			vector<double> AA00, AA0, AA1, AA2;
			clock_t start,end;
			DEBUG_PRINT_V0d(1, AlphaBeta[0], "AlphaBeta");

			Actions_TriaxialStackel_Fudge AA_TF1(pFPot, AlphaBeta[0], AlphaBeta[1]);
			AA00 = AA_TF1.actions(XX);
			Actions_TriaxialStackel_Fudge AA_TF2(pDPot, AlphaBeta[0], AlphaBeta[1]);
			AA0 = AA_TF2.actions(XX);

			start = clock();
			string Delta_lmn_name1 = pSTAGE->path_gm+"intermediate/snapshot_"+to_string(snapshot)+"_lmn_foci_Pot.txt";
			lmn_orb AA_lmn_1(pFPot, 0.05, 160., 32, true, false, Delta_lmn_name1);
			// AA_lmn_1.fillDeltagrids(Delta_lmn_name1);
			lmn_orb AA_lmn(pDPot, 0.05, 160., 32, true, false, Delta_lmn_name1); //??
			AA_lmn.readDeltagrids(Delta_lmn_name1);
			end = clock();
			printf("time load_lmn = %f\n", (double)(end-start)/1e6);
			AA1 = AA_lmn.actions(XX, AlphaBeta);
			start = clock();
			AA2 = AA_lmn.actions(XX); //use the calculated
			end = clock();
			printf("time AA_lmn = %f\n", (double)(end-start)/1e6);

			print_vec(AA00);
			print_vec(AA0);
			print_vec(AA1);
			print_vec(AA2);
			exit(0);
		}
		*/

		Write_aa = (struct write_angleaction *) malloc(sizeof(struct write_angleaction)*(N_ptcs));
		VecDoub xv0(2*Dim);

		//:: class of actions/frequencies/angles methods
		// Actions_Spherical AA_SS_FP(pFPot); //but not ...(Potential_JS*)
		Actions_Spherical_DataPotential AA_SS_FP(pFPot);
		Actions_AxisymmetricStackel_Fudge AA_AF_FP(pFPot,Alpha);
		Actions_TriaxialStackel_Fudge AA_TF_FP1(pFPot,Alpha,Beta);
		lmn_orb AA_TF_FP(pFPot, 0.5, 200., 32, true, false, Delta_lmn_name);
		// lmn_orb AA_TF_FP(pFPot, 0.05, 160., 32, true, false, Delta_lmn_name);
		AA_TF_FP.ABC_presentCalculation = {TACT_semiaxis_Alpha, TACT_semiaxis_Beta};
		int is_fillDeltagrids = 0; //disjunctor //input_other_3
		// int is_fillDeltagrids = 1; //disjunctor //input_other_3
DEBUG_PRINT_I(9901);
		if(is_fillDeltagrids){
			AA_TF_FP.fillDeltagrids(Delta_lmn_name); //?? pFPot, lmn_orb, fill, read by DP
		}else{
			AA_TF_FP.readDeltagrids(Delta_lmn_name);
		}
DEBUG_PRINT_I(9902);
		//() no AA_TEPPOD_FP in AA_TF_FP
		Actions_Genfunc AA_GF_FP(pFPot,"triaxial");
DEBUG_PRINT_I(9903);

		Actions_Spherical_DataPotential AA_SS_DP(pDPot);
		Actions_AxisymmetricStackel_Fudge AA_AF_DP(pDPot,Alpha);
		Actions_TriaxialStackel_Fudge AA_TF_DP1(pDPot, Alpha, Beta);
		lmn_orb AA_TF_DP(pDPot, 0.5, 200., 32, true, false, Delta_lmn_name); //??: can not integrate orbit in gsl when data potential
		AA_TF_DP.ABC_presentCalculation = {TACT_semiaxis_Alpha, TACT_semiaxis_Beta};
DEBUG_PRINT_I(9908);
		if(1){ //Cannot use foci method of TACT because orbit wrong shape in data potential
			// AA_TF_DP.fillDeltagrids(Delta_lmn_name); //TACT
			AA_TF_DP.readDeltagrids(Delta_lmn_name); //y_cut
		}
		// AA_TEPPOD_DP in AA_TF_DP //in the upper class
		// Actions_Genfunc AA_GF_DP(pDPot, "triaxial");
		// VecDoub de1 = AA_TF_DP.actions({-2.441926e+00, -2.867677e+02, -9.096738e+01, 3.425838e+01, -3.101219e+02, -9.737949e+01});
		// DEBUG_PRINT_V1d(0, de1, "de1");

		// bool is_record_tau_ptau = 0; //all
		bool is_record_tau_ptau = 1; //debug

		//: modify code
		double* inputAlphaBeta = nullptr;
		std::vector<std::vector<double>> vecAlphaBeta;
		std::string previous_actions_file = path_gm_1+"aa/snapshot_"+to_string(snapshot)+".action.method_all.txt";
		bool is_exist_previous_actions_file = false; //note that it has default manner
		// assign value of vecAlphaBeta from previous_actions_file and close file;
DEBUG_PRINT_V0d(10, 0, "after_Deltagrids");



		//:: VecDoub of actions/frequencies/angles by formula potential or data potential
		VecDoub Value_Actions_SS_FP, Value_AnglesFrequencies_SS_FP;
		VecDoub Value_Actions_AF_FP, Value_AnglesFrequencies_AF_FP;
		VecDoub Value_Actions_TF_FP, Value_AnglesFrequencies_TF_FP;
		VecDoub Value_Actions_TEPPOD_FP, Value_AnglesFrequencies_TEPPOD_FP; //not provided, need orbit in formula potential
		VecDoub Value_Actions_GF_FP, Value_AnglesFrequencies_GF_FP;

		VecDoub Value_Actions_SS_DP, Value_AnglesFrequencies_SS_DP;
		VecDoub Value_Actions_AF_DP, Value_AnglesFrequencies_AF_DP;
		VecDoub Value_Actions_TF_DP, Value_AnglesFrequencies_TF_DP;
		VecDoub Value_Actions_TEPPOD_DP, Value_AnglesFrequencies_TEPPOD_DP;
		VecDoub Value_Actions_GF_DP, Value_AnglesFrequencies_GF_DP; //not provided, need data potential in O2GF



		////(2) MPI: calculation distribution
		int remainder = N_ptcs%(comm_sz);
		// int interval = N_ptcs/comm_sz+(int)(N_ptcs*0.001/comm_sz);
		int interval = N_ptcs/comm_sz;
		int local_from, local_count, total_count = 0;

		////[main start
		if(my_rank==0){
			
			int my_from = interval*(comm_sz-1);
DEBUG_PRINT_V0d(1, my_from, "my_from rank0");
			int my_count = N_ptcs - my_from;
			int my_endout = my_from + my_count;
			if(my_count>interval*2){
				std::cerr<<"Bad task distribution because of too many ranks, which lead "
					"the count of tasks in rank 0 is much more than the count of tasks in rank other.\n";
			}
			printf("MPI tasks distribution: total number of tasks to do is %d, common size is %d, task interval is %d, "
				"the remainder is %d, rest = %d.\n", 
				N_ptcs, comm_sz, interval, remainder, my_count);
			printf("my_rank is %d: zero my_count = %d, my_from = %d, my_endout = %d.\n", 0, my_count, my_from, my_endout);
			printf("Now actions calculation ...\n\n");
DEBUG_PRINT_V0d(1, pFPot->Phi({1.,1.,1., 1.,1.,1.}), "pFPot->Phi({1.,1.,1., 1.,1.,1.})");
DEBUG_PRINT_I(52);

			////(1)particles
			for(int i=my_from;i<my_endout;i++)
			{
				//:: calculate
				int ID = i+ID_start;
				pDPot->set_partical_ID(ID);
				// pDPot->set_time(t0);

				xv0 = {pDPot->pSTAGE->SS[s]->P[ID].Pos[0], pDPot->pSTAGE->SS[s]->P[ID].Pos[1], pDPot->pSTAGE->SS[s]->P[ID].Pos[2], 
					pDPot->pSTAGE->SS[s]->P[ID].Vel[0], pDPot->pSTAGE->SS[s]->P[ID].Vel[1], pDPot->pSTAGE->SS[s]->P[ID].Vel[2] };
				// for(int ix=0;ix<6;ix++){ //something uncalculatable when zero??
				// 	if(abs(xv0[ix])<Err) xv0[ix] = Err;
				// }
				// DEBUG_PRINT_V1d(1, xv0, "xv0");

				int B0 = onlyRun_Actionmethod % 2; //(bool)Is run Sperical and Fudge in FPot
				int B1 = (onlyRun_Actionmethod - B0)/2 % 2; //(bool)Is run TEPPOD(none) and O2GF in FPot
				int B2 = (onlyRun_Actionmethod - B0 - B1*2)/2/2 % 2; //(bool)Is run Sperical and Fudge in DPot
				int B3 = (onlyRun_Actionmethod - B0 - B1*2 - B2*2*2)/2/2/2 % 2; //(bool)Is run TEPPOD and O2GF(none) in DPot
				if((bool)B0){
					// Value_Actions_SS_FP = AA_SS_FP.actions(xv0);
					// Value_AnglesFrequencies_SS_FP = AA_SS_FP.angles_and_freqs(xv0);
					// Value_Actions_AF_FP = AA_AF_FP.actions(xv0);
					// Value_AnglesFrequencies_AF_FP = AA_AF_FP.angles(xv0);
					Value_Actions_SS_FP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_SS_FP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_AF_FP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_AF_FP.resize(Dim*2, NotCalculatedActionsValue);

					Value_Actions_TF_FP = AA_TF_FP.actions(xv0);
					Value_AnglesFrequencies_TF_FP = AA_TF_FP.angles(xv0);
				}else{
					Value_Actions_SS_FP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_SS_FP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_AF_FP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_AF_FP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_TF_FP.resize(Dim+1, NotCalculatedActionsValue);
					Value_AnglesFrequencies_TF_FP.resize(Dim*2+5, NotCalculatedActionsValue);
				}
				if((bool)B1){
					Value_Actions_TEPPOD_FP.resize(LENGTH_TEPPOD, NotCalculatedActionsValue);
					Value_AnglesFrequencies_TEPPOD_FP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_GF_FP = AA_GF_FP.actions(xv0);
					Value_AnglesFrequencies_GF_FP = AA_GF_FP.angles(xv0);
				}else{
					Value_Actions_TEPPOD_FP.resize(LENGTH_TEPPOD, 0.);
					Value_AnglesFrequencies_TEPPOD_FP.resize(6, 0.);
					Value_Actions_GF_FP.resize(3, 0.);
					Value_AnglesFrequencies_GF_FP.resize(6, 0.);
				}
				if((bool)B2){
					// Value_Actions_SS_DP = AA_SS_DP.actions(xv0);
					// Value_AnglesFrequencies_SS_DP = AA_SS_DP.angles_and_freqs(xv0);
					// Value_Actions_AF_DP = AA_AF_DP.actions(xv0);
					// Value_AnglesFrequencies_AF_DP = AA_AF_DP.angles(xv0);
					Value_Actions_SS_DP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_SS_DP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_AF_DP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_AF_DP.resize(Dim*2, NotCalculatedActionsValue);

					//: modify code
					// if is_exist_previous_actions_file is false //keep inputAlphaBeta nullptr for lmn_orb::actions(const VecDoub& x, void* params);
					// if is_exist_previous_actions_file is true  //inputAlphaBeta[0] = vecAlphaBeta[ID-1][0], inputAlphaBeta[1] = vecAlphaBeta[ID-1][1];
					Value_Actions_TF_DP = AA_TF_DP.actions(xv0, inputAlphaBeta);
					Alpha = AA_TF_DP.ABC_presentCalculation[0], //it has been changed in AA_TF_DP.actions()
					Beta = AA_TF_DP.ABC_presentCalculation[1], 
					Gamma = Gamma; //to record focus

					if(is_record_tau_ptau){
						Actions_TriaxialStackel_Fudge AA_TF_DP_tau_ptau(pDPot, Alpha, Beta);
						AA_TF_DP_tau_ptau.reset_AA_integrating_data();
						AA_TF_DP_tau_ptau.is_record_action_integrand = 1;
DEBUG_PRINT_I(6631);

						AA_TF_DP_tau_ptau.actions(xv0); //repeat as AA_TF_DP.actions() with same foci
						for(int swit_AD = 0;swit_AD<3;swit_AD++){ //tau = lambda, mu, nu
							int count_AD = AA_TF_DP_tau_ptau.PAD[swit_AD].size();
							DEBUG_PRINT_V0d(1, count_AD, "count_AD");
							vector<struct data_debug> vdd_AD;
							vdd_AD.resize(count_AD);
							for(int i_AD=0;i_AD<count_AD;i_AD++){ //wrong i make cannot Pot.Phi() when gether
								auto AD = AA_TF_DP_tau_ptau.PAD[swit_AD][i_AD];
								vdd_AD[i_AD].value_double.push_back(AD.tau); //index 6
								vdd_AD[i_AD].value_double.push_back(AD.phichi);
								vdd_AD[i_AD].value_double.push_back(AD.Ints[0]); //index 8
								vdd_AD[i_AD].value_double.push_back(AD.Ints[1]);
								vdd_AD[i_AD].value_double.push_back(AD.Ints[2]);
								vdd_AD[i_AD].value_double.push_back(AD.ptau); //index 11
								vdd_AD[i_AD].value_double.push_back(AD.ptau_return);
							}
							string filename_debug_AD = path_gm_1
								+"debug/tau_ptau_snapshot_"+to_string(snapshot)+"_particle_"+to_string(ID)+"_swit_"+to_string(swit_AD)+".txt";
							write_data_debug(vdd_AD, filename_debug_AD);
						}
						
						AA_TF_DP_tau_ptau.reset_AA_integrating_data();
DEBUG_PRINT_I(6632);
					}
					Value_AnglesFrequencies_TF_DP = AA_TF_DP.angles(xv0);
DEBUG_PRINT_I(63);
				}else{
					Value_Actions_SS_DP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_SS_DP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_AF_DP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_AF_DP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_TF_DP.resize(Dim+1, NotCalculatedActionsValue);
					Value_AnglesFrequencies_TF_DP.resize(Dim*2+5, NotCalculatedActionsValue);
				}
				if((bool)B3){
// DEBUG_PRINT_I(64);
// DEBUG_PRINT_V0d(1, pDPot->pSTAGE->SS[s]->P[ID].Pos[0], "pDPot->pSTAGE->SS[s]->P[ID].Pos[0]");
					Value_Actions_TEPPOD_DP = AA_TF_DP.actions(ID, t0, xv0);
// DEBUG_PRINT_I(641);
					Value_AnglesFrequencies_TEPPOD_DP.resize(Dim*2, NotCalculatedActionsValue); //??
					// pDPot->pSTAGE->write_orbitApproxPeriod();
					pDPot->pSTAGE->reset_orbitdata();
// DEBUG_PRINT_I(65);
					Value_Actions_GF_DP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_GF_DP.resize(Dim*2, NotCalculatedActionsValue);
				}else{
					Value_Actions_TEPPOD_DP.resize(LENGTH_TEPPOD, NotCalculatedActionsValue);
					Value_AnglesFrequencies_TEPPOD_DP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_GF_DP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_GF_DP.resize(Dim*2, NotCalculatedActionsValue);
				}
				Value_Actions_SS_FP.push_back(		NotCalculatedActionsType); //to let the length be at least 4, similarly hereinafter
				Value_Actions_AF_FP.push_back(		NotCalculatedActionsType);
				Value_Actions_TF_FP.push_back(		NotCalculatedActionsType);
				Value_Actions_TEPPOD_FP.push_back(	NotCalculatedActionsType);
				Value_Actions_GF_FP.push_back(		NotCalculatedActionsType);
				Value_Actions_SS_DP.push_back(		NotCalculatedActionsType);
				Value_Actions_AF_DP.push_back(		NotCalculatedActionsType);
				Value_Actions_TF_DP.push_back(		NotCalculatedActionsType);
				Value_Actions_TEPPOD_DP.push_back(	NotCalculatedActionsType);
				Value_Actions_GF_DP.push_back(		NotCalculatedActionsType);
				
				//:: gather
				//gather basic info, from index 0, counts 6+2
				for(int ii=0;ii<6;ii++) Write_aa[i].particle_xv0[ii] = xv0[ii]; //init phase position
				Write_aa[i].particle_ID = ID;
				Write_aa[i].mass = pDPot->pSTAGE->SS[s]->P[ID].Mass;
				//gather other info, from index 108, counts 6
				Write_aa[i].particle_otherInfo[0] = pDPot->pSTAGE->SS[s]->P[ID].Type;
				Write_aa[i].particle_otherInfo[1] = pFPot->Phi(xv0);
				Write_aa[i].particle_otherInfo[2] = pDPot->Phi(xv0);
				Write_aa[i].particle_otherInfo[3] = Alpha;
				Write_aa[i].particle_otherInfo[4] = Beta;
				Write_aa[i].particle_otherInfo[5] = Gamma;

				//AA about B0, from index 8, counts (4+6)*3
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_SS_FP[ii] = Value_Actions_SS_FP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_SS_FP[ii] = Value_AnglesFrequencies_SS_FP[ii];
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_AF_FP[ii] = Value_Actions_AF_FP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_AF_FP[ii] = Value_AnglesFrequencies_AF_FP[ii];
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TF_FP[ii] = Value_Actions_TF_FP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TF_FP[ii] = Value_AnglesFrequencies_TF_FP[ii];
				
				//AA about B1, from index 38, counts (4+6)*2
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TEPPOD_FP[ii] = Value_Actions_TEPPOD_FP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TEPPOD_FP[ii] = Value_AnglesFrequencies_TEPPOD_FP[ii];
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_GF_FP[ii] = Value_Actions_GF_FP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_GF_FP[ii] = Value_AnglesFrequencies_GF_FP[ii];

				//AA about B2, from index 58, counts (4+6)*3
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_SS_DP[ii] = Value_Actions_SS_DP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_SS_DP[ii] = Value_AnglesFrequencies_SS_DP[ii];
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_AF_DP[ii] = Value_Actions_AF_DP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_AF_DP[ii] = Value_AnglesFrequencies_AF_DP[ii];
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TF_DP[ii] = Value_Actions_TF_DP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TF_DP[ii] = Value_AnglesFrequencies_TF_DP[ii];
				
				//AA about B3, from index 88, counts (4+6)*2
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TEPPOD_DP[ii] = Value_Actions_TEPPOD_DP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TEPPOD_DP[ii] = Value_AnglesFrequencies_TEPPOD_DP[ii];
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_GF_DP[ii] = Value_Actions_GF_DP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_GF_DP[ii] = Value_AnglesFrequencies_GF_DP[ii];
				//:: gather end, totally counts 114 (the most datatype is double).

				//:: Tell AA about particle_ID has been calculated
				// printf("ID_%d(%d, %f) ", ID, my_rank, (double)(i-my_from)/interval);
				cerr<<"ID_"<<ID<<"("<<my_rank<<", "<<(double)(i-my_from)/interval<<") ";
				fflush(stdout);
			}
			printf("\n\n");
			total_count += my_count;
DEBUG_PRINT_I(66);

			//recive from other rank:
			for(int source=1;source<comm_sz;source++){
				/* comparations:
				at rank other (send):	~		at rank 0 (recive):
				my_rank							source
				my_from							local_from
				my_count						local_count
				my_sum							local_sum
				&S[my_from]						&SS[local_from]
				: the struct vector is recollectived */
				local_from = interval*(source-1);
				local_count = interval;
				MPI_Recv(&Write_aa[local_from], 
					sizeof(struct write_angleaction)*local_count, MPI_BYTE, source, 0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				total_count += local_count;
			}
// DEBUG_PRINT_I(66);
		}else{ //if(my_rank!=0)

			int my_from = interval*(my_rank-1);
			int my_count = interval;
			int my_endout = my_from + my_count;
			printf("my_rank is %d: here my_count = %d, my_from = %d, my_endout = %d.\n", 
				my_rank, my_count, my_from, my_endout);
			printf("Now actions calculation ...\n\n");

			for(int i=my_from;i<my_endout;i++)
			{
				//:: calculate
				int ID = i+ID_start;
				pDPot->set_partical_ID(ID);
				// pDPot->set_time(t0);

				xv0 = {pDPot->pSTAGE->SS[s]->P[ID].Pos[0], pDPot->pSTAGE->SS[s]->P[ID].Pos[1], pDPot->pSTAGE->SS[s]->P[ID].Pos[2], 
					pDPot->pSTAGE->SS[s]->P[ID].Vel[0], pDPot->pSTAGE->SS[s]->P[ID].Vel[1], pDPot->pSTAGE->SS[s]->P[ID].Vel[2] };
				// for(int ix=0;ix<6;ix++){ //something uncalculatable when zero??
				// 	if(abs(xv0[ix])<Err) xv0[ix] = Err;
				// }

				int B0 = onlyRun_Actionmethod % 2; //(bool)Is run Sperical and Fudge in FPot
				int B1 = (onlyRun_Actionmethod - B0)/2 % 2; //(bool)Is run TEPPOD(none) and O2GF in FPot
				int B2 = (onlyRun_Actionmethod - B0 - B1*2)/2/2 % 2; //(bool)Is run Sperical and Fudge in DPot
				int B3 = (onlyRun_Actionmethod - B0 - B1*2 - B2*2*2)/2/2/2 % 2; //(bool)Is run TEPPOD and O2GF(none) in DPot
				if((bool)B0){
					// Value_Actions_SS_FP = AA_SS_FP.actions(xv0);
					// Value_AnglesFrequencies_SS_FP = AA_SS_FP.angles_and_freqs(xv0);
					// Value_Actions_AF_FP = AA_AF_FP.actions(xv0);
					// Value_AnglesFrequencies_AF_FP = AA_AF_FP.angles(xv0);
					Value_Actions_SS_FP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_SS_FP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_AF_FP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_AF_FP.resize(Dim*2, NotCalculatedActionsValue);

					Value_Actions_TF_FP = AA_TF_FP.actions(xv0);
					Value_AnglesFrequencies_TF_FP = AA_TF_FP.angles(xv0);
				}else{
					Value_Actions_SS_FP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_SS_FP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_AF_FP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_AF_FP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_TF_FP.resize(Dim+1, NotCalculatedActionsValue);
					Value_AnglesFrequencies_TF_FP.resize(Dim*2+5, NotCalculatedActionsValue);
				}
				if((bool)B1){
					Value_Actions_TEPPOD_FP.resize(LENGTH_TEPPOD, NotCalculatedActionsValue);
					Value_AnglesFrequencies_TEPPOD_FP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_GF_FP = AA_GF_FP.actions(xv0);
					Value_AnglesFrequencies_GF_FP = AA_GF_FP.angles(xv0);
				}else{
					Value_Actions_TEPPOD_FP.resize(LENGTH_TEPPOD, 0.);
					Value_AnglesFrequencies_TEPPOD_FP.resize(6, 0.);
					Value_Actions_GF_FP.resize(3, 0.);
					Value_AnglesFrequencies_GF_FP.resize(6, 0.);
				}
				if((bool)B2){
					// Value_Actions_SS_DP = AA_SS_DP.actions(xv0);
					// Value_AnglesFrequencies_SS_DP = AA_SS_DP.angles_and_freqs(xv0);
					// Value_Actions_AF_DP = AA_AF_DP.actions(xv0);
					// Value_AnglesFrequencies_AF_DP = AA_AF_DP.angles(xv0);
					Value_Actions_SS_DP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_SS_DP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_AF_DP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_AF_DP.resize(Dim*2, NotCalculatedActionsValue);

					//: modify code
					// if is_exist_previous_actions_file is false //keep inputAlphaBeta nullptr for lmn_orb::actions(const VecDoub& x, void* params);
					// if is_exist_previous_actions_file is true  //inputAlphaBeta[0] = vecAlphaBeta[ID-1][0], inputAlphaBeta[1] = vecAlphaBeta[ID-1][1];
					Value_Actions_TF_DP = AA_TF_DP.actions(xv0, inputAlphaBeta);
					Alpha = AA_TF_DP.ABC_presentCalculation[0], 
					Beta = AA_TF_DP.ABC_presentCalculation[1], 
					Gamma = Gamma; //to record focus
					Value_AnglesFrequencies_TF_DP = AA_TF_DP.angles(xv0);
				}else{
					Value_Actions_SS_DP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_SS_DP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_AF_DP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_AF_DP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_TF_DP.resize(Dim+1, NotCalculatedActionsValue);
					Value_AnglesFrequencies_TF_DP.resize(Dim*2+5, NotCalculatedActionsValue);
				}
				if((bool)B3){
					Value_Actions_TEPPOD_DP = AA_TF_DP.actions(ID, t0, xv0);
					Value_AnglesFrequencies_TEPPOD_DP.resize(Dim*2, NotCalculatedActionsValue); //??
					// pDPot->pSTAGE->write_orbitApproxPeriod();
					pDPot->pSTAGE->reset_orbitdata();

					Value_Actions_GF_DP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_GF_DP.resize(Dim*2, NotCalculatedActionsValue);
				}else{
					Value_Actions_TEPPOD_DP.resize(LENGTH_TEPPOD, NotCalculatedActionsValue);
					Value_AnglesFrequencies_TEPPOD_DP.resize(Dim*2, NotCalculatedActionsValue);
					Value_Actions_GF_DP.resize(Dim, NotCalculatedActionsValue);
					Value_AnglesFrequencies_GF_DP.resize(Dim*2, NotCalculatedActionsValue);
				}
				Value_Actions_SS_FP.push_back(		NotCalculatedActionsType); //to let the length be at least 4, similarly hereinafter
				Value_Actions_AF_FP.push_back(		NotCalculatedActionsType);
				Value_Actions_TF_FP.push_back(		NotCalculatedActionsType);
				Value_Actions_TEPPOD_FP.push_back(	NotCalculatedActionsType);
				Value_Actions_GF_FP.push_back(		NotCalculatedActionsType);
				Value_Actions_SS_DP.push_back(		NotCalculatedActionsType);
				Value_Actions_AF_DP.push_back(		NotCalculatedActionsType);
				Value_Actions_TF_DP.push_back(		NotCalculatedActionsType);
				Value_Actions_TEPPOD_DP.push_back(	NotCalculatedActionsType);
				Value_Actions_GF_DP.push_back(		NotCalculatedActionsType);
				
				//:: gather
				//gather basic info, from index 0, counts 6+2
				for(int ii=0;ii<6;ii++) Write_aa[i].particle_xv0[ii] = xv0[ii]; //init phase position
				Write_aa[i].particle_ID = ID;
				Write_aa[i].mass = pDPot->pSTAGE->SS[s]->P[ID].Mass;
				//gather other info, from index 108, counts 6
				Write_aa[i].particle_otherInfo[0] = pDPot->pSTAGE->SS[s]->P[ID].Type;
				Write_aa[i].particle_otherInfo[1] = pFPot->Phi(xv0);
				Write_aa[i].particle_otherInfo[2] = pDPot->Phi(xv0);
				Write_aa[i].particle_otherInfo[3] = Alpha;
				Write_aa[i].particle_otherInfo[4] = Beta;
				Write_aa[i].particle_otherInfo[5] = Gamma;

				//AA about B0, from index 8, counts (4+6)*3
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_SS_FP[ii] = Value_Actions_SS_FP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_SS_FP[ii] = Value_AnglesFrequencies_SS_FP[ii];
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_AF_FP[ii] = Value_Actions_AF_FP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_AF_FP[ii] = Value_AnglesFrequencies_AF_FP[ii];
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TF_FP[ii] = Value_Actions_TF_FP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TF_FP[ii] = Value_AnglesFrequencies_TF_FP[ii];
				
				//AA about B1, from index 38, counts (4+6)*2
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TEPPOD_FP[ii] = Value_Actions_TEPPOD_FP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TEPPOD_FP[ii] = Value_AnglesFrequencies_TEPPOD_FP[ii];
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_GF_FP[ii] = Value_Actions_GF_FP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_GF_FP[ii] = Value_AnglesFrequencies_GF_FP[ii];

				//AA about B2, from index 58, counts (4+6)*3
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_SS_DP[ii] = Value_Actions_SS_DP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_SS_DP[ii] = Value_AnglesFrequencies_SS_DP[ii];
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_AF_DP[ii] = Value_Actions_AF_DP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_AF_DP[ii] = Value_AnglesFrequencies_AF_DP[ii];
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TF_DP[ii] = Value_Actions_TF_DP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TF_DP[ii] = Value_AnglesFrequencies_TF_DP[ii];
				
				//AA about B3, from index 88, counts (4+6)*2
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_TEPPOD_DP[ii] = Value_Actions_TEPPOD_DP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_TEPPOD_DP[ii] = Value_AnglesFrequencies_TEPPOD_DP[ii];
				for(int ii=0;ii<LENGTH_ACTIONS_WRITE;ii++) Write_aa[i].Value_Actions_GF_DP[ii] = Value_Actions_GF_DP[ii];
				for(int ii=0;ii<LENGTH_ANGLESFREQUENCIES_WRITE;ii++) Write_aa[i].Value_AnglesFrequencies_GF_DP[ii] = Value_AnglesFrequencies_GF_DP[ii];
				//:: gather end, totally counts 114 (the most datatype is double).

				//:: Tell AA about particle_ID has been calculated
				// printf("ID_%d(%d, %f) ", ID, my_rank, (double)(i-my_from)/interval);
				cerr<<"ID_"<<ID<<"("<<my_rank<<", "<<(double)(i-my_from)/interval<<") ";
				fflush(stdout);
			}
			printf("\n\n");

			// MPI_Barrier(MPI_COMM_WORLD);
			printf("before MPI_Send()\n");
			MPI_Send(&Write_aa[my_from], sizeof(struct write_angleaction)*my_count, MPI_BYTE, 0, 0,MPI_COMM_WORLD);
			printf("after MPI_Send()\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
		printf("\nmy_rank (%d): Calculation actions ... done.\n\n", my_rank);
		////main end]



		////(3) write
		if(my_rank==0){
			if(total_count!=N_ptcs){
				std::cerr<<"Wrong total count of particles: total_count("<<total_count<<") != N_ptcs("<<N_ptcs<<"). Please check!\n";
				exit(0);
			}
			printf("\ntotal_count = %d.\n", total_count);

			write_action_data(Write_aa, snapshot, path_gm_1, N_ptcs); //write data points actions to txt
		}
		free(Write_aa);
		////TASK1 end]
		MPI_Barrier(MPI_COMM_WORLD);
DEBUG_PRINT_I(7);



		////(4) potential compare
		if(my_rank==0){
			printf("Calculate potential to compare.\n");
			for(int j=0;j<3;j++){ //x, y, z, 3 files
				int x_count = 100;
				double x_min = 1e-2, x_max = 1e3; //?? when it bigger then 2e3, the rotate potential will be invert
				VecDoub xlist = GJY::gen_logspace1D(x_min, x_max, x_count);
				double P = 0.;
				VecDoub F(3, 0.);
				VecDoub X(3, 0.);
				vector<struct data_debug> vdd;
				vdd.resize(x_count);

				for(int i=0;i<x_count;i++){ //points on one axis
					X = {0., 0., 0., 0., 0., 0.};
					X[j] = xlist[i];
					DEBUG_PRINT_V1d(1, X, "X");
					vdd[i].xv = X;
					
					//(1) ineria frame potential, interia frame force
					pSTAGE->is_preprocess_rotation = true;
					// DPot.set_algorithm(4); //SCF: 4
					// P = pSTAGE->SS[0]->potential_SCF(X);
					// F = pSTAGE->SS[0]->forces_SCF(X);
					P = pSTAGE->potential_t(X, t0, 1, 0, 4);
					F = pSTAGE->forces_t(X, t0, 1, 0, 4);
					vdd[i].value_double.push_back(P);
					vdd[i].value_double.push_back(F[0]);
					vdd[i].value_double.push_back(F[1]);
					vdd[i].value_double.push_back(F[2]);

					// DPot.set_algorithm(AlgorithmPot); //DS: 0 or 1 (without substracting self)
					P = pSTAGE->potential_t(X, t0, 1, 0, 1);
					F = pSTAGE->forces_t(X, t0, 1, 0, 0);
					vdd[i].value_double.push_back(P);
					vdd[i].value_double.push_back(F[0]);
					vdd[i].value_double.push_back(F[1]);
					vdd[i].value_double.push_back(F[2]);

					//(2) rotating frame potential, rotating frame force
					pSTAGE->is_preprocess_rotation = false; //ineria frame potential
					// DPot.set_algorithm(4); //SCF: 4
					P = pSTAGE->potential_t(X, t0, 1, 0, 4);
					F = pSTAGE->forces_t(X, t0, 1, 0, 4);
					vdd[i].value_double.push_back(P);
					vdd[i].value_double.push_back(F[0]);
					vdd[i].value_double.push_back(F[1]);
					vdd[i].value_double.push_back(F[2]);

					// DPot.set_algorithm(AlgorithmPot); //DS: 0 or 1 (without substracting self)
					P = pSTAGE->potential_t(X, t0, 1, 0, 1);
					F = pSTAGE->forces_t(X, t0, 1, 0, 0);
					vdd[i].value_double.push_back(P);
					vdd[i].value_double.push_back(F[0]);
					vdd[i].value_double.push_back(F[1]);
					vdd[i].value_double.push_back(F[2]);

				}
				string filename_debug = path_gm_1+"intermediate/potential_compare_"
					+to_string(snapshot)+"_"+to_string(j)+".txt";
				write_data_debug(vdd, filename_debug);
			}
			pSTAGE->is_preprocess_rotation = true;
		}



		////(5) write kernel density of mass and actions
		if(my_rank==0){
			/*
			//another file, or in python instead of this slow-debug-snale CC++ //??
			std::array<std::array<double, Dim>, N_total> xdata0, xdata1; //dynamic tree?? //x-tree and J-tree
			wtfh = (struct write_firsthand *) malloc(sizeof(struct write_firsthand)*(N_total));
			wtsh = (struct write_secondhand *) malloc(sizeof(struct write_secondhand)*(N_total));
			int WhatCannonical;

			WhatCannonical = 2;
			pSTAGE->SS[s]->load_to_firsthand_from_PD(); //??
			for(int i=0;i<N_total;i++){
				int iP = i+1;
				xdata0[i][0] = pSTAGE->SS[s]->P[iP].Pos[0];
				xdata0[i][1] = pSTAGE->SS[s]->P[iP].Pos[1];
				xdata0[i][2] = pSTAGE->SS[s]->P[iP].Pos[2];
			}
			KDtree<double, N_total, Dim> kdt0(&xdata0);
			pSTAGE->SS[s]->loadtree(&kdt0);
			// string suffix = "DF_CartesianMass";
			pSTAGE->SS[s]->write_secondhand_all_txt(WhatCannonical, 0, 0, 0);
			// pSTAGE->SS[s]->remove_tree(); //then after {}, the array will be release
			free(wtfh);
			free(wtsh);

			WhatCannonical = 5; //WhatCannonical
			for(int WP=0;WP<2;WP++){ //WhatPotential {0,1}
				for(int WS=2;WS<3;WS++){ //WhatSymmetry, {0,1,2}
					for(int WA=0;WA<3;WA++){ //WhatActionmethod {0,1,2}
						wtfh = (struct write_firsthand *) malloc(sizeof(struct write_firsthand)*(N_total));
						wtsh = (struct write_secondhand *) malloc(sizeof(struct write_secondhand)*(N_total));
						int isExistFile = pSTAGE->SS[s]->read_firsthand_all_txt(WhatCannonical, WP, WS, WA);
						if(isExistFile!=0){
							continue;
						}
						for(int i=0;i<N_total;i++){
							xdata1[i][0] = wtfh[i].QP[3]; 
							xdata1[i][1] = wtfh[i].QP[4]; 
							xdata1[i][2] = wtfh[i].QP[5];
						}
						KDtree<double, N_total, Dim> kdt1(&xdata1);
						pSTAGE->SS[s]->loadtree1(&kdt1);
						// suffix = "DF_actionUnit";
						pSTAGE->SS[s]->write_secondhand_all_txt(WhatCannonical, WP, WS, WA);
						// pSTAGE->SS[s]->remove_tree();
						free(wtfh);
						free(wtsh);
					}
				}
			}
			*/
		}
		MPI_Barrier(MPI_COMM_WORLD);
DEBUG_PRINT_I(8);



		////5. TASK2. samples and state density
		////This is another task. 
		////[TASK2 start
		//space xv density ~ space OJ density
		//each SFFP, TEPPOD(should orbit, slow) O2GF(slow); count 1e6 //??
		if(is_run_samples){
			////(1)
			// auto pAA = &AA_TF_FP;
			xv0.resize(6, 0.);
			VecDoub Value_Actions_samples, Value_AnglesFrequencies_samples;

			////(2)
			int N_samps = read_action_samples(Write_action_samples, snapshot, path_gm_1, N_action_samples); //-1
			// if(N_action_samples<N_samps){printf("Warning_: Too less samples that be specified.\n");}
			int N_1 = N_samps / comm_sz;
			int N_0 = N_samps - N_1*comm_sz;
			write_angleaction** pWrite_action_samples = new struct write_angleaction*[comm_sz];
			// MPI_Barrier(MPI_COMM_WORLD);

			////(3)
			if(my_rank==0){
				////:(1)
				int my_count1 = N_0;
				int my_from = N_1*(comm_sz-1);
				int my_endout = my_from+my_count1;
				pWrite_action_samples[my_rank] = new struct write_angleaction[my_count1];
				for(int source=1;source<comm_sz;source++){
					int sr_from = N_1*(source-1);
					int sr_count = N_1;
DEBUG_PRINT_V0d(1, my_rank, "before send A");
					MPI_Send(&Write_action_samples[sr_from], 
						sizeof(struct write_angleaction)*sr_count, MPI_BYTE, source, 
						0,MPI_COMM_WORLD);
DEBUG_PRINT_V0d(1, my_rank, "affter send A");
				}

				for(int i=0;i<my_count1;i++){
					memcpy(&xv0[0], &pWrite_action_samples[my_rank][i].particle_xv0[0], 
						sizeof(pWrite_action_samples[my_rank][i].particle_xv0[0])*2*Dim);
					Value_Actions_samples = AA_TF_FP.actions(xv0); //,Alpha,Beta
					Value_AnglesFrequencies_samples = AA_TF_FP.angles(xv0); //,Alpha,Beta
					memcpy(&pWrite_action_samples[my_rank][i].Value_Actions_TF_FP[0], &Value_Actions_samples[0], 
						sizeof(pWrite_action_samples[my_rank][i].Value_AnglesFrequencies_TF_FP[0])*2*Dim);
					memcpy(&pWrite_action_samples[my_rank][i].Value_AnglesFrequencies_TF_FP[0], &Value_AnglesFrequencies_samples[0], 
						sizeof(pWrite_action_samples[my_rank][i].Value_AnglesFrequencies_TF_FP[0])*2*Dim);
					printf("ID_%d(%d, %f) ", i+1, my_rank, (double)(i+1-my_from)/my_count1);
				}

				memcpy(&Write_action_samples[my_from], &pWrite_action_samples[my_rank][0], 
					sizeof(struct write_angleaction)*my_count1);
				delete[] pWrite_action_samples[my_rank];

				////:(2)
				for(int source=1;source<comm_sz;source++){
					int sr_from = N_1*(source-1);
					int sr_count = N_1;
DEBUG_PRINT_V0d(1, my_rank, "before recv B");
					MPI_Recv(&Write_action_samples[sr_from], 
						sizeof(struct write_angleaction)*sr_count, MPI_BYTE, source, 
						0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
DEBUG_PRINT_V0d(1, my_rank, "affter recv B");
				}
			}else{ //if(my_rank!=0)

				int my_count1 = N_0;
				int my_from = N_1*(my_rank-1);
				int my_endout = my_from+my_count1;
DEBUG_PRINT_I(81);
				pWrite_action_samples[my_rank] = new struct write_angleaction[my_count1];
DEBUG_PRINT_V0d(1, my_rank, "before recv A");
DEBUG_PRINT_I(811);
				MPI_Recv(&pWrite_action_samples[my_rank][0], 
					sizeof(struct write_angleaction)*my_count1, MPI_BYTE, 0, 
					0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
DEBUG_PRINT_V0d(1, my_rank, "affter recv A");
DEBUG_PRINT_I(82);

				for(int i=0;i<my_count1;i++){
					memcpy(&xv0[0], &pWrite_action_samples[my_rank][i].particle_xv0[0], 
						sizeof(pWrite_action_samples[my_rank][i].particle_xv0[0])*2*Dim);
					Value_Actions_samples = AA_TF_FP.actions(xv0); //,Alpha,Beta
					Value_AnglesFrequencies_samples = AA_TF_FP.angles(xv0); //,Alpha,Beta
					memcpy(&pWrite_action_samples[my_rank][i].Value_Actions_TF_FP[0], &Value_Actions_samples[0], 
						sizeof(pWrite_action_samples[my_rank][i].Value_AnglesFrequencies_TF_FP[0])*2*Dim);
					memcpy(&pWrite_action_samples[my_rank][i].Value_AnglesFrequencies_TF_FP[0], &Value_AnglesFrequencies_samples[0], 
						sizeof(pWrite_action_samples[my_rank][i].Value_AnglesFrequencies_TF_FP[0])*2*Dim);
					printf("ID_%d(%d, %f) ", i+1, my_rank, (double)(i+1-my_from)/my_count1);
				}

DEBUG_PRINT_I(83);
DEBUG_PRINT_V0d(1, my_rank, "before send B");
				MPI_Send(&pWrite_action_samples[my_rank][0], 
					sizeof(struct write_angleaction)*my_count1, MPI_BYTE, 0, 
					0,MPI_COMM_WORLD);
DEBUG_PRINT_V0d(1, my_rank, "affter send B");
				delete[] pWrite_action_samples[my_rank]; //??
DEBUG_PRINT_I(84);
			}
			if(my_rank==0){
				bool is_samples = true;
				write_action_data(Write_action_samples, snapshot, path_gm_1, 
					N_action_samples, is_samples);
			}
			delete[] Write_action_samples;
		}
		////TASK2 end]

		MPI_Barrier(MPI_COMM_WORLD);
DEBUG_PRINT_I(13);
	}

	////8. fit DF of AA
	// [...]
DEBUG_PRINT_I(14);

	////9. end
	MPI_Barrier(MPI_COMM_WORLD);
	if(my_rank==0){
		printf("\nDone.\n");
	}
DEBUG_PRINT_I(15);
    MPI_Finalize();
DEBUG_PRINT_I(16);
	return 0;
}



// /*
// 	////compare potentials and actions
// 	// VecDoub xv0_debug = {1., 10., 1., 220., 10., 1.};
// 	// for(){}

// 	////gJ
// 	// read samples, simply senn it as dd of dV1/dV2

// 	////fJ
// 	// is_correct, gJ3 and fJ select //in python

// 	////collision, diffusion, resonance, phase mixing scales
// 	// [...]

// 	////stream
// 	// [...]

// 	////laws of self-adaptive multi-agents evolution/game
// 	// [...]
	
// 	////ABCDEFG
// 	// [...]
// */