
#include "DataInterface.h"
// #include "out.h"



int main(int argc, char* argv[]){

DEBUG_PRINT_I(11);
	////1. input
	//:: in linux shell: 
	// # cd path/of/aa/; make clean && cd .. && make LAPACK=1 TORUS=1 && cd aa;  
	// # mpirun -np 4 mains/./data.exe  
	// # <input arguments below> #as arguments of ./../aa/step2_run.bat
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
	int N_ptcs 					= atoi(argv[17]); 	//the max index of particles //N_allPtcs
	
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
	int snapshot_previous   = atof(argv[29]);
DEBUG_PRINT_I(12);


	////3. snapshot settings
	string path_IC = "../../../../"
		"GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/"
		"IC_param.txt";
	Stage STAGE(path_IC);
	Stage* pSTAGE = &STAGE;
DEBUG_PRINT_I(13);

	////calculate
	string path_gm_1 = pSTAGE->path_gm;
	printf("path_gm: %s", path_gm_1.data());
	pSTAGE->load_multi_snapshots(t_init, t_final, dt_load, dt_step, 0, 0);
DEBUG_PRINT_I(14);
	double Alpha = TACT_semiaxis_Alpha, Beta = TACT_semiaxis_Beta;
DEBUG_PRINT_I(2);

	// ////potentials presetting
	// Potential_JS DPot(&STAGE);
	// Potential_JS* pDPot = &DPot;
	// //pDPot->Phi(..., Potential_other*=nullptr) //??
	// pDPot->set_algorithm(AlgorithmPot);
	// DEBUG_PRINT_V0d(1, AlgorithmPot, "the main AlgorithmPot");

	// // int N_calculate_load = (int)((t_end_run-t_start_run)/dt_run);
	// vector<Potential_JS> vFPot; //??
DEBUG_PRINT_I(3);

	for(double t0=t_start_run;t0<t_end_run;t0+=dt_run){ //time
		DEBUG_PRINT_V0d(1, t0, "t0 for");
		DEBUG_PRINT_V0d(1, t_init, "t_init");
		int s = t_to_loadsnapshot(t0, dt_load, t_init, 0.); //load
		DEBUG_PRINT_V0d(1, s, "s");
		int snapshot = pSTAGE->SS[s]->snap; //snapshot
		// pDPot->set_time(t0); //Note: One should set time before calculate potential to use the right snapshot of time
		pSTAGE->SS[s]->generate_kdtree_pos(); //prepare kdtree from particle data
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

		////(2) formula potential
		Galaxy_components GA; //GA();
		GA.read_fit_DF_x_mass(path_gm_1, snapshot);
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

		// // Density_MDPLEP rho(); //??
		// // TestDensity_NFW rho(GA.M_scale_comp[icmp], GA.scaled_length_fit_comp[icmp], 
		// //   {GA.axis_ratio_x_fit_comp[icmp], GA.axis_ratio_y_fit_comp[icmp], GA.axis_ratio_z_fit_comp[icmp]}
		// // ); //wrong energy
		// Density_DoublePowerLaw rho(GA.scaled_density_fit_comp[icmp], GA.scaled_length_fit_comp[icmp], 
		// 	{GA.axis_ratio_x_fit_comp[icmp], GA.axis_ratio_y_fit_comp[icmp], 
		// 	GA.axis_ratio_z_fit_comp[icmp]}, GA.powerA_fit_comp[icmp], GA.powerB_fit_comp[icmp]
		// );
		// // Density_Einasto rho(GA.scaled_density_fit_comp[icmp], GA.scaled_length_fit_comp[icmp], 
		// // 	{GA.axis_ratio_x_fit_comp[icmp], GA.axis_ratio_y_fit_comp[icmp], 
		// // 	GA.axis_ratio_z_fit_comp[icmp]}, GA.powerA_fit_comp[icmp]
		// // );

		string input_ME_name = path_gm_1+"input_ME.txt";
		VecDoub input_ME = read_pure_number(input_ME_name, 10);
DEBUG_PRINT_I(51);

		double GM = conv::G*GA.M_scale_comp[icmp];
		double rhos = GA.scaled_density_fit_comp[icmp];
		double rs = GA.scaled_length_fit_comp[icmp];
		double qx = GA.axis_ratio_x_fit_comp[icmp], 
			qy = GA.axis_ratio_y_fit_comp[icmp], 
			qz = GA.axis_ratio_z_fit_comp[icmp];

		// NFW Pot_NFW_Exc(GM, rs, qy, qz);
		// TestDensity_NFW rho_NFW(GM, rs, {qx, qy, qz});
		// TriaxialPotential Pot_NFW_TS(&rho_NFW, (double)input_ME[6]);
		// MultipoleExpansion Pot_NFW_ME(&rho_NFW, (int)input_ME[0], (int)input_ME[1], (int)input_ME[2], 
		// 	(int)input_ME[3], (double)input_ME[4], (double)input_ME[5], (double)input_ME[6], 
		// 	(bool)input_ME[7], (bool)input_ME[8], (bool)input_ME[9]
		// );
		// VecDoub a(9, 0.);
		// a[0] = rho_NFW.density_m2(0.1);
		// a[1] = rho_NFW.density_m2(1.);
		// a[2] = rho_NFW.density_m2(10.);

		// Density_DoublePowerLaw rho_DPL(GA.scaled_density_fit_comp[icmp], GA.scaled_length_fit_comp[icmp], 
		// 	{GA.axis_ratio_x_fit_comp[icmp], GA.axis_ratio_y_fit_comp[icmp], GA.axis_ratio_z_fit_comp[icmp]}, 
		// 	1., 2. //not fitted powers instead of NFW powers
		// 	// GA.powerA_fit_comp[icmp], GA.powerB_fit_comp[icmp]
		// );
		// MultipoleExpansion Pot_DPL_ME(&rho_DPL, (int)input_ME[0], (int)input_ME[1], (int)input_ME[2], 
		// 	(int)input_ME[3], (double)input_ME[4], (double)input_ME[5], (double)input_ME[6], 
		// 	(bool)input_ME[7], (bool)input_ME[8], (bool)input_ME[9]
		// );
		// a[3] = rho_DPL.density_m2(0.1);
		// a[4] = rho_DPL.density_m2(1.);
		// a[5] = rho_DPL.density_m2(10.);

		// Density_Einasto rho_SER(GA.scaled_density_fit_comp[icmp], GA.scaled_length_fit_comp[icmp], 
		// 	{GA.axis_ratio_x_fit_comp[icmp], GA.axis_ratio_y_fit_comp[icmp], GA.axis_ratio_z_fit_comp[icmp]}, 
		// 	GA.powerA_fit_comp[icmp]
		// );
		// MultipoleExpansion Pot_Sersic_ME(&rho_SER, (int)input_ME[0], (int)input_ME[1], (int)input_ME[2], 
		// 	(int)input_ME[3], (double)input_ME[4], (double)input_ME[5], (double)input_ME[6], 
		// 	(bool)input_ME[7], (bool)input_ME[8], (bool)input_ME[9]
		// );
		// a[6] = rho_SER.density_m2(0.1);
		// a[7] = rho_SER.density_m2(1.);
		// a[8] = rho_SER.density_m2(10.);
		// DEBUG_PRINT_V1d(1, a, "rho at rq2 = 1. or 10.");
		// //result: DEBUG_PRINT: 0.0264736 0.00669819 0.0264736 0.00119085 0.614154 0.00761373 (rho at rq2 = 1. or 10.)



		//// [1] potential compare 1
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
DEBUG_PRINT_I(52);

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


		
		//// [2] potential compare 2

	}

	return 0;
}



// int main(int argc, char* argv[]){

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
// 	int is_preprocessed			= atoi(argv[7]); 	//0: not <1>; 1: write and centerize //0
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
	
// 	int is_run_samples 			= atoi(argv[18]); 	//samples for action state density
// 	int N_action_samples		= atoi(argv[19]);

// 	int is_DEBUG				= atoi(argv[20]);

// 	//:: other
// 	// double changedParticle_mass	= atof(argv[21]);	//1. //0.1



// 	////3. snapshot settings
// 	string path_work = get_workpath();
// 	string path_base = path_work+"0prog/gadget/Gadget-2.0.7/galaxy_general/";
// 	string path_IC = "IC_param.txt";
// 	Stage STAGE(path_base, path_IC);
// 	Stage* pSTAGE = &STAGE;

// 	////calculate
// 	pSTAGE->load_multi_snapshots(t_init, t_final, dt_load, dt_step, 0, 0);
// 	double Alpha = TACT_semiaxis_Alpha, Beta = TACT_semiaxis_Beta;

// 	for(double t0=t_start_run;t0<t_end_run;t0+=dt_run){ //time
// 		int s = t_to_loadsnapshot(t0, dt_load, t_init, 0.); //load
// 		int snapshot = pSTAGE->SS[s]->snap; //snapshot
// 		// int calculateIndex = (int)((t0-t_start_run)/dt_run); //calculate
// 		DEBUG_PRINT_V0d(1, t0, "t0");
// 		DEBUG_PRINT_V0d(1, s, "s");
// 		DEBUG_PRINT_V0d(1, snapshot, "snapshot");

// 		////(1) data potential
// 		#ifdef DEBUG_GJY
// 		pSTAGE->SS[s]->preprocess(); //to get triaxialize matix of this s~snapshot
// 		#endif
		
// 		DEBUG_PRINT_V0d(10, s, "before xtree");
// 		std::array<std::array<double, Dim>, N_total> xdata; //dynamic tree?? //x-tree and J-tree
// 		for(int i=0;i<N_total;i++){
// 			int iP = i+1;
// 			xdata[i][0] = pSTAGE->SS[s]->P[iP].Pos[0];
// 			xdata[i][1] = pSTAGE->SS[s]->P[iP].Pos[1];
// 			xdata[i][2] = pSTAGE->SS[s]->P[iP].Pos[2];
// 		}
// 		KDtree<double, N_total, Dimension> kdt(&xdata);
// 		pSTAGE->SS[s]->loadtree(&kdt);
// 		DEBUG_PRINT_V0d(10, s, "after xtree");

// 		// potential and actions





// 		////small main
// 		// write kernel density of mass and actions
// 		if(1){
// 			//another file, or in python instead of this slow-debug-snale CC++ //??
// 			std::array<std::array<double, Dim>, N_total> xdata0, xdata1; //dynamic tree?? //x-tree and J-tree
// 			wtfh = (struct write_firsthand *) malloc(sizeof(struct write_firsthand)*(N_total));
// 			wtsh = (struct write_secondhand *) malloc(sizeof(struct write_secondhand)*(N_total));
// 			int WhatCannonical;

// 			WhatCannonical = 2;
// 			pSTAGE->SS[s]->load_to_firsthand_from_PD(); //??
// 			for(int i=0;i<N_total;i++){
// 				int iP = i+1;
// 				xdata0[i][0] = pSTAGE->SS[s]->P[iP].Pos[0];
// 				xdata0[i][1] = pSTAGE->SS[s]->P[iP].Pos[1];
// 				xdata0[i][2] = pSTAGE->SS[s]->P[iP].Pos[2];
// 			}
// 			KDtree<double, N_total, Dim> kdt0(&xdata0);
// 			pSTAGE->SS[s]->loadtree(&kdt0);
// 			// string suffix = "DF_CartesianMass";
// 			pSTAGE->SS[s]->write_secondhand_all_txt(WhatCannonical, 0, 0, 0);
// 			// pSTAGE->SS[s]->remove_tree(); //then after {}, the array will be release
// 			free(wtfh);
// 			free(wtsh);

// 			WhatCannonical = 5; //WhatCannonical
// 			for(int WP=0;WP<2;WP++){ //WhatPotential {0,1}
// 				for(int WS=2;WS<3;WS++){ //WhatSymmetry, {0,1,2}
// 					for(int WA=0;WA<3;WA++){ //WhatActionmethod {0,1,2}
// 						wtfh = (struct write_firsthand *) malloc(sizeof(struct write_firsthand)*(N_total));
// 						wtsh = (struct write_secondhand *) malloc(sizeof(struct write_secondhand)*(N_total));
// 						int isExistFile = pSTAGE->SS[s]->read_firsthand_all_txt(WhatCannonical, WP, WS, WA);
// 						if(isExistFile!=0){
// 							continue;
// 						}
// 						for(int i=0;i<N_total;i++){
// 							xdata1[i][0] = wtfh[i].QP[3]; 
// 							xdata1[i][1] = wtfh[i].QP[4]; 
// 							xdata1[i][2] = wtfh[i].QP[5];
// 						}
// 						KDtree<double, N_total, Dim> kdt1(&xdata1);
// 						pSTAGE->SS[s]->loadtree1(&kdt1);
// 						// suffix = "DF_actionUnit";
// 						pSTAGE->SS[s]->write_secondhand_all_txt(WhatCannonical, WP, WS, WA);
// 						// pSTAGE->SS[s]->remove_tree();
// 						free(wtfh);
// 						free(wtsh);
// 					}
// 				}
// 			}
// 		}

// 		printf("snapshot_%d done.\n", snapshot);
// 	}
// 	printf("Done.\n");
// }
