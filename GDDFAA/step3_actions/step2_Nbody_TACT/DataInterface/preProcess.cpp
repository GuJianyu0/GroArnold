// #include "out.h"
#include "DataInterface.h"



int main(int argc, char* argv[]){

DEBUG_PRINT_I(11);
	////1. input
	//:: in linux shell: 
	// # cd path/of/aa/; make clean && cd .. && make LAPACK=1 TORUS=1 && cd aa;  
	// # mpirun -np 4 mains/./data.exe  
	// # <input arguments below>
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
	int N_ptcs 					= atoi(argv[17]); 	//the max index of particles //10000: N_all
	
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

	std::cout<<"nothing ...\n";



/*
	////3. snapshot settings
	string path_work = get_workpath();
	string path_base = path_work+"GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/";
	string path_IC = "IC_param.txt";
	string path_current = getcwd(NULL, 0);
	DEBUG_PRINT_V0d(10, path_work, "path_work");
	DEBUG_PRINT_V0d(10, path_base, "path_base");
	DEBUG_PRINT_V0d(10, path_current, "path_current");

	Stage STAGE(path_base, path_IC);
	Stage* pSTAGE = &STAGE;

	////calculate
	pSTAGE->load_multi_snapshots(t_init, t_final, dt_load, dt_step, 0, 0);
	double Alpha = TACT_semiaxis_Alpha, Beta = TACT_semiaxis_Beta;

	// ////potentials presetting
	// Potential_JS DPot(&STAGE);
	// Potential_JS* pDPot = &DPot;
	// //pDPot->Phi(..., Potential_other*=nullptr) //??
	// pDPot->set_algorithm(AlgorithmPot);

	// // int N_calculate_load = (int)((t_end_run-t_start_run)/dt_run);
	// vector<Potential_JS> vFPot; //??
	// // Initialization

	for(double t0=t_start_run;t0<t_end_run;t0+=dt_run){ //time
		int s = t_to_loadsnapshot(t0, dt_load, t_init, 0.); //load
		int snapshot = pSTAGE->SS[s]->snap; //snapshot
		// int calculateIndex = (int)((t0-t_start_run)/dt_run); //calculate
		DEBUG_PRINT_V0d(1, t0, "t0");
		DEBUG_PRINT_V0d(1, s, "s");
		DEBUG_PRINT_V0d(1, snapshot, "snapshot");

		#ifdef DEBUG_GJY
		pSTAGE->SS[s]->preprocess(); //to get triaxialize matix of this s~snapshot
		#endif

		////(1) data potential
		// //!!?? debug preprocess: false when multiload, true now
		// #ifdef DEBUG_GJY
		// pDPot->pSTAGE->SS[s]->preprocess();
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
DEBUG_PRINT_I(3);



		////some
		double Alpha = TACT_semiaxis_Alpha, Beta = TACT_semiaxis_Beta, Gamma = -1.;
		DEBUG_PRINT_V1d(1, (VecDoub){Alpha, Beta, Gamma}, "ABC from file");
		Alpha = input_other_1, Beta = input_other_2;
		DEBUG_PRINT_V1d(1, (VecDoub){Alpha, Beta, Gamma}, "ABC from input");
		pSTAGE->set_ConfocalEllipsoidalCoordSys_focus(Alpha, Beta);
		int ID = ID_start;
		int count_maxTimeStep = 9000;
		pSTAGE->set_particle_data_time(ID, t0, count_maxTimeStep); //to get STAGE.PT[]

		char wt_fname_dbg[200];
		sprintf(wt_fname_dbg, "%s0prog/gadget/Gadget-2.0.7/%sorbit/orbit_particle_%d_endless%s.txt", 
			(char*)get_workpath().data(), modelPath, ID, input_other_3.data());
		FILE* fp = fopen(wt_fname_dbg, "w");
		if(fp==nullptr){
			printf("Cannot open file %s. Now exit.\n", wt_fname_dbg);
			exit(0);
		}
		string file_info = "##write: column: xv(double[6]) tau_and_metric(double[6]) "
			"tau_and_dottau(double[6]) tau_and_ptau(double[6]) "
			"tau_and_ptau_debug(double[6]) potential_and_energy."
			"#rank: file name ...+_tau+int means tau=lambda(0), mu(1), nu(2); "
			"in each file, leaf_up: each column_info in orbits, then leaf_down";
		fprintf(fp, "%s\n", file_info.data());
		// for(auto a:STAGE.PT){
		// 	VecDoub XV1 = {a.Pos[0], a.Pos[1], a.Pos[2], 
		// 		a.Vel[0], a.Vel[1], a.Vel[2] }; //xv 6
		for(int i=0;i<count_maxTimeStep;i++){
			VecDoub XV1 = {STAGE.PT[i].Pos[0], STAGE.PT[i].Pos[1], STAGE.PT[i].Pos[2], 
				STAGE.PT[i].Vel[0], STAGE.PT[i].Vel[1], STAGE.PT[i].Vel[2] }; //xv 6
			VecDoub TD1 = STAGE.CS->xv2tau(XV1); //tau dot_tau 6
			VecDoub PS1 = STAGE.CS->tau2PSQ(TD1); //tau and PSQ 6
			VecDoub TP1 = STAGE.CS->xv2tp(XV1); //tau p_tau 6
			
			for(auto aa:XV1) fprintf(fp, "%e ", aa); //from index 0
			fprintf(fp, "    ");
			for(auto aa:TD1) fprintf(fp, "%e ", aa); //from index 6, 6+3
			fprintf(fp, "    ");
			for(auto aa:PS1) fprintf(fp, "%e ", aa); //from index 12+3
			fprintf(fp, "    ");
			for(auto aa:TP1) fprintf(fp, "%e ", aa); //from index 18+3
			fprintf(fp, "    ");

			//??
			fprintf(fp, "%e ", STAGE.CS->alpha()); //from index 18+3
			fprintf(fp, "%e ", STAGE.CS->beta()); //from index 18+3
			fprintf(fp, "%e ", STAGE.CS->gamma()); //from index 18+3
			fprintf(fp, "    ");
			fprintf(fp, "%e ", STAGE.PT[i].Time); //from index 18+3
			fprintf(fp, "    ");
			// // double P = STAGE.potential_t(XV1, t0_one, ID_one);
			// double P = i->P[ID].Pot;
			// double E = P +0.5*(XV1[3]*XV1[3]+XV1[4]*XV1[4]+XV1[5]*XV1[5]);
			// fprintf(fp, "%e %e ", P, E);
			fprintf(fp, "\n");
			// if( !( PS1[3]>0 && PS1[4] && PS1[5]>0 ) ){
			// 	DEBUG_PRINT_V1d(2, PS1, "PS1");
			// }
		}
		fclose(fp);
		printf("Write file %s ... done.\n", wt_fname_dbg);
	}
*/

/*
	////3. small funcs
	//C++ malloc(): invalid size (unsorted)
	string st = "./input_ME.txt";
	//expect: 5000, 12, 12, -1, 1., 0.001, 1000., false, true, true
	auto a = read_pure_number(st, 10);
	std::cout<<a.size()<<" "<<(bool)a[9]<<" "<<a[-1]
		<<" "<<a.at(a.size()-1)<<"\n";
*/

/*
	////2. set
	// SnapshotVec SV;
	// pSnapshotVec SS;
	// SnapshotVec SV = SnapshotVec(N_load);
	// pSnapshotVec SS = pSnapshotVec(N_load);
	string path_work = get_workpath();
	string path_base = path_work+"0prog/gadget/Gadget-2.0.7/galaxy_general/";
	string path_IC = "IC_param.txt";
	Stage STAGE(path_base, path_IC);
	Stage* pSTAGE = &STAGE;

	// //write all //do not run
	// pSTAGE->write_particleData_byStep(t_init, t_final, dt_step, dt_step, 
	// 	is_writeSnapshot, is_affineTransformation, -1, 300, -1);



	////calculate
DEBUG_PRINT_I(3);
	pSTAGE->load_multi_snapshots(t_init, t_final, dt_load, dt_step, 1, 0);
	double Alpha = TACT_semiaxis_Alpha, Beta = TACT_semiaxis_Beta;
	printf("Time of loaded_snapshot 0 = %e.\n", pSTAGE->SS[1]->P[1].Time);

DEBUG_PRINT_I(4);
	for(int i=0;i<N_ptcs;i++){
		int ID = i+ID_start;
		pSTAGE->search_orbitPesudoPeriod(ID, t_start_run, Alpha, Beta, 10000);
		auto bot = pSTAGE->get_badOrbitType();
		DEBUG_PRINT_V0d(10, bot, "bot");
		auto NPP = pSTAGE->get_countSamples_orbitApproxPeriod();
		DEBUG_PRINT_V1d(1, NPP[0], "NPP0");
		DEBUG_PRINT_V1d(1, NPP[1], "NPP1");
		DEBUG_PRINT_V1d(1, NPP[2], "NPP2");
		auto AA = pSTAGE->integrate_actions(0, 0);
		DEBUG_PRINT_V1d(1, AA, "AA");
		pSTAGE->write_orbitApproxPeriod();
		pSTAGE->reset_orbitdata();
	}
*/

/*
	for(double t0=t_start_run;t0<t_end_run;t0+=dt_run){ //time
		int s = t_to_loadsnapshot(t0, dt_load, t_init, 0.); //load
		int snapshot = pSTAGE->SS[s]->snap; //snapshot

		////kernel rho and fJ
		std::array<std::array<double, 3>, N_total> xdata0, xdata1; //dynamic tree?? //x-tree and J-tree
		wtfh = (struct write_firsthand *) malloc(sizeof(struct write_firsthand)*(N_total+64));
		wtsh = (struct write_secondhand *) malloc(sizeof(struct write_secondhand)*(N_total+64));
		int WhatCannonical;

		WhatCannonical = 2;
		pSTAGE->SS[s]->load_to_firsthand_from_PD();
		for(int i=0;i<N_total;i++){
			int iP = i+1;
			xdata0[i][0] = pSTAGE->SS[s]->P[iP].Pos[0];
			xdata0[i][1] = pSTAGE->SS[s]->P[iP].Pos[1];
			xdata0[i][2] = pSTAGE->SS[s]->P[iP].Pos[2];
		}
		KDtree<double, N_total, Dimension> kdt0(&xdata0);
		pSTAGE->SS[s]->loadtree(&kdt0);
		pSTAGE->SS[s]->write_secondhand_all_txt(WhatCannonical, 0, 0, 0);
		// pSTAGE->SS[s]->remove_tree(); //then after {}, the array will be release
		free(wtfh);
		free(wtsh);

		WhatCannonical = 5; //WhatCannonical
		for(int WP=0;WP<2;WP++){ //WhatPotential
			for(int WS=0;WS<3;WS++){ //WhatSymmetry
				for(int WA=0;WA<3;WA++){ //WhatActionmethod
					wtfh = (struct write_firsthand *) malloc(sizeof(struct write_firsthand)*(N_total+64));
					wtsh = (struct write_secondhand *) malloc(sizeof(struct write_secondhand)*(N_total+64));
					int isExistFile = pSTAGE->SS[s]->read_firsthand_all_txt(WhatCannonical, WP, WS, WA);
					if(isExistFile!=0){
						continue;
					}
					for(int i=0;i<N_total;i++){
						xdata1[i][0] = wtfh[i].QP[3]; xdata1[i][1] = wtfh[i].QP[4]; xdata1[i][2] = wtfh[i].QP[5]; //J-tree
					}
					KDtree<double, N_total, Dimension> kdt1(&xdata1);
					pSTAGE->SS[s]->loadtree1(&kdt1);
					pSTAGE->SS[s]->write_secondhand_all_txt(WhatCannonical, WP, WS, WA); //char* info
					// pSTAGE->SS[s]->remove_tree();
					free(wtfh);
					free(wtsh);
				}
			}
		}

		free(Write_aa);
	}
*/



	printf("\nDone.\n");
	return 0;
}