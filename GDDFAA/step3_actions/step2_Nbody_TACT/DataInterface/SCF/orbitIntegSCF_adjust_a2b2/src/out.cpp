//// main
extern "C"{
#include"F_to_C.h"
}
#include"adjust_foci.h"
#include"potential_wrapper.h"
#include<time.h>
// #include<iostream>
// #include<iomanip>
#include<mpi.h>
using namespace std;
using namespace UTILITIES;

const double r_range_min = 5e-1; //1e1 nan while 1e1+0.1 not
const double r_range_max = 2.e2;
// const double r_range_min = 9e0+0.1;
// const double r_range_max = 1.5e2+0.1;
// const int N_grid = 4; //debug
const int N_grid = 32; //used

int main(int argc, char* argv[]){

	////MPI settings
	int my_rank, comm_sz;
    MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    //// prepare
    SCFORB::get_parameter_(); //note: to get parameters about SCF potential
    double delta_y = (r_range_max-r_range_min)/(N_grid);

    // struct data_debug_sendrecv* Write_aa;
    // Write_aa = (struct data_debug_sendrecv *) malloc(sizeof(struct data_debug_sendrecv)*(N_grid));

    ////MPI distribution
    int N_task = N_grid;
    MPI_distribution_assistant MDS(N_task, comm_sz);
    // MDS.distribute_by_direct_cutting();
    MDS.distribute_by_give_in_turn();
    // MDS.print_tasks_IDs();
    auto tasks_IDs = MDS.get_tasks_IDs();

    // int remainder = N_grid%(comm_sz);
    // int interval = N_grid/comm_sz;
    // int local_from, local_count, total_count = 0;

    ////MPI ranks
    if(my_rank==0){ //rank main

        clock_t t_run_start, t_run_end;
        double t_run;
        t_run_start = clock();

        //// distribute
        MDS.print_tasks_IDs();
        MDS.print_somerank_info(my_rank);

        struct data_debug_sendrecv* Write_aa;
        int N_struct_malloc = MDS.data_struct_malloc_count(my_rank);
        Write_aa = (struct data_debug_sendrecv *) malloc(sizeof(struct data_debug_sendrecv)*(N_struct_malloc));
        //[learn code] Do not use vector which has not fixed size
        // malloc globally; or malloc each and reset index of calculating and sending in unmain rank, like the below line
        // Write_aa = (struct data_debug_sendrecv *) malloc(sizeof(struct data_debug_sendrecv)*(my_count));

        // int my_from = interval*(comm_sz-1);
        // int my_count = N_grid - my_from;
        // int my_endout = my_from + my_count;
        // if(my_count>interval*2){
        //     std::cerr<<"Bad task distribution because of too many ranks, which lead "
        //         "the count of tasks in rank 0 is much more than the count of tasks in rank other.\n";
        // }
        // printf("MPI tasks distribution: total number of tasks to do is %d, common size is %d, task interval is %d, "
        //     "the remainder is %d, count_rank_0 = %d.\n", 
        //     N_grid, comm_sz, interval, remainder, my_count);
        // printf("my_rank is %d: zero my_count = %d, my_from = %d, my_endout = %d.\n", 0, my_count, my_from, my_endout);
        
        //// tasks of my_rank
        printf("Now actions calculation ...\n\n");
        int my_task_entered = 0;
        for(int i : tasks_IDs[my_rank]){ //note here the i is global task ID instead of loop index
            int idataloop = MDS.data_struct_tmp_index_locally(my_task_entered, my_rank);
            my_task_entered += 1;

			//[] global task i begin
            //// 1. prepare
            string orbitdata_num_b2 = "orbit/orbit_"+to_string(i)+"_b2.dat"; // orbit data (it will be rewritten for each energy)
            string orbitdata_num_a2 = "orbit/orbit_"+to_string(i)+"_a2.dat";
            double D1 = 0., D2 = 0.;
            char Fname_orbit_num_b2[50], Fname_orbit_num_a2[50];
            sprintf(Fname_orbit_num_b2, "%s", orbitdata_num_b2.data());
            sprintf(Fname_orbit_num_a2, "%s", orbitdata_num_a2.data());
            std::cout<<Fname_orbit_num_b2<<", "<<Fname_orbit_num_a2<<"\n";
            
            //Always start from a clean orbit file so we don't append
            //\ on top of any previous run (which may contain fat lines).
            //\ C's remove() is already available via stdio.h from F_to_C.h.
            remove(orbitdata_num_b2.c_str());
            remove(orbitdata_num_a2.c_str());

            SCFAEF<3> SCFAEF_O; //note: wrapper class to get foci~energy table
            double ef[N_var];
            double yy = r_range_min+delta_y*i; //note: y cut of each time (corresponded to an energy)
            cout<<"\nAt grid "<<i<<", yy = "<<yy<<":\n";

            //// 2. b2
            //[step]1: get closed orbit and search the min-max axis for foci
            // remove(orbitdata.data());
            SCFORB::__orbintegz_MOD_main_for_orbit_integrating(&yy, Fname_orbit_num_b2); //note: wrapper to get the closed orbit, from Fortran
            SCFAEF_O.load_orbit_data(Fname_orbit_num_b2, true); //note: load orbit data; this do not influence variable in main_1_b2() when reading data
            D2 = SCFAEF_O.foci_by_shape(1); //note: search the min-max axis for foci, if no result of main_1_b2
            SCFAEF_O.main_1_b2(ef, Fname_orbit_num_b2); //foci
            std::cout<<"\n\n\n\nef1: "<<ef[0]<<" "<<ef[1]<<" "<<D2<<"\n";
            // copy_file(orbitdata.data(), orbitdata_num_b2.data()); //cp Orbit.dat Orbit_grid_${i}_a.dat
            
            //[step]2: get energy of that y cut and prepare to write
            double x0b = SCFAEF_O.xyz_grid[0][0], //?? what energy
            x1b = SCFAEF_O.xyz_grid[0][1],
            x2b = SCFAEF_O.xyz_grid[0][2];
            double vx0b = SCFAEF_O.vvv_grid[0][0], 
            vx1b = SCFAEF_O.vvv_grid[0][1], 
            vx2b = SCFAEF_O.vvv_grid[0][2];
            double x012b[] = {x0b, x1b, x2b};
            double potb;
            SCFORB::get_pot_(x012b, &potb);
            double energyb = 0.5*l2norm_real<double>({vx0b,vx1b,vx2b})+potb;
            std::cout<<"energyb: "<<energyb<<"\n";
            SCFAEF_O.reset_grid();
            
            //// 3. a2
            //[step]1: get closed orbit and search the min-max axis for foci
            // remove(orbitdata.data());
            SCFORB::__orbintegx_MOD_main_for_orbit_integrating(&yy, Fname_orbit_num_a2);
            SCFAEF_O.load_orbit_data(Fname_orbit_num_a2, true);
            D1 = SCFAEF_O.foci_by_shape(0);
            SCFAEF_O.main_2_a2(ef, Fname_orbit_num_a2); //foci
            std::cout<<"\n\n\n\nef2: "<<ef[0]<<" "<<ef[1]<<" "<<D1<<"\n";
            // copy_file(orbitdata.data(), orbitdata_num_a2.data());

            //[step]2: get energy of that y cut and prepare to write
            double x0 = SCFAEF_O.xyz_grid[0][0], //?? what energy
            x1 = SCFAEF_O.xyz_grid[0][1],
            x2 = SCFAEF_O.xyz_grid[0][2];
            double vx0 = SCFAEF_O.vvv_grid[0][0], 
            vx1 = SCFAEF_O.vvv_grid[0][1], 
            vx2 = SCFAEF_O.vvv_grid[0][2];
            double x012[] = {x0, x1, x2};
            double pot;
            SCFORB::get_pot_(x012, &pot);
            double energy = 0.5*l2norm_real<double>({vx0,vx1,vx2})+pot;
            std::cout<<"energy: "<<energy<<"\n";
            SCFAEF_O.reset_grid();
            
            double b2m_minvar = ef[0], a2m_minvar = ef[1];
            double b2m = -D1-1., a2m = -D2+b2m;
            printf("%le %le %le; %le %le; %le %le \n", energy, b2m_minvar, a2m_minvar, b2m, a2m, yy, pot);
            printf("%le %le %le; %le %le %le \n", x0, x1, x2, vx0, vx1, vx2);

            //// 4. record
            Write_aa[idataloop].value_double[0] = energy;       //energy of best foci orbit //note that energyb~~energy so we use energy only
            Write_aa[idataloop].value_double[1] = b2m_minvar;   //-b^2 by minvar
            Write_aa[idataloop].value_double[2] = a2m_minvar;   //-a^2 by minvar
            Write_aa[idataloop].value_double[3] = -1.;          //-c^2
            Write_aa[idataloop].value_double[4] = yy;           //y_cut
            Write_aa[idataloop].value_double[5] = b2m;          //-b^2 by shape
            Write_aa[idataloop].value_double[6] = a2m;          //-a^2 by shape
            Write_aa[idataloop].value_double[7] = 0.;           //nothing
			//[] global task i end

            // //[] debug task i begin
            // Write_aa[idataloop].value_double[0] = i;       //energy of best foci orbit //note that energyb~~energy so we use energy only
            // Write_aa[idataloop].value_double[1] = i;   //-b^2 by minvar
            // Write_aa[idataloop].value_double[2] = i;   //-a^2 by minvar
            // Write_aa[idataloop].value_double[3] = i;          //-c^2
            // Write_aa[idataloop].value_double[4] = i;           //y_cut
            // Write_aa[idataloop].value_double[5] = i;          //-b^2 by shape
            // Write_aa[idataloop].value_double[6] = i;          //-a^2 by shape
            // Write_aa[idataloop].value_double[7] = i;           //nothing
            // //[] debug task i end

            // std::cerr<<"task_"<<i<<"("<<my_rank<<", "<<(double)(i-my_from)/interval<<") ";
            std::cerr<<"task_"<<i<<" ";
        }
		// total_count += my_count;
	    MPI_Barrier(MPI_COMM_WORLD);

        //// MPI sendrecv
        for(int source=1;source<comm_sz;source++){ //recive from unmain rank
            /* comparations:
            at rank other (send):	~		at rank 0 (recive):
            my_rank							source
            my_from							local_from
            my_count						local_count
            my_sum							local_sum
            &S[my_from]						&SS[local_from]
            : the struct vector is recollectived */
            // local_from = interval*(source-1);
            // local_count = interval;
            // total_count += local_count;

            int local_from = MDS.data_struct_local_from(source);
            int local_count = MDS.data_struct_local_count(source);
            printf("data recv: local_from = %d, local_count = %d\n", local_from, local_count);
            MPI_Recv(
                &Write_aa[local_from], sizeof(struct data_debug_sendrecv)*local_count, 
                MPI_BYTE, source, 0,MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        }
        std::cout<<"Calculation, done.\n";

        //// record
        string filename = "some_lmn_foci_Pot.txt";
        //\ note: [filename] is the output file of the foci~energy table for actions, 
        //\ each line is "#energy alpha beta delta1 delta2 vy yy "
        int N_eachline = 8;
        write_data_debug(Write_aa, N_grid, N_eachline, filename);
	    free(Write_aa);

        t_run_end = clock();
        t_run = (double)(t_run_end-t_run_start)/1e6;
        std::cout<<"Time used: "<<t_run<<" sec.\n";

    }else{ //rank unmain //?? rank 1 sagment fault

        MDS.print_somerank_info(my_rank);
        // int my_from = interval*(my_rank-1);
        // int my_count = interval;
        // int my_endout = my_from + my_count;
        // printf("my_rank is %d: here my_count = %d, my_from = %d, my_endout = %d.\n", 
        //     my_rank, my_count, my_from, my_endout);

        struct data_debug_sendrecv* Write_aa;
        int N_struct_malloc = MDS.data_struct_malloc_count(my_rank);
        Write_aa = (struct data_debug_sendrecv *) malloc(sizeof(struct data_debug_sendrecv)*(N_struct_malloc));

        //// task of my_rank
        printf("Now actions calculation ...\n\n");
        int my_task_entered = 0;
        for(int i : tasks_IDs[my_rank]){ //note here the i is global task ID instead of loop index
            int idataloop = MDS.data_struct_tmp_index_locally(my_task_entered, my_rank);
            my_task_entered += 1;

			//[] global task i begin
            //// 1. prepare
            string orbitdata_num_b2 = "orbit/orbit_"+to_string(i)+"_b2.dat"; // orbit data (it will be rewritten for each energy)
            string orbitdata_num_a2 = "orbit/orbit_"+to_string(i)+"_a2.dat";
            double D1 = 0., D2 = 0.;
            char Fname_orbit_num_b2[50], Fname_orbit_num_a2[50];
            sprintf(Fname_orbit_num_b2, "%s", orbitdata_num_b2.data());
            sprintf(Fname_orbit_num_a2, "%s", orbitdata_num_a2.data());
            std::cout<<Fname_orbit_num_b2<<", "<<Fname_orbit_num_a2<<"\n";
            
            //Always start from a clean orbit file so we don't append
            //\ on top of any previous run (which may contain fat lines).
            //\ C's remove() is already available via stdio.h from F_to_C.h.
            remove(orbitdata_num_b2.c_str());
            remove(orbitdata_num_a2.c_str());

            SCFAEF<3> SCFAEF_O; //note: wrapper class to get foci~energy table
            double ef[N_var];
            double yy = r_range_min+delta_y*i; //note: y cut of each time (corresponded to an energy)
            cout<<"\nAt grid "<<i<<", yy = "<<yy<<":\n";

            //// 2. b2
            //[step]1: get closed orbit and search the min-max axis for foci
            // remove(orbitdata.data());
            SCFORB::__orbintegz_MOD_main_for_orbit_integrating(&yy, Fname_orbit_num_b2); //note: wrapper to get the closed orbit, from Fortran
            SCFAEF_O.load_orbit_data(Fname_orbit_num_b2, true); //note: load orbit data; this do not influence variable in main_1_b2() when reading data
            D2 = SCFAEF_O.foci_by_shape(1); //note: search the min-max axis for foci, if no result of main_1_b2
            SCFAEF_O.main_1_b2(ef, Fname_orbit_num_b2); //foci
            std::cout<<"\n\n\n\nef1: "<<ef[0]<<" "<<ef[1]<<" "<<D2<<"\n";
            // copy_file(orbitdata.data(), orbitdata_num_b2.data()); //cp Orbit.dat Orbit_grid_${i}_a.dat
            
            //[step]2: get energy of that y cut and prepare to write
            double x0b = SCFAEF_O.xyz_grid[0][0], //?? what energy
            x1b = SCFAEF_O.xyz_grid[0][1],
            x2b = SCFAEF_O.xyz_grid[0][2];
            double vx0b = SCFAEF_O.vvv_grid[0][0], 
            vx1b = SCFAEF_O.vvv_grid[0][1], 
            vx2b = SCFAEF_O.vvv_grid[0][2];
            double x012b[] = {x0b, x1b, x2b};
            double potb;
            SCFORB::get_pot_(x012b, &potb);
            double energyb = 0.5*l2norm_real<double>({vx0b,vx1b,vx2b})+potb;
            std::cout<<"energyb: "<<energyb<<"\n";
            SCFAEF_O.reset_grid();
            
            //// 3. a2
            //[step]1: get closed orbit and search the min-max axis for foci
            // remove(orbitdata.data());
            SCFORB::__orbintegx_MOD_main_for_orbit_integrating(&yy, Fname_orbit_num_a2);
            SCFAEF_O.load_orbit_data(Fname_orbit_num_a2, true);
            D1 = SCFAEF_O.foci_by_shape(0);
            SCFAEF_O.main_2_a2(ef, Fname_orbit_num_a2); //foci
            std::cout<<"\n\n\n\nef2: "<<ef[0]<<" "<<ef[1]<<" "<<D1<<"\n";
            // copy_file(orbitdata.data(), orbitdata_num_a2.data());

            //[step]2: get energy of that y cut and prepare to write
            double x0 = SCFAEF_O.xyz_grid[0][0], //?? what energy
            x1 = SCFAEF_O.xyz_grid[0][1],
            x2 = SCFAEF_O.xyz_grid[0][2];
            double vx0 = SCFAEF_O.vvv_grid[0][0], 
            vx1 = SCFAEF_O.vvv_grid[0][1], 
            vx2 = SCFAEF_O.vvv_grid[0][2];
            double x012[] = {x0, x1, x2};
            double pot;
            SCFORB::get_pot_(x012, &pot);
            double energy = 0.5*l2norm_real<double>({vx0,vx1,vx2})+pot;
            std::cout<<"energy: "<<energy<<"\n";
            SCFAEF_O.reset_grid();
            
            double b2m_minvar = ef[0], a2m_minvar = ef[1];
            double b2m = -D1-1., a2m = -D2+b2m;
            printf("%le %le %le; %le %le; %le %le \n", energy, b2m_minvar, a2m_minvar, b2m, a2m, yy, pot);
            printf("%le %le %le; %le %le %le \n", x0, x1, x2, vx0, vx1, vx2);

            //// 4. record
            Write_aa[idataloop].value_double[0] = energy;       //energy of best foci orbit //note that energyb~~energy so we use energy only
            Write_aa[idataloop].value_double[1] = b2m_minvar;   //-b^2 by minvar
            Write_aa[idataloop].value_double[2] = a2m_minvar;   //-a^2 by minvar
            Write_aa[idataloop].value_double[3] = -1.;          //-c^2
            Write_aa[idataloop].value_double[4] = yy;           //y_cut
            Write_aa[idataloop].value_double[5] = b2m;          //-b^2 by shape
            Write_aa[idataloop].value_double[6] = a2m;          //-a^2 by shape
            Write_aa[idataloop].value_double[7] = 0.;           //nothing
			//[] global task i end

            // //[] debug task i begin
            // Write_aa[idataloop].value_double[0] = i;       //energy of best foci orbit //note that energyb~~energy so we use energy only
            // Write_aa[idataloop].value_double[1] = i;   //-b^2 by minvar
            // Write_aa[idataloop].value_double[2] = i;   //-a^2 by minvar
            // Write_aa[idataloop].value_double[3] = i;          //-c^2
            // Write_aa[idataloop].value_double[4] = i;           //y_cut
            // Write_aa[idataloop].value_double[5] = i;          //-b^2 by shape
            // Write_aa[idataloop].value_double[6] = i;          //-a^2 by shape
            // Write_aa[idataloop].value_double[7] = i;           //nothing
            // //[] debug task i end

            // std::cerr<<"task_"<<i<<"("<<my_rank<<", "<<(double)(i-my_from)/interval<<") ";
            std::cerr<<"task_"<<i<<" ";
        }
	    MPI_Barrier(MPI_COMM_WORLD);

        //// MPI sendrecv
        int data_from = 0;
        // int data_from = MDS.data_struct_local_from(my_rank);
        int data_count = MDS.data_struct_local_count(my_rank);
        printf("data send: data_from = %d, data_count = %d\n", data_from, data_count);
        MPI_Send(
            &Write_aa[data_from], sizeof(struct data_debug_sendrecv)*data_count, 
            MPI_BYTE, 0, 0,MPI_COMM_WORLD
        );
	    free(Write_aa);
    
    }


    MPI_Finalize();
    return 0;
}

/*

*/

//// old
// double dd(double *a){
//     return a[1];
// }
// void main(){
//     double a[] = {0, 1.1, 2.2};
//     double dd_value = dd(a); //ok
//     std::cout<<"dd: "<<dd_value<<"\n";
// }

// int main(int argc, char* argv[])
// {
//     SCFORB::get_parameter_(); //note: to get parameters about SCF potential
    
//     //pot display
//     double poti, ai[Dim], adoti[Dim];
//     double xcar_target[] = {10., 1., -2.};
//     double vcar_target[] = {20., 20., 20.};
//     SCFORB::get_pot_(xcar_target, &poti);
//     SCFORB::get_force_(xcar_target, vcar_target, ai, adoti);
//     std::cout<<"poti    : "<<poti<<"\n";
//     std::cout<<"ai      : "<<ai[0]<<" "<<ai[1]<<" "<<ai[2]<<"\n";
//     std::cout<<"adoti   : "<<adoti[0]<<" "<<adoti[1]<<" "<<adoti[2]<<"\n";

//     double xcar_target1[] = {0., 0., 1.}; //nan for poti
//     // double xcar_target1[] = {1e-1, 1e-1, 1e0};
//     // double xcar_target1[] = {1e-56, 1e-56, 1e0};
//     // double xcar_target1[] = {1e-8, 1e-8, 1.}; //nan for ai and adoti
//     double vcar_target1[] = {20., 20., 20.};
//     SCFORB::get_pot_(xcar_target1, &poti);
//     SCFORB::get_force_(xcar_target1, vcar_target1, ai, adoti);
//     std::cout<<"poti    : "<<poti<<"\n";
//     std::cout<<"ai      : "<<ai[0]<<" "<<ai[1]<<" "<<ai[2]<<"\n";
//     std::cout<<"adoti   : "<<adoti[0]<<" "<<adoti[1]<<" "<<adoti[2]<<"\n";

//     const int ni = 4;
//     double x_samples[ni], y_samples[ni];
//     for(int i=0;i<ni;++i){
//         x_samples[i] = 1.*(i-1.);
//         y_samples[i] = sin(x_samples[i]);
//     }
//     double x_target = x_samples[1]+0.01, y_target, ydx_target;
//     int ni_tem = ni, m = 1; //[learn code] can not ocnvert "int const *" to "int *"; *a_form -- &a_conctent 
//     double tmp;
//     tmp = x_samples[1]; x_samples[1] = x_samples[2]; x_samples[2] = tmp;
//     tmp = y_samples[1]; y_samples[1] = y_samples[2]; y_samples[2] = tmp;
//     SCFORB::interpolate_3o_spline_1d_(x_samples, y_samples, &ni_tem, 
//         &x_target, &y_target, &ydx_target, &m);
//     std::cout<<"xt: "<<x_target<<"\n";
//     std::cout<<"yf: "<<sin(x_target)<<"\n";
//     std::cout<<"yt: "<<y_target<<"\n";
// }