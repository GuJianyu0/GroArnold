#include "DataInterface.h"

///// variables and functions
int     modelId;
char    modelInfo[MaxCharactersInString];
char    modelPath[MaxCharactersInString];
double  Mass_vir;
int     components;
int     N_allPtcs;
double  TACT_semiaxis_Alpha, TACT_semiaxis_Beta, TACT_semiaxis_Gamma = -1.;
double  flatx_comp[MaxGalaxyComponents], flaty_comp[MaxGalaxyComponents], flatz_comp[MaxGalaxyComponents];
int     N_comp[MaxGalaxyComponents];
int     type_comp[MaxGalaxyComponents];
double  frac_mass_comp[MaxGalaxyComponents];
double  m_target_comp[MaxGalaxyComponents];
double  softening_comp[MaxGalaxyComponents];
double	softening_type[6];
double  scale_length_comp[MaxGalaxyComponents];
double  spin_L[MaxGalaxyComponents];
double  other_params[MaxGalaxyComponents];
const VecDoub* const Substance_of_VecDoub_as_nullptr = nullptr;

struct write_angleaction* Write_aa, * Write_action_samples;
struct write_firsthand* wtfh;
struct write_secondhand* wtsh;
// struct AA_integrating_data;
// struct AA_motion_data;

int t_to_loadsnapshot(double t, double dt_load1, double t_init1, double t_alignment){
	return (int)round( (t - t_init1 - t_alignment)/dt_load1 ); //round(0.999...XX000...) -> 1.
}
double loadsnapshot_to_t(int snapshot, double dt_load1, double t_init1, double t_alignment){
	return snapshot/dt_load1 + t_init1 + t_alignment;
}
int t_to_snapshot(double t, double dt_step1, double t_init1, double t_alignment){
	return (int)round( (t - t_init1 - t_alignment)/dt_step1 );
}
double snapshot_to_t(int snapshot, double dt_step1, double t_init1, double t_alignment){
	return snapshot/dt_step1 + t_init1 + t_alignment;
}

double interp_linear_2d_2points(double x, double x1, double y1, double x2, double y2){
    if(abs(x1-x2)<Err){
        return (y1+y2)/2.;
    }else{
        return ( (x2-x)*y1 + (x-x1)*y2 )/(x2-x1);
    }
}

double parabola_2d_3points_y_x(double x, const VecDoub& points){
    double x1 = points[0], y1 = points[1], x2 = points[2], y2 = points[3], x3 = points[4], y3 = points[5];
    return (x-x1)*(x-x2)/((x3-x1)*(x3-x2))*y3 + (x-x1)*(x-x3)/((x2-x1)*(x2-x3))*y2 + (x-x2)*(x-x3)/((x1-x2)*(x1-x3))*y1;
}

void standardize_ellipcoor_range_with_axislength(double& tau, const int& swit, const double& alpha, const double& beta, const double& gamma){
    double t = tau;
    //lammda>-alpha>mu>-beta>nu>-gamma
    //first max, then min
    if(swit==0){ //lambda
        t = selectMax(t, -alpha);
    }else if(swit==1){ //mu
        t = selectMax(t, -beta);
        t = selectMin(t, -alpha);
    }else if(swit==2){ //nu
        t = selectMax(t, -gamma);
        t = selectMin(t, -beta);
    }else{
        printf("The coordinate swit value: 0,1,2. Not done, leap it.");
    }
    tau = t;
}

//read and write
void initialize_write_angleaction(struct write_angleaction& WAA){
    for(int j=0;j<2*Dim;j++){WAA.particle_xv0[j] = 0.;}
    WAA.particle_ID = -1;
    WAA.mass = 0.;
    for(int j=0;j<6;j++){WAA.particle_otherInfo[j] = 0.;}

    for(int j=0;j<LENGTH_ACTIONS_WRITE;j++){WAA.Value_Actions_SS_FP[j] = 0.;}
    for(int j=0;j<LENGTH_ACTIONS_WRITE;j++){WAA.Value_Actions_AF_FP[j] = 0.;}
    for(int j=0;j<LENGTH_ACTIONS_WRITE;j++){WAA.Value_Actions_TF_FP[j] = 0.;}
    for(int j=0;j<LENGTH_ACTIONS_WRITE;j++){WAA.Value_Actions_TEPPOD_FP[j] = 0.;}
    for(int j=0;j<LENGTH_ACTIONS_WRITE;j++){WAA.Value_Actions_GF_DP[j] = 0.;}

    for(int j=0;j<LENGTH_ACTIONS_WRITE;j++){WAA.Value_Actions_SS_DP[j] = 0.;}
    for(int j=0;j<LENGTH_ACTIONS_WRITE;j++){WAA.Value_Actions_AF_DP[j] = 0.;}
    for(int j=0;j<LENGTH_ACTIONS_WRITE;j++){WAA.Value_Actions_TF_DP[j] = 0.;}
    for(int j=0;j<LENGTH_ACTIONS_WRITE;j++){WAA.Value_Actions_TEPPOD_DP[j] = 0.;}
    for(int j=0;j<LENGTH_ACTIONS_WRITE;j++){WAA.Value_Actions_GF_DP[j] = 0.;}

    for(int j=0;j<LENGTH_ANGLESFREQUENCIES_WRITE;j++){WAA.Value_AnglesFrequencies_SS_FP[j] = 0.;}
    for(int j=0;j<LENGTH_ANGLESFREQUENCIES_WRITE;j++){WAA.Value_AnglesFrequencies_AF_FP[j] = 0.;}
    for(int j=0;j<LENGTH_ANGLESFREQUENCIES_WRITE;j++){WAA.Value_AnglesFrequencies_TF_FP[j] = 0.;}
    for(int j=0;j<LENGTH_ANGLESFREQUENCIES_WRITE;j++){WAA.Value_AnglesFrequencies_TEPPOD_FP[j] = 0.;}
    for(int j=0;j<LENGTH_ANGLESFREQUENCIES_WRITE;j++){WAA.Value_AnglesFrequencies_GF_DP[j] = 0.;}

    for(int j=0;j<LENGTH_ANGLESFREQUENCIES_WRITE;j++){WAA.Value_AnglesFrequencies_SS_DP[j] = 0.;}
    for(int j=0;j<LENGTH_ANGLESFREQUENCIES_WRITE;j++){WAA.Value_AnglesFrequencies_AF_DP[j] = 0.;}
    for(int j=0;j<LENGTH_ANGLESFREQUENCIES_WRITE;j++){WAA.Value_AnglesFrequencies_TF_DP[j] = 0.;}
    for(int j=0;j<LENGTH_ANGLESFREQUENCIES_WRITE;j++){WAA.Value_AnglesFrequencies_TEPPOD_DP[j] = 0.;}
    for(int j=0;j<LENGTH_ANGLESFREQUENCIES_WRITE;j++){WAA.Value_AnglesFrequencies_GF_DP[j] = 0.;}
}

int write_data_debug(const vector<struct data_debug>& vdd, string filename)
{
    char wt_fname[MaxCharactersInString];
    sprintf(wt_fname, "%s", filename.data());
    FILE* fp = fopen(wt_fname, "w");
    if(fp==nullptr){
        printf("Cannot open file %s. Now exit.\n", wt_fname);
        exit(0);
    }
    string data_descreption = "##write: column: xv(double[6]) Potential1 Forces11 12 13 "
        "Potential2 ... Potential3 ... Potential3 ... "
        "#not used: tau_and_metric(double[6]) tau_and_dottau(double[6]) "
        "tau_and_ptau(double[6]) tau_and_ptau_debug(double[6]) potential_and_energy.";
    fprintf(fp, "%s\n", data_descreption.data());
    for(auto d : vdd){
        for(auto a:d.xv){fprintf(fp, "%le ", a);} //count 6
        fprintf(fp, "    ");
        for(auto a:d.value_double){fprintf(fp, "%le ", a);} //main data
        fprintf(fp, "    ");
        for(auto a:d.info){fprintf(fp, "%le ", a);} //count 6 default
        // fprintf(fp, "    ");
        // for(auto a:d.value_int){fprintf(fp, "%d ", a);} //count 0 default
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Write debug file %s ... done.\n", wt_fname);
    return 0;
}

/*  Parameter file for initialization of TACT class MultipoleExpansion. 
    One can read "%le" one by one. The order of paramters should be correct. 
    There cannot be strings in file, while there can be "\n" in file. 
*/
VecDoub read_pure_number(string filename, int N_v){
    FILE *fp = nullptr;
    char wt_fname[MaxCharactersInString];
    sprintf(wt_fname, "%s", filename.data());
    fp = fopen(wt_fname, "r");
    if(fp==nullptr){
        printf("Cannot open file \"%s\".\n", wt_fname);
        return {-1};
    }
    VecDoub a;
    int i = 0;
    double b = 0.;
    while(i<N_v){
        fscanf(fp, "%le", &b);
        a.push_back(b);
        i++;
    }
    // while(!feof(fp){i++}
DEBUG_PRINT_V1d(1, a, "read");
    fclose(fp);
    return a;
}

int read_action_samples(struct write_angleaction*& WAA, //the syntax need "struct ..."
    int snapshot, string path_gm_1, int N_specify) //??*& yinyong //??change to vector for security
{
    char wt_fname[MaxCharactersInString];
    sprintf(wt_fname, "%stxt/snapshot_%d.samples.txt", path_gm_1.data(), snapshot);
    
    FILE *fp = nullptr;
    fp = fopen(wt_fname, "r");
    if(fp==nullptr){
        printf("Cannot open file \"%s\".\n", wt_fname);
        return -1;
    }

    int N_samps = 1;
    if(N_specify>0){ //when specify the length
        N_samps = N_specify;
    }
    while(!feof(fp)){ //let the count of samples to be count of lines of the file
        SKIP_UNTILENDL(fp);
        N_samps++;
    }
    fclose(fp);
    WAA = new struct write_angleaction[N_samps];

    fp = fopen(wt_fname, "r"); //reopen
    double a[10];
    for(int i=0;i<N_samps;i++){
        fscanf(fp, "%le %le %le %le %le %le %le%*[^\n]%*c", 
            &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6]);
        SKIP_UNTILENDL(fp);

        initialize_write_angleaction(WAA[i]);
        for(int j=0;j<Dim*2;j++){
            WAA[i].particle_xv0[j] = a[j];
        }
        WAA[i].mass = a[6];
        // WAA[i].particle_otherInfo[0] = a[7];
    }
    fclose(fp);
    return N_samps;
}

// int write_action_data(const write_angleaction*& WAA, int snapshot, string path_gm_1, int N)
int write_action_data(const write_angleaction* WAA, int snapshot, string path_gm_1, int N, bool is_samples)
{
    char wt_fname[MaxCharactersInString];
    string smps = "";
    if(is_samples){smps = "_samples";}
    sprintf(wt_fname, "%saa/snapshot_%d.action%s.method_all.txt", path_gm_1.data(), snapshot, smps.data());

    string file_info = "##write: #xv(double[6]);     ID(int)     "
        "mass(double).                 "
        "Values of actions[3+1], frequecies[3] and angles[3], i.e. "
        "(From index 12) AA by Sperical, Axisymmetric, Triaxial Fudge in FPot; "
        "(From index 42) AA by TEPPOD(not provided) and O2GF in FPot; "
        "(From index 62) AA by Sperical, Axisymmetric, Triaxial Fudge in DPot; "
        "(From index 92) AA by TEPPOD and O2GF(not provided) in DPot. "
        "particle_other_info[6], i.e., particle type, formula potential at current position, "
        "data potential at current position, alpha, beta and gamma of when "
        "calculate current action by Triaxial Fudge in DPot. "
        "#The main index: ([0] ..., [12] ..., [42] ..., [72] ..., [92], ...); "
        "The totally count is 112.";
    std::ofstream outfile;
    outfile.open(wt_fname);

    outfile<<file_info.data()<<"\n";
    for(int i=0;i<N;i++){
        //xv
        for(auto aa : WAA[i].particle_xv0) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        outfile<<"    ";
        //other info
        outfile<<WAA[i].particle_ID<<" ";
        outfile<<WAA[i].mass<<" ";
        outfile<<"    ";
        outfile<<"    ";
        outfile<<"    ";
        outfile<<"    ";
        outfile<<"    ";
        outfile<<"    ";

        //AA B0
        for(auto aa : WAA[i].Value_Actions_SS_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_AnglesFrequencies_SS_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_Actions_AF_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_AnglesFrequencies_AF_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_Actions_TF_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_AnglesFrequencies_TF_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        outfile<<"    ";

        //AA B1
        for(auto aa : WAA[i].Value_Actions_TEPPOD_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_AnglesFrequencies_TEPPOD_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_Actions_GF_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_AnglesFrequencies_GF_FP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        outfile<<"    ";
        outfile<<"    ";
        outfile<<"    ";
        outfile<<"    ";
        outfile<<"    ";

        //AA B2
        for(auto aa : WAA[i].Value_Actions_SS_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_AnglesFrequencies_SS_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_Actions_AF_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_AnglesFrequencies_AF_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_Actions_TF_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_AnglesFrequencies_TF_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        outfile<<"    ";

        //AA B3
        for(auto aa : WAA[i].Value_Actions_TEPPOD_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_AnglesFrequencies_TEPPOD_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_Actions_GF_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        for(auto aa : WAA[i].Value_AnglesFrequencies_GF_DP) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"    ";
        outfile<<"    ";
        outfile<<"    ";
        outfile<<"    ";
        outfile<<"    ";
        for(auto aa : WAA[i].particle_otherInfo) outfile<<scientific<<setprecision(6)<<aa<<" ";
        outfile<<"#endl\n";
    }
    outfile.close();

    printf("\nWrite actions to \"%s\", done.\n\n\n", wt_fname);
    return 0;
}

int write_value_txt(double** valuestruct, int N_writeline, int N_variables, char* fname, int tag){
	char wt_fname[200];
	sprintf(wt_fname, "./0DEBUG/DAA_%s.debug.txt", fname);
	FILE* fp = fopen(wt_fname, "w");
	if(fp==nullptr){
		printf("Cannot open file, exit.\n");
        exit(0);
	}
	fprintf(fp, "##DAA\n");
	// for(auto i:*(double (*)[10]) valuestruct){}
	for(int l=0;l<N_writeline;l++){
	    for(int v=0;v<N_variables;v++){
            fprintf(fp, "%e ", valuestruct[l][v]);
        }
        fprintf(fp, "\n");
	}
	printf("Write file %s ... done.\n", wt_fname);
	fclose(fp);
	return 1;
}

void combine_actions_output(string f, string f1, string f2, int during_replace, int NP){

    int NPP;
    if(NP>N_allPtcs){
        printf("Too many particles to be more than total count, set it as N_allPtcs.\n");
        NPP = N_allPtcs+1;
    }else{
        NPP = NP+1;
    }

    const char* wt_fname_dbg = f.data();
    // FILE* fp = fopen(f.data(), "w");
    FILE* fp = fopen(wt_fname_dbg, "w");
    if(fp==nullptr){
        printf("Cannot open file %s. Donot write.\n", wt_fname_dbg);
        exit(0);
    }
    fprintf(fp, "##write\n");

    const char* rd_fname_dbg1 = f1.data();
    // FILE* fp1 = fopen(f1.data(), "r");
    FILE* fp1 = fopen(rd_fname_dbg1, "r");
    if(fp1==nullptr){
        printf("Cannot open file %s. Donot write.\n", rd_fname_dbg1);
        exit(0);
    }
    SKIP_UNTILENDL(fp1);

    const char* rd_fname_dbg2 = f2.data();
    // FILE* fp2 = fopen(f2.data(), "r");
    FILE* fp2 = fopen(rd_fname_dbg2, "r");
    if(fp2==nullptr){
        printf("Cannot open file %s. Donot write.\n", rd_fname_dbg2);
        exit(0);
    }
    SKIP_UNTILENDL(fp2);

    for(int i=1;i<NPP;i++){
        float v[100], v1[20], v2[20];
        int vd[100];
        fscanf(fp2, 
            "%*e %*e %*e %*e %*e %*e     %*d %*e %*e %*e %*e %*d     "
            "%e %e %e %e     %e %e %e %e %e %e %e %e %e %e %e "
            // "        " //donot need space count consistence
            "%*d %*d %*d %*d %*d %*d     %*e %*e %*e %*e %*e %*e %*e %*e %*e %*e %*e %*e\n", 
            &v2[0], &v2[1], &v2[2], &v2[3], &v2[4], &v2[5], &v2[6], &v2[7], &v2[8], &v2[9], 
            &v2[10], &v2[11], &v2[12], &v2[13], &v2[14]
        );
        fscanf(fp1, 
            "%e %e %e %e %e %e     %d     %e %e %e %e %e "
            // "                " //donot need space count consistence
            "%e %e %e     %e %e %e %e %e %e         "
            "%e %e %e     %e %e %e %e %e %e         "
            "%e %e %e %e     %e %e %e %e %e %e %e %e %e %e %e "
            // "                " //donot need space count consistence
            "%e %e %e     %e %e %e %e %e %e         "
            "%e %e %e     %e %e %e %e %e %e         "
            "%e %e %e %e     %e %e %e %e %e %e %e %e %e %e %e", 
            &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &vd[6], &v[7], &v[8], &v[9], 
            &v[10], &v[11], &v[12], &v[13], &v[14], &v[15], &v[16], &v[17], &v[18], &v[19], 
            &v[20], &v[21], &v[22], &v[23], &v[24], &v[25], &v[26], &v[27], &v[28], &v[29], 
            &v[30], &v[31], &v[32], &v[33], &v[34], &v[35], &v[36], &v[37], &v[38], &v[39], 
            &v[40], &v[41], &v[42], &v[43], &v[44], &v[45], &v[46], &v[47], &v[48], &v[49], 
            &v[50], &v[51], &v[52], &v[53], &v[54], &v[55], &v[56], &v[57], &v[58], &v[59], 
            &v[60], &v[61], &v[62], 
            &v1[0], &v1[1], &v1[2], &v1[3], &v1[4], &v1[5], &v1[6], &v1[7], &v1[8], &v1[9], //actions[]
            &v1[10], &v1[11], &v1[12], &v1[13], &v1[14] //actions[]
        );
        SKIP_UNTILENDL(fp1);
        // DEBUG_PRINT_V0d(1, v[0], "v[0]");
        // DEBUG_PRINT_V0d(1, v[1], "v[1]");
        // DEBUG_PRINT_V0d(2, v1[0], "v1[0]");
        for(int i=0;i<during_replace;i++){
            v1[i] = v2[i];
        }
        fprintf(fp, 
            "%e %e %e %e %e %e     %d     %e %e %e %e %e "
            "                "
            "%e %e %e     %e %e %e %e %e %e         "
            "%e %e %e     %e %e %e %e %e %e         "
            "%e %e %e %e     %e %e %e %e %e %e %e %e %e %e %e "
            "                "
            "%e %e %e     %e %e %e %e %e %e         "
            "%e %e %e     %e %e %e %e %e %e         "
            "%e %e %e "
            "%e     %e %e %e %e %e %e %e %e %e %e %e \n", 
            v[0], v[1], v[2], v[3], v[4], v[5], vd[6], v[7], v[8], v[9], 
            v[10], v[11], v[12], v[13], v[14], v[15], v[16], v[17], v[18], v[19], 
            v[20], v[21], v[22], v[23], v[24], v[25], v[26], v[27], v[28], v[29], 
            v[30], v[31], v[32], v[33], v[34], v[35], v[36], v[37], v[38], v[39], 
            v[40], v[41], v[42], v[43], v[44], v[45], v[46], v[47], v[48], v[49], 
            v[50], v[51], v[52], v[53], v[54], v[55], v[56], v[57], v[58], v[59], 
            v[60], v[61], v[62], 
            v1[0], v1[1], v1[2], v1[3], v1[4], v1[5], v1[6], v1[7], v1[8], v1[9], //actions[]
            v1[10], v1[11], v1[12], v1[13], v1[14] //actions[]
        );
    }

    fclose(fp);
    fclose(fp1);
    fclose(fp2);
    printf("Write file %s ... done.\n", wt_fname_dbg);
}

int read_params(string path_IC_1)
{
    //load and assign value
    FILE *fp=fopen(path_IC_1.data(), "r");
    if(!fp){
	    printf("Cannot open file \"%s\".\n", path_IC_1.data());
        exit(0);
        return -1;
    }

    //tell what values
    char nameAllParams[MaxClassParams][MaxCharactersInString];
    char nameAllParams_index[MaxClassParams][MaxGalaxyComponents][MaxCharactersInString];
    char nameAllParams_other[MaxClassParams][MaxCharactersInString];
    int sizeAllParams[MaxClassParams];
    int sizeAllParams_index[MaxClassParams][MaxGalaxyComponents];
    int sizeAllParams_other[MaxClassParams];
    const void* addrAllParams[MaxClassParams];
    const void* addrAllParams_index[MaxClassParams][MaxGalaxyComponents];
    const void* addrAllParams_other[MaxClassParams];
    int nt = 0, nt0 = 0;

    strcpy(nameAllParams[nt], "modelId");
    addrAllParams[nt] = &modelId; //GlobalActions.
    sizeAllParams[nt] = 1; //int
    nt0++, nt++;

    strcpy(nameAllParams[nt], "modelInfo");
    addrAllParams[nt] = &modelInfo;
    sizeAllParams[nt] = 0; //char*
    nt0++, nt++;

    strcpy(nameAllParams[nt], "modelPath");
    addrAllParams[nt] = &modelPath;
    sizeAllParams[nt] = 0; //char*
    nt0++, nt++;

    strcpy(nameAllParams[nt], "Mass_vir");
    addrAllParams[nt] = &Mass_vir;
    sizeAllParams[nt] = 2; //double
    nt0++, nt++;

    strcpy(nameAllParams[nt], "components");
    addrAllParams[nt] = &components;
    sizeAllParams[nt] = 1; //int
    nt0++, nt++;

    //TACT semiaxis; may be changed in TACT when needed
    strcpy(nameAllParams[nt], "TACT_semiaxis_Alpha");
    addrAllParams[nt] = &TACT_semiaxis_Alpha;
    sizeAllParams[nt] = 2; //double
    nt0++, nt++;
    strcpy(nameAllParams[nt], "TACT_semiaxis_Beta");
    addrAllParams[nt] = &TACT_semiaxis_Beta;
    sizeAllParams[nt] = 2; //double
    nt0++, nt++;
    strcpy(nameAllParams[nt], "TACT_semiaxis_Gamma");
    addrAllParams[nt] = &TACT_semiaxis_Gamma;
    sizeAllParams[nt] = 2; //double
    nt0++, nt++;
    
    //softening by particle type instead of components
    strcpy(nameAllParams[nt], "softening_type_gas");
    addrAllParams[nt] = &(softening_type[0]);
    sizeAllParams[nt] = 2; //double
    nt0++, nt++;
    strcpy(nameAllParams[nt], "softening_type_halo");
    addrAllParams[nt] = &(softening_type[1]);
    sizeAllParams[nt] = 2; //double
    nt0++, nt++;
    strcpy(nameAllParams[nt], "softening_type_disk");
    addrAllParams[nt] = &(softening_type[2]);
    sizeAllParams[nt] = 2; //double
    nt0++, nt++;
    strcpy(nameAllParams[nt], "softening_type_bulge");
    addrAllParams[nt] = &(softening_type[3]);
    sizeAllParams[nt] = 2; //double
    nt0++, nt++;
    strcpy(nameAllParams[nt], "softening_type_stars");
    addrAllParams[nt] = &(softening_type[4]);
    sizeAllParams[nt] = 2; //double
    nt0++, nt++;
    strcpy(nameAllParams[nt], "softening_type_bndry");
    addrAllParams[nt] = &(softening_type[5]);
    sizeAllParams[nt] = 2; //double
    nt0++, nt++;

    int nt11 = 0;
    for(int j=0;j<MaxGalaxyComponents;j++){ //the component index in the name are begin with 1 instead of 0
        char name_temp[MaxCharactersInString];
        int nt1 = 0;

        sprintf(name_temp, "N_comp%d", j + 1); //kaifa
        strcpy(nameAllParams_index[nt1][j], name_temp);
        addrAllParams_index[nt1][j] = &N_comp[j];
        sizeAllParams_index[nt1][j] = 1; //int
        nt1++, nt++;

        sprintf(name_temp, "type_comp%d", j + 1);
        strcpy(nameAllParams_index[nt1][j], name_temp);
        addrAllParams_index[nt1][j] = &type_comp[j];
        sizeAllParams_index[nt1][j] = 2; //double
        nt1++, nt++;

        sprintf(name_temp, "frac_mass_comp%d", j + 1);
        strcpy(nameAllParams_index[nt1][j], name_temp);
        addrAllParams_index[nt1][j] = &frac_mass_comp[j];
        sizeAllParams_index[nt1][j] = 2; //double
        nt1++, nt++;

        sprintf(name_temp, "softening_comp%d", j + 1);
        strcpy(nameAllParams_index[nt1][j], name_temp);
        addrAllParams_index[nt1][j] = &softening_comp[j];
        sizeAllParams_index[nt1][j] = 2; //double
        nt1++, nt++;

        sprintf(name_temp, "scale_length_comp%d", j + 1);
        strcpy(nameAllParams_index[nt1][j], name_temp);
        addrAllParams_index[nt1][j] = &scale_length_comp[j];
        sizeAllParams_index[nt1][j] = 2; //double
        nt1++, nt++;

        sprintf(name_temp, "spin_L%d", j + 1);
        strcpy(nameAllParams_index[nt1][j], name_temp);
        addrAllParams_index[nt1][j] = &spin_L[j];
        sizeAllParams_index[nt1][j] = 2; //double
        nt1++, nt++;

        sprintf(name_temp, "flatx_comp%d", j + 1);
        strcpy(nameAllParams_index[nt1][j], name_temp);
        addrAllParams_index[nt1][j] = &flatx_comp[j];
        sizeAllParams_index[nt1][j] = 2; //double
        nt1++, nt++;
        sprintf(name_temp, "flaty_comp%d", j + 1);
        strcpy(nameAllParams_index[nt1][j], name_temp);
        addrAllParams_index[nt1][j] = &flaty_comp[j];
        sizeAllParams_index[nt1][j] = 2; //double
        nt1++, nt++;
        sprintf(name_temp, "flatz_comp%d", j + 1);
        strcpy(nameAllParams_index[nt1][j], name_temp);
        addrAllParams_index[nt1][j] = &flatz_comp[j];
        sizeAllParams_index[nt1][j] = 2; //double
        nt1++, nt++;

        nt11 = nt1;
    }

    float f; int d; char s[MaxCharactersInString];
    char s0[MaxCharactersInString], leftLine[MaxCharactersInString];
    int maxline = 500;
    for(int i=1;i<maxline+1;i++){
        // int l = fscanf(fp, "%s", s0);
        // std::cout<<s0[0]<<std::endl;
        // printf("read line %d:\n", i);
        fscanf(fp, "%s %s", s0, s);
        if(s0[0]=='#' || s0[0]=='%' || s0[0]=='!' || (s0[0]=='/' && s0[1]=='/')){ //comment: skip
            fscanf(fp, "%[^\n]%*c", leftLine);
            // char c = '\0';
            // while(c!='\n') fscanf(fp, "%c", &c); //alter method to end line, bad
            // printf("notes line: first character = %s\n", s0);
            // printf("left characters: %s\n", leftLine);
        }else{ //variables: read value as certain type into certain variable

            for(int c=0;c<nt0;c++){ //c: nt0 or nt1
                if(!strcmp(s0, nameAllParams[c])){
                    // printf("Now find param %s.\n", s0);
                    if(sizeAllParams[c]==1)
                        *((int *)addrAllParams[c]) = atoi(s);
                    else if(sizeAllParams[c]==2)
                        *((double *)addrAllParams[c]) = atof(s);
                    else
                        strcpy((char*)addrAllParams[c], s);
                }
            }
            for(int c=0;c<nt11;c++){ //c: nt0 or nt1
                for(int j=0;j<MaxGalaxyComponents;j++){
                    if(!strcmp(s0, nameAllParams_index[c][j])){ //vary bad eff
                        // printf("Now find param_index %s.\n", s0);
                        if(sizeAllParams_index[c][j]==1)
                            *((int *)addrAllParams_index[c][j]) = atoi(s);
                        else if(sizeAllParams_index[c][j]==2)
                            *((double *)addrAllParams_index[c][j]) = atof(s);
                        else
                            strcpy((char*)addrAllParams_index[c][j], s);
                    }
                }
            }
            fscanf(fp, "%[^\n]%*c", leftLine);
            // printf("param line: name = %s, value = %s\n", s0, s);
            // printf("left characters: %s\n", leftLine);
        }
    }

    //derived params
    std::cout<<"\n\n\n\nsoftening_type[1]: "<<softening_type[1]<<"\n";
    int Nt = 0;
    for(int i=0;i<nt11;i++){
        Nt += N_comp[i];
        m_target_comp[i] = Mass_vir*frac_mass_comp[i]/N_comp[i];
    }
    N_allPtcs = Nt; //note: this is read from auto txt file instead of DICE .g1 file and it should same with NumPart
    for(int i=0;i<MaxGalaxyComponents;i++){
        other_params[i] = 0.; //to be adden...
    }
    fclose(fp);
	printf("Read file \"%s\" ... done.\n", path_IC_1.data());

	printf("Some initial parameters:\n");
    std::cout<<"model Id "<<modelId<<std::endl;
    std::cout<<"model Info: "<<modelInfo<<std::endl;
    std::cout<<"model Path: "<<modelPath<<std::endl;
    std::cout<<"total mass: "<<Mass_vir<<std::endl;
    std::cout<<"count of particles from txt: "<<N_allPtcs<<std::endl;
    std::cout<<"count of components: "<<components<<std::endl;
    std::cout<<"softening_type[0]: "<<softening_type[0]<<"kpc"<<std::endl;
    std::cout<<"scalelength[0]: "<<scale_length_comp[0]<<"kpc"<<std::endl;
    std::cout<<std::endl;
    return 0;
}



//// class member
int Stage::load_multi_snapshots(double t_init1, double t_final1, 
    double dt_load1, double dt_step1, int is_witeSnapshot, int is_preprocessed1)
{
    this->reset_SnapshotsData();
    this->set_info_snapshot(t_init1, t_final1, dt_load1, dt_step1);
    this->SV.resize(this->N_load); //??
    this->SS.resize(this->N_load);

DEBUG_PRINT_I(211);
    if(N_load<1){
        std::cout<<"(N_load<1), there is no snapshot to load, please check. Exit.\n";
        exit(0);
    }
	for(int s=0;s<this->N_load;s+=1){
		int ss = this->Initial_snapshot+s*this->Load_Step;
        double tt = this->t_init+s*this->dt_load;
        printf("load_%d, snapshot_%d, time_%e.\n", s, ss, tt);
		// SV[s].set_path(path_gm); //tree_pot //to set the folder where the data of a galaxy model is
		// SV[s].set_snapshot(ss, tt);
		if(!SV[s]) SV[s].reset(new Snapshot());
		SV[s]->set_path(path_gm); //to set the folder where the data of a galaxy model is
		SV[s]->set_snapshot(ss, tt);
DEBUG_PRINT_I(2111);
		// SV[s].load();
		SV[s]->load();
DEBUG_PRINT_I(2112);
        if(is_witeSnapshot==1){ //to write original simulation output
DEBUG_PRINT_I(2113);
            // SV[s].write_PD_txt();
            SV[s]->write_PD_txt();
        }
		if(is_preprocessed1==1){
DEBUG_PRINT_I(2114);
			// SV[s].preprocess();
			SV[s]->preprocess();
		}
		// SS[s] = &SV[s]; //each pointer
        SS[s] = SV[s].get(); //each pointer
	}

DEBUG_PRINT_I(213);
	double Alpha = TACT_semiaxis_Alpha, Beta = TACT_semiaxis_Beta, Gamma = -1.;
	this->set_ConfocalEllipsoidalCoordSys_focus(Alpha, Beta);
	return 0;
}



////==============================================
////            difination of class mumbers
////==============================================

int Snapshot::allocate_memory(void){

    printf("allocating memory...\n");
    if(!(P = (struct particle_data *) malloc(NumPart * sizeof(struct particle_data)))){
            fprintf(stderr, "failed to allocate memory.\n");
            exit(0);
    }
    P--;				/* start with offset 1 */

    if(!(Id = (int*) malloc(NumPart * sizeof(int)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(0);
    }
    Id--;				/* start with offset 1 */

    printf("allocating memory...done\n");
    return 0;
}

int Snapshot::load_snapshot(char *fname, int files){

    FILE *fd;
    char buf[MaxCharactersInString];
    int i, j, k, dummy, ntot_withmasses;
    int t, n, off, pc, pc_new, pc_sph;

    #define SKIP fread(&dummy, sizeof(dummy), 1, fd);

    for(i = 0, pc = 1; i < files; i++, pc = pc_new)
    {
        //header
        if(files > 1)
	        sprintf(buf, "%s.%d", fname, i);
        else
	        sprintf(buf, "%s", fname);

        if(!(fd = fopen(buf, "r")))
	    {
	        printf("can't open file `%s`\n", buf);
	        exit(0); //gjy changed
            // return 1;
	    }

        printf("reading `%s' ...\n", buf);
        fflush(stdout);

        fread(&dummy, sizeof(dummy), 1, fd);
        fread(&header1, sizeof(header1), 1, fd);
        fread(&dummy, sizeof(dummy), 1, fd);

        if(files == 1)
	    {
	        for(k = 0, NumPart = 0, ntot_withmasses = 0; k < 6; k++)
	            NumPart += header1.npart[k];
	        Ngas = header1.npart[0];
	    }
        else
	    {
	        for(k = 0, NumPart = 0, ntot_withmasses = 0; k < 6; k++)
	            NumPart += header1.npartTotal[k];
	        Ngas = header1.npartTotal[0];
	    }

        for(k = 0, ntot_withmasses = 0; k < 6; k++)
	    {
	        if(header1.mass[k] == 0)
	            ntot_withmasses += header1.npart[k];
	    }

        if(i == 0)
	        allocate_memory();


        ////particle data
        //positions
        //gjy note: P[].Pos begin
        SKIP;
        // printf("%ld %ld\n", sizeof(int), sizeof(float));
        // printf("pos0 dummy = %d\n", dummy);
        for(k = 0, pc_new = pc; k < 6; k++)
	    {
	        for(n = 0; n < header1.npart[k]; n++)
	        {
	            fread(&P[pc_new].Pos[0], sizeof(float), 3, fd); //gjy note: P[].Pos
	            pc_new++;
	        }
	    }
        SKIP;
        // printf("pos1 dummy = %d\n", dummy);
        //gjy note: P[].Pos end

        //velocities
        SKIP;
        // printf("vel0 dummy = %d\n", dummy);
        for(k = 0, pc_new = pc; k < 6; k++)
	    {
	        for(n = 0; n < header1.npart[k]; n++)
	        {
	            fread(&P[pc_new].Vel[0], sizeof(float), 3, fd); //gjy note: P[].Vel
	            pc_new++;
	        }
	        }
        SKIP;
        // printf("vel1 dummy = %d\n", dummy);

        //ID
        SKIP;
        // printf("id0  dummy = %d\n", dummy);
        for(k = 0, pc_new = pc; k < 6; k++)
	    {
	        for(n = 0; n < header1.npart[k]; n++)
	        {
	            fread(&Id[pc_new], sizeof(int), 1, fd);
	            pc_new++;
	        }
	    }
        SKIP;
        // printf("id1  dummy = %d\n", dummy);

        //mass //IO_MASS
        if(ntot_withmasses > 0)
	    SKIP;
        // printf("mas0 dummy = %d\n", dummy);
        for(k = 0, pc_new = pc; k < 6; k++)
	    {
	        for(n = 0; n < header1.npart[k]; n++)
	        {
	            P[pc_new].Type = k;

	            if(header1.mass[k] == 0)
		            fread(&P[pc_new].Mass, sizeof(float), 1, fd); //gjy note: P[].Mass if has mass, but type is ok before
	            else
		            P[pc_new].Mass = header1.mass[k];
	            pc_new++;
	        }
	    }
        if(ntot_withmasses > 0)
	    SKIP;
        // printf("mas1 dummy = %d\n", dummy);

        ////gas type
        //energy, density, Ne //or IO_U, IO_RHO, IO_HSML??
        if(header1.npart[0] > 0) //gjy note: It has been judged by this if() that whether gas and U,RHO,HSML exist.
	    {
	        SKIP;
            // printf("u0   dummy = %d\n", dummy); //4000 particles * 4 sizeof(int) = 16000 dummy, in data_20210314
	        for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
	        {
	            fread(&P[pc_sph].U, sizeof(float), 1, fd);
	            pc_sph++;
	        }
	        SKIP;
            // printf("u1   dummy = %d\n", dummy);

	        SKIP;
            // printf("rho0 dummy = %d\n", dummy);
	        for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
	        {
	            fread(&P[pc_sph].Rho, sizeof(float), 1, fd);
	            pc_sph++;
	        }
	        SKIP;
            // printf("rho1 dummy = %d\n", dummy);

	        if(header1.flag_cooling)
	        {
	            SKIP;
                // printf("hsm0 dummy = %d\n", dummy);
	            for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
		        {
		            fread(&P[pc_sph].Hsml, sizeof(float), 1, fd);
		            pc_sph++;
		        }
	            SKIP;
                // printf("hsm1 dummy = %d\n", dummy);
	        }
	        else //not by read
	            for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
	            {
		            P[pc_sph].Hsml = 1.0;
		            pc_sph++;
	            }

            //Ne??
	        SKIP;
            // printf("ne0  dummy = %d\n", dummy);
	        for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
	        {
	            fread(&P[pc_sph].Ne, sizeof(float), 1, fd);
	            pc_sph++;
	        }
	        SKIP;
            // printf("ne1  dummy = %d\n", dummy);
	    }

        //// all output particle data
        //potential
        SKIP;
        // printf("pot0 dummy = %d\n", dummy);
        for(k = 0, pc_new = pc; k < 6; k++)
	    {
	        for(n = 0; n < header1.npart[k]; n++)
	        {
	            fread(&P[pc_new].Pot, sizeof(float), 1, fd);
	            pc_new++;
	        }
	    }
        SKIP;
        // printf("pot1 dummy = %d\n", dummy);

        //acc
        SKIP;
        // printf("acc0 dummy = %d\n", dummy);
        for(k = 0, pc_new = pc; k < 6; k++)
	    {
	        for(n = 0; n < header1.npart[k]; n++)
	        {
	            fread(&P[pc_new].Acc[0], sizeof(float), 3, fd); //gjy note: P[].Vel
	            pc_new++;
	        }
	    }
        SKIP;
        // printf("acc1 dummy = %d\n", dummy);


        //IO_DTENTR?? dA/dt, A is the entropy of gas
	    SKIP;
        // printf("ent0 dummy = %d\n", dummy);
	    for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
	    {
	        fread(&P[pc_sph].dAdt, sizeof(float), 1, fd);
	        pc_sph++;
	    }
	    SKIP;
        // printf("ent1 dummy = %d\n", dummy);


        //IO_TSTP
        //gjy note: are these Metal and Age??
        //the order is inverted??
        SKIP;
        // printf("age1 dummy = %d\n", dummy);
        for(k = 0, pc_new = pc; k < 6; k++)
	    {
	        for(n = 0; n < header1.npart[k]; n++)
	        {
	            fread(&P[pc_new].Age, sizeof(float), 1, fd);
	            pc_new++;
	        }
	    }
        SKIP;
        // printf("age1 dummy = %d\n", dummy);

        //Metal??
        SKIP;
        // printf("met0 dummy = %d\n", dummy);
        for(k = 0, pc_new = pc; k < 6; k++)
	    {
	        for(n = 0; n < header1.npart[k]; n++)
	        {
	            fread(&P[pc_new].Metal, sizeof(float), 1, fd);
	            pc_new++;
	        }
	    }
        SKIP;
        // printf("met1 dummy = %d\n", dummy);
    }

    fclose(fd);

    Time = header1.time;
    Redshift = header1.time;
    return 0;
}

int Snapshot::reordering(void){

    int i, j;
    int idsource, idsave, dest;
    struct particle_data psave, psource;
    
    printf("reordering...\n");
    for(i = 1; i <= NumPart; i++)
    {
        if(Id[i] != i)
	    {
	        psource = P[i];
	        idsource = Id[i];
	        dest = Id[i];

	        do
	        {
	            psave = P[dest];
	            idsave = Id[dest];

	            P[dest] = psource;
	            Id[dest] = idsource;

	            if(dest == i)
		            break;

	            psource = psave;
	            idsource = idsave;

	            dest = idsource;
	        }
	        while(1);
	    }
    }
    printf("reordering...done\n");

    Id++;
    free(Id);

    printf("space for particle ID freed\n");
    return 0;
}



int Snapshot::write_gadget_ics_known(int ID_change, double rate, char pathload[MaxCharactersInString]){

    char fname[MaxCharactersInString];
    if(pathload!=nullptr) sprintf(fname, "%s%s_%03d", pathload, bname, snap);
    else sprintf(fname, "%s%s_%03d", path_gm.data(), bname, snap);

    FILE *fp1, *fp2;
    int dummy, ntot_withmasses, NumPart, ptype;
    unsigned long int i, j, k;
    int t, n, off, pc, pc_new, pc_sph;
    int files = 1;
    char buf[MaxCharactersInString];
    #define SKIP2 fwrite(&dummy, sizeof(dummy), 1, fp1); //kaifa

    // // ////settings by use //gjy add
    // // AllVars.MaxCompNumber = 4;
    // // AllVars.redshift = 0.0;
    // // AllVars.h = 0.71;
    // // AllVars.Omega_m = 0.3;
    // // AllVars.Omega_l = 0.7;
    // // AllVars.Omega_k = 0.0;
    // // Set everything to zero and overwrite it later if needed.
    // header1.npart[0] = 0;
    // header1.npart[1] = 0;
    // header1.npart[2] = 0;
    // header1.npart[3] = 0;
    // header1.npart[4] = 0;
    // header1.npart[5] = 0;
    // header1.npartTotal[0] = 0;
    // header1.npartTotal[1] = 0;
    // header1.npartTotal[2] = 0;
    // header1.npartTotal[3] = 0;
    // header1.npartTotal[4] = 0;
    // header1.npartTotal[5] = 0;
    // header1.mass[0] = 0.0;
    // header1.mass[1] = 0.0;
    // header1.mass[2] = 0.0;
    // header1.mass[3] = 0.0;
    // header1.mass[4] = 0.0;
    // header1.mass[5] = 0.0;

    // // Set the header values to some defaults.
    // header1.npart[0] = (int)gal->num_part[0];
    // header1.npart[1] = (int)gal->num_part[1];
    // header1.npart[2] = (int)gal->num_part[2];
    // header1.npart[3] = (int)gal->num_part[3];
    // header1.npartTotal[0] = (int)gal->num_part[0];
    // header1.npartTotal[1] = (int)gal->num_part[1];
    // header1.npartTotal[2] = (int)gal->num_part[2];
    // header1.npartTotal[3] = (int)gal->num_part[3];
    // header1.time = 1.0 / (1.0 + AllVars.redshift);
    // header1.redshift = AllVars.redshift;
    // header1.flag_sfr = 0.0;
    // header1.flag_feedback = 0.0;
    // header1.flag_cooling = 0.0;
    // header1.num_files = 1;
    // header1.BoxSize = 0.0;
    // header1.Omega0 = AllVars.Omega_m;
    // header1.OmegaLambda = AllVars.Omega_l;
    // header1.HubbleParam = AllVars.h;

    // if (!(P = (struct particle_data *) malloc(gal->ntot_part * sizeof(struct particle_data)))) {
    //     fprintf(stderr, "Unable to create particle data structure in memory.");
    //     exit(0);
    // }
    // P--;
    //// a new sys
    // j = 1;
    // // We need to transfer in the order of particle type as defined in GADGET2.
    // // 0->Gas 1->Disk 2->Halo etc.
    // for (ptype = 0; ptype < 10; ptype++) {
    //     for (k = 0; k < AllVars.MaxCompNumber; k++) {
    //         if (gal->comp_type[k] == ptype && gal->comp_npart[k] > 0) {
    //             if (gal->comp_delete[k] == 0) {
    //                 for (i = gal->comp_start_part[k];
    //                     i < gal->comp_start_part[k] + gal->comp_npart[k]; ++i) {
    //                     P[j].Pos[0] = gal->x[i];
    //                     P[j].Pos[1] = gal->y[i];
    //                     P[j].Pos[2] = gal->z[i];
    //                     P[j].Vel[0] = gal->vel_x[i];
    //                     P[j].Vel[1] = gal->vel_y[i];
    //                     P[j].Vel[2] = gal->vel_z[i];
    //                     P[j].U = gal->u[i];
    //                     P[j].Rho = gal->rho[i];
    //                     P[j].Mass = gal->mass[i];
    //                     P[j].Type = gal->comp_type[k];
    //                     P[j].Metal = gal->metal[i];
    //                     P[j].Age = gal->age[i];
    //                     P[j].Id = j;
    //                     ++j;
    //                 }

    //             } else {
    //                 header1.npart[ptype] -= gal->comp_npart[k];
    //                 header1.npartTotal[ptype] -= gal->comp_npart[k];
    //                 gal->ntot_part -= gal->comp_npart[k];
    //             }
    //         }
    //     }
    // }

    //// We have read data to hearder and *P.
    //// change particle data, set galaxy* only for number, write.
    int ID_change_one = ID_change;
    // DEBUG_PRINT_V0d(1, this->P[ID_change_one].Mass, "");
    // DEBUG_PRINT_V0d(1, rate, "");
    this->P[ID_change_one].Mass *= rate;
    // DEBUG_PRINT_V0d(1, this->P[ID_change_one].Mass, "");

    ////write all pd to IC file
    int N_tot_part = N_allPtcs; //??
    int N_tot_part_stars = 0;
    // header1.npart; //done
    for(int i_comp=0;i_comp<components;i_comp++){
        if(type_comp[i_comp]>=2){ //particle with type (>=2) are stars
            N_tot_part_stars += N_comp[i_comp];
        }
    }

    for (i = 0, pc = 1; i < files; i++, pc = pc_new) {
        if (files > 1) //gjy changed
            sprintf(buf, "%s_gg_file%d", fname, (int)i);
        else
            sprintf(buf, "%s_gg", fname);

        if (!(fp1 = fopen(buf, "w"))) {
            fprintf(stderr, "can't open file `%s`\n", buf);
            exit(0);
        }

        fflush(stdout);
        // Header
        dummy = sizeof(header1);
        fwrite(&dummy, sizeof(dummy), 1, fp1);
        fwrite(&header1, sizeof(header1), 1, fp1);
        fwrite(&dummy, sizeof(dummy), 1, fp1);

        for (k = 0, ntot_withmasses = 0; k < 6; k++) {
            if (header1.mass[k] == 0)
                ntot_withmasses += header1.npart[k];
        }

        // Positions
        dummy = 3 * sizeof(float) * N_tot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Pos[0], sizeof(float), 3, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // Velocities
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Vel[0], sizeof(float), 3, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // Identifiers
        dummy = sizeof(int) * N_tot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Id, sizeof(int), 1, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // Mass
        if (ntot_withmasses > 0) {
            dummy = sizeof(float) * N_tot_part;
            SKIP2;
        }
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                if (header1.mass[k] == 0)
                    fwrite(&P[pc_new].Mass, sizeof(float), 1, fp1);
                else
                    P[pc_new].Mass = header1.mass[k];
                pc_new++;
            }
        }
        if (ntot_withmasses > 0) {
            SKIP2;
        }

        //// Gas specific datablocks
        if (header1.npart[0] > 0) {
            // Internal energy
            dummy = sizeof(float) * header1.npart[0];
            SKIP2;
            for (n = 0, pc_sph = pc; n < header1.npart[0]; n++) {
                fwrite(&P[pc_sph].U, sizeof(float), 1, fp1);
                pc_sph++;
            }
            SKIP2;
            // Density
            SKIP2;
            for (n = 0, pc_sph = pc; n < header1.npart[0]; n++) {
                fwrite(&P[pc_sph].Rho, sizeof(float), 1, fp1);
                pc_sph++;
            }
            SKIP2;
        }

        //// added, two
        //Potential
        dummy = sizeof(float) * N_tot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Pot, sizeof(float), 1, fp1);
                pc_new++;
            }
        }
        SKIP2;

        //Accelerations
        dummy = 3 * sizeof(float) * N_tot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Acc[0], sizeof(float), 3, fp1);
                pc_new++;
            }
        }
        SKIP2;

        ////others no use
        // Metallicity
        dummy = sizeof(int) * N_tot_part; //gjy note: int??
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Metal, sizeof(float), 1, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // Age for star particles
        if (N_tot_part_stars > 0) {
            dummy = sizeof(int) * (N_tot_part_stars);
            SKIP2;
            for (k = 2, pc_new = header1.npart[0] + header1.npart[1] + 1; k < 6; k++) {
                for (n = 0; n < header1.npart[k]; n++) {
                    fwrite(&P[pc_new].Age, sizeof(float), 1, fp1);
                    pc_new++;
                }
            }
            SKIP2;
        }

    }

    // P++;
    // free(P);
    fclose(fp1);
    printf("Wrote %s ... done.\n", buf);
    return 0;
}



int Snapshot::write_gadget1_ics(galaxy *gal, char *fname){

    FILE *fp1, *fp2;
    int dummy, ntot_withmasses, NumPart, ptype;
    unsigned long int i, j, k;
    int t, n, off, pc, pc_new, pc_sph;
    int files = 1;
    char buf[MaxCharactersInString];
    #define SKIP2 fwrite(&dummy, sizeof(dummy), 1, fp1);

    // ////settings by use //gjy add
    // AllVars.MaxCompNumber = 4;
    // AllVars.redshift = 0.0;
    // AllVars.h = 0.71;
    // AllVars.Omega_m = 0.3;
    // AllVars.Omega_l = 0.7;
    // AllVars.Omega_k = 0.0;
    // Set everything to zero and overwrite it later if needed.
    header1.npart[0] = 0;
    header1.npart[1] = 0;
    header1.npart[2] = 0;
    header1.npart[3] = 0;
    header1.npart[4] = 0;
    header1.npart[5] = 0;
    header1.npartTotal[0] = 0;
    header1.npartTotal[1] = 0;
    header1.npartTotal[2] = 0;
    header1.npartTotal[3] = 0;
    header1.npartTotal[4] = 0;
    header1.npartTotal[5] = 0;
    header1.mass[0] = 0.0;
    header1.mass[1] = 0.0;
    header1.mass[2] = 0.0;
    header1.mass[3] = 0.0;
    header1.mass[4] = 0.0;
    header1.mass[5] = 0.0;

    // Set the header values to some defaults.
    header1.npart[0] = (int)gal->num_part[0];
    header1.npart[1] = (int)gal->num_part[1];
    header1.npart[2] = (int)gal->num_part[2];
    header1.npart[3] = (int)gal->num_part[3];
    header1.npartTotal[0] = (int)gal->num_part[0];
    header1.npartTotal[1] = (int)gal->num_part[1];
    header1.npartTotal[2] = (int)gal->num_part[2];
    header1.npartTotal[3] = (int)gal->num_part[3];
    header1.time = 1.0 / (1.0 + AllVars.redshift);
    header1.redshift = AllVars.redshift;
    header1.flag_sfr = 0.0;
    header1.flag_feedback = 0.0;
    header1.flag_cooling = 0.0;
    header1.num_files = 1;
    header1.BoxSize = 0.0;
    header1.Omega0 = AllVars.Omega_m;
    header1.OmegaLambda = AllVars.Omega_l;
    header1.HubbleParam = AllVars.h;

    if (!(P = (struct particle_data *) malloc(gal->ntot_part * sizeof(struct particle_data)))) {
        fprintf(stderr, "Unable to create particle data structure in memory.");
        exit(0);
    }
    P--;

    // Transfer the particle data from the DICE galaxy data structure to the
    // Gadget position_data structure.
    j = 1;
    // We need to transfer in the order of particle type as defined in GADGET2.
    // 0->Gas 1->Disk 2->Halo etc.
    for (ptype = 0; ptype < 10; ptype++) {
        for (k = 0; k < AllVars.MaxCompNumber; k++) {
            if (gal->comp_type[k] == ptype && gal->comp_npart[k] > 0) {
                if (gal->comp_delete[k] == 0) {
                    for (i = gal->comp_start_part[k];
                        i < gal->comp_start_part[k] + gal->comp_npart[k]; ++i) {
                        P[j].Pos[0] = gal->x[i];
                        P[j].Pos[1] = gal->y[i];
                        P[j].Pos[2] = gal->z[i];
                        P[j].Vel[0] = gal->vel_x[i];
                        P[j].Vel[1] = gal->vel_y[i];
                        P[j].Vel[2] = gal->vel_z[i];
                        P[j].U = gal->u[i];
                        P[j].Rho = gal->rho[i];
                        P[j].Mass = gal->mass[i];
                        P[j].Type = gal->comp_type[k];
                        P[j].Metal = gal->metal[i];
                        P[j].Age = gal->age[i];
                        P[j].Id = j;
                        ++j;
                    }
                } else {
                    header1.npart[ptype] -= gal->comp_npart[k];
                    header1.npartTotal[ptype] -= gal->comp_npart[k];
                    gal->ntot_part -= gal->comp_npart[k];
                }
            }
        }
    }

    for (i = 0, pc = 1; i < files; i++, pc = pc_new) {
        if (files > 1)
            sprintf(buf, "%s.g1.%d", fname, (int)i);
        else
            sprintf(buf, "%s.g1", fname);

        if (!(fp1 = fopen(buf, "w"))) {
            fprintf(stderr, "can't open file `%s`\n", buf);
            exit(0);
        }

        fflush(stdout);
        // Header
        dummy = sizeof(header1);
        fwrite(&dummy, sizeof(dummy), 1, fp1);
        fwrite(&header1, sizeof(header1), 1, fp1);
        fwrite(&dummy, sizeof(dummy), 1, fp1);

        for (k = 0, ntot_withmasses = 0; k < 6; k++) {
            if (header1.mass[k] == 0)
                ntot_withmasses += header1.npart[k];
        }

        // Positions
        dummy = 3 * sizeof(float) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Pos[0], sizeof(float), 3, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // Velocities
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Vel[0], sizeof(float), 3, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // Identifiers
        dummy = sizeof(int) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Id, sizeof(int), 1, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // Mass
        if (ntot_withmasses > 0) {
            dummy = sizeof(float) * gal->ntot_part;
            SKIP2;
        }
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                if (header1.mass[k] == 0)
                    fwrite(&P[pc_new].Mass, sizeof(float), 1, fp1);
                else
                    P[pc_new].Mass = header1.mass[k];
                pc_new++;
            }
        }
        if (ntot_withmasses > 0) {
            SKIP2;
        }

        //// Gas specific datablocks
        if (header1.npart[0] > 0) {
            // Internal energy
            dummy = sizeof(float) * header1.npart[0];
            SKIP2;
            for (n = 0, pc_sph = pc; n < header1.npart[0]; n++) {
                fwrite(&P[pc_sph].U, sizeof(float), 1, fp1);
                pc_sph++;
            }
            SKIP2;
            // Density
            SKIP2;
            for (n = 0, pc_sph = pc; n < header1.npart[0]; n++) {
                fwrite(&P[pc_sph].Rho, sizeof(float), 1, fp1);
                pc_sph++;
            }
            SKIP2;
        }

        //// added, two
        //Potential
        dummy = sizeof(float) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Pot, sizeof(float), 1, fp1);
                pc_new++;
            }
        }
        SKIP2;

        //Accelerations
        dummy = 3 * sizeof(float) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Acc[0], sizeof(float), 3, fp1);
                pc_new++;
            }
        }
        SKIP2;

        ////others no use
        // Metallicity
        dummy = sizeof(int) * gal->ntot_part; //gjy note: int??
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
            for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Metal, sizeof(float), 1, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // Age for star particles
        if (gal->ntot_part_stars > 0) {
            dummy = sizeof(int) * (gal->ntot_part_stars);
            SKIP2;
            for (k = 2, pc_new = header1.npart[0] + header1.npart[1] + 1; k < 6;
                k++) {
                for (n = 0; n < header1.npart[k]; n++) {
                fwrite(&P[pc_new].Age, sizeof(float), 1, fp1);
                pc_new++;
                }
            }
            SKIP2;
    }

    }
    P++;
    free(P);
    fclose(fp1);
    return 0;
}

int Snapshot::write_gadget2_ics(galaxy *gal, char *fname) {

    FILE *fp1, *fp2;
    int dummy, nextblock, bytes_per_blockelement, ntot_withmasses, NumPart, ptype;
    unsigned long int i, j, k;
    int t, n, off, pc, pc_new, pc_sph;
    int files = 1;
    char buf[MaxCharactersInString];
    #define SKIP2 fwrite(&dummy, sizeof(dummy), 1, fp1);

    // Set everything to zero and overwrite it later if needed.
    header1.npart[0] = 0;
    header1.npart[1] = 0;
    header1.npart[2] = 0;
    header1.npart[3] = 0;
    header1.npart[4] = 0;
    header1.npart[5] = 0;
    header1.npartTotal[0] = 0;
    header1.npartTotal[1] = 0;
    header1.npartTotal[2] = 0;
    header1.npartTotal[3] = 0;
    header1.npartTotal[4] = 0;
    header1.npartTotal[5] = 0;
    header1.mass[0] = 0.0;
    header1.mass[1] = 0.0;
    header1.mass[2] = 0.0;
    header1.mass[3] = 0.0;
    header1.mass[4] = 0.0;
    header1.mass[5] = 0.0;

    // Set the header values to some defaults.
    header1.npart[0] = (int)gal->num_part[0];
    header1.npart[1] = (int)gal->num_part[1];
    header1.npart[2] = (int)gal->num_part[2];
    header1.npart[3] = (int)gal->num_part[3];
    header1.npartTotal[0] = (int)gal->num_part[0];
    header1.npartTotal[1] = (int)gal->num_part[1];
    header1.npartTotal[2] = (int)gal->num_part[2];
    header1.npartTotal[3] = (int)gal->num_part[3];
    header1.time = 0.0;
    header1.redshift = AllVars.redshift;
    header1.flag_sfr = 0.0;
    header1.flag_feedback = 0.0;
    header1.flag_cooling = 0.0;
    header1.num_files = 1;
    header1.BoxSize = 0.0;
    header1.Omega0 = AllVars.Omega_m;
    header1.OmegaLambda = AllVars.Omega_l;
    header1.HubbleParam = AllVars.h;

    if (!(P = (struct particle_data *) malloc(gal->ntot_part * sizeof(struct particle_data)))) {
        fprintf(stderr, "Unable to create particle data structure in memory.");
        exit(0);
    }
    P--;



    ////load *P by DICE: struct::galaxy
    // Transfer the particle data from the DICE galaxy data structure to the
    // Gadget position_data structure.
    j = 1;
    // We need to transfer in the order of particle type as defined in GADGET2.
    // 0->Gas 1->Disk 2->Halo etc. (GADGET format -- 0=Gas,1=Halo,2=Disk,3=Bulge,4=Stars)
    for (ptype = 0; ptype < 10; ptype++) {
        for (k = 0; k < AllVars.MaxCompNumber; k++) {
        if (gal->comp_type[k] == ptype && gal->comp_npart[k] > 0) {
            if (gal->comp_delete[k] == 0) {
            for (i = gal->comp_start_part[k];
                i < gal->comp_start_part[k] + gal->comp_npart[k]; ++i) {
                P[j].Pos[0] = gal->x[i];
                P[j].Pos[1] = gal->y[i];
                P[j].Pos[2] = gal->z[i];
                P[j].Vel[0] = gal->vel_x[i];
                P[j].Vel[1] = gal->vel_y[i];
                P[j].Vel[2] = gal->vel_z[i];
                P[j].U = gal->u[i];
                P[j].Rho = gal->rho[i];
                P[j].Mass = gal->mass[i];
                P[j].Type = gal->comp_type[k];
                P[j].Metal = gal->metal[i];
                P[j].Age = gal->age[i];
                P[j].Id = j;
                P[j].Hsml = 0.1;
                ++j;
            }
            } else {
            header1.npart[ptype] -= gal->comp_npart[k];
            header1.npartTotal[ptype] -= gal->comp_npart[k];
            gal->ntot_part -= gal->comp_npart[k];
            }
        }
        }
    }



    for(i = 0, pc = 1; i < files; i++, pc = pc_new){ //main loop begin; 其实也不是循环, 一个文件就只循环一次, 接下来同一类的变量是挨着存储的
        //some if
        if (files > 1)
            sprintf(buf, "%s.g2.%d", fname, (int)i);
        else
            sprintf(buf, "%s.g2", fname);
        if (!(fp1 = fopen(buf, "w"))) {
            fprintf(stderr, "can't open file `%s`\n", buf);
            exit(0);
        }
        fflush(stdout);

        dummy = sizeof(int) + 4 * sizeof(char);
        SKIP2; //把现在dummy地址上的值, 写入到fd1的一个dummy大小的空间中; 这个SKIP2在整体文件内容的作用, 似乎就是记录接下来所要空间的大小, 而且首尾夹着
        fwrite("HEAD", sizeof(char), 4, fp1);
        nextblock = sizeof(header1) + 2 * sizeof(int);
        fwrite(&nextblock, sizeof(int), 1, fp1);
        SKIP2;

        // Header
        dummy = sizeof(header1);
        SKIP2;
        fwrite(&header1, sizeof(header1), 1, fp1); //把hearder写进去
        SKIP2;

        for (k = 0, ntot_withmasses = 0; k < 6; k++) {
        if (header1.mass[k] == 0)
            ntot_withmasses += header1.npart[k];
        }

        // Positions
        dummy = sizeof(int) + 4 * sizeof(char);
        SKIP2;
        fwrite("POS ", sizeof(char), 4, fp1);
        bytes_per_blockelement = 3 * sizeof(float);
        nextblock = gal->ntot_part * bytes_per_blockelement + 2 * sizeof(int);
        fwrite(&nextblock, sizeof(int), 1, fp1); //这句写的是 int数量=ntot_part*3float+2int, 记录下一块区域的大小, 即两个dummy加一堆三维坐标
        SKIP2;

        dummy = 3 * sizeof(float) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Pos[0], sizeof(float), 3, fp1);
            pc_new++;
        }
        }
        SKIP2;

        // Velocities
        dummy = sizeof(int) + 4 * sizeof(char);
        SKIP2;
        fwrite("VEL ", sizeof(char), 4, fp1);
        bytes_per_blockelement = 3 * sizeof(float);
        nextblock = gal->ntot_part * bytes_per_blockelement + 2 * sizeof(int);
        fwrite(&nextblock, sizeof(int), 1, fp1);
        SKIP2;

        dummy = 3 * sizeof(float) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Vel[0], sizeof(float), 3, fp1);
            pc_new++;
        }
        }
        SKIP2;

        // Identifiers
        dummy = sizeof(int) + 4 * sizeof(char);
        SKIP2;
        fwrite("ID  ", sizeof(char), 4, fp1);
        bytes_per_blockelement = sizeof(int);
        nextblock = gal->ntot_part * bytes_per_blockelement + 2 * sizeof(int);
        fwrite(&nextblock, sizeof(int), 1, fp1);
        SKIP2;

        dummy = sizeof(int) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Id, sizeof(int), 1, fp1);
            pc_new++;
        }
        }
        SKIP2;

        // Mass
        dummy = sizeof(int) + 4 * sizeof(char);
        SKIP2;
        fwrite("MASS", sizeof(char), 4, fp1);
        bytes_per_blockelement = sizeof(float);
        nextblock = gal->ntot_part * bytes_per_blockelement + 2 * sizeof(int);
        fwrite(&nextblock, sizeof(int), 1, fp1);
        SKIP2;

        dummy = sizeof(float) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Mass, sizeof(float), 1, fp1);
            pc_new++;
        }
        }
        SKIP2;

        ////Gas specific datablocks, three
        if (header1.npart[0] > 0) {

        // Internal energy
        dummy = sizeof(int) + 4 * sizeof(char);
        SKIP2;
        fwrite("U   ", sizeof(char), 4, fp1);
        bytes_per_blockelement = sizeof(float);
        nextblock = header1.npart[0] * bytes_per_blockelement + 2 * sizeof(int);
        fwrite(&nextblock, sizeof(int), 1, fp1);
        SKIP2;

        dummy = sizeof(float) * header1.npart[0];
        SKIP2;
        for (n = 0, pc_sph = pc; n < header1.npart[0]; n++) {
            fwrite(&P[pc_sph].U, sizeof(float), 1, fp1);
            pc_sph++;
        }
        SKIP2;

        // Density
        dummy = sizeof(int) + 4 * sizeof(char);
        SKIP2;
        fwrite("RHO ", sizeof(char), 4, fp1);
        bytes_per_blockelement = sizeof(float);
        nextblock = header1.npart[0] * bytes_per_blockelement + 2 * sizeof(int);
        fwrite(&nextblock, sizeof(int), 1, fp1);
        SKIP2;

        dummy = sizeof(float) * header1.npart[0];
        SKIP2;
        for (n = 0, pc_sph = pc; n < header1.npart[0]; n++) {
            fwrite(&P[pc_sph].Rho, sizeof(float), 1, fp1);
            pc_sph++;
        }
        SKIP2;

        // HSML
        dummy = sizeof(int) + 4 * sizeof(char);
        SKIP2;
        fwrite("HSML", sizeof(char), 4, fp1);
        bytes_per_blockelement = sizeof(float);
        nextblock = header1.npart[0] * bytes_per_blockelement + 2 * sizeof(int);
        fwrite(&nextblock, sizeof(int), 1, fp1);
        SKIP2;

        dummy = sizeof(float) * header1.npart[0];
        SKIP2;
        for (n = 0, pc_sph = pc; n < header1.npart[0]; n++) {
            fwrite(&P[pc_sph].Hsml, sizeof(float), 1, fp1);
            pc_sph++;
        }
        SKIP2;
        }

        ////added, two
        //Potential
        dummy = sizeof(int) + 4 * sizeof(char);
        SKIP2;
        fwrite("POS ", sizeof(char), 4, fp1);
        bytes_per_blockelement = 1 * sizeof(float);
        nextblock = gal->ntot_part * bytes_per_blockelement + 2 * sizeof(int);
        fwrite(&nextblock, sizeof(int), 1, fp1);
        SKIP2;

        dummy = 3 * sizeof(float) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Pot, sizeof(float), 1, fp1);
            pc_new++;
        }
        }
        SKIP2;

        //Accelerations
        dummy = sizeof(int) + 4 * sizeof(char);
        SKIP2;
        fwrite("POS ", sizeof(char), 4, fp1);
        bytes_per_blockelement = 3 * sizeof(float);
        nextblock = gal->ntot_part * bytes_per_blockelement + 2 * sizeof(int);
        fwrite(&nextblock, sizeof(int), 1, fp1);
        SKIP2;

        dummy = 3 * sizeof(float) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Pos[0], sizeof(float), 3, fp1);
            pc_new++;
        }
        }
        SKIP2;

        ////others no use
        // Metallicity
        dummy = sizeof(int) + 4 * sizeof(char);
        SKIP2;
        fwrite("Z   ", sizeof(char), 4, fp1);
        bytes_per_blockelement = sizeof(float);
        nextblock = gal->ntot_part * bytes_per_blockelement + 2 * sizeof(int);
        fwrite(&nextblock, sizeof(int), 1, fp1);
        SKIP2;

        dummy = sizeof(float) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Metal, sizeof(float), 1, fp1);
            pc_new++;
        }
        }
        SKIP2;

        // Age for star particles
        if (gal->ntot_part_stars > 0) {
        dummy = sizeof(int) + 4 * sizeof(char);
        SKIP2;
        fwrite("AGE ", sizeof(char), 4, fp1);
        bytes_per_blockelement = sizeof(float);
        nextblock =
            gal->ntot_part_stars * bytes_per_blockelement + 2 * sizeof(int);
        fwrite(&nextblock, sizeof(int), 1, fp1);
        SKIP2;

        dummy = sizeof(float) * gal->ntot_part_stars;
        SKIP2;
        for (k = 2, pc_new = header1.npart[0] + header1.npart[1] + 1; k < 6;
            k++) {
            for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Age, sizeof(float), 1, fp1);
            pc_new++;
            }
        }
        SKIP2;
        //potential 引力势并没有存储, 可在gadget算
        }
        fclose(fp1);
    } //main loop end
    P++;
    free(P);
    // free(Ids);
    return 0;
}

int Snapshot::write_gadget1_ics_manul(galaxy *gal, struct particle_data* P_r, char *fname){

    FILE *fp1, *fp2;
    int dummy, ntot_withmasses, NumPart, ptype;
    unsigned long int i, j, k;
    int t, n, off, pc, pc_new, pc_sph;
    int files = 1;
    char buf[MaxCharactersInString];
    #define SKIP2 fwrite(&dummy, sizeof(dummy), 1, fp1);

    ////gjy add: manual settings
    AllVars.MaxCompNumber = 4;
    AllVars.redshift = 0.0;
    AllVars.h = 0.71;
    AllVars.Omega_m = 0.3;
    AllVars.Omega_l = 0.7;
    AllVars.Omega_k = 0.0;
    //to set numbers
    for(int k=0;k<6;k++){
        gal->num_part[k] = 0;
    }
    gal->num_part[1] = 2; //halo of this IC //if <120, Gadget2 will endrun(11): 
    gal->ntot_part = 0;
    for(int k=0;k<6;k++){
        gal->ntot_part += gal->num_part[k];
    }
    //set particle data *P_w in the next for(...)
    //manual settings end

    // Set everything to zero and overwrite it later if needed.
    header1.npart[0] = 0;
    header1.npart[1] = 0;
    header1.npart[2] = 0;
    header1.npart[3] = 0;
    header1.npart[4] = 0;
    header1.npart[5] = 0;
    header1.npartTotal[0] = 0;
    header1.npartTotal[1] = 0;
    header1.npartTotal[2] = 0;
    header1.npartTotal[3] = 0;
    header1.npartTotal[4] = 0;
    header1.npartTotal[5] = 0;
    header1.mass[0] = 0.0;
    header1.mass[1] = 0.0;
    header1.mass[2] = 0.0;
    header1.mass[3] = 0.0;
    header1.mass[4] = 0.0;
    header1.mass[5] = 0.0;

    // Set the header values to some defaults.
    header1.npart[0] = (int)gal->num_part[0];
    header1.npart[1] = (int)gal->num_part[1];
    header1.npart[2] = (int)gal->num_part[2];
    header1.npart[3] = (int)gal->num_part[3];
    header1.npartTotal[0] = (int)gal->num_part[0];
    header1.npartTotal[1] = (int)gal->num_part[1];
    header1.npartTotal[2] = (int)gal->num_part[2];
    header1.npartTotal[3] = (int)gal->num_part[3];
    header1.time = 1.0 / (1.0 + AllVars.redshift);
    header1.redshift = AllVars.redshift;
    header1.flag_sfr = 0.0;
    header1.flag_feedback = 0.0;
    header1.flag_cooling = 0.0;
    header1.num_files = 1;
    header1.BoxSize = 0.0;
    header1.Omega0 = AllVars.Omega_m;
    header1.OmegaLambda = AllVars.Omega_l;
    header1.HubbleParam = AllVars.h;

    if (!(P = (struct particle_data *) malloc(gal->ntot_part * sizeof(struct particle_data)))) {
        fprintf(stderr, "Unable to create particle data structure in memory.");
        exit(0);
    }
    P--;

    // Transfer the particle data from the DICE galaxy data structure to the
    // Gadget position_data structure.
    // We need to transfer in the order of particle type as defined in GADGET2.
    // 0->Gas 1->Disk 2->Halo etc.
    // j=1;
    // for (ptype = 0; ptype < 10; ptype++) {
    //   for (k = 0; k < AllVars.MaxCompNumber; k++) {
    //     if (gal->comp_type[k] == ptype && gal->comp_npart[k] > 0) {
    //       if (gal->comp_delete[k] == 0) {
    //         for (i = gal->comp_start_part[k]; i < gal->comp_start_part[k] + gal->comp_npart[k]; ++i) {
    //           ;
    //         }
    //       } else {
    //         header1.npart[ptype] -= gal->comp_npart[k];
    //         header1.npartTotal[ptype] -= gal->comp_npart[k];
    //         gal->ntot_part -= gal->comp_npart[k];
    //       }
    //     }
    //   }
    // }

            // j = 1;
            // for(int ii=0; ii<6; ii++){
            //   printf("Number: header1.npartTotal[%d] = %d\n", ii,header1.npartTotal[ii]); //gjy add
            //   for(int jj=0; jj<header1.npartTotal[ii];jj++){
            //     // printf("\nhere1 %e\n", P_r[j].Pos[0]); //gjy add
            //     //gjy add: manual settings, particle data
            //     P[j].Pos[0] = P_r[j].Pos[0]; //The *P_w is the particle data to write, while the *P is the particle data we read from snapshot before.
            //     P[j].Pos[1] = P_r[j].Pos[1];
            //     P[j].Pos[2] = P_r[j].Pos[2];
            //     P[j].Vel[0] = P_r[j].Vel[0];
            //     P[j].Vel[1] = P_r[j].Vel[1];
            //     P[j].Vel[2] = P_r[j].Vel[2];

            //     P[j].Mass   = P_r[j].Mass;
            //     P[j].Type   = P_r[j].Type; //??

            //     P[j].U      = P_r[j].U;
            //     P[j].Rho    = P_r[j].Rho;
            //     // P[j].Hsml   = P_r[j].Hsml; //??

            //     P[j].Pot    = P_r[j].Pot; //extra added
            //     P[j].Acc[0] = P_r[j].Acc[0]; //extra added
            //     P[j].Acc[1] = P_r[j].Acc[1];
            //     P[j].Acc[2] = P_r[j].Acc[2];

            //     P[j].Metal  = P_r[j].Metal;
            //     P[j].Age    = P_r[j].Age;

            //     P[j].Id     = j; //Id j is better than original Id of *P
            //     ++j;
            //   }
            // }

                //Now we set a simplist model -- a pair of stable circular binary stars 
                //with: r0=1kpc, v0=10km/s, G=43007.1(this Gadget2), then m=4*v0^2*r0/G, and Phi_rel=-Gm/(2*r).
                j = 1;
                P[j].Pos[0] = 0.; //x, y, z
                P[j].Pos[1] = 1.;
                P[j].Pos[2] = 0.;
                P[j].Vel[0] = -10.; //v_x, v_y, v_z
                P[j].Vel[1] = 0.;
                P[j].Vel[2] = 0.;

                P[j].Mass   = 0.009300789869579674; //mass
                P[j].Type   = 1; //halo

                P[j].U      = 0.;
                P[j].Rho    = 0.;
                P[j].Hsml   = 0.; //??

                P[j].Pot    = -200.; //donot influent
                P[j].Acc[0] = 0.; //donot influent
                P[j].Acc[1] = 0.;
                P[j].Acc[2] = 0.;

                P[j].Metal  = 0.;
                P[j].Age    = 0.;
                P[j].Id     = 1; //Id j is better than original Id of *P

                j = 2;
                P[j].Pos[0] = 0.; //x, y, z
                P[j].Pos[1] = -1.;
                P[j].Pos[2] = 0.;
                P[j].Vel[0] = 10.; //v_x, v_y, v_z
                P[j].Vel[1] = 0.;
                P[j].Vel[2] = 0.;

                P[j].Mass   = 0.009300789869579674; //mass
                P[j].Type   = 1; //halo

                P[j].U      = 0.;
                P[j].Rho    = 0.;
                P[j].Hsml   = 0.; //??

                P[j].Pot    = -200.; //donot influent
                P[j].Acc[0] = 0.; //donot influent
                P[j].Acc[1] = 0.;
                P[j].Acc[2] = 0.;

                P[j].Metal  = 0.;
                P[j].Age    = 0.;
                P[j].Id     = 2; //Id j is better than original Id of *P
    // //results
    // >>> E1 = 0.5*0.009301*v2+p
    // >>> v2=100.; p=-200.;
    // >>> E1
    // -91.16753585470386
    // >>> v2=(-9.091153**2 +4.151279**2 +0.000000**2); p=-90.863319;
    // >>> E1
    // -91.16753585470386



    ////now writing...
    for (i = 0, pc = 1; i < files; i++, pc = pc_new) {
        if (files > 1)
            sprintf(buf, "%s.g1.%d", fname, (int)i);
        else
            sprintf(buf, "%s.g1", fname);
        if (!(fp1 = fopen(buf, "w"))) {
            fprintf(stderr, "can't open file `%s`\n", buf);
            exit(0);
        }
        fflush(stdout);

        // Header
        dummy = sizeof(header1);
        fwrite(&dummy, sizeof(dummy), 1, fp1);
        fwrite(&header1, sizeof(header1), 1, fp1);
        fwrite(&dummy, sizeof(dummy), 1, fp1);

        for (k = 0, ntot_withmasses = 0; k < 6; k++) {
        if (header1.mass[k] == 0)
            ntot_withmasses += header1.npart[k];
        }

        // Positions
        dummy = 3 * sizeof(float) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Pos[0], sizeof(float), 3, fp1);
            pc_new++;
        }
        }
        SKIP2;

        // Velocities
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Vel[0], sizeof(float), 3, fp1);
            pc_new++;
        }
        }
        SKIP2;

        // Identifiers
        dummy = sizeof(int) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Id, sizeof(int), 1, fp1);
            pc_new++;
        }
        }
        SKIP2;

        // Mass
        if (ntot_withmasses > 0) {
        dummy = sizeof(float) * gal->ntot_part;
        SKIP2;
        }
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            if (header1.mass[k] == 0)
            fwrite(&P[pc_new].Mass, sizeof(float), 1, fp1);
            else
            P[pc_new].Mass = header1.mass[k];
            pc_new++;
        }
        }
        if (ntot_withmasses > 0) {
        SKIP2;
        }

        //// Gas specific datablocks
        if (header1.npart[0] > 0) {
        // Internal energy
        dummy = sizeof(float) * header1.npart[0];
        SKIP2;
        for (n = 0, pc_sph = pc; n < header1.npart[0]; n++) {
            fwrite(&P[pc_sph].U, sizeof(float), 1, fp1);
            pc_sph++;
        }
        SKIP2;
        // Density
        SKIP2;
        for (n = 0, pc_sph = pc; n < header1.npart[0]; n++) {
            fwrite(&P[pc_sph].Rho, sizeof(float), 1, fp1);
            pc_sph++;
        }
        SKIP2;
        }

        //Hsml is not writen??

        //// added, two
        //Potential
        dummy = sizeof(float) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Pot, sizeof(float), 1, fp1);
            pc_new++;
        }
        }
        SKIP2;

        //Accelerations
        dummy = 3 * sizeof(float) * gal->ntot_part;
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Acc[0], sizeof(float), 3, fp1);
            pc_new++;
        }
        }
        SKIP2;

        ////others no use
        // Metallicity
        dummy = sizeof(int) * gal->ntot_part; //gjy note: int??
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++) {
        for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Metal, sizeof(float), 1, fp1);
            pc_new++;
        }
        }
        SKIP2;

        // Age for star particles
        if (gal->ntot_part_stars > 0) {
        dummy = sizeof(int) * (gal->ntot_part_stars);
        SKIP2;
        for (k = 2, pc_new = header1.npart[0] + header1.npart[1] + 1; k < 6;
            k++) {
            for (n = 0; n < header1.npart[k]; n++) {
            fwrite(&P[pc_new].Age, sizeof(float), 1, fp1);
            pc_new++;
            }
        }
        SKIP2;
        
        }
    }

    // P++; //gjy note: if used latter, donot change
    // free(P);
    int id = 1;
    printf("%e %d %e %e %e\n", P[id].Pos[0], P[id].Id, P[id].Mass, P[id].Pot, P[id].Acc[2]);
    fclose(fp1);

    // printf("Here is ics1. Now we will print the value of some variables.\n"); //gjy add
    // printf("多少多少多少多少: \n\n\n\n\n\ngal->num_part(part[1] is halo) = %d %d %d %d\n", (int)gal->num_part[0],(int)gal->num_part[1],(int)gal->num_part[2],(int)gal->num_part[3]);
    // printf("AllVars.MaxCompNumber, (int)gal->ntot_part, gal->ntot_part_stars = %d %d %d\n", AllVars.MaxCompNumber, (int)gal->ntot_part, (int)gal->ntot_part_stars);
    return 0;
}



int Snapshot::dispersion_discrete_all(int coor, int nn){

        char input_fname[MaxCharactersInString];
        // if(input_fname == nullptr)
        sprintf(input_fname, "%stxt/%s_%03d.secondhand.txt", path_gm.data(), bname, snap); //default path
        FILE *fp = fopen(input_fname, "w");
        if(fp==nullptr){
            printf("Cannot open file %s.\n", input_fname);
            return -1;
        };

        printf("Calculating discrete dispersions of all particles ...\n");
        for(int id=0+1;id<NumPart+1;id++){
            //data analysis for each particle
			VecDoub x_tgt_arr = {P[id].Pos[0], P[id].Pos[1], P[id].Pos[2]};
			vector<int> nearest_idx = pkdtree->find_k_nearest(nn, x_tgt_arr);
			// vector<int> nearest_min = pkdtree->find_k_nearest(1, x_tgt_arr);

            vector<double> dispersion = {0.,0.,0.}, vbar = {0.,0.,0.}, v2bar = {0.,0.,0.}; //of this target id
            double rho_itp = 0.; double potential_itp = 0.; double what_itp = 0.; //of this target id

            for(auto inn:nearest_idx){
                int i = inn;
                vector<double> Rpzvvv = {P[i].Pos[0], P[i].Pos[1], P[i].Pos[2], P[i].Vel[0], P[i].Vel[1], P[i].Vel[2]};
                // printf("Rpzvvv[3] b = %e\n", Rpzvvv[3]);
                // Rpzvvv = CoorTrans(Rpzvvv, coor); //??
                // printf("Rpzvvv[3] e = %e\n", Rpzvvv[3]);
                for(int ix=0;ix<Dimension;ix++){
                    vbar[ix] += Rpzvvv[ix+3];
                    v2bar[ix] += pow(Rpzvvv[ix+3],2);
                }

                double dr = distance_l2norm({P[id].Pos[0], P[id].Pos[1], P[id].Pos[2]}, {P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]});
                // printf("the knn: %d %e %e %e\n", i, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                // double h = 140.; //2.8*0.05;
                // double h = 2.8*softening_comp[P[i].Type]; //?? P[i].Type
                int imin = nearest_idx[nn-1]; //max??
                double h = distance_l2norm({P[id].Pos[0], P[id].Pos[1], P[id].Pos[2]}, {P[imin].Pos[0], P[imin].Pos[1], P[imin].Pos[2]});
                // double h1 = distance_l2norm({P[id].Pos[0], P[id].Pos[1], P[id].Pos[2]}, {P[imin+1].Pos[0], P[imin+1].Pos[1], P[imin+1].Pos[2]});
                // printf("h = %e, h1 = %e, imin = %d, P[imin].Pos[0] = %e\n", h, h1, imin, P[imin].Pos[0]);
                // printf("dr = %e, nid = %d, h = %e\n", dr, inn, h);
                // printf("Mass = %e, Weight_SPH = %e\n", P[i].Mass, Weight_SPHSmoothingKernel(dr,h));
                rho_itp += P[i].Mass*Weight_SPHSmoothingKernel(dr,h); //here P[id] is the target
            }
            for(int ix=0;ix<Dimension;ix++){
                vbar[ix] /= nn;
                v2bar[ix] /= nn;
                dispersion[ix] = sqrt( v2bar[ix]-vbar[ix]*vbar[ix] );
            }
            printf("ID_%d, %e\n\n", id, rho_itp);

            //write for each particle
            fprintf(fp, "%e %e %e %e %e %e    %d %d %e        %g %g %g        %e %e %e    %e %e %e    %e %e %e",    
                P[id].Pos[0],P[id].Pos[1],P[id].Pos[2],    P[id].Vel[0],P[id].Vel[1],P[id].Vel[2],    
                P[id].Id, P[id].Type,    P[id].Mass,    
                rho_itp, potential_itp, what_itp,    
                vbar[0], vbar[1], vbar[2],   v2bar[0], v2bar[1], v2bar[2],   dispersion[0], dispersion[1], dispersion[2]    
            );
            // for(int inn=0;inn<nn;inn++) fprintf(fp, "%d ", nearest_idx[inn]);
            fprintf(fp, "\n");
        }

        fclose(fp);
        printf("Write directly analysis second-hand data to file %s ... done.\n", input_fname);
        return 0;
}

vector<double> Snapshot::dispersion_interp(const vector<double>& x, int nn){
        
        char input_fname[MaxCharactersInString];
        // if(input_fname == nullptr)
        sprintf(input_fname, "%stxt/%s_%03d.secondhand.txt", path_gm.data(), bname, snap); //default path
        FILE *fp = fopen(input_fname, "r");
        if(!fp){
            printf("Faled to open file %s.\n", input_fname);
            return {0.,0.,0.};
        }
        fclose(fp);

        //matrix and target position
		VecDoub x_tgt_arr = {x[0], x[1], x[2]};
		vector<int> nearest_idx = pkdtree->find_k_nearest(nn, x_tgt_arr);
		MatrixXd Pts(nn, Dimension);
		VectorXd Vals_x(nn), Vals_y(nn), Vals_z(nn);
		VectorXd x_tgt_mat(Dimension);
		x_tgt_mat<<x[0], x[1], x[2];
        vector<double> dispersion_tgt = {0.,0.,0.};

        //read line where correspoding particle data located, Line[i] stores id=i (1 to NumPart)
    	for(int i=0;i<nn;i++){
            //reorder nearest_idx
            // float toolman; int toolman_d;
            float x0,y0,z0, sx0, sy0, sz0;

            FILE *fp = fopen(input_fname, "r");
            for(int line=1;line<nearest_idx[i];line++) fscanf(fp, "%*[^\n]%*c");
            fscanf(fp, "%e %e %e %*f %*f %*f    %*d %*d %*f        %*f %*f %*f    %*f %*f %*f    %e %e %e        ", &x0, &y0, &z0, &sx0, &sy0, &sz0);
            // for(int inn=0;inn<nn;inn++) fscanf(fp, "%d ", &idx);
            fscanf(fp, "\n");
            vector<double> position = {x0,y0,z0};
            vector<double> dispersion = {sx0,sy0,sz0};
            // printf("nearest_idx: %d\n", nearest_idx[i]);
            // printVector(position);
            // printVector(dispersion);
            fclose(fp);

    		for(int ix=0;ix<Dimension;ix++){
   	    		Pts(i,ix) = position[ix];
    		}
        	Vals_x(i) = dispersion[0];
        	Vals_y(i) = dispersion[1];
        	Vals_z(i) = dispersion[2];
   		}
        
        //interpolation //it is better to interpolation because of a position distribution
        //It is better to reanalysis position interaction by metric, then in RBF_interp myfunc(init); now simplify them all just by l2norm.
		double r0 = 1.;
		RBF_Gauss imq(r0); //bad, better
		RBF_interp myfunc_x(Pts, Vals_x, imq, 0);
		dispersion_tgt[0] = myfunc_x.interp(x_tgt_mat);
		RBF_interp myfunc_y(Pts, Vals_y, imq, 0);
		dispersion_tgt[1] = myfunc_y.interp(x_tgt_mat);
		RBF_interp myfunc_z(Pts, Vals_z, imq, 0);
		dispersion_tgt[2] = myfunc_z.interp(x_tgt_mat);

        return dispersion_tgt;
}



int Snapshot::process_tosecondhand(const double* const QP, int WhereWrite, int WhatCannonical, double* h_mat, int nknn){

		//:: Q: x or angle Thata->Omega, 3; P: v or action J, 3
		VecDoub Q_target_arr = {QP[0], QP[1], QP[2]};
		VecDoub P_target_arr = {QP[3], QP[4], QP[5]};
		VecDoub QP_target_arr = {QP[0], QP[1], QP[2], QP[3], QP[4], QP[5]};
        vector<int> nearest_idx;
        if(WhatCannonical==0){ //tree for x: 3*N
            // this->loadtree(pk, 0); //??
            nearest_idx = this->pkdtree->find_k_nearest(nknn, Q_target_arr);
        }else if(WhatCannonical==1){ //tree for v: 3*N
            nearest_idx = this->pkdtree1->find_k_nearest(nknn, P_target_arr);
        }else if(WhatCannonical==2){ //tree for x rho: 3*N
		    nearest_idx = this->pkdtree->find_k_nearest(nknn, Q_target_arr);
        }else if(WhatCannonical==3){ //tree for xv: 6*N
		    nearest_idx = this->pkdtree2->find_k_nearest(nknn, QP_target_arr);
        }

        else if(WhatCannonical==4){ //tree for O: 3*N
            nearest_idx = this->pkdtree->find_k_nearest(nknn, Q_target_arr);
        }else if(WhatCannonical==5){ //tree for J: 3*N
		    nearest_idx = this->pkdtree1->find_k_nearest(nknn, P_target_arr);
        }
        else if(WhatCannonical==6){ //tree for O Theta: 3*N
		    nearest_idx = this->pkdtree->find_k_nearest(nknn, Q_target_arr);
        }else if(WhatCannonical==7){ //tree for OJ: 6*N
		    nearest_idx = this->pkdtree2->find_k_nearest(nknn, QP_target_arr);
        }else{
			printf("Only x-v or angle-action mode provided now!\n"); //??
            exit(0);
			return -1;
		}

        // printf("Before sort():\n");
        // for(auto k : nearest_idx){
        //     double dd = distance_l2norm({QP[0], QP[1], QP[2]}, {wtfh[k].QP[0], wtfh[k].QP[1], wtfh[k].QP[2]});
        //     printf("%d(%e) ", k, dd);
        // }
        // printf("\n");
        // printf("After sort():\n");
        // sort(nearest_idx.begin(),nearest_idx.end());
        // for(auto k : nearest_idx){
        //     double dd = distance_l2norm({QP[0], QP[1], QP[2]}, {wtfh[k].QP[0], wtfh[k].QP[1], wtfh[k].QP[2]});
        //     printf("%d(%e) ", k, dd);
        // }
        // printf("\n");
        // exit(0);
        int inear = nearest_idx[0];
        int ifar = nearest_idx[nknn-1];
        //:: xv, H: search the neareast particle whose information are seemed as information on this target grid
        //:: it is better by knn and interpolation loosely
        double barP[3], barP2[3], sigP[3], xv[6];
        double DF, HJ, H, E_Gadget, I0;
        VecDoub dr_arr(nknn), m_arr(nknn);

        if(WhatCannonical==2){ //rho
            barP[0]=barP[1]=barP[2]=0., barP2[0]=barP2[1]=barP2[2]=0., sigP[0]=sigP[1]=sigP[2]=0., DF = 0.;

            int index = 0;
            for(auto i:nearest_idx){
                double QPknn[6] = {wtfh[i].QP[0], wtfh[i].QP[1], wtfh[i].QP[2], 
                                    wtfh[i].QP[3], wtfh[i].QP[4], wtfh[i].QP[5]};
                for(int ix=0;ix<Dimension;ix++){
                    barP[ix] += QPknn[ix+3];
                    barP2[ix] += pow(QPknn[ix+3],2);
                }
                //:: the kernel function W here is only versus r; max dr as bandwidth
                dr_arr[index] = distance_l2norm({QP[0], QP[1], QP[2]}, {QPknn[0], QPknn[1], QPknn[2]});
                m_arr[index] = wtfh[i].m;
                index ++;
            }
            double dr_max = *max_element(dr_arr.begin(),dr_arr.end());
            for(int i=0;i<index;i++){
                // printf("%e / %e, %e\n", dr_arr[i], dr_max, m_arr[i]);
                DF += Weight_SPHSmoothingKernel(dr_arr[i],dr_max)*m_arr[i];
            }
            // printf("rho = %e\n", DF);
            // exit(0);

            // for(auto i:nearest_idx){
            //     double QPknn[6] = {wtfh[i].QP[0], wtfh[i].QP[1], wtfh[i].QP[2], 
            //                         wtfh[i].QP[3], wtfh[i].QP[4], wtfh[i].QP[5]};
            //     for(int ix=0;ix<Dimension;ix++){
            //         barP[ix] += QPknn[ix+3];
            //         barP2[ix] += pow(QPknn[ix+3],2);
            //     }
            //     double h, dr;
            //     double m = wtfh[i].m;
            //     if(h_mat==nullptr){
            //         //:: the kernel function W here is only versus r; max dr as bandwidth
            //         h  = distance_l2norm({QP[0], QP[1], QP[2]}, {wtfh[ifar].QP[0], wtfh[ifar].QP[1], wtfh[ifar].QP[2]});
            //         dr = distance_l2norm({QP[0], QP[1], QP[2]}, {QPknn[0], QPknn[1], QPknn[2]});
            //         // if(dr<0. || h<=0.){ //not normal //??
            //         //     printf("Bad SPH smoothing length. Please check! Wrong kdtree prog."
            //         //         "\ndr = %e, h = %e, i = %d, ifar = %d\n", dr, h, i, ifar);
            //         //     printf("target: %e %e %e\n", QP[0], QP[1], QP[2]);
            //         //     printf("itemp : %e %e %e\n", wtfh[i].QP[0], wtfh[i].QP[1], wtfh[i].QP[2]);
            //         //     printf("far   : %e %e %e\n", wtfh[ifar].QP[0], wtfh[ifar].QP[1], wtfh[ifar].QP[2]);
            //         //     // exit(0);
            //         //     // swap(ifar,inear);
            //         //     // h  = distance_l2norm({QP[0], QP[1], QP[2]}, {wtfh[ifar].QP[0], wtfh[ifar].QP[1], wtfh[ifar].QP[2]});
            //         // }
            //         // printf("[%e] [%e] [%e]; ", m, QP[0], QPknn[0]);
            //         // if(i==inear || i==ifar) printf("the knn: %d %e %e %e; %e %e\n", i, wtfh[i].QP[3], wtfh[i].xv[0], QPknn[3], h, dr);

            //         DF += Weight_SPHSmoothingKernel(dr,h)*m;
            //     } else {
            //         //:: h matrix //??
            //         DF += 0.;
            //     }
            //     // DEBUG_PRINT_C("\n\n\n\n");
            // }

            for(int ix=0;ix<Dimension;ix++){
                barP[ix] /= nknn;
                barP2[ix] /= nknn;
                sigP[ix] = sqrt( barP2[ix]-barP[ix]*barP[ix] );
            }
            for(int i=0;i<6;i++) xv[i] = wtfh[inear].xv[i];
            H = 0.5*pow(distance_l2norm({xv[3],xv[4],xv[5]}),2) +wtfh[inear].phi;
            for(int i=0;i<3;i++) wtsh[WhereWrite].bar_PvQ[i] = barP[i];
            for(int i=0;i<3;i++) wtsh[WhereWrite].sig_PvQ[i] = sigP[i];
            wtsh[WhereWrite].DF_Q = DF;
        }

        if(WhatCannonical==5){ //DF_J
            barP[0]=barP[1]=barP[2]=0., barP2[0]=barP2[1]=barP2[2]=0., sigP[0]=sigP[1]=sigP[2]=0., DF = 0.;
            for(int i=0;i<6;i++) xv[i] = wtfh[inear].xv[i];
            H = 0.5*pow(distance_l2norm({xv[3],xv[4],xv[5]}),2) +wtfh[inear].phi;
            E_Gadget = 0.; //0.5*pow(distance_l2norm({xv[3],xv[4],xv[5]}),2) +wtfh[inear].phi_Gadget;
            I0 = H;

            int index = 0;
            for(auto i:nearest_idx){
                double QPknn[6] = {wtfh[i].QP[0], wtfh[i].QP[1], wtfh[i].QP[2], 
                                    wtfh[i].QP[3], wtfh[i].QP[4], wtfh[i].QP[5]};
                for(int ix=0;ix<Dimension;ix++){
                    barP[ix] += QPknn[ix+3];
                    barP2[ix] += pow(QPknn[ix+3],2);
                }
                //:: the kernel function W here is only versus r; max dr as bandwidth
                dr_arr[index] = distance_l2norm({QP[3], QP[4], QP[5]}, {QPknn[3], QPknn[4], QPknn[5]});
                m_arr[index] = wtfh[i].m;
                index ++;
            }
            double dr_max = *max_element(dr_arr.begin(),dr_arr.end());
            for(int i=0;i<index;i++){
                // printf("%e / %e\n", dr_arr[i], dr_max);
                DF += Weight_SPHSmoothingKernel(dr_arr[i],dr_max);
            }

            // for(auto i:nearest_idx){
            //     double QPknn[6] = {wtfh[i].QP[0], wtfh[i].QP[1], wtfh[i].QP[2], 
            //                         wtfh[i].QP[3], wtfh[i].QP[4], wtfh[i].QP[5]};
            //     for(int ix=0;ix<Dimension;ix++){
            //         barP[ix] += QPknn[ix+3];
            //         barP2[ix] += pow(QPknn[ix+3],2);
            //     }
            //     //// now determinate the bandwidth of kernel, i.e. the max distance to knn list
            //     //:: 6D data and its bandwith should bave a 6D tree
            //     // printf("ifar = %d, wtfh[ifar].xv[3] = %e, wtfh[ifar].phi = %e\n", ifar, wtfh[ifar].xv[3], wtfh[ifar].phi);
            //     double h, dr;
            //     if(h_mat==nullptr){
            //         //:: the kernel function W here is only versus r; max dr as bandwidth
            //         h = distance_l2norm({QP[3], QP[4], QP[5]}, {wtfh[ifar].QP[3], wtfh[ifar].QP[4], wtfh[ifar].QP[5]});
            //         dr = distance_l2norm({QP[3], QP[4], QP[5]}, {QPknn[3], QPknn[4], QPknn[5]});
            //         DF += Weight_SPHSmoothingKernel(dr,h); //*1
            //         printf("%d: %e /// %e\n", i, dr, h);
            //         // if(i==inear || i==ifar) printf("the knn: %d %e %e %e; %e %e\n", i, wtfh[i].QP[3], wtfh[i].xv[0], QPknn[3], h, dr);
            //     } else {
            //         //:: h matrix //??
            //         DF += 0.;
            //     }
            // }
            // exit(0);

            DF /= NumPart; //number -> frequency -> probability
            for(int ix=0;ix<Dimension;ix++){
                barP[ix] /= nknn;
                barP2[ix] /= nknn;
                sigP[ix] = sqrt( barP2[ix]-barP[ix]*barP[ix] );
            }
            for(int i=0;i<3;i++) wtsh[WhereWrite].bar_PvQ[i] = barP[i];
            for(int i=0;i<3;i++) wtsh[WhereWrite].sig_PvQ[i] = sigP[i];
            wtsh[WhereWrite].DF_P = DF;
            HJ = QP[0]*QP[3] +QP[1]*QP[4] +QP[2]*QP[5] +0.; //this formula is only for harmonic oscillator
            wtsh[WhereWrite].H_OJ = HJ;
        }

        // printf("wtfh near: %d %e %e %e %e\n", inear, wtfh[0].xv[0], wtfh[0].QP[3], wtfh[inear].xv[0], wtfh[inear].QP[3]);
		for(int i=0;i<6;i++) wtsh[WhereWrite].xv[i] = xv[i];
		for(int i=0;i<6;i++) wtsh[WhereWrite].QP[i] = QP[i];
		wtsh[WhereWrite].H = H;
        wtsh[WhereWrite].I[0] = H, wtsh[WhereWrite].I[1] = 0., wtsh[WhereWrite].I[2] = 0.;
		return 0;
}

int Snapshot::load_to_firsthand_from_PD(){

        int N = NumPart;
    	for(int i=0;i<N;i++){
            int ii = i+1;
            wtfh[i].xv[0] = P[ii].Pos[0], wtfh[i].xv[1] = P[ii].Pos[1], wtfh[i].xv[2] = P[ii].Pos[2],
            wtfh[i].xv[3] = P[ii].Vel[0], wtfh[i].xv[4] = P[ii].Vel[1], wtfh[i].xv[5] = P[ii].Vel[2],
            wtfh[i].QP[0] = P[ii].Pos[0], wtfh[i].QP[1] = P[ii].Pos[1], wtfh[i].QP[2] = P[ii].Pos[2],
            wtfh[i].QP[3] = P[ii].Vel[0], wtfh[i].QP[4] = P[ii].Vel[1], wtfh[i].QP[5] = P[ii].Vel[2],
            wtfh[i].ID = P[ii].Id, wtfh[i].m = P[ii].Mass, wtfh[i].phi = P[ii].Pot;
   		}
        printf("Load to write_firsthand from particle data ... done.\n");
		return 0;
}

/*  Read actions.
    @param WhatCannonical:
    2: positions x - tree;
    5: actions J - tree.
    @param WhatPotential:
    1: formula potential;
    2: data potential.
    @param WhatSymmetry:
    See switch.
    @param WhatActionmethod:
    1: Stackel Fudge;
    2: Integrate elliptical coordinates in a pseudo period orbit.
    3: Combine method, J_\lambda by 1,  J_\mu and J_\nu by 2.
*/
int Snapshot::read_firsthand_all_txt(int WhatCannonical, int WhatPotential, int WhatSymmetry, int WhatActionmethod){

        char input_fname[MaxCharactersInString];
        sprintf(input_fname, "%saa/%s_%03d.action.method_all.txt", path_gm.data(), bname, snap); //default path
        FILE *fp = fopen(input_fname, "r");
        if(!fp){
            printf("Fail to open file %s.\n", input_fname);
            exit(0);
            return -1;
        }

        if(!( 
            (WhatCannonical==2 || WhatCannonical==5) 
            && (WhatPotential==0 || WhatPotential==1)
            && (WhatSymmetry>=0)
            && (WhatActionmethod>=0 && WhatActionmethod<=2)
         )){
            printf("No such of tags provided. Donot write the file; continue.\n");
            return -2;
        }

        fscanf(fp, "%*[^\n]%*c"); //to remove the first line where the notes are
        //:: read line where correspoding particle data located, Line[i] stores id=i (txtdata indexes start from 1)
        //:: we select DPot instead of FPot here
        //:: when reading: double ~ le or lg or lf, and be wrong without l
        //:: .6: 6 decimal places instead of valid number (for sentific notation they are the same)
        //:: %e exp, %f float, %g auto
        int N = NumPart;
        // /*
    	for(int i=0;i<N;i++){
            //:: read
            int id; double m,p;
            VecDoub x, a, fo;
            FSCANF_V0d_VECDOUBLE(fp, x, DimTwo); //xv
            fscanf(fp, "%le %le ", &p, &m); //ID(%d), mass
            id = (int)p;

            for(int k=0;k<WhatPotential;k++){ //0 or 1
                SKIP_AA0(fp); //WhatSymmetry 0 //not used ??
                SKIP_AA0(fp); //WhatSymmetry 1
                SKIP_AA0(fp); //WhatActionmethod 0, StackelFudge
                SKIP_AA0(fp); //WhatActionmethod 1, TEPPOD
                SKIP_AA0(fp); //WhatActionmethod 2, O2GF
            }

            SKIP_AA0(fp); //WhatSymmetry 0
            SKIP_AA0(fp); //WhatSymmetry 1
            switch(WhatActionmethod){ //the distribution type of galaxy
                case 1: { //WhatActionmethod 1, TEPPOD
                    SKIP_AA0(fp);
                    FSCANF_V0d_VECDOUBLE(fp, a, LENGTH_ACTIONS_WRITE);
                    FSCANF_V0d_VECDOUBLE(fp, fo, LENGTH_ANGLESFREQUENCIES_WRITE);
                    SKIP_AA0(fp); //the rest
                    break;
                }
                case 2: { //WhatActionmethod 2, O2GF
                    SKIP_AA0(fp);
                    SKIP_AA0(fp);
                    FSCANF_V0d_VECDOUBLE(fp, a, LENGTH_ACTIONS_WRITE);
                    FSCANF_V0d_VECDOUBLE(fp, fo, LENGTH_ANGLESFREQUENCIES_WRITE);
                    break;
                }
                case 3: { //other like "quasileaner", "disorder", not provided
                    printf("No such now.\n");
                    exit(0);
                    break;
                }
                default: { //WhatActionmethod 0, StackelFudge
                    FSCANF_V0d_VECDOUBLE(fp, a, LENGTH_ACTIONS_WRITE);
                    a[1] = abs(a[1]); //might |Lz|
                    FSCANF_V0d_VECDOUBLE(fp, fo, LENGTH_ANGLESFREQUENCIES_WRITE);
                    SKIP_AA0(fp); //the rest
                    SKIP_AA0(fp); //the rest
                    break;
                }
            }

            fscanf(fp, "%*le %*le %le ", &p); //type(%d) Phi_F Phi_D Alpha Beta Gamma
            SKIP_UNTILENDL(fp);

            //:: load them to data struct
            for(int ii=0;ii<DimTwo;ii++){wtfh[i].xv[ii] = x[ii];}
            wtfh[i].ID = id, wtfh[i].m = m, wtfh[i].phi = p;
            
            wtfh[i].QP[3] = a[0], wtfh[i].QP[4] = a[1], wtfh[i].QP[5] = a[2];
            wtfh[i].type_actionJudje = a[3];
            wtfh[i].Theta[0] = fo[0], wtfh[i].Theta[1] = fo[1], wtfh[i].Theta[2] = fo[2];
            wtfh[i].QP[0] = fo[3], wtfh[i].QP[1] = fo[4], wtfh[i].QP[2] = fo[5];
            
            if(WhatCannonical==2){
                for(int j=0;j<DimTwo;j++){
                    wtfh[i].QP[j] = wtfh[i].xv[j];
                }
            }
            // printf("ID_%d(%e) ", i+1, wtfh[i].xv[0]);
   		}
        // */
        fclose(fp);
        printf("Load one-hand data from file %s ... done.\n", input_fname);
		return 0;
	}

int Snapshot::write_secondhand_all_txt(int WhatCannonical, int WhatPotential, int WhatSymmetry, int WhatActionmethod){

        char output_fname[MaxCharactersInString];
        sprintf(output_fname, "%saa/%s_%03d.secondhand_C%dP%dS%dA%d.txt", path_gm.data(), bname, snap, WhatCannonical, WhatPotential, WhatSymmetry, WhatActionmethod); //default path
        FILE *fp = fopen(output_fname, "w");
        if(fp==nullptr){
            printf("Cannot open file %s.\n", output_fname);
            return -1;
        }

        //:: write for each particle; it maybe chanegd by need
        int N = N_allPtcs;
        fprintf(fp, "## secondhand data: xv(6)     Omega(3) J(3) Theta(3)     ID_int DF H HJ \n");
        for(int i=0;i<N;i++){
            double QP[6] = {wtfh[i].QP[0], wtfh[i].QP[1], wtfh[i].QP[2], wtfh[i].QP[3], wtfh[i].QP[4], wtfh[i].QP[5]};
            this->process_tosecondhand(QP, i, WhatCannonical); //from wtfh to wtsh
            wtsh[i].ID = i+1;
            fprintf(fp, "%e %e %e %e %e %e     %e %e %e %e %e %e %e %e %e     %d %e %e %e     %e %e %e \n",    
                wtsh[i].xv[0], wtsh[i].xv[1], wtsh[i].xv[2], wtsh[i].xv[3], wtsh[i].xv[4], wtsh[i].xv[5],   
                wtsh[i].QP[0], wtsh[i].QP[1], wtsh[i].QP[2], wtsh[i].QP[3], wtsh[i].QP[4], wtsh[i].QP[5],   
                wtsh[i].Theta[0], wtsh[i].Theta[1], wtsh[i].Theta[2],       
                wtsh[i].ID, wtsh[i].DF_Q, wtsh[i].DF_P, wtsh[i].H_OJ, wtsh[i].I[0], wtsh[i].I[1], wtsh[i].I[2]
            );
            printf("ID_%d ", i+1);
        }
        fclose(fp);
        printf("\nWrite second-hand data to file %s ... done.\n", output_fname);
        return 0;
    }







//:: main
int main3(){
    // A=MatrixXd::Random(k_nn,k_nn); //Zero(m,n,V); //Ones(); //start from A(0,0)
    // B=MatrixXd::Random(k_nn,1);
    // C=A.ldlt().solve(B); //Cholesky decompose and solve linear eqs
}
