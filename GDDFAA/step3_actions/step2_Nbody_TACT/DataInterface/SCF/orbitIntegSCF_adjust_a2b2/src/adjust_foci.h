///////////////////////////////////////////////////////////
//// To adjust b2.
//// Author: Shiyan Zhong
///////////////////////////////////////////////////////////

#ifndef _ADJUST_FOCI_
#define _ADJUST_FOCI_

#include<stdio.h>
#include<math.h>
#include<gsl/gsl_poly.h>
#include"utilities.h"

////macros and consts
#define NMAX 1000000
#define R_macro 0.61803399
#define C_macro (1.0-R_macro)
#define TINY 1e-8
#define N_var 5
#define SHFT2(a,b,c) (a)=(b);(b)=(c);
#define SHFT3(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

const double a2_init = 1e2; //1e8; //gjy add
const double b2_init = 2.0;
const double c2_init = 1.0;
const double xstep = 1e0; //kpc
// const double xstep = 1.0e4; //kpc
const int ncall_max = 1000;



//wrapper class to get foci~energy table
template<int dim>
class SCFAEF{
public:
    // double xN[NMAX]; //[learn code] There can not be too much array in class
    double* xN = new double[NMAX];
    double* yN = new double[NMAX];
    double* zN = new double[NMAX];
    double* timeN = new double[NMAX];
    int ndata, ncall, N_pd = 0;
    double t_max = 100.0; // Gyr
    double xstep_min = 1e-3;
    double xstep_max = 1e3;
    double b2_min = c2_init+TINY;
    double a2_min = b2_min+TINY;
    double a2_max = 1e5;
    double b2_max = a2_max-TINY;
    vector<vector<double>> xyz_grid, vvv_grid;
    vector<double> t_grid;
    double a2_a2b2c2_txt=0., b2_a2b2c2_txt=0., c2_a2b2c2_txt=0.; //from old code txt to class variable
    double delta1_delta_i_txt=0., delta2_delta_i_txt=0.; //from old code txt to class variable

    SCFAEF(){
        //[learn code] should reset ncall when use
        //[learn code] the "Orbit.dat" is bigger and bigger because of fortran
        reset_grid();
    }

    ~SCFAEF(){
        reset_grid();
        delete xN;
        delete yN;
        delete zN;
        delete timeN;
    }

    void reset_grid();
    void reset_value();
    void reset_txtval();

    void solve_eq_3(double x[3], double abc2[3], double res[3]);
    void assign_lmn(double tmp[3], double abc2[3], double lmn[3]);
    double compute_dla(double abc2[3]);
    double golden_section_search(double ax, double bx, double cx, double abc2[3]);

    void load_orbit_data(char* fname, bool is_center=true); //centeralized in this function
    vector<double> get_orbit_info_by_idx(int idx);

    double foci_by_shape(int swit);

    int main_1_b2(double ef[N_var], char* fname);
    int main_2_a2(double ef[N_var], char* fname);
    int main_3_xyz_to_lmn(char* fname);
};





//templates cannot be defined in different files ??
template<int dim>
void SCFAEF<dim>::reset_grid(){
    vector<vector<double>>().swap(xyz_grid);
    vector<vector<double>>().swap(vvv_grid);
    vector<double>().swap(t_grid);
    N_pd = 0;
}

template<int dim>
void SCFAEF<dim>::reset_value(){
    // ndata = 0;
    ncall = 0;
}

template<int dim>
void SCFAEF<dim>::reset_txtval(){
    a2_a2b2c2_txt = 0.;
    b2_a2b2c2_txt = 0.;
    c2_a2b2c2_txt = 0.;
    delta1_delta_i_txt = 0.;
    delta2_delta_i_txt = 0.;
}

template<int dim>
void SCFAEF<dim>::load_orbit_data(char* fname, bool is_center)
{
    // FILE* fp = nullptr;
    // printf("Read orbit data ...\n");
    // if( !(fp=fopen(fname,"r")) ){
    //     printf("Can not open %s!\n", fname);
    //     exit(1);
    // }
    // // while( fscanf(finp,"%le %le %le %le %le %le %le %le\n",
    // //     &timeN[idx], &xN[idx], &yN[idx], &zN[idx], &rtmp, &rtmp, &rtmp, &rtmp)!=EOF && idx<NMAX)
    // // {
    // //     idx += 1;
    // // }
    // int idx = 0, fsc = EOF;
    // double ttmp = 0., rtmp = 0.;
    // vector<double> xtmp(dim), vtmp(dim);
    // reset_grid();
    // do{
    //     fscanf(fp,"%le ", &ttmp);
    //     for(int i=0;i<dim;i++){
    //         fscanf(fp,"%le ", &xtmp[i]);
    //     }
    //     for(int i=0;i<dim;i++){
    //         fscanf(fp,"%le ", &vtmp[i]);
    //     }
    //     fsc = fscanf((fp), "%*[^\n]%*c");
    //     t_grid.push_back(ttmp);
    //     xyz_grid.push_back(xtmp);
    //     vvv_grid.push_back(vtmp);
    //     ++idx;
    // }while(fsc!=EOF);
    // fclose(fp);
    // N_pd = idx;
    // std::cout<<"Count of orbit data points: "<<N_pd<<"\n";

    // //[] centeralize the orbit data
    // if(is_center){
    //     auto center_pos = UTILITIES::calculate_mass_center_unitmass<double, dim>(xyz_grid);
    //     UTILITIES::translate_vector<double, dim>(xyz_grid, center_pos);
    //     std::cout<<"The orbit data has been centralized with mean data point weight.\n";
    // }

    printf("Read orbit data ...\n");
    FILE* fp = fopen(fname,"r");
    if(!fp){
        printf("Can not open %s!\n", fname);
        exit(1);
    }

    // Read the entire file as a flat stream of doubles.
    // Each record is: t, x(1:dim), v(1:dim), pot  -> total 1 + 2*dim + 1 numbers.
    const int n_per_rec = 1 + 2*dim + 1; // here: 1 (t) + 3 (x) + 3 (v) + 1 (pot) = 8

    std::vector<double> buf;
    buf.reserve(1024);

    double val;
    while(fscanf(fp, "%le", &val) == 1){
        buf.push_back(val);
    }
    fclose(fp);

    if(buf.size() % n_per_rec != 0){
        fprintf(stderr,
            "Warning: orbit file %s has %zu doubles, not a multiple of %d. "
            "Truncating the last incomplete record.\n",
            fname, buf.size(), n_per_rec);
    }

    reset_grid();
    const size_t n_rec = buf.size() / n_per_rec;
    t_grid.reserve(n_rec);
    xyz_grid.reserve(n_rec);
    vvv_grid.reserve(n_rec);

    for(size_t k = 0; k < n_rec; ++k){
        size_t base = k * n_per_rec;
        double ttmp = buf[base + 0];

        std::vector<double> xtmp(dim), vtmp(dim);
        for(int i = 0; i < dim; ++i){
            xtmp[i] = buf[base + 1 + i];
        }
        for(int i = 0; i < dim; ++i){
            vtmp[i] = buf[base + 1 + dim + i];
        }
        // buf[base + 1 + 2*dim] is pot; we ignore it

        t_grid.push_back(ttmp);
        xyz_grid.push_back(xtmp);
        vvv_grid.push_back(vtmp);
    }
    N_pd = static_cast<int>(n_rec);
    std::cout<<"Count of orbit data points: "<<N_pd<<"\n";

    //[] centralize the orbit data
    if(is_center){
        auto center_pos = UTILITIES::calculate_mass_center_unitmass<double, dim>(xyz_grid);
        UTILITIES::translate_vector<double, dim>(xyz_grid, center_pos);
        std::cout<<"The orbit data has been centralized with mean data point weight.\n";
    }
}

template<int dim>
vector<double> SCFAEF<dim>::get_orbit_info_by_idx(int idx){
    return {xyz_grid[idx][0], xyz_grid[idx][1], xyz_grid[idx][2], 
        xyz_grid[idx][3], xyz_grid[idx][4], xyz_grid[idx][5]};
}

/*  To search the max vertical length difference of the near ellip picture
    @param swit:
        swit==0: orbit in yz plane;
        swit==1: orbit in xz plane.
*/
template<int dim>
double SCFAEF<dim>::foci_by_shape(int swit)
{
    if(!(swit==0 || swit==1)){
        std::cout<<"foci_by_shape(): Wrong index of orbit tag. "
            "The value of @param swit can be only (int)0 and (int)1. "
            "Please check. Exit.\n";
        exit(1);
    }
    double max_vertical_diff = -1.;
    int N_angleunit = 90;
    double rot = 0., rot_where_long = -1.;
    for(int i=0;i<N_angleunit;i++){
        rot = i*pi_8/N_angleunit/2.;
        vector<double> ang = {0., 0., rot};
        auto xyz_grid_1 = UTILITIES::SO3_vector<double, dim>(xyz_grid, ang);
        auto xt = UTILITIES::transpose(xyz_grid_1);

        //?? use line intersected dy dx

	    double y1max = (*max_element(xt[swit].begin(), xt[swit].end()))
            -(*min_element(xt[swit].begin(), xt[swit].end()));
	    double y2max = (*max_element(xt[swit+1].begin(), xt[swit+1].end()))
            -(*min_element(xt[swit+1].begin(), xt[swit+1].end()));
        double diff = sqrt(abs(y1max*y1max-y2max*y2max)); //the sqrt difference of long and short axis of the closed orbit
        if(max_vertical_diff<diff){
            max_vertical_diff = diff;
            rot_where_long = rot;
        }
        // max_vertical_diff = selectMax(max_vertical_diff, diff);
        // rot_where_long = selectMax(rot_where_long, rot);
    }
    std::cout<<"\nrotated angle from initial to orbit-shaped long axis for foci: "<<rot_where_long<<"\n\n";
    if(rot<pi_8/4.){
        std::cout<<"Orbit wrong shape with axis ratio! Use the foci by the max ratio.\n";
    }
    return max_vertical_diff;
}



//求解一元三次方程
template<int dim>
void SCFAEF<dim>::solve_eq_3(double x[3], double abc2[3], double res[3])
{
    int i;
    double a[4]; //3次多项式的系数，4项
    double z[6]; //3次多项式的根，包括实部和虚部

    double x2,y2,z2,alpha,beta,gamma;
    double fc;

    //计算多项式的系数
    x2 = x[0]*x[0];
    y2 = x[1]*x[1];
    z2 = x[2]*x[2];
    alpha = -1.0*abc2[0];
    beta  = -1.0*abc2[1];
    gamma = -1.0*abc2[2];

    a[0]=alpha*beta*gamma - beta*gamma*x2 - alpha*gamma*y2 - alpha*beta*z2;
    a[1]=alpha*beta + alpha*gamma + beta*gamma - (beta+gamma)*x2 - (alpha+gamma)*y2 - (alpha+beta)*z2;
    a[2]=alpha + beta + gamma - x2 - y2 - z2;
    a[3]=1.0;

    gsl_poly_complex_workspace *w
        = gsl_poly_complex_workspace_alloc(4);

    gsl_poly_complex_solve(a,4,w,z);
    gsl_poly_complex_workspace_free(w);

    for(i=0;i<3;i++)
    {
        res[i]=z[2*i];
        //printf("z%d = %.3f + %.3f j\n", i, z[2*i], z[2*i+1]);
    }

    /*
    //验证结果
    for(i=0;i<3;i++)
    {
        fc = x2/(lmn[i]+alpha) + y2/(lmn[i]+beta) + z2/(lmn[i]+gamma);
        printf("%.3f\n", fc);
    }
    */
}

//确定lambda,mu,nu的值
template<int dim>
void SCFAEF<dim>::assign_lmn(double tmp[3], double abc2[3], double lmn[3])
{
    int i,j,flag_nu,flag_mu,flag_lambda;
    double a2,b2,c2,val,lambda, mu, nu;

    a2 = abc2[0];
    b2 = abc2[1];
    c2 = abc2[2];

    // 将tmp[:]按照降序排列，直接对应到la,mu,nu
    for(i=0;i<3;i++)
        for(j=i+1;j<3;j++)
        if(tmp[i]<tmp[j])
        {
            val=tmp[i]; tmp[i]=tmp[j]; tmp[j]=val;
        }

    lmn[0]=tmp[0]; //lambda
    lmn[1]=tmp[1]; //mu
    lmn[2]=tmp[2]; //nu

    if( lmn[0] < lmn[1] )
    {
        printf("Error: la < mu\n la=%.16f  mu=%.16f\n", lmn[0], lmn[1]);
        printf("tmp= %.6f %.6f %.6f\n", tmp[0],tmp[1],tmp[2]);
    }
    if( lmn[1] < lmn[2] )
    {
        printf("Error: mu < nu\n mu=%.16f  nu=%.16f\n", lmn[1], lmn[2]);
        printf("tmp= %.6f %.6f %.6f\n", tmp[0],tmp[1],tmp[2]);
    }
}

//Compute delta_lambda
template<int dim>
double SCFAEF<dim>::compute_dla(double abc2[3])
{
    double x[3], lmn[3], tmp[3];
    double a2,b2,c2,la,mu,nu;
    double la_max=-1.0,la_min=1.0e20;
    int    i;

    //找出 la_max, la_min
    for(i=0;i<ndata;i++){
    
        if( timeN[i] > t_max ) break;
        x[0]=xN[i];
        x[1]=yN[i];
        x[2]=zN[i];
        // printf("sa: %e %e, ", la_min, la_max);
        solve_eq_3(x,abc2,tmp);
        // printf("sa: %e %e %e, ", tmp[0], tmp[1], tmp[2]);
        assign_lmn(tmp,abc2,lmn);
        // printf("sa: %e %e %e \n", lmn[0], lmn[1], lmn[2]);
        la = lmn[0]; mu=lmn[1]; nu=lmn[2];

        if( la > la_max ) la_max=la;
        if( la < la_min ) la_min=la;
    }

    // printf("a2 = %.4f b2 = %.4f c2 = %.4f,  Delta_la = %.16f\n", 
    //     abc2[0], abc2[1], abc2[2], la_max-la_min);
    // printf("xn: %e %e ", xN[0], xN[ndata-1]);
    // printf("sa2: %e %e %e, ", tmp[0], tmp[1], tmp[2]);
    // printf("sa3: %e %e %e \n", lmn[0], lmn[1], lmn[2]);
    // printf("sa1: %d %e %e, ", ndata, la_min, la_max);
    // if(la_max-la_min>1e-2) exit(1);
    return la_max-la_min;
}

//Copy from Numerical Recipe 2nd Ed. Chapter 10.1
template<int dim>
double SCFAEF<dim>::golden_section_search(double ax, double bx, double cx, double abc2[3])
{
    double f1,f2,x0,x1,x2,x3,tol=1.0e-7;
    x0=ax;
    x3=cx;
    if( fabs(cx-bx) > fabs(bx-ax) ){
        x1=bx;
        x2=bx+C_macro*(cx-bx);
    }else{
        x2=bx;
        x1=bx-C_macro*(bx-ax);
    }
    
    abc2[1]=x1;
    f1 = compute_dla(abc2); ncall++;
    abc2[1]=x2;
    f2 = compute_dla(abc2); ncall++;

    while( fabs(x3-x0) > tol*(fabs(x1)+fabs(x2)) ){
        if( f2 < f1 ){
            SHFT3(x0,x1,x2,R_macro*x1+C_macro*x3)
            f1=f2;
            abc2[1]=x2;
            f2 = compute_dla(abc2); ncall++;
        }else{
            SHFT3(x3,x2,x1,R_macro*x2+C_macro*x0)
            f2=f1;
            abc2[1]=x1;
            f1 = compute_dla(abc2); ncall++;
        }
    }

    if(f1<f2){
        printf("Delta lambda min = %3f\n", f1);
        return x1;
    }else{
        printf("Delta lambda min = %3f\n", f2);
        return x2;
    }
}

//main_1_b2
// vector<double> main_1_b2(void)
template<int dim>
int SCFAEF<dim>::main_1_b2(double ef[N_var], char* fname)
{
    FILE* finp;
    FILE* fout;
    int i;
    double x[3], abc2[3], lmn[3], tmp[3];
    double x1,x2,x3; //用于搜索使Delta_la最小化的b2值
    double dla1,dla2,dla3;
    double a2,b2,c2,la,mu,nu,delta1,delta2;
    double la_max=-1.0, la_min=1.0e20;
    double rtmp;

    // //read initial value
    // if( !(finp=fopen("a2b2c2.txt","r")) )
    // {
    //     printf("Can not open a2b2c2.txt!\n");
    //     exit(1);
    // }
    // fscanf(finp,"%lE\n%lE\n%lE\n",&a2,&b2,&c2);
    // fclose(finp);
    a2 = a2_init, b2 = b2_init, c2 = c2_init;
    printf("a2, b2, c2 11: %f %f %f\n", a2,b2,c2);


    //############################################################
    // 读入轨道数据
    printf("读入轨道数据并确定大致搜索区间...\n");
    if( !(finp=fopen(fname,"r")) ) //"Orbit.dat"
    {
        printf("Can not open %s!\n", fname);
        exit(1);
    }
    
    i=0;
    while( fscanf(finp,"%lE %lE %lE %lE %lE %lE %lE %lE\n",
        &timeN[i], &xN[i], &yN[i], &zN[i], &rtmp, &rtmp, &rtmp, &rtmp)!=EOF && i<NMAX)
    {
        i += 1;
    }
    fclose(finp);
    ndata=i;
    //############################################################
    // 找出包围了最小值的区间，x1,x2,x3


    abc2[0] = a2;
    abc2[2] = c2;
    x1 = c2; //b^2
    abc2[1] = x1 + 1.0e-6; //b^2
    dla1 = compute_dla(abc2); ncall++;

    x2 = x1 + xstep;
    abc2[1] = x2; //b^2
    dla2 = compute_dla(abc2); ncall++;

    while(1){
        x3 = x2+xstep;
        if( x3 >= a2 ){
            printf("Warning: b2 = a2, 请增加a^2的值.\n"
                "Now let a2=a2_init*(2.0+TINY), b2=a2_init*1.5 "
                "and end to search the best b2.\n");
            x3 = a2;

            // exit(1);
            b2 = a2_init*1.5, a2 = a2_init*2.0;
            ef[0] = -b2, ef[1] = -a2, ef[2] = -c2, ef[3] = 0., ef[4] = 0.;
            // if( !(fout=fopen("a2b2c2.txt","w")) ) //gjy add
            // {
            //     printf("Can not open a2b2c2.txt!\n");
            //     exit(1);
            // }
            // fprintf(fout,"%.5f\n%.5f\n%.5f\n", a2, b2, c2);
            // fclose(fout);
            this->a2_a2b2c2_txt = a2, this->b2_a2b2c2_txt = b2, this->c2_a2b2c2_txt = c2;
            printf("abc: 111, %f %f %f\n", a2,b2,c2);

            delta1 = sqrt(a2-b2);
            delta2 = sqrt(b2-c2);
            // if( !(fout=fopen("delta_i.txt","w")) )
            // {
            //     printf("Can not create delta_i.txt!\n");
            //     exit(1);
            // }
            // fprintf(fout,"%.5f\n%.5f\n", delta1, delta2);
            // fclose(fout);
            delta1_delta_i_txt = delta1, delta2_delta_i_txt = delta2;

            reset_value();
            return 1;
        }
        
        abc2[1] = x3; //b^2
        dla3 = compute_dla(abc2); ncall++;
        
        if( dla1 > dla2 && dla2 < dla3 )
            break; //找到搜索区间了
        x1 = x2, x2 = x3;
        dla1 = dla2, dla2 = dla3;

        // printf("ncncncnc = %d ", ncall);
        if(ncall>ncall_max){ //gjy add
            printf("It reaches to the max count of calling compute_dla. Let it break.\n", x1, x2, x3);
            break;
        }
    }
    printf("abc2[012]: a2 = %.4f, b2 = %.4f, c2 = %.4f, ncall = %d\n", abc2[0], abc2[1], abc2[2], ncall);

    printf("搜索区间: %.5f, %.5f, %.5f\n", x1, x2, x3);
    printf("compute_dla() 调用次数 = %d\n", ncall);
    b2 = golden_section_search(x1,x2,x3,abc2);
    printf("b^2 =  %.5f\n", b2);
    printf("compute_dla() 调用次数 = %d\n", ncall);


    // if( !(fout=fopen("a2b2c2.txt","w")) )
    // {
    //     printf("Can not open a2b2c2.txt!\n");
    //     exit(1);
    // }
    // fprintf(fout,"%.5f\n%.5f\n%.5f\n",a2,b2,c2);
    // fclose(fout);
    this->a2_a2b2c2_txt = a2, this->b2_a2b2c2_txt = b2, this->c2_a2b2c2_txt = c2;
    printf("abc: 12, %f %f %f\n", a2,b2,c2);


    delta1 = sqrt(a2-b2);
    delta2 = sqrt(b2-c2);
    // if( !(fout=fopen("delta_i.txt","w")) )
    // {
    //     printf("Can not create delta_i.txt!\n");
    //     exit(1);
    // }
    // fprintf(fout,"%.5f\n%.5f\n", delta1, delta2);
    // fclose(fout);
    delta1_delta_i_txt = delta1, delta2_delta_i_txt = delta2;

    ef[0] = -b2, ef[1] = -a2, ef[2] = -c2, ef[3] = delta2, ef[4] = delta1;
    reset_value();
    return 0;
}

//main_2_a2
// vector<double> main_2_a2(void)
template<int dim>
int SCFAEF<dim>::main_2_a2(double ef[N_var], char* fname)
{
    FILE* finp;
    FILE* fout;
    int i;
    double x[3], abc2[3], lmn[3], tmp[3];
    double x1,x2,x3; //用于搜索使Delta_la最小化的b2值
    double dla1,dla2,dla3;
    double a2,b2,c2,la,mu,nu,delta1,delta2;
    double la_max=-1.0, la_min=1.0e20;
    double rtmp;


    // if( !(finp=fopen("a2b2c2.txt","r")) )
    // {
    //     printf("Can not open a2b2c2.txt!\n");
    //     exit(1);
    // }
    // fscanf(finp,"%lE\n%lE\n%lE\n",&a2,&b2,&c2);
    // fclose(finp);
    a2 = this->a2_a2b2c2_txt, b2 = this->b2_a2b2c2_txt, c2 = this->c2_a2b2c2_txt;
    printf("abc: 21, %f %f %f\n", a2,b2,c2);


    //############################################################
    // 读入轨道数据
    printf("读入轨道数据并确定大致搜索区间...\n");
    if( !(finp=fopen(fname,"r")) ) //"Orbit.dat"
    {
        printf("Can not open %s!\n", fname);
        exit(1);
    }
    
    i=0;
    while( fscanf(finp,"%lE %lE %lE %lE %lE %lE %lE %lE\n",
        &timeN[i], &xN[i], &yN[i], &zN[i], &rtmp, &rtmp, &rtmp, &rtmp)!=EOF && i<NMAX)
    {
        i += 1;
    }
    fclose(finp);
    ndata=i;
    //############################################################
    // 找出包围了最小值的区间，x1,x2,x3
    
    
    abc2[1] = b2;
    abc2[2] = c2;
    x1 = b2; //a^2
    abc2[0] = b2+0.1; //a^2
    dla1 = compute_dla(abc2); ncall++;

    x2 = x1 + xstep;
    abc2[0] = x2; //a^2
    dla2 = compute_dla(abc2); ncall++;

    while(1){
        x3 = x2+xstep;
        abc2[0] = x3; //a^2
        dla3 = compute_dla(abc2); ncall++;
        
        if( dla1 > dla2 && dla2 < dla3 ) break; //找到搜索区间了
        x1 = x2; x2 = x3;
        dla1 = dla2; dla2 = dla3;

        if(ncall>ncall_max){ //gjy add
            printf("It reaches to the max count of calling compute_dla. Let it break.\n", x1, x2, x3);
            break;
        }
    }
    printf("搜索区间: %.5f, %.5f, %.5f\n", x1, x2, x3);
    printf("compute_dla() 调用次数 = %d\n", ncall);

    a2 = golden_section_search(x1,x2,x3,abc2);
    printf("a^2 =  %.5f\n", a2);
    printf("compute_dla() 调用次数 = %d\n", ncall);

    // if( !(fout=fopen("a2b2c2.txt","w")) )
    // {
    //     printf("Can not open a2b2c2.txt!\n");
    //     exit(1);
    // }
    // fprintf(fout,"%.5f\n%.5f\n%.5f\n",a2,b2,c2);
    // fclose(fout);
    this->a2_a2b2c2_txt = a2, this->b2_a2b2c2_txt = b2, this->c2_a2b2c2_txt = c2;
    printf("abc: 22, %f %f %f\n", a2,b2,c2);



    delta1 = sqrt(a2-b2);
    delta2 = sqrt(b2-c2);
    // if( !(fout=fopen("delta_i.txt","w")) )
    // {
    //     printf("Can not create delta_i.txt!\n");
    //     exit(1);
    // }
    // fprintf(fout,"%.5f\n%.5f\n", delta1, delta2);
    // fclose(fout);
    delta1_delta_i_txt = delta1, delta2_delta_i_txt = delta2;

    /*
    printf("a2 = %.2f, b2 = %.2f, c2=%.2f\n", a2, b2, c2);
    printf("la_max = %.5f, la_min = %.5f, Delta la=%.5f\n", la_max, la_min, la_max-la_min);
    printf("Finish, Output file : Metrics.csv!\n");
    */

    ef[0] = -b2, ef[1] = -a2, ef[2] = -c2, ef[3] = delta2, ef[4] = delta1;
    reset_value();
    return 0;
}

//main_3_xyz_to_lmn: two file, about b2 and a2
template<int dim>
int SCFAEF<dim>::main_3_xyz_to_lmn(char* fname)
{
    FILE* finp;
    FILE* fout;
    int i;
    double x[3], abc2[3], lmn[3], tmp[3];
    double P2,Q2,R2; //度规系数
    double a2,b2,c2,la,mu,nu,delta1,delta2;
    double la_max=-1.0, la_min=1.0e20;
    double rtmp, timeN, t_max;

    // if( !(finp=fopen("a2b2c2.txt","r")) )
    // {
    //     printf("Can not open a2b2c2.txt!\n");
    //     exit(1);
    // }
    // fscanf(finp,"%lE\n%lE\n%lE\n",&a2,&b2,&c2);
    // fclose(finp);
    a2 = this->a2_a2b2c2_txt, b2 = this->b2_a2b2c2_txt, c2 = this->c2_a2b2c2_txt;


    abc2[0] = a2;
    abc2[1] = b2;
    abc2[2] = c2;

    t_max = 100.0; // Gyr

    //############################################################

    if( !(finp=fopen(fname,"r")) ) //"Orbit.dat"
    {
        printf("Can not open %s!\n", fname);
        exit(1);
    }

    
    //############################################################

    i=0;
    while( fscanf(finp,"%lE %lE %lE %lE %lE %lE %lE %lE\n",
        &timeN, &x[0], &x[1], &x[2], &rtmp, &rtmp, &rtmp, &rtmp)!=EOF && i<NMAX)
    {
        if( timeN > t_max ) break;

        i++;
        solve_eq_3(x,abc2,tmp);
        assign_lmn(tmp,abc2,lmn);
        la = lmn[0]; mu=lmn[1]; nu=lmn[2];

        if( la == a2 )
        P2 = 1.0e20;
        else
        P2 = (la-mu)*(la-nu)/(4.0*(la-a2)*(la-b2)*(la-c2));
        
        if( la > la_max ) la_max=la;
        if( la < la_min ) la_min=la;

        if( mu == a2 || mu==b2)
        Q2 = 1.0e20;
        else
        Q2 = (mu-nu)*(mu-la)/(4.0*(mu-a2)*(mu-b2)*(mu-c2));

        if( nu == b2 || mu==c2)
        R2 = 1.0e20;
        else
        R2 = (nu-la)*(nu-mu)/(4.0*(nu-a2)*(nu-b2)*(nu-c2));


    }
    fclose(finp);

    delta1 = sqrt(a2-b2);
    delta2 = sqrt(b2-c2);
    // if( !(fout=fopen("delta_i.txt","w")) )
    // {
    //     printf("Can not create delta_i.txt!\n");
    //     exit(1);
    // }
    // fprintf(fout,"%.5f\n%.5f\n", delta1, delta2);
    // fclose(fout);
    delta1_delta_i_txt = delta1, delta2_delta_i_txt = delta2;

    printf("a2 = %.2f, b2 = %.2f, c2=%.2f\n", a2, b2, c2);
    printf("la_max = %.5f, la_min = %.5f, Delta la=%.5f\n", la_max, la_min, la_max-la_min);
    printf("Finish, Output file : delta_i.txt.\n");
    
    reset_value();
    return 0;
}

#endif