/*  ===================================================================================
    Author: Jianyu Gu
    Filename: DataInterface.h
    Description: An interface from Gadget simulation data to TACT action calculation
    ===================================================================================
*/

#ifndef _DATAINTERFACE_ //otherwise, recompilation
#define _DATAINTERFACE_

#include <math.h>
#include <vector>
#include <cmath>
#include <array>
#include <iostream>
#include <fstream>
#include <string>
#include "string.h"
#include <iomanip>
#include <memory>
#include <time.h> //to get time
#include <stdio.h> //to call getcwd()
#include <unistd.h> //to call getcwd()
#include <pwd.h> //to get user name
#include <Eigen/Eigen>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include "Gadget2FormatData_io.h"
extern "C"{
#include "SCF/SCF_coeff_pot/inc.h"
}
// #include "kdtree_knn/KDtree.hpp"
// #include "kdtree_knn/metrics.hpp"
#include "utils_basic.h"
#include "kdtree_knn.h"
#include "tree_potential.h"
#include "../general/coordtransforms/inc/coordsys.h"
#include "../general/coordtransforms/inc/coordtransforms.h"

using namespace std;
using namespace Eigen;



/////////////////////////////settings (should be in param files):
#define DEBUG_GJY
// #define POTENTIAL_INTERP
#define POTENTIAL_SCF

////#define
#define Dim 3 //One cannot changed to other dimension values in this project
#define DimTwo 6 //Dim*2
#define MinInterpPoints 32
#define MaxInterpPoints 64
#define CountSmallTimeStep 10 //9
#define CountLargeTimeStep 100
#define MaxClassParams 20
#define MaxSettingParams 200
#define MaxGalaxyComponents 10
#define LENGTH_TEPPOD 26
#define LENGTH_ACTIONS_WRITE 4
#define LENGTH_ANGLESFREQUENCIES_WRITE 6
#define UnitConvert_time 1.022 //(km/s)*Gyr/kpc = 1./3.0857e16 * 3.1536e7*1e9
#define MaxInterpDistance 5. //kpc
#define Err 1.e-18
// #define Err1 1.e-5
#define Err1 1.e-2
#define RadiusMinConsidered 5e-2
#define RadiusMaxConsidered 4e2
#define NotCalculatedActionsValue (-std::numeric_limits<double>::infinity())
#define NotCalculatedActionsType (-99)

////useful tools
typedef double TypeRealNumber;

#define isSign(a) 		( (a)>=0? 1 : 0 )
#define Sign_(a) 		( (a)>0? 1 : ((a)<0? -1 : 0) )
#define selectMin(a,b) 	( (a)<=(b)? (a) : (b) )
#define selectMax(a,b) 	( (a)<=(b)? (b) : (a) )

#define SKIP_AA0(fp) fscanf((fp), "%*le %*le %*le %*le     %*le %*le %*le %*le %*le %*le ");
#define SKIP_AA1(fp) fscanf((fp), "%*e %*e %*e     %*e %*e %*e %*e %*e %*e ");
#define SKIP_AA3(fp) fscanf((fp), "%*e %*e %*e %*e     %*e %*e %*e %*e %*e %*e %*e %*e %*e %*e %*e ");
#define SKIP_ENDL(fp) fscanf((fp), "\n");
#define SKIP_UNTILENDL(fp) fscanf((fp), "%*[^\n]%*c");
#define FSCANF_V0d_INT(fp, a) fscanf((fp), "%d ", &a);
#define FSCANF_V0d_FLOAT(fp, a) fscanf((fp), "%e ", &a);
#define FSCANF_V0d_DOUBLE(fp, a) fscanf((fp), "%le ", &a);
#define FSCANF_V0d_VECDOUBLE(fp, a, n) {\
	VecDoub().swap((a));\
	for(int i=0;i<(n);i++){\
		(a).push_back(0.);\
		fscanf((fp), "%le ", &((a)[i]));\
	}\
}

#define DEBUG_PRINT_I(v) {std::cout<<"Here_"<<(v)<<"\n"; fflush(stdout);}
// #define DEBUG_PRINT_V0d(is_continue, v) {\
// 	std::cout<<"DEBUG_PRINT: "<<(v)<<"\n";\
// 	if((is_continue)<0.1){exit(0);}\
// }
#define DEBUG_PRINT_V0d(is_continue, v, c) {\
	std::cout<<"DEBUG_PRINT: ";\
	std::cout<<(v)<<" ";\
	std::cout<<"("<<(c)<<")";\
	if((is_continue)>1.1){\
		std::cout<<"\n";\
	}\
	else if((is_continue)>0.1){\
		std::cout<<"    ";\
	}\
	else{\
		std::cout<<"\n";\
		exit(0);\
	}\
    fflush(stdout);\
}
template<typename T>
int DEBUG_PRINT_V1d(const int& is_continue, const std::vector<T> &v, 
	string c)
{ //not need extern and it can only defined here
	std::cout<<"DEBUG_PRINT: ";
	for(auto i:v) std::cout<<i<<" ";
	std::cout<<"("<<c<<")\n";
	if(is_continue>1.1){
		int I;
		std::cin>>I;
		if(I==0){exit(0);}
	}
	else if(is_continue>0.1){;}
	else{exit(0);}
    fflush(stdout);
	return is_continue;
}



////global
extern int      modelId; //about which model to tell, model identity
extern char     modelInfo[MaxCharactersInString]; //model information
extern char     modelPath[MaxCharactersInString]; //model path to load and put //example: GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/
extern double   Mass_vir; //Units: 1e10Msol, then G=43009.15694
extern int      components; //number of components
extern int      N_allPtcs; //total number of all particles
extern double   TACT_semiaxis_Alpha, TACT_semiaxis_Beta, TACT_semiaxis_Gamma;

extern int      N_comp[MaxGalaxyComponents]; //number of each component of particles //index (i) in prog ~ index (i+1) in paramsfile
extern int		type_comp[MaxGalaxyComponents]; //type of each component of particles
extern double   frac_mass_comp[MaxGalaxyComponents]; //mass fraction of each
extern double   m_target_comp[MaxGalaxyComponents]; //mass of target particle of each
extern double   softening_comp[MaxGalaxyComponents]; //softening of each //unused
extern double	softening_type[6]; //softening of type
extern double   scale_length_comp[MaxGalaxyComponents]; //scale length of each
extern double   spin_L[MaxGalaxyComponents]; //spin of each
extern double   flatx_comp[MaxGalaxyComponents], flaty_comp[MaxGalaxyComponents], flatz_comp[MaxGalaxyComponents]; 
					//approximate axis ratio (flattening) of the triaxial galaxy
extern double   other_params[MaxGalaxyComponents]; //other params to spare

const int       Dimension = 3; //dimention number of space, 3d
const int       files = 1; //number of files per snapshot; in this prog, one can only use value 1, one file
const char      bname[MaxCharactersInString] = "snapshot";
const char      name_type[][MaxCharactersInString] = {"gas", "halo", "stars", "bndry"};
// const int 		N_total = 1000000; //no use
extern const VecDoub* const Substance_of_VecDoub_as_nullptr;

////namespace
namespace GJY{
	VecDoub samples_compared(int rs, int ns);

	//linear space 1d
	template<typename c>
	void GenLinspace1D(c*& v, c a, c b, int N){
		c d = (b-a)/(N-1);
		for(int i=0;i<N;i++){
			v[i] = a + i*d;
			// printf("GenLinspace1D(): %d %e\n", i, v[i]);
		}
	}

	template<typename c> //float, double; not complex number
	void gen_logspace1D(c*& v, c a, c b, int N, c basenumber=10.){
		if(a<=0||b<=0){
			printf("gen_logspace1D(): The number by log set should be positive.\n");
			exit(1);
		}
		c loga = log(a)/log(basenumber);
		c logb = log(b)/log(basenumber);
		c d = (logb-loga)/(N-1);
		for(int i=0;i<N;i++){
			v[i] = loga + i*d;
			v[i] = exp(v[i]*log(basenumber));
		}
	}

	template<typename c>
	vector<c> gen_logspace1D(c a, c b, int N, c basenumber=10.){
		if(a<=0||b<=0){
			printf("gen_logspace1D(): The number by log set should be positive.\n");
			exit(1);
		}
		vector<c> v(N);
		c loga = log(a)/log(basenumber);
		c logb = log(b)/log(basenumber);
		c d = (logb-loga)/(N-1);
		for(int i=0;i<N;i++){
			v[i] = loga + i*d;
			v[i] = exp(v[i]*log(basenumber));
		}
		return v;
	}
}

//////////////////////////////functions and symbols:
typedef vector<double> VecDoub;
typedef vector< vector<double> > MatDoub;

extern int write_data_debug(const vector<struct data_debug>& vdd, string filename);
extern VecDoub read_pure_number(string path, int N_v);
extern int read_params(string path_IC_1);
extern int write_value_txt(double** valuestruct, int N_writeline, int N_variables, 
	char* fname, int tag=0); //[learn code]: ["double *a"] means (*a) is double instead that a is * or a is double*; it is same like function pointer array.
extern void initialize_write_angleaction(struct write_angleaction& WAA);
extern int read_action_samples(struct write_angleaction*& WAA, int snapshot, string path_gm_1, int N_specify);
extern int write_action_data(const write_angleaction* WAA, int snapshot, string path_gm_1, int N, bool is_samples=false); //to write all actions

extern int t_to_loadsnapshot(double t, double dt_load1, double t_init1, double t_alignment=0.); //From t to the front snapshot ID; the later snapshot should +1
extern double loadsnapshot_to_t(int snapshot, double dt_load1, double t_init1, double t_alignment=0.); //From snapshot to t
extern int t_to_snapshot(double t, double dt_step1, double t_init1_notused=0., double t0_alignment=0.);
extern double snapshot_to_t(int snapshot, double dt_step1, double t_init1_notused=0., double t0_alignment=0.);



/* 	If some correspond elements from A to B has all changed sign, 
	it return true; otherwise, it return false. A product p0*p1<=0 
	(p0!=0) means "p0 changed sign to p1".
*/
template<typename T>
bool is_ChangedSign_vecAny(const std::vector<T> &A, const std::vector<T> &B){
	bool x = true;
	for(int i=3;i<6;i++){
		x = (x && A[i]*B[i]<=0);
	}
	return x;
}

//interpolation of between 2 points //mainly for potential on t
extern double interp_linear_2d_2points(double x, double x1, double y1, double x2, double y2);

//parabola_2d_3points_y_x: params(double* points) are form of {x1,y1,x2,y2,x3,y3}.
extern double parabola_2d_3points_y_x(double x, const VecDoub& points);

//standardize_ellipcoor_range_with_axislength
extern void standardize_ellipcoor_range_with_axislength(double& tau, const int& swit, 
	const double& alpha, const double& beta, const double& gamma);

//combine actions output between method_fudge and method_directorbit
extern void combine_actions_output(string f1, string f2, string f, int during_replace, int NP);

////inline
inline std::string getUserName(){
    struct passwd* pwd;
    uid_t userid;
    userid = getuid();
    pwd = getpwuid(userid);
    return pwd->pw_name;
}
inline std::string get_workpath(){ //to use: func("%s/XX", get_workpath().data());
    std::string workpath;
	string path_current = getcwd(NULL, 0); //the folder where the .exe is (decided by how the .exe be make)
    if(getUserName()=="darkgaia") //the original user
        workpath = "../../../../";
    else //other user
        workpath = "../../../../"; //the folder XXXX_work that where GDDFAA is; the relatve path is from XXXX_work/GDDFAA/actions/TACT/aa/main.exe
        // workpath = "/home/"+getUserName()+"/"+"workroom/0prog/GDDFAA_work/";
		// workpath = "your/path/";
    return workpath;
}

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {
	// initialize original index locations
	vector<size_t> idx(v.size());
	iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	// using std::stable_sort instead of std::sort
	// to avoid unnecessary index re-orderings
	// when v contains elements of equal values 
	stable_sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
	return idx;
}

template <typename T>
vector<T> assign_by_indexes(){
	;
}

//overload operator for data class "+" ??
template<typename T>
// typename std::enable_if<std::is_same<std::nullptr_t, T>::value || std::is_pointer<T>::value>::type
T lpnorm_real_with_distance_and_coef(const vector<T>* pa, const vector<T>* pb=nullptr, 
	const vector<T>* pc=nullptr, double pp=2)
{
	T sum = (T)0;
	vector<T> a = *pa, b, c;
	size_t n = a.size();
	if(pp==0){ //to return the count of elements that is none zero
		int idx = 0;
		for(auto i:a){
			if(abs(i)<Err){
				idx++;
			}
		}
		return (T)idx;
	}else{
		if(pp==std::numeric_limits<T>::infinity()){ //to return the max element
			return *max_element(a.begin(), a.end());
		}
	}

	if(pb==nullptr){
		b.resize(n, (T)0);
	}else{
		b = *pb;
	}
	if(pc==nullptr){
		c.resize(n, (T)1);
	}else{
		c = *pc;
	}
	if( !( n>=b.size() && n>=c.size() ) ){
		std::cerr<<"The sizes of vectors are not equal. Exit.\n";
		exit(1);
	}
	for(int i=0;i<a.size();i++){ //to return the usual norm of a real number vector
		sum += abs(pow((a[i]-b[i])*c[i], pp));
	}
	return (T)pow(sum, 1./pp);
}

inline double distance_l2norm(const vector<double>& a){
	double sum = 0.;
	for(int i=0;i<a.size();i++){
		sum += (a[i]-0)*(a[i]-0);
	}
	return sqrt(sum);
}
inline double distance_l2norm(const vector<double>& a, const vector<double>& b){ //array<>
	double sum = 0.;
	if(a.size() != b.size()){
		cerr<<"Unequal size! return value 0."<<endl;
		return 0;
	}
	for(int i=0;i<a.size();i++){
		sum += (a[i]-b[i])*(a[i]-b[i]);
	}
	return sqrt(sum);
}

inline VecDoub CartesianToSPolar(const VecDoub& Cartesian){ //from Sanders TACT
	double r = sqrt(Cartesian[0]*Cartesian[0]+Cartesian[1]*Cartesian[1]+Cartesian[2]*Cartesian[2]);
    VecDoub SPolar = {r,atan2(Cartesian[1],Cartesian[0]),acos(Cartesian[2]/r)};
    if(Cartesian.size()==3)	return SPolar;
    SPolar.push_back((Cartesian[3]*cos(SPolar[1])+Cartesian[4]*sin(SPolar[1]))*sin(SPolar[2])+cos(SPolar[2])*Cartesian[5]); //add velocities
    SPolar.push_back(-Cartesian[3]*sin(SPolar[1])+Cartesian[4]*cos(SPolar[1]));
    SPolar.push_back((Cartesian[3]*cos(SPolar[1])+Cartesian[4]*sin(SPolar[1]))*cos(SPolar[2])-sin(SPolar[2])*Cartesian[5]);
    return SPolar;
}

inline VecDoub SPolarToCartesian(const VecDoub& Spherical){
    VecDoub SPolar = {Spherical[0]*sin(Spherical[2])*cos(Spherical[1]),
    					Spherical[0]*sin(Spherical[2])*sin(Spherical[1]),
    					Spherical[0]*cos(Spherical[2])};
    if(Spherical.size()==3)	return SPolar;
    return SPolar;
}

template<class c>
inline void print_vec(const std::vector<c> &a){
    for(unsigned int i=0;i<a.size();i++) std::cout<<a[i]<<" "; std::cout<<"\n";
}

template<typename T>
vector<T> read_pure_number_template(string filename, int N_v){
    FILE *fp = nullptr;
    fp = fopen(filename.data(), "r");
    if(fp==nullptr){
        printf("Cannot open file \"%s\".\n", filename.data());
        exit(0);
    }
    VecDoub a;
    int i = 0;
    T b;
    while(i<N_v){
        if(typeid(b).name()=="int"){
            fscanf(fp, "%d", &b);
        }
        if(typeid(b).name()=="float"){
            fscanf(fp, "%e", &b);
        }
        if(typeid(b).name()=="double"){
            fscanf(fp, "%le", &b);
        }
        if(typeid(b).name()=="string"){ //??
            char* bchar;
            fscanf(fp, "%s", &bchar);
            b = bchar;
        }else{
            cout<<"The typeid name is "<<typeid(b).name()<<", No such type provided. Exit.\n";
            exit(-1);
        }
        a.push_back(b);
        i++;
    }
    // while(!feof(fp){i++}
DEBUG_PRINT_V1d(1, a, "read");
    fclose(fp);
    return a;
}

inline VecDoub CoorTrans(const VecDoub& xv, int coor){
    auto xv_ = xv;
	switch(coor){
		case 0:	{ //Cartesian
		    break;
		}
        case 1: { //SphericalPolar
            xv_ = conv::CartesianToSphericalPolar(xv);
            break;
        }
        case 2: { //Polar
            xv_ = conv::CartesianToPolar(xv);
            break;
        }
        default: { //other
            printf("No such coordinate set, use Cartesian ...\n");
            break;
        }
    }
    return xv_;
}



///////////////////////////////interp funcs:
////RBF interpolation (Ref: <<C++ Recipes>> E3. Chap3.7)
struct RBF_fn{
/* base class template for any particular radial basis function(RBF) */
	virtual double rbf(double r) = 0;
};
struct RBF_multiquadric: RBF_fn{
/* RBF of multiquadric */
	double r02;
	RBF_multiquadric(double scale=1.): r02(sqrt(scale)){} //scale ~ soften??
	double rbf(double r){ return sqrt(sqrt(r)+r02); }
};
struct RBF_inversemultiquadric: RBF_fn{
/* RBF of inverse multiquadric */
	double r02;
	RBF_inversemultiquadric(double scale=1.): r02(sqrt(scale)){}
	double rbf(double r){ return 1./sqrt(sqrt(r)+r02); }
};
struct RBF_Gauss: RBF_fn{
/* RBF of Gauss kernel */
	double r0;
	RBF_Gauss(double scale=1.): r0(scale){}
	double rbf(double r){ return exp(-0.5*sqrt(r/r0)); }
};
struct RBF_thinplate: RBF_fn{
/* RBF of thin-plate spline */
	double r0;
	RBF_thinplate(double scale=1.): r0(scale){}
	double rbf(double r){ return r <= 0. ? 0. : sqrt(r)*log(r/r0); }
};

struct RBF_interp {
/* Radial basis function interpolation using n points in dim dimensions. */
	int n, dim; 		  //number of points; dimention of points' space
	const MatrixXd &pts;  //position of points
	const VectorXd &vals; //function values
	VectorXd w; 		  //weight of each RBF
	RBF_fn &fn; 		  //select of RBF
	bool normalization;   //is normalization

	RBF_interp(MatrixXd &ptss, VectorXd &valss, RBF_fn &func, bool nrbf=false):
	dim(ptss.cols()), n(ptss.rows()) , pts(ptss), vals(valss), w(n), fn(func), normalization(nrbf){
		int i,j;
		double sum;
		MatrixXd RBF(n,n);
		VectorXd RHS(n);
		for(i=0;i<n;i++){
			sum = 0.;
			for(j=0;j<n;j++){
				RBF(i,j) = fn.rbf( distance_l2norm_(pts.row(i), pts.row(j)) );
				sum += RBF(i,j);
			}
			// VectorXd testVec(3); testVec=pts.row(i).transpose(); //or omit .transpose() for VectorX_ assignment
			// cout<<"testVec: \n"<<testVec<<endl;
			if(normalization) RHS[i] = sum*vals[i];
			else RHS[i] = vals[i];
		}
		w = RBF.ldlt().solve(RHS);
		// cout<<"RBF=\n"<<RBF<<endl;
		// cout<<"w=\n"<<w<<endl;
	}

	double interp(const VectorXd &pt){
		double fval, sum=0., sumw=0.;
		if(pt.size() != dim) throw("RBF_interp bad size!");
		for(int i=0;i<n;i++){
			fval = fn.rbf( distance_l2norm_(pt, pts.row(i)) );
			sumw += w[i]*fval;
			sum += fval;
		}
		return normalization ? sumw/sum : sumw;
	}

	double distance_l2norm_(const VectorXd& p1, const VectorXd& p2){
		double sum = 0.;
		for(int i=0;i<dim;i++){
			sum += (p1[i]-p2[i])*(p1[i]-p2[i]);
		}
		// cout<<"distance_l2norm_(): p1,p2: "<<sqrt(sum)<<endl;
		return sqrt(sum);
	}
};



///////////////////////////////classes of data:
class InterpSplineP3D1{
public:
	VecDoub x_samples;
	VecDoub y_samples;
	int N_nodes;
	int n;
	MatrixXd Coef; //coefficients of spline3 interpolation
	// friend class Snapshot;
	
	InterpSplineP3D1(){}
	void reset_samples_and_coefs(const VecDoub& x_samples1, const VecDoub& y_samples1){
		N_nodes = x_samples1.size();
		if( N_nodes!=y_samples1.size() ){
			printf("Sizes of x samples and y samples are not equal. Exit.\n");
			exit(1);
		}
		DEBUG_PRINT_I(443);
		x_samples = x_samples1;
		y_samples = y_samples1;
		sort(x_samples.begin(), x_samples.end()); //??
		n = N_nodes-1;
		// int n = N_nodes-1; //[learn code] donot redefine, otherwise the n 
		//\ might be a uncontrollable value such as very much, then the prog 
		//\ may occupy much memory and let the system stuck and crash
		DEBUG_PRINT_V0d(10, n, "n");
		Coef = MatrixXd::Zero(4, n);
		calculate_Coef();
		cout<<"Coef: \n"<<Coef<<"\n";
	};

	void calculate_Coef(){
		// DEBUG_PRINT_V0d(10, n, "X1");
		VectorXd X; X = VectorXd::Zero(n+1); //x
		VectorXd Y(n+1); Y = VectorXd::Zero(n+1); //y
		VectorXd H(n); H = VectorXd::Zero(n); //dx
		VectorXd G(n); G = VectorXd::Zero(n); //dy
		MatrixXd HH(n+1,n+1); HH = MatrixXd::Zero(n+1,n+1); //matrix, width
		VectorXd M(n+1); M = VectorXd::Zero(n+1); //to solve, second serive at nods
		VectorXd V(n+1); V = VectorXd::Zero(n+1); //right hand side
		for(int i=0;i<n+1;i++){
			X(i) = x_samples[i];
			Y(i) = y_samples[i];
		}
		for(int i=0;i<n;i++){
			H(i) = X(i+1)-X(i);
			G(i) = Y(i+1)-Y(i);
			if(abs(H(i))<Err){H(i)=Err;} //notenote: to avoid same-x points and bad orfer points
		}
		for(int i=1;i<=n-1;i++){
			HH(i,i-1) = H(i-1), HH(i,i) = (H(i-1)+H(i))*2, HH(i,i+1) = H(i-1); 
			//notenote: ()*2 instead of ()/2, be careful
			V(i) = G(i)/H(i)-G(i-1)/H(i-1);
		}
		//::boundary condition 1: ''=0;
		HH(0,0) = 1., HH(n,n) = 1.;
		//::boundary condition 2: '=C;
		// HH(0,0) = 2*H(0), HH(0,1) = H(0), HH(n-1,n) = H(n-1), HH(n,n) = 2*H(n-1);
		//::boundary condition 3: '''=0;
		// HH(0,0) = -H(1), HH(0,1) = H(0)+H(1), HH(0,2) = -H(0);
		// HH(n,n-2) = -H(n-1), HH(n,n-1) = H(n-2)+H(n-1), HH(n,n) = -H(n-2);
		M = HH.lu().solve(V*6); //notenote: lu() is better than ldlt(), the later may not have M(0)==0

		VectorXd A(n); A = VectorXd::Zero(n);
		VectorXd B(n); B = VectorXd::Zero(n);
		VectorXd C(n); C = VectorXd::Zero(n);
		VectorXd D(n); D = VectorXd::Zero(n);
		for(int i=0;i<n;i++){
			A(i) = Y(i);
			B(i) = (Y(i+1)-Y(i))/H(i) -M(i)*H(i)/2 -(M(i+1)-M(i))*H(i)/6;
			C(i) = M(i)/2;
			D(i) = (M(i+1)-M(i))/H(i)/6;
			Coef(0,i) = A(i), Coef(1,i) = B(i), Coef(2,i) = C(i), Coef(3,i) = D(i);
		}
	}

	double target_interp(double tau){
		int i = 0;
		double ptau = 0.;
		double taumin = x_samples[0];
		double taumax = x_samples[n];
		double dx;
		if( !( taumin<=tau && tau<=taumax ) ){ //swit and leaf index //sort before
			// printf("Out of range of inner interpolation, please check! Now use the boundary points.\n");
			// printf("tau_target = %e, tau_min = %e, tau_max = %e\n", tau, taumin, taumax);
			// return 0.;
			if(abs(tau-taumin)<Err){
				dx = taumin;
			}else{
				dx = taumax;
			}
		}else{
			while( !(x_samples[i]<=tau && tau<=x_samples[i+1]) ){
				i++;
				if(i>=n-1){break;}
			}
			dx = tau-x_samples[i];
		}
		// DEBUG_PRINT_V0d(10, i, "interp interval i");
		ptau = Coef(0,i) +Coef(1,i)*dx +Coef(2,i)*dx*dx +Coef(3,i)*dx*dx*dx;
		return ptau;
	}
};

class TriaxializeCoordinate{ //preprocess_snapshot
private:
public:
	VecDoub C; //mean_coordinates_Cartesian
	VecDoub O; //total_rotation_frequencies
	MatrixXd T; //redirection_mainAxis_interia

	TriaxializeCoordinate(){
		reset_parameters();
	}

	int calculate_parameters(int snapshot, int tag=0){
		/* It is in python programs but not provided here now. */
		return 0;
	}

	int read_parameters_preprocess(int snapshot, string path_gm_1){
	    char wt_fname[MaxCharactersInString];
		sprintf(wt_fname, "%saa/snapshot_%d_triaxialize.txt", path_gm_1.data(), snapshot);
		FILE *fp = fopen(wt_fname, "r");
		if(fp==nullptr){
			printf("Cannot open file \"%s\" for parameters for preprocess snapshot.\n", wt_fname);
			printf("Do not preprocess it. Set the parameters trivial.\n");
			fflush(stdout);
			reset_parameters();
			return -1;
		}
		double a[9];
		SKIP_UNTILENDL(fp);
        fscanf(fp, "%le %le %le %le %le %le%*[^\n]%*c", 
			&a[0], &a[1], &a[2], &a[3], &a[4], &a[5]);
		for(int i=0;i<Dim*2;i++){
			C[i] = a[i];
		}
        fscanf(fp, "%le %le %le %le %le %le %le %le %le%*[^\n]%*c", 
			&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7], &a[8]);
		for(int i=0;i<Dim;i++){
			for(int j=0;j<Dim;j++){
				T(i,j) = a[Dim*i+j];
			}
		}
        fscanf(fp, "%le %le %le%*[^\n]%*c", 
			&a[0], &a[1], &a[2]);
		for(int i=0;i<Dim;i++){
			O[i] = a[i];
		}
        fclose(fp);
		printf("Read file `%s` for parameters for preprocess snapshot ... Done.\n", wt_fname);
		return 0;
	}

	VecDoub centralize(const VecDoub& xv){
		auto xv_ = xv;
		for(int i=0;i<Dim*2;i++){
			xv_[i] -= C[i];
		}
		return xv_;
	}

	VecDoub run_total_rotation(const VecDoub& xv){
		auto xv_ = xv;
		VecDoub x = {xv[0], xv[1], xv[2]};
		VecDoub v = {xv[3], xv[4], xv[5]};
		VecDoub dv = {
			O[1]*x[2]-O[2]*x[1], 
			O[2]*x[0]-O[0]*x[2], 
			O[0]*x[1]-O[1]*x[0]
		};
		for(int i=0;i<Dim;i++){ //only velocities
			xv_[i+Dim] = v[i]-dv[i];
		}
		// DEBUG_PRINT_V1d(1, xv, "xv");
		// DEBUG_PRINT_V1d(1, dv, "dv");
		// DEBUG_PRINT_V1d(0, xv_, "xv_");
		return xv_;
	}

	VecDoub run_redirection_mainAxis(const VecDoub& xv){
		auto xv_ = xv;
		MatrixXd X(1, Dim), V(1, Dim), X_(1, Dim), V_(1, Dim);
		// MatrixXd X(Dim, 1), V(Dim, 1), X_(Dim, 1), V_(Dim, 1);
		for(int i=0;i<Dim;i++){
			X(i) = xv[i];
			V(i) = xv[i+Dim];
		}
		X_ = X*T;
        // DEBUG_PRINT_V0d(10, X, "X");
        // DEBUG_PRINT_V0d(10, T, "T");
        // DEBUG_PRINT_V0d(10, X_, "X_");
		V_ = V*T;
		for(int i=0;i<Dim;i++){
			xv_[i] = X_(i);
			xv_[i+Dim] = V_(i);
		}
		return xv_;
	}

	VecDoub new_coordinate(const VecDoub& xv, 
		bool is_total_rotation=false, bool is_redirection_mainAxis=true)
	{
		// DEBUG_PRINT_V1d(1, xv, "new_coordinate(): xv 0");
		// DEBUG_PRINT_V1d(1, C, "C");
		// DEBUG_PRINT_V0d(10, T, "T");
		// DEBUG_PRINT_V1d(1, O, "O");
		auto xv_ = centralize(xv);
		// DEBUG_PRINT_V1d(1, xv_, "new_coordinate(): xv_ 1");
		if(is_redirection_mainAxis){
			xv_ = run_redirection_mainAxis(xv_);
		}
		// DEBUG_PRINT_V1d(1, xv_, "new_coordinate(): xv_ 2");
		if(is_total_rotation){
			xv_ = run_total_rotation(xv_);
		}
		// DEBUG_PRINT_V1d(1, this->O, "rotation");
		// DEBUG_PRINT_V1d(0, xv_, "new_coordinate(): xv_ 3");
		return xv_;
	}

	void reset_parameters(){
		// C.resize(Dim*2, 0.);
		// O.resize(Dim, 0.);
		// T.resize(Dim, Dim);
		C = {0., 0., 0., 0., 0., 0.};
		O = {0., 0., 0.};
		T = MatrixXd::Zero(Dim, Dim);
		for(int i=0;i<Dim;i++){T(i,i) = 1.;} //unit, do nothing
	}
};

class Galaxy_components{ //note: this is for FPot, but FPot is not used for calculation
private:

public:
	Galaxy_components(){
		;
	}
	
	int count_components = 0;
	int model_ID_comp[MaxGalaxyComponents];
	double scaled_density_fit_comp[MaxGalaxyComponents]; //all 1.
	double scaled_length_fit_comp[MaxGalaxyComponents];
	double axis_ratio_x_fit_comp[MaxGalaxyComponents]; //not used
	double axis_ratio_y_fit_comp[MaxGalaxyComponents];
	double axis_ratio_z_fit_comp[MaxGalaxyComponents];
	double powerA_fit_comp[MaxGalaxyComponents];
	double powerB_fit_comp[MaxGalaxyComponents];
	double powerC_fit_comp[MaxGalaxyComponents];
	double M_expected_comp[MaxGalaxyComponents]; //not used
	double M_scale_comp[MaxGalaxyComponents];
	// vector<Potential_JS_analytical> P; //should before TACT/pot/potential.h

	int read_fit_DF_x_mass(string path_gm_1, int snapshot){
		if(count_components!=0){
			printf("Please reset class Galaxy_components before rereading.\n");
			exit(1);
		}
	    char wt_fname[MaxCharactersInString];
		sprintf(wt_fname, "%saa/example_snapshot_DF_params_fit.xv.txt", path_gm_1.data()); //this is only an example
		// sprintf(wt_fname, "%saa/snapshot_%d_DF_params_fit.xv.txt", path_gm_1.data(), snapshot);
		FILE *fp = nullptr;
		fp = fopen(wt_fname, "r");
    	// fp = nullptr; //to use expected setting value
		if(fp==nullptr){
			printf("Cannot open file \"%s\" for parameters for profile formula.\n", wt_fname);
			printf("Do not read fit parameters. Use the expected parameters instead.\n");
			fflush(stdout);
			// count_components = components; //??
			count_components = 1; //now only one component for FPot
			for(int i=0;i<count_components;i++){
				scaled_length_fit_comp[i] = scale_length_comp[i];
				axis_ratio_x_fit_comp[i] = flatx_comp[i];
				axis_ratio_y_fit_comp[i] = flaty_comp[i];
				axis_ratio_z_fit_comp[i] = flatz_comp[i];
				model_ID_comp[i] = 0; //set all 0
				powerA_fit_comp[i] = 0.2; //read more to MPE-DF??
DEBUG_PRINT_V0d(1, powerA_fit_comp[i], "powerA_fit_comp[i] fixed");
				powerB_fit_comp[i] = 2.0;
				powerC_fit_comp[i] = 1.7;
				M_expected_comp[i] = Mass_vir*frac_mass_comp[i]; //by calculate total mass of a component in a snapshot
				M_scale_comp[i] = M_expected_comp[i]/2.; //?? frac 2.
				scaled_density_fit_comp[i] = M_scale_comp[i]/(4./3.*pi_8*pow(scaled_length_fit_comp[i], 3));
			}
			return -1;
		}

		int N_skipline = 0;
		for(int i=0;i<N_skipline;i++){ //skip N_skipline lines about others
			SKIP_UNTILENDL(fp);
		}

		count_components = 1;
        // fscanf(fp, "%d %*[^\n]%*c", &count_components); //??
		if(count_components>MaxGalaxyComponents){
			printf("Too many components of classical gravitational system in this prog. "
				"Set the count to be %d", MaxGalaxyComponents);
			count_components = MaxGalaxyComponents;
		}
        // fscanf(fp, "%*[^\n]%*c");
		for(int i=0;i<count_components;i++){ //??
        	fscanf(fp, "%d%*[^\n]%*c", &(model_ID_comp[i]));
        	fscanf(fp, "%le%*[^\n]%*c", &(axis_ratio_y_fit_comp[i]));
        	fscanf(fp, "%le%*[^\n]%*c", &(axis_ratio_z_fit_comp[i]));
        	fscanf(fp, "%le%*[^\n]%*c", &(powerA_fit_comp[i]));
        	fscanf(fp, "%le%*[^\n]%*c", &(powerB_fit_comp[i]));
        	fscanf(fp, "%le%*[^\n]%*c", &(scaled_length_fit_comp[i]));
        	fscanf(fp, "%le%*[^\n]%*c", &(scaled_density_fit_comp[i]));
DEBUG_PRINT_V0d(1, powerA_fit_comp[i], "powerA_fit_comp[i] read");
DEBUG_PRINT_V0d(1, scaled_density_fit_comp[i], "scaled_density_fit_comp[i] read");
			axis_ratio_x_fit_comp[i] = 1.;
			M_expected_comp[i] = 1.; //by calculate total mass of a component in a snapshot
			M_scale_comp[i] = 4./3.*pi_8*scaled_density_fit_comp[i]*pow(scaled_length_fit_comp[i], 3);
    		//: fit params of simple DPL: "coef_free_p1", "coef_free_p2", "power_free_1", "power_free_2", "density_scale"
			//: TACT potential is by GM instead of GMm, Gadget output is also GM
			//\ in units of TACT 1e10Msol, GP(G*M_tot,r_scale) //??
			//\ not analytical formula potential grid by Poisson solver??
		}
        fclose(fp);

		return 0.;
	};

	void reset(){
		count_components = 0;
	}
};



class Snapshot{
private:


	//// The below private funcs in this class are adapted from Gadget2 and DICE
    /* this routine allocates the memory for the 
    * particle data.
    */
    int allocate_memory(void);

    ////gjy note; the main func that read snapshot to particle data *P
    /* this routine loads particle data from Gadget's default
    * binary file format. (A snapshot may be distributed
    * into multiple files.
    */
    int load_snapshot(char *fname, int files);

    /* This routine brings the particles back into
    * the order of their ID's.
    * NOTE: The routine only works if the ID's cover
    * the range from 1 to NumPart !
    * In other cases, one has to use more general
    * sorting routines.
    */
    int reordering(void);

    /* here the particle data is at your disposal 
    */
    int do_what_you_want(void){
        return 0;
    }

    /* this template shows how one may convert from Gadget's units
    * to cgs units.
    * In this example, the temperate of the gas is computed.
    * (assuming that the electron density in units of the hydrogen density
    * was computed by the code. This is done if cooling is enabled.)
    */
    int unit_conversion(void){
        return 0;
    }

    // This function frees the memory used when the load_snapshot function is
    // called. In general, users will not know that P has to be incremented
    // because it was decremented and an attempt to free it will result in
    // a segmentation fault. This prevents the problem.
    int unload_snapshot(){
        P++;
        free(P);
        return 0;
    }

    // Function to write initial conditions to file in default Gadget2 format.
    // The code was originally an input routine, read_snapshot.c, provided by
    // Volker Springel with the Gadget2 source code. It has been hacked into a
    // write routine.
    int write_gadget1_ics(galaxy *gal, char *fname);

    ////gjy note: it seems for hdf5 with block name, now we doNOT use it.
    // Function to write initial conditions to file in default Gadget2 format.
    // The code was originally an input routine, read_snapshot.c, provided by
    // Volker Springel with the Gadget2 source code. It has been hacked into a
    // write routine.
    int write_gadget2_ics(galaxy *gal, char *fname);

    ////gjy note: Write data by manufactely, we use this.
    // Function to write initial conditions to file in default Gadget2 format.
    // The code was originally an input routine, read_snapshot.c, provided by
    // Volker Springel with the Gadget2 source code. It has been hacked into a
    // write routine.
    int write_gadget1_ics_manul(galaxy *gal, struct particle_data* P_r, char *fname);



public:

	double time; //this data
    int snap; //this data
    struct io_header_1 header1;
    int *Id;
    int NumPart = 0, Ngas; 	//1000000; // NumPart here is the total count because all type particles 
							//\ in the "same" time are in one snapshot; potential by adding is by for(i NumPart){P[i].information}
	double Time, Redshift;
    string path_gm = ""; //member of class Stage //path of data of a running galaxy model
    // char path_gm[MaxCharactersInString];
	particle_data* P; // = new particle_data;

    int N_knn = MinInterpPoints; //MaxInterpPoints;
    std::vector<std::vector<double>> kdtree_points_pos;
    std::vector<std::vector<double>> kdtree_points_v;
    std::vector<std::vector<double>> kdtree_points_posv;
    KDTreeKNN* pkdtree  = nullptr; // 3D x
    KDTreeKNN* pkdtree1 = nullptr; // 3D v
    KDTreeKNN* pkdtree2 = nullptr; // 6D (x,v)

    //: tree gravity potential (octree, monopole only)
    //\ note: use shared_ptr (NOT unique_ptr) because Snapshot is stored in std::vector
    //\ and legacy code likely requires Snapshot to be copy-constructible during vector resize/realloc.
    // std::unique_ptr<treegrav::OctreePotential> pot_tree_;
	std::shared_ptr<treegrav::OctreePotential> pot_tree_;
    bool pot_tree_built_ = false; //tree_pot

	Snapshot(){
		// std::array<std::array<double, Dimension>, N_total> x_data; //static??
    	// for(int i=1;i<N_total+1;i++){
        // 	x_data[i][0] = P[i].Pos[0]; x_data[i][1] = P[i].Pos[1]; x_data[i][2] = P[i].Pos[2]; //or snapshot_data[i][0]
    	// }
		// kdtree.Set(&x_data);
	    /* notes:
	        Snapshot(std::array<std::array<double, Dimension>, N_total>* x_data): kdtree(&x_data) { //wrong.
	        error: C++ requires a type specifier for all declarations: You may forget type declaration.
	        error: no default constructor exists for class "...": non static inner class cannot intialized here; but here you need to intialize, so you can add static.
	        kdtree.Set(&data); : const Kdtree::member can only initialized in costructor instead of function; static Kdtree<...> kdtree : not needed.
	        If just want an intializtion in constructor with Snapshot::data and a type convert, a better choose is to add a functionand return the new type.
        */
	}

	~Snapshot(){
		// printf("~Snapshot() called.\n");
		unload_snapshot(); //donot put this func in load()
    	// for(int j=0;j<N_total;j++)
        // 	delete []snapshot_data[j];
    	// delete []snapshot_data;
	}



	// version multicomp: SCF multi-component support
	// List of SCF coefficient files, one per galaxy component
	std::vector<std::string> scf_component_files;
	// Which component's coefficients are currently loaded into the Fortran SCF (get_pot_)
	int scf_loaded_comp = -1;
	// The single filename the Fortran SCF expects to read (legacy): <path_gm>/intermediate/scficoef
	std::string scf_workfile;

	/*
		Function to write initial conditions to file in default Gadget2 format by changing 
		known particle_data without the struct member galaxy* gal. After changing *P of a 
		snapshot, write it to IC file.
		Note: This is an old version, not used.
	*/
	int write_gadget_ics_known(int ID_change, double rate, char pathload[MaxCharactersInString]=nullptr);
	
    // void set_path(char path_1[MaxCharactersInString]=nullptr){ //??
    // void set_path(const char* path_1=nullptr){ //??
    //     if(path_1==nullptr){
	// 		sprintf(path_gm, "path_gm/%s", modelPath);
    //     }
    //     else{
    //         strcpy(path_gm, path_1);
    //     }
    //     printf("model path is %s\n", path_gm);
    // }
    void set_path(const string& path_1=""){ //??
        if(path_1==""){
			path_gm = "path_gm_example/"+(string)modelPath;
        }
        else{
            path_gm = path_1;
        }
        printf("model path is %s\n", path_gm.data());
    }
	void set_snapshot(int ss, double tt){
		this->snap = ss;
		this->time = tt;
	}

	// void load(char pathload[MaxCharactersInString]=nullptr){ //this func must be called if Data_Potential used
	// void load(char pathload[MaxCharactersInString]=nullptr){ //this func must be called if Data_Potential used
	// 	char input_fname[MaxCharactersInString];
	// 	if(pathload!=nullptr) sprintf(path_gm, "%s", pathload);
    // 	sprintf(input_fname, "%ssnapshot/%s_%03d", path_gm, bname, snap);
    //     printf("loading %s ...\n", input_fname);
	// 	load_snapshot(input_fname, files); //cannot open file: in this func
  	// 	// int id = 1;
	// 	// printf("%e %d %e %e %e\n", P[id].Pos[0], P[id].Id, P[id].Mass, P[id].Pot, P[id].Acc[2]);
  	// 	// reordering();			/* call this routine only if your ID's are set properly */
	// 	for(int id=1;id<NumPart+1;id++){
	// 		P[id].Id = id; //rewrite id
	// 		P[id].Time = this->time; //rewrite time
	// 	}
  	// 	// unit_conversion();		/* optional stuff */
  	// 	// do_what_you_want();
    //     printf("loading IC ... done.\n");
	// }
	void load(const string& pathload=""){ //this func must be called if Data_Potential used
		char input_fname[MaxCharactersInString];
		if(pathload!="") this->path_gm = pathload;
    	sprintf(input_fname, "%ssnapshot/%s_%03d", path_gm.data(), bname, snap);
        printf("loading %s ...\n", input_fname);
		load_snapshot(input_fname, files); //cannot open file: in this func
		printf("total particle count: N_allPtcs = %d, NumPart = %d\n", N_allPtcs, NumPart);
		if (N_allPtcs != NumPart) { // Use NumPart (actual snapshot count) rather than N_allPtcs (IC-configured count)
			printf("Warning: N_allPtcs != NumPart, "
			    "Triaxialization will iterate over NumPart to match allocated P[]\n"
			);
		}
		// int id = 1;
		// printf("%e %d %e %e %e\n", P[id].Pos[0], P[id].Id, P[id].Mass, P[id].Pot, P[id].Acc[2]);
  		reordering();			/* call this routine only if your ID's are set properly */
		for(int id=1;id<NumPart+1;id++){
			// P[id].Id = id; //rewrite id
			P[id].Time = this->time; //rewrite time
		}

        printf("particle types displaying: \n");
        for(int id=1;id<NumPart+1;id+=1000){
            printf("P[%d].Type = %d ", id, P[id].Type);
        }
        printf("\n");
		// DEBUG_PRINT_V0d(0, NumPart, "NumPart"); //debug

  		// unit_conversion();		/* optional stuff */
  		// do_what_you_want();
        printf("loading IC ... done.\n");
	}

    // void load_PeterBerczik_PlummerIC(double M, double rs, double vs){ //only Plummer halo
    //     this->snap = 0;
    //     float x0,y0,z0, vx0, vy0, vz0, m0; int ID0;
    //     this->NumPart = N_total;
    //     float R = rs*16./3./pi_8;
    //     float V = 2*sqrt( conv_G*M/rs );
    //     printf("PlummerBP: R = %e, V = %e\n", R, V);
    //     allocate_memory(); //memory alloc for P[] is necessary, numb should be large
    //     char pathload[MaxCharactersInString] = "PlummerBP_IC/data.inp";
    //     FILE *fp = fopen(pathload, "r");
    //     for(int line=1;line<4;line++) fscanf(fp, "%*[^\n]%*c");
    //     for(int i=1;i<N_total+1;i++){
    //         fscanf(fp, "%d   %e   %e %e %e   %e %e %e \n", &ID0, &m0, &x0, &y0, &z0, &vx0, &vy0, &vz0);
    //         // fscanf(fp, "%06d   % .16E   % .16E % .16E % .16E   "
	// 		//	"% .16E % .16E % .16E \n", &ID0, &E0, &x0, &y0, &z0, &vx0, &vy0, &vz0);
    //         // printf("%d %d %e %e %e\n", i, ID0, E0, x0, vx0);
    //         P[i].Id = ID0+1, P[i].Mass = m0*M;
    //         // P[i].Pos[0] = x0*rs, P[i].Pos[1] = y0*rs, P[i].Pos[2] = z0*rs;
    //         P[i].Vel[0] = vx0*vs, P[i].Vel[1] = vy0*vs, P[i].Vel[2] = vz0*vs;
    //         P[i].Pos[0] = x0*R, P[i].Pos[1] = y0*R, P[i].Pos[2] = z0*R;
    //     }
    //     fclose(fp);
    //     printf("loading IC ... done.\n");
    // }

	/*	Snapshot: Triaxialization.
	*/
	TriaxializeCoordinate TC; //in class Snapshot
	void triaxialize(){
		TC.reset_parameters(); //Reset each time is to avoid repeat preprocess-ing snapshot
		TC.read_parameters_preprocess(snap, path_gm);
		print_snapshot_info("tril3");
		for(int i=1;i<NumPart+1;i++){
			// printf("%d ", i);
			VecDoub xv = {P[i].Pos[0], P[i].Pos[1], P[i].Pos[2], 
				P[i].Vel[0], P[i].Vel[1], P[i].Vel[2]};
			VecDoub xv_ = TC.new_coordinate(xv, true, true);
			// VecDoub xv_ = TC.new_coordinate_fine(xv, true, true); //??
			P[i].Pos[0] = xv_[0], P[i].Pos[1] = xv_[1], P[i].Pos[2] = xv_[2], 
			P[i].Vel[0] = xv_[3], P[i].Vel[1] = xv_[4], P[i].Vel[2] = xv_[5];
		}
	}

    void adjust_center_rewrite(double move_x=0.,double move_y=0.,double move_z=0.){

        // this->snap = snap_id;
        vector<double> M = {0.,0.,0.};
        //center of mass, to adjust x; we donot use median of {x} temporarily.
        vector<double> MR = {0.,0.,0.};
        vector<double> R = {0.,0.,0.};
        //center of momentum, to adjust v
        vector<double> MV = {0.,0.,0.};
        vector<double> V = {0.,0.,0.};

        for(int id=0+1;id<NumPart+1;id++){
            M[0] += P[id].Mass;
            M[1] += P[id].Mass;
            M[2] += P[id].Mass;
            MR[0] += P[id].Mass*P[id].Pos[0];
            MR[1] += P[id].Mass*P[id].Pos[1];
            MR[2] += P[id].Mass*P[id].Pos[2];
            MV[0] += P[id].Mass*P[id].Vel[0];
            MV[1] += P[id].Mass*P[id].Vel[1];
            MV[2] += P[id].Mass*P[id].Vel[2];
        }
        R[0] = MR[0]/M[0];
        R[1] = MR[1]/M[1];
        R[2] = MR[2]/M[2];
        V[0] = MV[0]/M[0];
        V[1] = MV[1]/M[1];
        V[2] = MV[2]/M[2];
        printf("center of R = %e %e %e; of V = %e %e %e.\n", R[0], R[1], R[2], V[0], V[1], V[2]);

        for(int id=0+1;id<NumPart+1;id++){
            P[id].Pos[0] -= R[0];
            P[id].Pos[1] -= R[1];
            P[id].Pos[2] -= R[2];
            P[id].Vel[0] -= V[0];
            P[id].Vel[1] -= V[1];
            P[id].Vel[2] -= V[2];
        }
        printf("Ajusted center ... done.\n");

        //ajust another center as a deviation
        for(int id=0+1;id<NumPart+1;id++){
            P[id].Pos[0] -= -move_x;
            P[id].Pos[1] -= -move_y;
            P[id].Pos[2] -= -move_z;
            P[id].Vel[0] -= V[0];
            P[id].Vel[1] -= V[1];
            P[id].Vel[2] -= V[2];
        }
        printf("Ajusted another center ... done.\n");
    }

	VecDoub similar_NFW_param(){
		double M = 0., rs, qy, qz, rhos;
		double x2 = 0., y2 = 0., z2 = 0., rm = 0.;
		for(int i=1;i<NumPart+1;i++){
			x2 += P[i].Pos[0]*P[i].Pos[0];
			y2 += P[i].Pos[1]*P[i].Pos[1];
			z2 += P[i].Pos[2]*P[i].Pos[2];
			M += P[i].Mass;
			rm += sqrt(P[i].Pos[0]*P[i].Pos[0]+P[i].Pos[1]*P[i].Pos[1]+P[i].Pos[2]*P[i].Pos[2]);
		}
		qy = sqrt(y2/x2);
		qz = sqrt(z2/x2);
		rs = rm/NumPart;
		rhos = M/(4./3.*pi_8*pow(rs, 3));
		VecDoub p = {rhos, rs, qy, qz};
		DEBUG_PRINT_V1d(1, p, "{similar NFW: rhos, rs, qy, qz}");
		return {M, rs, qy, qz, rhos};
	}

	/* 	To adjust the galaxy that is read to struct Particle_data *P.
		Many ways without user interacting options.
	*/
	void adjust_galaxy(int how, double* params=nullptr){

        // this->snap = snap_id;
		switch (how){
			case 1:{
				float translation[3] = {-100., -80., -60.};
				for(int id=0+1;id<NumPart+1;id++){
					P[id].Pos[0] -= translation[0];
					P[id].Pos[1] -= translation[1];
					P[id].Pos[2] -= translation[2];
					P[id].Vel[0] -= 0.;
					P[id].Vel[1] -= 0.;
					P[id].Vel[2] -= 0.;
				}
				printf("Adjust galaxy: translation done.\n");
				break;
			}
			case 2:{
				float q[3] = {1.0, 1.0, 0.3};
				for(int id=0+1;id<NumPart+1;id++){
					P[id].Pos[0] *= q[0];
					P[id].Pos[1] *= q[1];
					P[id].Pos[2] *= q[2];
					P[id].Vel[0] *= sqrt(q[0]);
					P[id].Vel[1] *= sqrt(q[1]);
					P[id].Vel[2] *= sqrt(q[2]);
				}
				printf("Adjust galaxy: flattening-2 done.\n");
				break;
			}
			case 3:{
				float q[3] = {1.0, 0.5, 0.3};
				for(int id=0+1;id<NumPart+1;id++){
					P[id].Pos[0] *= q[0];
					P[id].Pos[1] *= q[1];
					P[id].Pos[2] *= q[2];
					P[id].Vel[0] *= sqrt(q[0]);
					P[id].Vel[1] *= sqrt(q[1]);
					P[id].Vel[2] *= sqrt(q[2]);
				}
				printf("Adjust galaxy: flattening-3 done.\n");
				break;
			}
			case 4:{
				#define LEthan(a,b) a<b? -1:1
				#define Split(a,f) a<f? f:(1-f)
				int count = 2, pm;
				float translation[3] = {-100., -80., -60.}, frac = 0.3, ff, rd;
				srand(9999); //srand(time(0)); //there are (1.-frac)/1. particles are throw away to generate another galaxy
				for(int id=0+1;id<NumPart+1;id++){
					rd = rand();
					pm = LEthan(rd,frac);
					ff = Split(rd,frac);
					P[id].Pos[0] -= translation[0]*pm;
					P[id].Pos[1] -= translation[1]*pm;
					P[id].Pos[2] -= translation[2]*pm;
					P[id].Vel[0] *= ff;
					P[id].Vel[1] *= ff;
					P[id].Vel[2] *= ff;
				}
				printf("Adjust galaxy: splitting into 2 galaxy done.\n");
				break;
			}
			default:{
				printf("Adjust galaxy: nothing done.\n");
				break;
			}
		}
	}

	/* 	Update potential of particles after transform posisions of particles in a snapshot.
	*/
	void PD_update_potential(){
		for(int i=1;i<NumPart+1;i++){
			P[i].Pot = potential_sum_particle({P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]}, i);
		}
	}

    int write_PD_txt(string disctrp=""){

        // this->snap = snap_id;
	    char wt_fname[MaxCharactersInString];
DEBUG_PRINT_I(3211);
	    sprintf(wt_fname, "%stxt/%s_%03d_%s.txt", path_gm.data(), bname, snap, disctrp.data());
DEBUG_PRINT_I(3212);
        FILE *fp = fopen(wt_fname, "w");
DEBUG_PRINT_I(3213);
        if(fp==nullptr) return -1;
        for(int id=1;id<NumPart+1;id++){
            fprintf(fp, "%e %e %e %e %e %e    %d %d %e %e    %e %e %e %e    %e %e %e %e    %e %e %e \n", 
				P[id].Pos[0], P[id].Pos[1], P[id].Pos[2],    P[id].Vel[0], P[id].Vel[1], P[id].Vel[2],    
				P[id].Id, P[id].Type, P[id].Mass, P[id].Time, 
				P[id].U, P[id].Rho, P[id].Hsml, P[id].Ne, 
				P[id].Pot,    P[id].Acc[0], P[id].Acc[1], P[id].Acc[2], 
				P[id].dAdt, P[id].Age, P[id].Metal 
            );
        }
        fclose(fp);
        printf("Write particle_data *P to file %s ... done.\n", wt_fname);
        return 0;
    }

    int read_PD_txt(){

        // this->snap = snap_id;
	    char wt_fname[MaxCharactersInString];
	    sprintf(wt_fname, "%stxt/%s_%03d.txt", path_gm.data(), bname, snap);
        FILE *fp = fopen(wt_fname, "r");
        if(fp==nullptr) return -1;
		// char info[MaxCharactersInString];
	    // fscanf(fp, "%[^\n]s", info);
		// printf("%s\n", info)
        for(int id=1;id<NumPart+1;id++){
			float x[3], v[3], mass, pot, acc[3], time; int ID, type; 
            fscanf(fp, "%e %e %e %e %e %e    %d %d %e    %*f %*f %*f %*f    %e %e %e %e    %*f %*f %*f     %e \n",    
            	&(x[0]),&(x[1]),&(x[2]),    &v[0],&v[1],&v[2],    
            	&ID, &type,    &mass,    
            	&pot,    &acc[0],&acc[1],&acc[2],     &time 
            );
            P[id].Pos[0] = x[0],P[id].Pos[1] = x[1],P[id].Pos[2] = x[2],    
			P[id].Vel[0] = v[0],P[id].Vel[1] = v[1],P[id].Vel[2] = v[2],    
            P[id].Id = ID, P[id].Type = type,    P[id].Mass = mass,    
            P[id].U = 0., P[id].Rho = 0., P[id].Hsml = 0., P[id].Ne = 0.,    
            P[id].Pot = pot,    P[id].Acc[0] = acc[0],P[id].Acc[1] = acc[1],P[id].Acc[2] = acc[2],    
            P[id].dAdt = 0., P[id].Age = 0., P[id].Metal = 0.,     P[id].Time = time;
        }
        fclose(fp);
        printf("Read particle_data *P from file %s ... done.\n", wt_fname);
        return 0;
    }

    int write_PD_toSCF(){
        // this->snap = snap_id;
	    char wt_fname[MaxCharactersInString];
	    sprintf(wt_fname, "%stxt/%s_%03d.SCF.txt", path_gm.data(), bname, snap);
        FILE *fp = fopen(wt_fname, "w");
        if(fp==nullptr) return 1;
        fprintf(fp, "%d \n%d \n%le \n", 100, NumPart, 0.01); //the first line
        for(int id=1;id<NumPart+1;id++){
            fprintf(fp, "%d %e    %e %e %e %e %e %e\n", 
				P[id].Id, P[id].Mass, 
				P[id].Pos[0], P[id].Pos[1], P[id].Pos[2], P[id].Vel[0], P[id].Vel[1], P[id].Vel[2] 
            );
        }
        fclose(fp);
        printf("Write particle_data *P to file %s ... done.\n", wt_fname);
        return 0;
    }



	// /* interpspline3 */
	InterpSplineP3D1 ISP1, ISP2;
	double r_min_sp3, r_scale_sp3, r_max_sp3;
	int N_grid_sp3; //count of radical grids, odd or (even-1)
	int tag_which_sample; //use which potential as samples
	VecDoub axis_ratio_sp3; //length Dim, triaxial potential with fixed axis ratio
	VecDoub r_grid_sp3; //length 3, user should set: r_min, r_scale*2., r_max
	VecDoub phi_grid_sp3; //length N_grid, the radical potential is spline3
	void RebuildPotentialSP3(const VecDoub& axis_ratio_sp31, double r_scale1, 
		double r_min1=2e-2, double r_max1=5e2, int N_grid1=9, int tag_which_sample1=0)
	{
		if(N_grid1<3){
			printf("Wrong value of grid count. Set as default value.\n");
			N_grid_sp3 = 5;
		}else{
			if(N_grid1%2==0){
				N_grid_sp3 = N_grid1-1;
			}else{
				N_grid_sp3 = N_grid1;
			}
		}
		r_min_sp3 = r_min1;
		r_scale_sp3 = r_scale1*2.;
		r_max_sp3 = r_max1;
		if( !( r_min_sp3<r_scale_sp3 && r_scale_sp3<r_max_sp3 ) ){
			printf("Wrong settings of radius. Exit.\n");
			exit(1);
		}
		if( !( axis_ratio_sp31.size()==3 ) ){
			printf("Wrong settings of axis ratio. Exit.\n");
			exit(1);
		}
		axis_ratio_sp3 = axis_ratio_sp31;
		tag_which_sample = tag_which_sample1;
		// DEBUG_PRINT_V1d(1, axis_ratio_sp3, "66511");
		reset_PotentialSP3();
		build_PotentialSP3();
	}
	void reset_PotentialSP3(){
DEBUG_PRINT_I(66512);
		axis_ratio_sp3.resize(Dim, 0.);
		r_grid_sp3.resize(N_grid_sp3, 0.);
		phi_grid_sp3.resize(N_grid_sp3, 0.);
		int Nl = (int)(N_grid_sp3-1)/2; //count of grid whose radius less than r_scale_sp3
		int Nr = Nl; //greater than, set the same count
		for(int i=0;i<Nl;i++){
			r_grid_sp3[i] = r_min_sp3+(r_scale_sp3-r_min_sp3)/Nl*i;
		}
		for(int i=Nl;i<N_grid_sp3;i++){
			r_grid_sp3[i] = r_scale_sp3+(r_max_sp3-r_scale_sp3)/(N_grid_sp3-1-Nl)*(i-Nl);
		}
		// DEBUG_PRINT_V1d(0, r_grid_sp3, "r_grid_sp3");
	};
	void build_PotentialSP3(){
		//[??] axis ratio by MOI
		//[??] python curve_fit init to emcee and parallel
		//[??] to shell and debug tact foci
		//[learn code] friend initialize or ss->many->stage or all in one??
DEBUG_PRINT_I(66513);
		VecDoub xs = r_grid_sp3, ys;
		for(auto r : xs){
			vector<VecDoub> x_six;
			x_six.push_back({r*axis_ratio_sp3[0], 0., 0.});
			x_six.push_back({-r*axis_ratio_sp3[0], 0., 0.});
			x_six.push_back({0., r*axis_ratio_sp3[1], 0.});
			x_six.push_back({0., -r*axis_ratio_sp3[1], 0.});
			x_six.push_back({0., 0., r*axis_ratio_sp3[2]});
			x_six.push_back({0., 0., -r*axis_ratio_sp3[2]});
			DEBUG_PRINT_V0d(1, r, "r");
			// DEBUG_PRINT_V0d(1, axis_ratio_sp3[2], "r");
			double pot = 0.;
			for(auto a : x_six){
				pot += potential_SCF(a);
			}
			// pot = 6.*sin(r);
			// double p = this->potential_SCF({Err, 0., 0.003}); //nan when at {(x,y,z)|x=0,y=0} ????
			// DEBUG_PRINT_V0d(1, p, "potscf");
			// DEBUG_PRINT_V0d(0, pot, "potscf");
			ys.push_back(pot/6.);
		}
		DEBUG_PRINT_V1d(1, xs, "xs");
		DEBUG_PRINT_V1d(1, ys, "ys");
		ISP1.reset_samples_and_coefs(xs, ys);
DEBUG_PRINT_I(66514);

		VecDoub().swap(ys);
		for(auto r : xs){
			vector<VecDoub> x_six;
			x_six.push_back({r*axis_ratio_sp3[0], 0., 0.});
			x_six.push_back({-r*axis_ratio_sp3[0], 0., 0.});
			x_six.push_back({0., r*axis_ratio_sp3[1], 0.});
			x_six.push_back({0., -r*axis_ratio_sp3[1], 0.});
			x_six.push_back({0., 0., r*axis_ratio_sp3[2]});
			x_six.push_back({0., 0., -r*axis_ratio_sp3[2]});
			// x_six.push_back({1.e-8, 1.e-8, r*axis_ratio_sp3[2]});
			// x_six.push_back({1.e-8, 1.e-8, -r*axis_ratio_sp3[2]});
			double fr = 0.;
			for(auto a : x_six){
				auto fr1 = forces_SCF(a);
				fr += lpnorm_real_with_distance_and_coef(&fr1);
				// DEBUG_PRINT_V0d(1, fr, "fr each"); //?? the SCF data might be not at center, another data (this is 4.0e5~1.6e6)
				// DEBUG_PRINT_V1d(1, fr1, "fr1 each");
			}
			// exit(0);
			ys.push_back(fr/6.);
		}
		cout<<"The samples of interp 1d grid:";
		DEBUG_PRINT_V1d(1, xs, "x_samples");
		DEBUG_PRINT_V1d(1, ys, "y_samples");
		ISP2.reset_samples_and_coefs(xs, ys);
	}
	double potential_TRSP3(const VecDoub& x){
		// DEBUG_PRINT_V1d(1, axis_ratio_sp3, "axis_ratio_sp3");
		// DEBUG_PRINT_V1d(1, x, "x_tgt");
		double xt, yt;
		xt = lpnorm_real_with_distance_and_coef(&x, Substance_of_VecDoub_as_nullptr, &axis_ratio_sp3, 2);
		// DEBUG_PRINT_V0d(1, xt, "x0");
		// yt = potential_SCF(x);
		// DEBUG_PRINT_V0d(0, yt, "y0");
		yt = ISP1.target_interp(xt);
		// DEBUG_PRINT_V0d(1, yt, "y");
		return yt;
	}
	VecDoub forces_TRSP3(const VecDoub& x){
		//This is by iso-potential surface of elliptical potential.
		double xt, yt;
		xt = lpnorm_real_with_distance_and_coef(&x, Substance_of_VecDoub_as_nullptr, &axis_ratio_sp3);
		DEBUG_PRINT_V0d(1, xt, "xt");
		DEBUG_PRINT_V1d(1, x, "x");
		auto fr1 = forces_SCF(x);
		DEBUG_PRINT_V1d(1, fr1, "fr10");
		yt = lpnorm_real_with_distance_and_coef(&fr1);
		DEBUG_PRINT_V0d(1, yt, "y0");
		yt = ISP2.target_interp(xt);
		DEBUG_PRINT_V0d(10, yt, "y");
		double rq = lpnorm_real_with_distance_and_coef(&x, Substance_of_VecDoub_as_nullptr, &axis_ratio_sp3);
		VecDoub yf(Dim, 0.);
		for(int i=0;i<Dim;i++){
			yf[i] = -(yt*x[i])/(rq*axis_ratio_sp3[i]*axis_ratio_sp3[i]); //??
		}
		DEBUG_PRINT_V1d(1, yf, "fr1yf");
		return yf;
	}



    /* kdtree prepare */
    void loadtree (KDTreeKNN* pk) { pkdtree  = pk; }
    void loadtree1(KDTreeKNN* pk) { pkdtree1 = pk; }
    void loadtree2(KDTreeKNN* pk) { pkdtree2 = pk; }

    void generate_kdtree_pos(int nn = MinInterpPoints){
        // Build 3D KD-tree on positions (indices are 0..NumPart-1)
        kdtree_points_pos.clear(); //note: the pointed-to memory is not touched in any way
        kdtree_points_pos.reserve(NumPart);
        for (int i=1; i<=NumPart; ++i)
            kdtree_points_pos.push_back({P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]});
        delete pkdtree;
		pkdtree = new KDTreeKNN(kdtree_points_pos);

        // std::vector<double> x_tgt = {1.,1.,1.};
        // auto nearest_id = pkdtree->find_k_nearest(std::min(nn, (int)kdtree_points_pos.size()), x_tgt);
        // for (auto id : nearest_id) printf("id = %d\n", id);
    }

	vector<int> get_available_knn(int& kk, std::array<double, 3>& x_tgt_arr){ //bad method, no use
    	int nknn = kk;
    	bool is_loop = 0; //no use
    	//?? add center xv
    	do{
    	   	double distance = 0., distance_max=0, distance_min=2*MaxInterpDistance;
    	    bool is_r_within=0, is_phi_within=0, is_theta_within=0, r_relative, phi_relative, theta_relative;
    	    vector<double> SP, SP1, SP_tgt;
    	    vector<int> nearest_idx = pkdtree->find_k_nearest(nknn, x_tgt_arr);
    	    vector<int> nearest_available;
    	    SP_tgt = CartesianToSPolar({x_tgt_arr[0], x_tgt_arr[1], x_tgt_arr[2]});
    	    SP1 = CartesianToSPolar({P[nearest_idx[0]].Pos[0], P[nearest_idx[0]].Pos[1], P[nearest_idx[0]].Pos[2]});
    	    r_relative = isSign(SP_tgt[0]-SP1[0]); 
			phi_relative = isSign(SP_tgt[1]-SP1[1]); 
			theta_relative = isSign(SP_tgt[2]-SP1[2]); //first ^ first = 0

    	    //find the particles whin MaxInterpDistance
    	    for(int k=0;k<nknn;k++){
    	        for(int j=0;j<Dimension;j++){
    	            distance += (x_tgt_arr[j] -P[nearest_idx[k]].Pos[j])*(x_tgt_arr[j] -P[nearest_idx[k]].Pos[j]);
     	        }
				distance = sqrt(distance);
     	    	// cout<<"distance of nn particle "<<k<<": "<<distance<<endl;
     	    	if(distance<=MaxInterpDistance){
        		    //record the points within range
            	    nearest_available.push_back(nearest_idx[k]);
            		//record the max and min distance, and SphericalPolar coordinate range that dicided by these sample points
                	distance_max = distance_max>distance ? distance_max : distance;
            	    distance_min = distance_min<distance ? distance_min : distance;
            	    SP = CartesianToSPolar({P[nearest_idx[k]].Pos[0], P[nearest_idx[k]].Pos[1], P[nearest_idx[k]].Pos[2]});
            	    r_relative ^= isSign(SP_tgt[0]-SP[0]); if(r_relative) is_r_within = 1; //just once is OK
           	    	phi_relative ^= isSign(SP_tgt[1]-SP[1]); if(phi_relative) is_phi_within = 1;
            	    theta_relative ^= isSign(SP_tgt[2]-SP[2]); if(theta_relative) is_theta_within = 1;
            		// cout<<"SP of nn particle "<<k<<": "<<SP[0]<<" "<<SP[1]<<" "<<SP[2]<<endl;
            	    // cout<<"SP of target point: "<<SP_tgt[0]<<" "<<SP_tgt[1]<<" "<<SP_tgt[2]<<endl;
            	}
        	}
        	// cout<<"is relative: "<<r_relative<<" "<<phi_relative<<" "<<theta_relative<<endl;
        	// cout<<"is within: "<<is_r_within<<" "<<is_phi_within<<" "<<is_theta_within<<endl;
        	// for(auto ns : nearest_idx) cout<<"ns: "<<ns<<endl;
        	if(distance_min>MaxInterpDistance || nearest_available.size()<5 || nknn>3*MinInterpPoints){
        	    is_loop = 0;
        	    return {0}; //Cannot inner interpolate, switch to add().
        	}
        	if(is_r_within && is_phi_within && is_theta_within){
            	is_loop = 0;
            	return nearest_available; //Inner interpolate.
        	}
        	else{
            	nknn += MinInterpPoints;
            	is_loop = 1;
        	}
    	}while(is_loop);
    	return {0};
	}

	/* discard */
    vector<double> dispersion_interp(const vector<double>& x, int nn = MaxInterpPoints); //only Cartesain x input provided

	/* discard */
    int dispersion_discrete_all(int coor = 0, int nn = MaxInterpPoints);
	

		
	/*	To calculate the probability density function of a target point by kernel density estimation(KDE) and k-nearest 
		neigubour method(KNN).
		Meanwile, calculate the secondhand data from target onhand data run by Gadget or TACT. 
		The data should be set before: see all the members of struct write_secondhand; 
		The tree should be generated before： such as the x-tree, the v-tree, the Omega-tree, the J-tree ... 
		The distance of the tree is euclidean. The list ID of tree's knn start from 0. 
		(more DF about f(\boldsymbol Coor, t) f(\boldsymbol coor_\mbox{Fourier or other transform}), may be continuded ... )

		param @ QP： the 6D target cannonical coordinates; 
		param @ WhereWrite: the index(start from 0) of wtsh where the secondhand data write;
		param @ WhatCannonical: the form of cannonical coordinates of data wtsh[].QP such as coordinate type, 
				Q-P order, is exist something; 
				the default value represent wtsh[].QP is {J_1, J_2, J_3, \Omega_1, Omega_2. Omega_3};
		param @ h_mat: the bandwith matrix of interpolation kernel function, unit, diagnal, no-symmetri; 
				the default is unit matix, the most far disdance of these knn;
		param @ nknn: the count set of knn.
	*/
    int process_tosecondhand(const double* const QP, int WhereWrite, int WhatCannonical, 
		double* h_mat=nullptr, int nknn=MaxInterpPoints);

	/*	To read firsthand data from .txt file which was run by gadget or TACT. 
		All the particles are considered.
	*/
	int read_firsthand_all_txt(int WhatCannonical, int WhatPotential, int WhatSymmetry, int WhatActionmethod);

	/*	To write secondhand data to .txt file which was processed by int this->process_tosecondhand(). 
		All the particles are considered.
	*/
	int write_secondhand_all_txt(int WhatCannonical, int WhatPotential, int WhatSymmetry, int WhatActionmethod);

	/*	To calculate the data about knn together.
		The data are list on struct write_secondhand.
	*/
	int data_knn(const double* const QP, int WhereWrite, int WhatCannonical, 
		double* h_mat=nullptr, int nknn=MaxInterpPoints){
		// same with this->process_tosecondhand();
		return 0;
	}

	/*	To load particles data to the firsthand data.
	*/
	int load_to_firsthand_from_PD();

    /*
        To print the snapshot info.
    */
    void print_snapshot_info(string discpt=""){
        std::cout
            <<"Snapshot::print_snapshot_info(), "<<discpt<<":\n"
            <<snap<<" \n"
            <<time<<" \n"
            <<NumPart<<" \n"
        <<"\n";
        fflush(stdout);
    }

	/*	To wrap function for preprocess and loading SCF coef data.
	*/
	void preprocess(bool is_directly_get_parameter_=false)
	{
		printf("Do preprocess().\n");
		// //read anglular velocity of galaxy total rotation to modify rotating frame potential
		// TC.read_parameters_preprocess(snap, path_gm); //TC.O

		//triaxialize xv to calculate triaxial galaxy actions
		if(!is_directly_get_parameter_){
			printf("Do triaxialize() in each mpi process.\n");
			triaxialize(); //to update particle_data xv
			// write_PD_txt("preprocessed"); //the python code fit_galaxy_distribution_function.py has writen triaxialized snapshot //one should only run this line in mpi process 1
			#ifdef POTENTIAL_INTERP //donot run it default
			this->PD_update_potential(); //optional: refresh per-particle potential after coordinate transform
			#endif
		}

		if(1){ //tree_pot
			std::cout<<"Building tree ...\n";
			// Build a fresh tree after particle positions have been updated
			// (so it matches the snapshot coordinate system used by action computation).
			// pot_tree_.reset(new treegrav::OctreePotential(/*G=*/conv_G, /*theta=*/0.7, /*bucket=*/16, /*max_depth=*/32));
			pot_tree_.reset(new treegrav::OctreePotential(/*G=*/conv_G, /*theta=*/1.1, /*bucket=*/64, /*max_depth=*/32));
DEBUG_PRINT_I(2201);

			// You can tune theta/bucket for speed:
			//   smaller theta -> more accurate, slower
			//   larger theta -> faster, less accurate
			//   bucket 16~64 typical; bigger -> fewer nodes, slightly more leaf work
			// If your code uses G != 1, pass your G here.

			std::vector<treegrav::OctreePotential::Particle> tp;
			tp.resize((size_t)NumPart + 1); // keep 1-based for convenience
DEBUG_PRINT_I(2202);

			for (int i = 1; i <= NumPart; ++i) {
				treegrav::OctreePotential::Particle q;

				// ===== positions =====
				// Choose the actual fields in your particle_data:
				// If triaxialize() updates P[i].Pos[0..2]:
				q.x = P[i].Pos[0];
				q.y = P[i].Pos[1];
				q.z = P[i].Pos[2];

				// If instead it's P[i].xv[0..2], then use that:
				// q.x = P[i].xv[0]; q.y = P[i].xv[1]; q.z = P[i].xv[2];

				// ===== mass =====
				q.m = P[i].Mass;

				// ===== per-particle softening =====
				// If you have particle type 0..5:
				// int ty = P[i].Type;
				// q.eps = softening_type[ty];
				//
				// If you have per-particle softening stored directly (preferred):
				// q.eps = P[i].Softening;
				//
				// If you only have global softening:
				// q.eps = softening_type[0];

				int ty = P[i].Type;                  // adapt if your field name differs
				if (ty < 0) ty = 0;
				if (ty > 5) ty = 5;
				q.eps = softening_type[ty];

				tp[i] = q;
			}
DEBUG_PRINT_I(2203);

			// Build: tp is 1-based; pass first_index=1, n=NumPart.
			// reorder_particles=true is strongly recommended for speed (leaf contiguity).
			pot_tree_->build(tp.data(), NumPart, /*first_index=*/1, /*reorder_particles=*/true);
DEBUG_PRINT_I(2204);

			pot_tree_built_ = true;
			std::cout<<"Building tree, done.\n";
		}

        //load SCF parameters for potential  //gjy add: multi-component discovery
        DEBUG_PRINT_V0d(1, this->TC.O[0], "preprocess TC.O[0]");
DEBUG_PRINT_I(3221);
		#ifdef POTENTIAL_SCF

		//:: old version
		// // write_PD_toSCF(); //(s) //??
		// //calculate SCF coef //??
		// string path_gm_string = path_gm;
		// string path_coef = path_gm_string+"intermediate/scficoef"; //??
        // FILE *fp = fopen(path_coef.data(), "r");
		// // DEBUG_PRINT_V0d(1, fp, "fp");
        // if(fp==nullptr){
		// 	printf("Can not find coefficients file `%s` by Fortran prog SCF. Exit.\n", path_coef.data());
		// 	std::cout<<"path_work: "<<get_workpath()<<"\n";
		// 	std::cout<<"path_gm: "<<path_gm<<"\n";
		// 	std::cout<<"path_coef: "<<path_coef<<"\n";
		// 	exit(0);
		// }else{ //if there exist the coef file, run get_parameter_()
		// 	fclose(fp);
		// 	get_parameter_(); //(s);
		// }

		//:: new version: Build working filename and discover component files
        scf_workfile = path_gm + "intermediate/scficoef";
        scf_component_files.clear();
        // Try per-component patterns first; fall back to single-file legacy
        for(int c=0; c<components && (int)scf_component_files.size()<components; ++c){
            char cand1[MaxCharactersInString], cand2[MaxCharactersInString], cand3[MaxCharactersInString];
            sprintf(cand1, "%sintermediate/scficoef_comp%02d", path_gm.c_str(), c);
            sprintf(cand2, "%sintermediate/scficoef_comp%d",    path_gm.c_str(), c);
            sprintf(cand3, "%sintermediate/SCF_comp%d.txt",     path_gm.c_str(), c);
            FILE* fp=nullptr;
            if((fp=fopen(cand1,"r"))){ fclose(fp); scf_component_files.emplace_back(cand1); continue; }
            if((fp=fopen(cand2,"r"))){ fclose(fp); scf_component_files.emplace_back(cand2); continue; }
            if((fp=fopen(cand3,"r"))){ fclose(fp); scf_component_files.emplace_back(cand3); continue; }
        }
        if(scf_component_files.empty()){
            // Legacy: single combined file
            FILE* fp = fopen(scf_workfile.c_str(), "r");
            if(!fp){
                printf("Can not find SCF coefficients file `%s`. Exit.\n", scf_workfile.c_str());
                std::cout<<"path_work: "<<get_workpath()<<"\n";
                std::cout<<"path_gm: "<<path_gm<<"\n";
                std::cout<<"scf_workfile: "<<scf_workfile<<"\n";
                exit(0);
            }
            fclose(fp);
            scf_component_files.push_back(scf_workfile);
        }
        // Preload first component so SCF library is initialized (keeps previous behavior)
        load_scf_component(0);
        printf("SCF preprocess: discovered %d component file(s).\n", (int)scf_component_files.size());

		#endif
	}



	////===================================================
	////potential methods
	/* 	Potential: direct summation.
		The potential of a point in the space. 
		The potential is relative potential (without mass).
	*/
	double potential_sum(const VecDoub& x_tgt){
		double pot = 0.;
		int N1 = this->NumPart+1;
		double epsilon, h, h_inv, dx, dy, dz, r2, r, u, fac;
		for(int i=1;i<N1;i++){
    		epsilon = softening_type[P[i].Type]; //dmax??
      		h = epsilon*2.8;
      		h_inv = 1./h;
			dx = x_tgt[0]-P[i].Pos[0];
			dy = x_tgt[1]-P[i].Pos[1];
			dz = x_tgt[2]-P[i].Pos[2];
      		r2 = dx*dx+dy*dy+dz*dz;
      		r = sqrt(r2);
      		u = r*h_inv;
	  		fac = P[i].Mass * h_inv*Weight_splineSofteningKernel(u);
			pot += fac;
		}
		pot = conv_G*pot;
		return pot;
	}

	/* 	Potential: direct summation.
		For a test particle, its forces should remove the effort by itself.
		Remove the potential of the particle itself directly. Other algorithms (NOT 
		direct summation algorithm) do not consider it.
	*/
	double potential_sum_particle(const VecDoub& x_tgt, int ID){
		double h_inv = 1./softening_type[P[ID].Type]/2.8;
		// return potential_sum(x_tgt) - conv_G*P[ID].Mass *h_inv*(-2.8)*1.5;
		return potential_sum(x_tgt) - conv_G*P[ID].Mass *h_inv*(-2.8);
	}

	/* 	Potential: direct summation.
		The potential of a point in the space. 
		The potential is relative potential (without mass).
	*/
	double potential_sum1(const VecDoub& x_tgt){
		double pot = 0.;
		int N1 = this->NumPart+1;
		// DEBUG_PRINT_V0d(10, this->NumPart, "NumPart");
		for(int i=1;i<N1;i++){
			// DEBUG_PRINT_V0d(1, i, "i");
			// DEBUG_PRINT_V0d(1, P[i].Pos[0], "P[i].Pos[0]");
			double delta_r2 = pow((x_tgt[0]-P[i].Pos[0]),2) +pow((x_tgt[1]-P[i].Pos[1]),2) +pow((x_tgt[2]-P[i].Pos[2]),2);
			double Sr = - (delta_r2 + 1.5*pow(softening_type[P[i].Type],2)) / pow(delta_r2+pow(softening_type[P[i].Type], 2), 1.5);
			pot += P[i].Mass*Sr;
			// DEBUG_PRINT_V0d(10, delta_r2, "delta_r2");
			// DEBUG_PRINT_V0d(1, Sr, "Sr");
		}
		pot = conv_G*pot;
		return pot;
	}

	/* 	Potential: direct summation.
		The potential of a particle by the field. 
		Remove the potential of the particle itself directly. Other algorithms (NOT 
		direct summation algorithm) do not consider it.
	*/
	double potential_sum_particle1(const VecDoub& x_tgt, int ID){
		// DEBUG_PRINT_V0d(1, - conv_G*P[ID].Mass * (-1.5/softening_type[P[ID].Type]), "P_self");
		return potential_sum1(x_tgt) - conv_G*P[ID].Mass * (-1.5/softening_type[P[ID].Type]);
	}
	
	/* 	ambiguous
		Potential: direct summation.
		The meaning of this potential is ambiguous. 
		Remove the potential of the particle itself when the distance of the target point 
		and a particle is small enough.
	*/
	double potential_sum_ambiguous(const VecDoub& x_tgt){ //by directly adding with softening
		double pot = 0.;
		for(int i=0+1;i<NumPart+1;i++){
			double delta_r2 = pow((x_tgt[0]-P[i].Pos[0]),2) +pow((x_tgt[1]-P[i].Pos[1]),2) +pow((x_tgt[2]-P[i].Pos[2]),2);
			if(delta_r2>Err*Err){
				double Sr = - (delta_r2 + 1.5*pow(softening_type[P[i].Type],2)) / pow(delta_r2+pow(softening_type[P[i].Type], 2), 1.5);
				pot += P[i].Mass*Sr;
			}
		}
		pot = conv_G*pot;
		return pot;
	}

	/*	Potential: Self-consist field (SCF) method.
		Now only potential of a fixed snapshot is provided.
	*/
	double potential_SCF(const VecDoub& x_tgt){
    	double x = x_tgt[0], y = x_tgt[1], z = x_tgt[2];
		double pot;

		if( !(x_tgt[0]>Err1 && x_tgt[1]>Err1) ){
			double p = 0.;
			if(!(abs(x_tgt[0])>Err1)){
				x = Err1;
			}
			if(!(abs(x_tgt[1])>Err1)){
				y = Err1;
			}
        	get_pot_xyz_(&x, &y, &z, &pot);
			p += pot;

			if(!(abs(x_tgt[0])>Err1)){
				x = -Err1;
			}
			if(!(abs(x_tgt[1])>Err1)){
				y = -Err1;
			}
        	get_pot_xyz_(&x, &y, &z, &pot);
			p += pot;
			pot = p/2;
		}else{
        	get_pot_xyz_(&x, &y, &z, &pot);
		}
        // printf("potential_SCF(): %e; ", pot);
		return pot;
	}

    /*  Potential: Self-consistent field (SCF) method.
        version multicomp: Now returns the total potential from all components by summing 
        each component’s SCF contribution.
    */
    double potential_SCF_sumcomp(const VecDoub& x_tgt){
        // Re-discover (defensive) if not yet initialized
        #ifdef POTENTIAL_SCF
        if(scf_component_files.empty()){
            scf_workfile = path_gm + "intermediate/scficoef";
            // try to reuse preprocess path
            FILE* fp = fopen(scf_workfile.c_str(), "r");
            if(fp){ fclose(fp); scf_component_files.push_back(scf_workfile); }
        }
        #endif

        auto pot_single = [&](int comp)->double{
            double x = x_tgt[0], y = x_tgt[1], z = x_tgt[2], pot=0.0;
            // Ensure the SCF library has the right component loaded
            load_scf_component(comp);
            // Original small-axis guard retained
            if( !(x_tgt[0]>Err1 && x_tgt[1]>Err1) ){
                double p = 0.;
                if(!(abs(x_tgt[0])>Err1)) x =  Err1;
                if(!(abs(x_tgt[1])>Err1)) y =  Err1;
                get_pot_xyz_(&x, &y, &z, &pot); p += pot;
                if(!(abs(x_tgt[0])>Err1)) x = -Err1;
                if(!(abs(x_tgt[1])>Err1)) y = -Err1;
                get_pot_xyz_(&x, &y, &z, &pot); p += pot;
                pot = p*0.5;
            }else{
                get_pot_xyz_(&x, &y, &z, &pot);
            }
            return pot;
        };

        double pot_total = 0.0;
        for(int c=0; c<(int)scf_component_files.size(); ++c){
            pot_total += pot_single(c);
        }
        return pot_total;
    }

    /*  version multicomp: Helpers for multi-component SCF.
    	Copy file (binary-safe). Returns 0 on success, <0 on error.
    */
	int copy_file_(const std::string& src, const std::string& dst){
        FILE *fi=fopen(src.c_str(), "rb"); if(!fi) return -1;
        FILE *fo=fopen(dst.c_str(), "wb"); if(!fo){ fclose(fi); return -2; }
        char buf[1<<15];
        size_t n;
        while((n=fread(buf,1,sizeof(buf),fi))>0) fwrite(buf,1,n,fo);
        fclose(fi); fclose(fo);
        return 0;
    }

    /*  version multicomp: Ensure the Fortran SCF has component `comp` loaded (id into scf_component_files).
    	Returns 0 on success.
    */
	int load_scf_component(int comp){
        if(comp==scf_loaded_comp) return 0;
        if(comp<0 || comp>=(int)scf_component_files.size()){
            printf("load_scf_component(): invalid comp=%d\n", comp);
            exit(1);
        }
        if(scf_workfile.empty()) scf_workfile = path_gm + "intermediate/scficoef";
        const std::string &src = scf_component_files[comp];
        if(src != scf_workfile){
            int rc = copy_file_(src, scf_workfile);
            if(rc!=0){
                printf("load_scf_component(): failed to copy '%s' -> '%s' (rc=%d)\n",
                       src.c_str(), scf_workfile.c_str(), rc);
                exit(1);
            }
        }
        // (Re)initialize Fortran SCF with the selected coefficients
        get_parameter_();
        scf_loaded_comp = comp;
        return 0;
    }

	/* 	Potential: tree method.
	*/
	// double potential_tree(const VecDoub& x_tgt, int ID=-1){ //tree_pot
	// 	return 0.;
	// }

	double potential_tree(const VecDoub& x_tgt, int ID=-1){ //tree_pot
		if (!pot_tree_built_ || !pot_tree_) {
			std::cout<<"Warning: The potential tree has not been built. Here return zero potential.\n";
			return 0.;
		}
		double eps_target = 0.0;
		if(ID!=-1){
			eps_target = softening_type[P[ID].Type];
		}
		return pot_tree_->potential(x_tgt[0], x_tgt[1], x_tgt[2], eps_target, ID);
	}

	/*	Calculate potential on a 3D Cartesian coordinate by SPH(smooth particle hydro) method, 
		a kernel density function estimation mathod on k-nearest neighbours interpolation. 
		However, potential of position who is too far from the galaxy center (set as 10 times 
		of the main scalelength; in fact, the good position for interpolation might be surrounded 
		by samples and samples are too far) is calculated by direct summation method 
		this->potential_sum_ambiguous() for accuracy.
	*/
	double potential_SPH(const VecDoub& x, double* rho_target=nullptr){

		// double r_center = distance_l2norm(x); //Is condition for interpolation
		// if(r_center>8*scale_length_comp[0]) return potential_sum_ambiguous(x);
		// DEBUG_PRINT_V0d(1, x[0], "sph");

		double pot = 0., rho = 0.; double h, h_inv, dr;
		VecDoub arr = {x[0], x[1], x[2]};

        vector<int> nearest_idx = this->pkdtree->find_k_nearest(N_knn, arr);
		int ifar = nearest_idx[N_knn-1];
		// int inear = nearest_idx[0];
		// DEBUG_PRINT_V0d(1, "B", "sph");
        for(auto i:nearest_idx){
			// int ifar_Plus = ifar, i_Plus = i;
			int ifar_Plus = ifar+1, i_Plus = i+1;
            h  = distance_l2norm({x[0], x[1], x[2]}, {P[ifar_Plus].Pos[0], P[ifar_Plus].Pos[1], P[ifar_Plus].Pos[2]});
            dr = distance_l2norm({x[0], x[1], x[2]}, {P[i_Plus].Pos[0], P[i_Plus].Pos[1], P[i_Plus].Pos[2]});
			rho += Weight_SPHSmoothingKernel(dr,h)*P[i_Plus].Mass;
            pot += Weight_SPHSmoothingKernel(dr,h)*P[i_Plus].Mass*P[i_Plus].Pot;
			// acc[0] += Weight_SPHSmoothingKernel(dr,h)*P[i_Plus].Mass*P[i_Plus].Acc[0];
        }
        pot /= rho; //Poisson eq in cube and intepolation -> G*S*m

		// DEBUG_PRINT_V0d(1, "C", "sph");
		// *rho_target = rho;
        // printf("potential_SPH(): %e %e\n", pot, rho);
		return pot;
	}

	/* Potential by a certin formula. This is usually as a background potential. */
	double potential_formula(const VecDoub& x_tgt){
		return 0.;
	}
	// double potential_Poisson(const VecDoub& x_tgt){
	// 	//By Poisson equation. Not done.
	// 	return 0.;
	// }

	/* 	Potential: interpolation by RBF.
		To return the value of potential on x by interpolation on the snapshot data points; 
		for sparse points, directly add each particle's potential instead of interpolation. 
		(Then in Stage::potential_t(), interpolate two snapshots' potentials on t.)
		@parameter x_tgt: potential target position;
		@parameter algorithm: calculation algorithm to select:
		    algorithm=0: distance reciprocal weight average
    		algorithm=1: 3d linear interpolation
    		algorithm=2: 3d LSQ quadric poly fit
			algorithm=3: 3d cubic spline
			algorithm=4: 
    		algorithm=5: interface to Python scipy.interpolate.interp3d()??
    		otherwise: directly add each particle's potential */
	double potential_interp(const VecDoub& x_tgt, int algorithm = 0){

		//// get k neareat neighbours
		double potential_tgt = 0; //the certain potential of target position
		int nknn = 0;   //number of trial k neareat neighbours
		vector<int> nearest_statistics;

		//// calculate
		switch(algorithm){
			case 1: { //bad estimation result
				vector<int> nearest_idx = {1,1,1,1}; //??
				nknn = 4;
				MatrixXd A(nknn, 4); //Matrix of {1,x,y,z}_i, i=1,2,...,nknn
				VectorXd B(nknn);    //potential
				VectorXd C(nknn);    //solve of interp index
				VectorXd D(nknn);    //distance
				for(int i=0;i<nknn;i++){
					A(i,0)=1;
					for(int dim=0;dim<Dimension;dim++){
						A(i,dim+1) = P[nearest_idx[i]].Pos[dim];
					}
					B(i) = P[nearest_idx[i]].Pot;
					D(i) = sqrt( (A(i,1)-x_tgt[0])*(A(i,1)-x_tgt[0]) 
						+(A(i,2)-x_tgt[1])*(A(i,2)-x_tgt[1]) +(A(i,3)-x_tgt[2])*(A(i,3)-x_tgt[2]) );
				}
				C = A.ldlt().solve(B); //Cholesky decompose and solve linear eqs
				potential_tgt = C(0)+C(1)*x_tgt[0]+C(2)*x_tgt[1]+C(3)*x_tgt[2];
				// cout<<"xyz       A=\n"<<A<<endl;
				// cout<<"potential B=\n"<<B<<endl;
				// cout<<"solve     C=\n"<<C<<endl;
				// cout<<"disdance  D=\n"<<D<<endl;
				// cout<<"target potential = "<<potential_tgt<<endl;
				break;
			}
			case 2: { //LSQ
				potential_tgt = 0.;
				break;
			}
			case 3: { //3d cubic spline interpolation even FEM?? Not implemented...
				potential_tgt = 0.;
				break;
			}
			default: { //case 0: //too slow; bad for distant points
				nknn = MaxInterpPoints;

				VecDoub x_tgt_arr = {x_tgt[0],x_tgt[1],x_tgt[2]};
				vector<int> nearest_idx = pkdtree->find_k_nearest(nknn, x_tgt_arr);

				// double r0 = 10.;
				// int ifar = nearest_idx[nknn-1]; //+1;
				// double r0 = distance_l2norm({x_tgt[0], x_tgt[1], x_tgt[2]}, {P[ifar].Pos[0], P[ifar].Pos[1], P[ifar].Pos[2]});
				int index = 0;
				VecDoub dr_arr(nknn);
				for(auto i:nearest_idx){
					//:: the kernel function W here is only versus r; max dr as bandwidth
					dr_arr[index] = distance_l2norm({P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]}, x_tgt);
					index ++;
				}
				double r0 = *max_element(dr_arr.begin(),dr_arr.end());
				// for(int i=0;i<index;i++){
				//     // printf("%e / %e, %e\n", dr_arr[i], dr_max, m_arr[i]);
				//     DF += Weight_SPHSmoothingKernel(dr_arr[i],dr_max)*m_arr[i];
				// }

				MatrixXd Pts(nknn, Dimension);
				VectorXd Vals(nknn);
				VectorXd Dist(nknn);
				for(int i=0;i<nknn;i++){ //assign values
					int iPD = nearest_idx[i]; //+1;
					for(int dim=0;dim<Dimension;dim++){
						Pts(i,dim) = P[iPD].Pos[dim];
					}
					Vals(i) = P[iPD].Pot;
					// Dist(i)=sqrt( (Pts(i,0)-x_tgt_arr[0])*(Pts(i,0)-x_tgt_arr[0]) 
					//	+(Pts(i,1)-x_tgt_arr[1])*(Pts(i,1)-x_tgt_arr[1]) +(Pts(i,2)-x_tgt_arr[2])*(Pts(i,2)-x_tgt_arr[2]) );
				}
				// cout<<"points=\n"<<Pts<<endl;
				// cout<<"rhs=\n"<<Vals<<endl;
				// RBF_thinplate imq(r0); //bad
				// RBF_inversemultiquadric imq(r0);
				// RBF_multiquadric imq(r0); //middle
				RBF_Gauss imq(r0); //bad, better
				RBF_interp myfunc(Pts, Vals, imq, 0);
				VectorXd x_tgt_mat(Dimension);
				x_tgt_mat<<x_tgt[0], x_tgt[1], x_tgt[2];
				potential_tgt = myfunc.interp(x_tgt_mat); // /m_target_comp[component]; //TACT use the relative potential
				break;
			}
		}
		return potential_tgt;
	}
	////===================================================



	////===================================================
	////forces (accelerations, without mass) methods
	/* Forces by direct summation of all particles. */
	VecDoub forces_sum(const VecDoub& x_tgt){
		int IDMax = NumPart+1;
        VecDoub forces = {0.,0.,0.};
		int N1 = this->NumPart+1;
		double epsilon, h, h_inv, dx, dy, dz, r2, r, u, fac;
		for(int i=1;i<N1;i++){
			epsilon = softening_type[P[i].Type]; //dmax??
			h = epsilon*2.8;
			h_inv = 1./h;
			dx = x_tgt[0]-P[i].Pos[0];
			dy = x_tgt[1]-P[i].Pos[1];
			dz = x_tgt[2]-P[i].Pos[2];
			r2 = dx*dx+dy*dy+dz*dz;
			r = sqrt(r2);
			u = r*h_inv;
			fac = P[i].Mass * h_inv*h_inv*h_inv * Weight_splineSofteningKernel_derive_divid_u(u);
			// if(std::isnan(fac)){
			// 	std::cout<<"r: "<<r<<"\n";
			// 	exit(0);
			// }
			forces[0] += (-dx*fac);
			forces[1] += (-dy*fac);
			forces[2] += (-dz*fac);
		}
		return {conv_G*forces[0], conv_G*forces[1], conv_G*forces[2]};
	}

	/* 	Forces by direct summation of all particles. 
		For a test particle, its forces should remove the effort by itself.
	*/
	VecDoub forces_sum_particle(const VecDoub& x_tgt, int ID){
		int IDMax = NumPart+1;
        VecDoub forces = forces_sum(x_tgt);
		double epsilon, h, h_inv, dx, dy, dz, r2, r, u, fac;
		epsilon = softening_type[P[ID].Type]; //dmax??
		h = epsilon*2.8;
		h_inv = 1./h;
		dx = x_tgt[0]-P[ID].Pos[0];
		dy = x_tgt[1]-P[ID].Pos[1];
		dz = x_tgt[2]-P[ID].Pos[2];
		r2 = dx*dx+dy*dy+dz*dz;
		r = sqrt(r2);
		u = r*h_inv;
		fac = P[ID].Mass * h_inv*h_inv*h_inv * Weight_splineSofteningKernel_derive_divid_u(u);
		return {forces[0]+conv_G*dx*fac, forces[1]+conv_G*dy*fac, forces[2]+conv_G*dz*fac};
	}

	/* Forces by direct summation of all particles. */
	VecDoub forces_sum1(const VecDoub& x_tgt){
		int IDMax = NumPart+1;
        VecDoub forces = {0.,0.,0.};
		for(int i=1;i<IDMax;i++){
			double r2 = pow((x_tgt[0]-P[i].Pos[0]),2) +pow((x_tgt[1]-P[i].Pos[1]),2) +pow((x_tgt[2]-P[i].Pos[2]),2);
            double r = sqrt(r2);
            double epsilon2 = pow(softening_type[P[i].Type], 2);
			//corrected Plummer softening kernel (SF is derivation of Sr):
			// double Sr = - (r2 + 1.5*epsilon2) / pow(r2+epsilon2, 1.5);
			// double SF = - ( 3*r*pow(r2+epsilon2, 0.5)*(r2+1.5*epsilon2) 
			//	-2*r*(r2+epsilon2) ) / pow(r2+epsilon2, 2.5);
			//simple Plummer softening kernel (SF is derivation of Sr):
			// double Sr = -1./pow(r2+epsilon2, 0.5);
			double SF = r*pow(r2+epsilon2, -1.5); //simplest softening
			double Fi = - P[i].Mass*SF; //each Fi should be calculated independently
			forces[0] += Fi*(x_tgt[0]-P[i].Pos[0])/(r+Err); //wrong??
			forces[1] += Fi*(x_tgt[1]-P[i].Pos[1])/(r+Err);
			forces[2] += Fi*(x_tgt[2]-P[i].Pos[2])/(r+Err);
		}
		return {conv_G*forces[0], conv_G*forces[1], conv_G*forces[2]};
	}

	VecDoub forces_sum_particle1(const VecDoub& x_tgt, int ID){
		int IDMax = NumPart+1;
        VecDoub forces = {0.,0.,0.};
		for(int i=1;i<IDMax;i++){
			if(i==ID) continue;
			double r2 = pow((x_tgt[0]-P[i].Pos[0]),2) +pow((x_tgt[1]-P[i].Pos[1]),2) +pow((x_tgt[2]-P[i].Pos[2]),2);
            double r = sqrt(r2);
			// if(r<Err) continue;
            double epsilon2 = pow(softening_type[P[i].Type], 2);
			double SF = r*pow(r2+epsilon2, -1.5); //simplest softening
			double Fi = - P[i].Mass*SF; //each Fi should be calculated independently
			forces[0] += Fi*(x_tgt[0]-P[i].Pos[0])/(r); //+Err??
			forces[1] += Fi*(x_tgt[1]-P[i].Pos[1])/(r);
			forces[2] += Fi*(x_tgt[2]-P[i].Pos[2])/(r);
		}
		return {conv_G*forces[0], conv_G*forces[1], conv_G*forces[2]};
	}

	/* ambiguous */
	VecDoub forces_sum_ambiguous(const VecDoub& x_tgt){ //by directly adding with soften
        VecDoub forces = {0.,0.,0.};
		for(int i=0+1;i<NumPart+1;i++){
            // if(P[i].Type!=2) continue; //opt: for one type
			double r2 = pow((x_tgt[0]-P[i].Pos[0]),2) +pow((x_tgt[1]-P[i].Pos[1]),2) +pow((x_tgt[2]-P[i].Pos[2]),2);
            double r = sqrt(r2);
            double epsilon2 = pow(softening_type[P[i].Type], 2);
			if(r2>Err*Err){ //to exclude the particle itself
				// double Sr = - (r2 + 1.5*epsilon2) / pow(r2+epsilon2, 1.5);
				// double SF = - ( 3*r*pow(r2+epsilon2, 0.5)*(r2+1.5*epsilon2) -2*r*(r2+epsilon2) ) / pow(r2+epsilon2, 2.5); //SF is derivation of Sr
				double SF = r*pow(r2+epsilon2, -1.5); //simplest softening
				double Fi = - P[i].Mass*SF; //each Fi should be calculated independently
				forces[0] += Fi*(x_tgt[0]-P[i].Pos[0])/r;
				forces[1] += Fi*(x_tgt[1]-P[i].Pos[1])/r;
				forces[2] += Fi*(x_tgt[2]-P[i].Pos[2])/r;
                // printf("%d %e %e; %e %e %e; %e %e\n", i, P[i].Mass, epsilon2, r, abs(Sr), abs(SF), Fi, (x_tgt[0]-P[i].Pos[0])/r);
			}
		}
		return {conv_G*forces[0], conv_G*forces[1], conv_G*forces[2]};
	}

	VecDoub forces_SCF(const VecDoub& x_tgt){
    	double x = x_tgt[0], y = x_tgt[1], z = x_tgt[2];
		double accx, accy, accz;

		if( !( abs(x_tgt[0])>Err1 && abs(x_tgt[1])>Err1 && abs(x_tgt[2])>Err1 ) ){
			// DEBUG_PRINT_V1d(1, (VecDoub){x, y, z}, "xxx0");
			if(!(abs(x_tgt[0])>Err1)){ //[learn code] "!func(x)>0" is not "!(func(x)>0)"
				// cout<<x<<" "<<x_tgt[0]<<" "<<Err1<<" ";
				x = Err1; //modifying to avoid nan
			}
			if(!(abs(x_tgt[1]))>Err1){
				y = Err1;
			}
			if(!(abs(x_tgt[2]))>Err1){
				z = Err1;
			}
			// DEBUG_PRINT_V1d(1, (VecDoub){x, y, z}, "xxx1");
			double ax = 0., ay = 0., az = 0.;
        	get_acc_xyz_(&x, &y, &z, &accx, &accy, &accz);
			ax += accx, ay += accy, az += accz;
			// DEBUG_PRINT_V1d(1, (VecDoub){accx, accy, accz}, "fff1");
			
			if(!(abs(x_tgt[0])>Err1)){
				x = -Err1;
			}
			if(!(abs(x_tgt[1])>Err1)){
				y = -Err1;
			}
			if(!(abs(x_tgt[2])>Err1)){
				z = -Err1;
			}
        	get_acc_xyz_(&x, &y, &z, &accx, &accy, &accz);
			// DEBUG_PRINT_V1d(1, (VecDoub){x, y, z}, "xxx2");
			// DEBUG_PRINT_V1d(1, (VecDoub){accx, accy, accz}, "fff2");
			ax += accx, ay += accy, az += accz;
			accx = ax/2, accy = ay/2, accz = az/2;
		}else{
			get_acc_xyz_(&x, &y, &z, &accx, &accy, &accz);
		}
		// DEBUG_PRINT_V1d(1, (VecDoub){accx, accy, accz}, "accx, accy, accz");
    	return {accx, accy, accz};
	}

	VecDoub forces_SPH(const VecDoub& x_tgt){
    	return {0., 0., 0.}; //??
	}

	VecDoub forces_interp(const VecDoub& x_tgt, int algorithm=0){

		//// get k neareat neighbours
		double forces[3]; //the certain potential of target position

		int nknn = MinInterpPoints; //number of trial k neareat neighbours
		VecDoub x_tgt_arr = {x_tgt[0], x_tgt[1], x_tgt[2]};
		vector<int> nearest_idx = pkdtree->find_k_nearest(nknn, x_tgt_arr);
		MatrixXd Pts(nknn, Dimension);
		VectorXd Vals(nknn);
		VectorXd Dist(nknn);
		double r0 = 1.;
		for(int j=0;j<3;j++){
    		for(int i=0;i<nknn;i++){ //assign values
    	    	for(int dim=0;dim<Dimension;dim++){
   	        		Pts(i,dim) = P[nearest_idx[i]].Pos[dim];
    	    	}
        		Vals(i) = P[nearest_idx[i]].Pot; //Acc[dim]??
        		// Dist(i)=sqrt( (Pts(i,0)-x_tgt_arr[0])*(Pts(i,0)-x_tgt_arr[0]) 
				//	+(Pts(i,1)-x_tgt_arr[1])*(Pts(i,1)-x_tgt_arr[1]) 
				//	+(Pts(i,2)-x_tgt_arr[2])*(Pts(i,2)-x_tgt_arr[2]) );
   			}
			// RBF_thinplate imq(r0); //bad
			// RBF_inversemultiquadric imq(r0);
			// RBF_multiquadric imq(r0); //middle
			RBF_Gauss imq(r0); //bad, better
			// RBF_interp myfunc(Pts, Vals, imq, 0);
			RBF_interp myfunc(Pts, Vals, imq, 1);
			VectorXd x_tgt_mat(Dimension);
			x_tgt_mat<<x_tgt[0], x_tgt[1], x_tgt[2];
			forces[j] = myfunc.interp(x_tgt_mat);
		}
		return {forces[0], forces[1], forces[2]};
	}
	////===================================================



	////===================================================
	////Higher order derives of field.
	/* diag of second derive 3*3 of potential (without mass) */
	VecDoub Pxx_sum(const VecDoub& x, int coor){ //by directly adding with soften
	    if(coor == 0){ //Cartesian
		    printf("No result provided in Cartesian now!\n");
		    double Phi__xx = 0., Phi__yy = 0., Phi__zz = 0.;
		    return {Phi__xx, Phi__yy, Phi__zz};
	    }
	    if(coor == 2){ //Polar
            VecDoub forces = {0.,0.,0.};
		    for(int i=0+1;i<NumPart+1;i++){
			    double delta_r2 = pow((x[0]-P[i].Pos[0]),2) +pow((x[1]-P[i].Pos[1]),2) +pow((x[2]-P[i].Pos[2]),2);
                double delta_r = sqrt(delta_r2);
                double epsilon2 = pow(softening_type[P[i].Type], 2);
                VecDoub delta_Rpz = conv::CartesianToPolar({x[0]-P[i].Pos[0], x[1]-P[i].Pos[1], x[2]-P[i].Pos[2]});

			    if(delta_r2>Err){ //to exclude the particle itself
				    double k = conv_G*P[i].Mass;
				    forces[0] += k *pow(delta_r2+epsilon2, -2.5) *(2*delta_r2+2*epsilon2-3*delta_Rpz[0]*delta_Rpz[0]);
				    forces[1] += k *0.;
				    forces[2] += k *pow(delta_r2+epsilon2, -2.5) *(delta_Rpz[0]*delta_Rpz[0]+epsilon2-2*delta_Rpz[2]*delta_Rpz[2]);
			    }
		    }
		    return forces;
	    }
	    else{
		    printf("No result provided in such coordinate now!\n");
		    return {0.,0.,0.};
	    }
	}
	////===================================================

};



// typedef vector<Snapshot> SnapshotVec;
// typedef vector<Snapshot*> pSnapshotVec;
// typedef vector<Snapshot>* ptrSnapshotVec;

typedef std::vector<std::unique_ptr<Snapshot>> SnapshotVec; //tree_pot
typedef std::vector<Snapshot*> pSnapshotVec;
typedef std::vector<std::unique_ptr<Snapshot>>* ptrSnapshotVec;

class Stage{
public:
	string path_gm; //member of class Stage
	SnapshotVec SV;
	pSnapshotVec SS;

	int t_init;
	int t_final;
	double dt_load; //time interval between two loaded snapshots
	double dt_step; //time interval betwwen two simulation output neareast snapshots
	int Load_Step;
	int Initial_load; //for P[]
	int Final_load; //for P[]
	int N_load; //for P[]
	int Initial_snapshot; //for PT[]
	int Final_snapshot; //for PT[]
	int N_timepoints; //for PT[]

	bool is_preprocess_rotation; //for rotation frame

	// explicit 
	Stage(string path_IC_1){
		char paramfile[MaxCharactersInString];
		read_params(path_IC_1); //N_allPtcs has been calculated after now
		string path_work = get_workpath(); //../../../../
		path_gm = path_work+"GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"+modelPath; //filename_IC="IC_param_"${ModelName}".txt"
		CS = std::unique_ptr<ConfocalEllipsoidalCoordSys>(new ConfocalEllipsoidalCoordSys(-3.,-2.));
	};

	void set_info_snapshot(double ti, double tf, double dt_l, double dt_st){ //better in initialization function
		// // t_init = ti; //?? no use
		// this->t_init = ti;
		// t_final = tf;
		// dt_load = dt_l;
		// dt_step = dt_st;
		// Load_Step = (int)round(dt_load/dt_step);

		// Initial_load = t_to_loadsnapshot(ti, dt_load, t_init, 0.);
		// Final_load = t_to_loadsnapshot(tf, dt_load, t_init, 0.);
		// N_load = Final_load-Initial_load+1;
		// Initial_snapshot = t_to_snapshot(ti, dt_step, 0., 0.);
		// Final_snapshot = t_to_snapshot(tf, dt_step, 0., 0.);
		// N_timepoints = (int)round( (t_final-t_init) / (dt_step) );

		// VecDoub VEC = {Initial_snapshot, Final_snapshot, Initial_load, Final_load, 
		// 	dt_load, dt_step, Load_Step, N_timepoints};
		// DEBUG_PRINT_V1d(1, VEC, "snapshot setting");

		double tii = ti;
		this->t_init = ti; //?? no use
		// t_init = ti; //?? no use
		t_final = tf;
		dt_load = dt_l;
		dt_step = dt_st;
		Load_Step = (int)round(dt_load/dt_step);
DEBUG_PRINT_V0d(1, ti, "ti set_info_snapshot");
DEBUG_PRINT_V0d(1, tii, "tii set_info_snapshot");
DEBUG_PRINT_V0d(1, dt_load, "dt_load");

		Initial_load = t_to_loadsnapshot(ti, dt_load, tii, 0.);
		Final_load = t_to_loadsnapshot(tf, dt_load, tii, 0.);
		N_load = Final_load-Initial_load+1;
// DEBUG_PRINT_V0d(1, t_init, "t_init");
DEBUG_PRINT_V0d(1, tf, "Initial_load");
DEBUG_PRINT_V0d(1, Initial_load, "Initial_load");
DEBUG_PRINT_V0d(1, Final_load, "Initial_load");

		Initial_snapshot = t_to_snapshot(ti, dt_step, 0., 0.);
		Final_snapshot = t_to_snapshot(tf, dt_step, 0., 0.);
		N_timepoints = (int)round( (t_final-tii) / (dt_step) );
DEBUG_PRINT_V0d(1, t_final, "Initial_snapshot");
DEBUG_PRINT_V0d(1, dt_step, "Initial_snapshot");
DEBUG_PRINT_V0d(1, Initial_snapshot, "Initial_snapshot");
DEBUG_PRINT_V0d(1, Final_snapshot, "Initial_snapshot");
DEBUG_PRINT_V0d(1, N_timepoints, "Initial_snapshot");

		VecDoub VEC = {(double)Initial_snapshot, (double)Final_snapshot, (double)Initial_load, (double)Final_load, 
			dt_load, dt_step, (double)Load_Step, (double)N_timepoints};
		DEBUG_PRINT_V1d(1, VEC, "snapshot setting");
	}
	void reset_SnapshotsData(){
		SnapshotVec().swap(this->SV);
		pSnapshotVec().swap(this->SS);
	}
	
	int load_multi_snapshots(double t_init1, double t_final1, double dt_load1, double dt_step1, 
		int is_witeSnapshot, int is_preprocessed1);
	
	int write_particleData_byStep(double t_init1, double t_final1, double dt_load1, double dt_step1, 
		int is_witeSnapshot, int is_preprocessed1, int maxWrite=-1, int maxLoad=500, int wID=-1)
	{
		int NM = N_allPtcs;
		int maxWrite1;
		int ID_i = 1, ID_f = NM+1;
		int N_timepoints_pre = 9000; //set N_timepoints
		if(wID>NM){
			printf("Particle ID %d is out of range.\n", wID);
			return -2;
		}else if(wID>0){
			ID_i = wID, ID_f = wID+1;
		}
		if(0<maxWrite && maxWrite<maxLoad){
			printf("Too many snapshots to load.\n");
			return -3;
		}else if(maxWrite<=0){
			maxWrite1 = N_timepoints_pre;
		}else if(N_timepoints_pre<maxWrite){
			// printf("Use defualt count snapshots to load.\n");
			printf("Too many snapshots to write.\n");
			return -3;
		}else{
			maxWrite1 = maxWrite;
		}

	    char wt_fname[MaxCharactersInString];
		int loadSnapshot_i = 0, loadSnapshot_f, count = 0;
		double t_i = t_init, t_f = t_init;
		printf("Count of loaded and wroten snapshots once: %d, %d.\n", maxLoad, maxWrite1);
		printf("Total count of particles: %d; particle_IDs: [%d, %d): \n", NM, ID_i, ID_f);
		while(count<maxWrite1){
			if(maxWrite1 % maxLoad == 0){
				loadSnapshot_f = maxLoad;
			}else{
				loadSnapshot_f = maxWrite1 % maxLoad;
			}
DEBUG_PRINT_V0d(1, maxWrite1, "maxWrite1");
DEBUG_PRINT_V0d(1, maxLoad, "maxLoad");
DEBUG_PRINT_V0d(1, maxWrite1 % maxLoad, "maxWrite1 % maxLoad");
			t_f += dt_step*(loadSnapshot_f-loadSnapshot_i);
			printf("loadIndex: [%d, %d): \n", loadSnapshot_i, loadSnapshot_f);
			printf("time: [%e, %e): \n", t_i, t_f);
DEBUG_PRINT_I(21);
			// this->reset_SnapshotsData();
			this->load_multi_snapshots(t_i, t_f, dt_load, dt_step, is_witeSnapshot, is_preprocessed1);
DEBUG_PRINT_I(22);
			printf("loadIndex after: [%d, %d): \n", loadSnapshot_i, loadSnapshot_f);
			printf("time after: [%e, %e): \n", t_i, t_f);

			for(int id=ID_i;id<ID_f;id++){
				printf("particleID_%d ", id);
				sprintf(wt_fname, "%sparticle/particle_%d.txt", path_gm.data(), id);
				FILE *fp = nullptr;
				fp = fopen(wt_fname, "a+");
				if(fp==nullptr){
					printf("Cannot open file \"%s\".\n", wt_fname);
					return -1;
				}

				for(int s=loadSnapshot_i;s<loadSnapshot_f;s++){
					// printf("sss_%d, particleID_%d; ", s, id);
					fprintf(fp, "%e %e %e %e %e %e     %d %d %e %e     %e %e %e %e     %e %e %e %e     %e %e %e \n", 
						SS[s]->P[id].Pos[0], SS[s]->P[id].Pos[1], SS[s]->P[id].Pos[2], 
						SS[s]->P[id].Vel[0], SS[s]->P[id].Vel[1], SS[s]->P[id].Vel[2], 
						SS[s]->P[id].Id, SS[s]->P[id].Type,    SS[s]->P[id].Mass, SS[s]->P[id].Time, 
						SS[s]->P[id].U, SS[s]->P[id].Rho, SS[s]->P[id].Hsml, SS[s]->P[id].Ne, 
						SS[s]->P[id].Pot,    SS[s]->P[id].Acc[0], SS[s]->P[id].Acc[1], SS[s]->P[id].Acc[2], 
						SS[s]->P[id].dAdt, SS[s]->P[id].Age, SS[s]->P[id].Metal 
					);
				}
				fclose(fp);
				// printf("Write file \"%s\" ... done.\n", wt_fname);
			}
			printf("\n\n");
			count += maxLoad;
			t_i += dt_step*maxLoad;
		}
DEBUG_PRINT_I(25);
		return 0;
	}



	//=================================================
	//potential and forces
	/* potential */
	double potential_t(const VecDoub& x_target, const double& t, const int& ID, 
		int coor=0, int algorithm=0)
	{
// DEBUG_PRINT_I(66660);
		double pot;
		auto x = x_target;
		int snapshot_t1 = t_to_loadsnapshot(t, dt_load, t_init, 0.);
// DEBUG_PRINT_V0d(1, t, "666601 t");
// DEBUG_PRINT_V0d(1, dt_load, "666601 dt_load");
// DEBUG_PRINT_V0d(1, t_init, "666601 t_init");
// DEBUG_PRINT_V0d(1, snapshot_t1, "666601 snapshot_t1");
		// int snapshot_t2 = snapshot_t1+1;
		snapshot_t1 = selectMax(snapshot_t1, Initial_load);
		snapshot_t1 = selectMin(snapshot_t1, Final_load);
// DEBUG_PRINT_V0d(1, Initial_load, "666601 Initial_load");
// DEBUG_PRINT_V0d(1, Final_load, "666601 Final_load");
// DEBUG_PRINT_I(66661);

		switch(coor){ //input coor instead of output coor
			default: {break;} //0: Cartesian
			case 1: { //SPolar
				x = conv::SphericalPolarToCartesian(x_target);
				break;
			}
			case 2: { //CPolar
				x = conv::PolarToCartesian(x_target);
				break;
			}
		}
// DEBUG_PRINT_I(66662);

		switch(algorithm){
			default: { //0: direct summation
				// auto pot1 = SS[snapshot_t1]->potential_sum_particle({x[0], x[1], x[2]}, ID);
				// auto pot2 = SS[snapshot_t2]->potential_sum_particle({x[0], x[1], x[2]}, ID);
				// pot = interp_linear_2d_2points(t, t1, pot1, t2, pot2);
				// break;
				pot = SS[snapshot_t1]->potential_sum_particle({x[0], x[1], x[2]}, ID);
				break;
			}
			case 1: { //0: direct summation without substracting self particle potential
				pot = SS[snapshot_t1]->potential_sum({x[0], x[1], x[2]});
				break;
			}
			case 2: { //tree and kernel interpolation
				pot = SS[snapshot_t1]->potential_SPH({x[0], x[1], x[2]});
				break;
			}
			case 3: { //RBF interpolation
				pot = SS[snapshot_t1]->potential_interp({x[0], x[1], x[2]}, 0);
				break;
			}
			case 4: { //SCF //global, fix snapshot SCF
				// DEBUG_PRINT_V0d(1, snapshot_t1, "potential_SCF calling");
				pot = SS[snapshot_t1]->potential_SCF({x[0], x[1], x[2]});
// DEBUG_PRINT_I(66663);
				break;
			}
			case 5: { //spline3 interpolation for triaxial ellipsoid potential
				// DEBUG_PRINT_V0d(1, snapshot_t1, "potential_TRSP3 calling");
				pot = SS[snapshot_t1]->potential_TRSP3({x[0], x[1], x[2]});
				break;
			}
			case 6: { //tree summation
				pot = SS[snapshot_t1]->potential_tree({x[0], x[1], x[2]}, ID);
				break;
			}
			case 7: { //formula //?? call a function
				pot = 0.;
				break;
			}
		}
// DEBUG_PRINT_I(66664);

		if(is_preprocess_rotation){ //one must run this->SS[snapshot_t1]->preprocess(True) to get TC.O before
// DEBUG_PRINT_I(66665);
// DEBUG_PRINT_V0d(1, snapshot_t1, "pot_t snapshot_t1");
// DEBUG_PRINT_V0d(1, SS.size(), "pot_t SS.size");
// DEBUG_PRINT_V0d(1, SS[snapshot_t1]->TC.O[0], "pot_t TC.O[0]");
			double Ob0 = SS[snapshot_t1]->TC.O[0], Ob1 = SS[snapshot_t1]->TC.O[1], Ob2 = SS[snapshot_t1]->TC.O[2];
			double normOb2normx2 = (Ob0*Ob0+Ob1*Ob1+Ob2*Ob2)*(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
			double Obx2 = pow(Ob0*x[0]+Ob1*x[1]+Ob2*x[2], 2);
			double pot_rotate = -0.5*(normOb2normx2-Obx2);
			// DEBUG_PRINT_V1d(1, (VecDoub){Ob0, pot, pot_rotate, pot_rotate/pot}, "{Ob0, pot(the ineria), pot_rotate, pot_rotate/pot}");
			pot -= pot_rotate;
// DEBUG_PRINT_I(66666);
		}
// DEBUG_PRINT_I(66667);
		
		return pot;
	}

	VecDoub forces_t(const VecDoub& x_target, const double& t, const int& ID, 
		int coor=0, int algorithm=0)
	{
		VecDoub forces(3);
		auto x = x_target;
		int snapshot_t1 = t_to_loadsnapshot(t, dt_load, t_init, 0.);
		// int snapshot_t2 = snapshot_t1+1;
		snapshot_t1 = selectMax(snapshot_t1, Initial_load);
		snapshot_t1 = selectMin(snapshot_t1, Final_load);

		switch(coor){ //input coor instead of output coor
			default: {break;} //0: Cartesian
			case 1: { //SPolar
				x = conv::SphericalPolarToCartesian(x_target);
				break;
			}
			case 2: { //CPolar
				x = conv::PolarToCartesian(x_target);
				break;
			}
		}

		switch(algorithm){
			default: { //0: direct summation
				forces = SS[snapshot_t1]->forces_sum_particle({x[0], x[1], x[2]}, ID);
				break;
			}
			case 1: { //formula
				forces = {0., 0., 0.};
				break;
			}
			case 2: { //tree and kernel interpolation
				forces = SS[snapshot_t1]->forces_SPH({x[0], x[1], x[2]});
				break;
			}
			case 3: { //RBF interpolation
				forces = SS[snapshot_t1]->forces_interp({x[0], x[1], x[2]}, 0);
				break;
			}
			case 4: { //SCF //global, fix snapshot SCF
				forces = SS[snapshot_t1]->forces_SCF({x[0], x[1], x[2]});
				break;
			}
			case 5: { //spline3 interpolation for triaxial ellipsoid potential
				// forces = SS[snapshot_t1]->forces_sum_particle1({x[0], x[1], x[2]}, ID);
				forces = SS[snapshot_t1]->forces_TRSP3({x[0], x[1], x[2]});
				break;
			}
		}

		VecDoub F = {x[0],x[1],x[2], forces[0],forces[1],forces[2]};
		switch(coor){ //not vector
			default: {break;} //Cartesian
			case 1: {
				F = conv::CartesianToSphericalPolar(F); //SPolar //only the last three as vector need to coordinate transform
				break;
			}
			case 2: {
				F = conv::CartesianToPolar(F); //CPolar
				break;
			}
		}

		if(is_preprocess_rotation){ //one must run this->SS[snapshot_t1]->preprocess(True) to get TC.O before
			; //?? it has no rotating frame modification; the Stackel Fudge method need only potential
		}

		// print_vec(Coor);
		// print_vec(F_Cartesian);
		// print_vec(F);
		return {F[3],F[4],F[5]};
	}

	VecDoub Pxx_t(const VecDoub& x, const double& t, int coor){ //the coor are in Snapshot::Pxx_sum()
		int t1=t-1000, t2=t-1000; //tt??
		auto Pxx = SS[0]->Pxx_sum({x[0], x[1], x[2]}, coor); //t ??
		return Pxx;
	}

	double kineticEnergy(const VecDoub& xv){ //must xv = {x,y,z, vx,vy,vz};
		return 0.5*( xv[3]*xv[3] + xv[4]*xv[4] + xv[5]*xv[5] );
	}

	VecDoub angularMomentCartesian(const VecDoub& xv){ //must xv = {x,y,z, vx,vy,vz};
		VecDoub L(3);
		L[0] = xv[1]*xv[5] + xv[2]*xv[4];
		L[1] = xv[2]*xv[3] + xv[0]*xv[5];
		L[2] = xv[0]*xv[4] + xv[1]*xv[3];
		return L;
	}
	//=================================================



	//=================================================
	double t_temp = -255.; /* The time of the particle; temporary. */
	int particle_ID = -255; /* The ID of the particle; temporary. */
	void set_info_timetemp(double t){
		t_temp = t;
	}
	void set_info_particle(int ID){
		particle_ID = ID;
	}

	particle_data* PT;
	/*	To set struct particle_data* PT with preprocess parameter at time t.
		Each time when running TEPPOD for a particle, this function will be called.
	*/
	int set_particle_data_time(int ID, double t, int count_maxTimeStep=10000)//, bool is_preprocess_PT=true)
	{
		int Load = t_to_loadsnapshot(t, dt_load, t_init, 0.); //Note: each time when searching orbit
		if(PT){
			free(PT);
			PT = nullptr;
		}
		PT = (struct particle_data *) malloc(count_maxTimeStep * sizeof(struct particle_data));
	    
		char wt_fname[MaxCharactersInString];
		sprintf(wt_fname, "%sparticle/particle_%d.txt", path_gm.data(), ID);
		FILE *fp = fopen(wt_fname, "r");
		if(fp==nullptr){
			printf("Cannot open file \"%s\". These files might not be write before. Exit.\n", wt_fname);
			exit(0);
			return -1;
		}
		int i = 0;
		while(!feof(fp) && i<count_maxTimeStep){
            fscanf(fp, "%e %e %e %e %e %e  %d %d %e %e  %e %e %e %e  %e %e %e %e  %e %e %e \n", 
				&(PT[i].Pos[0]), &(PT[i].Pos[1]), &(PT[i].Pos[2]), 
				&(PT[i].Vel[0]), &(PT[i].Vel[1]), &(PT[i].Vel[2]), 
				&(PT[i].Id), &(PT[i].Type), &(PT[i].Mass), &(PT[i].Time), 
				&(PT[i].U), &(PT[i].Rho), &(PT[i].Hsml), &(PT[i].Ne), 
				&(PT[i].Pot),    &(PT[i].Acc[0]), &(PT[i].Acc[1]), &(PT[i].Acc[2]), 
				&(PT[i].dAdt), &(PT[i].Age), &(PT[i].Metal) 
            );

			VecDoub xv = {PT[i].Pos[0], PT[i].Pos[1], PT[i].Pos[2], 
				PT[i].Vel[0], PT[i].Vel[1], PT[i].Vel[2]};
			VecDoub xv_ = this->SS[Load]->TC.new_coordinate(xv, true, true); //?? do this of not preprocessed
			// VecDoub xv_ = TC.new_coordinate_fine(xv, true, true); //??
			PT[i].Pos[0] = xv_[0], PT[i].Pos[1] = xv_[1], PT[i].Pos[2] = xv_[2], 
			PT[i].Vel[0] = xv_[3], PT[i].Vel[1] = xv_[4], PT[i].Vel[2] = xv_[5];
			//The other info will be not right
// DEBUG_PRINT_V0d(1, Load, "Load");
// DEBUG_PRINT_V1d(1, this->SS[Load]->TC.C, "TC.C");
// DEBUG_PRINT_V1d(1, xv, "xv");
// DEBUG_PRINT_V1d(0, xv_, "xv_");
			
			++i;
		}
		fclose(fp);
		printf("Read file \"%s\" ... done.\n", wt_fname);
		return 0;
	}

	VecDoub orbitIntegrating(const VecDoub& xv, double t, int ID, double Dt, 
		int step, bool is_forward, int alg=0)
	{
		//estimate epicycle-R period to timestep //??
		auto xv_ = xv;
		int isfw = is_forward? 1:-1;

		if(alg==0){ //simply add; fixed timestep
			double dt = Dt/step;
			double dt_sameUnit = dt*UnitConvert_time;
			for(int i=0;i<step;i++){
				// printf("each_forces ");
				VecDoub acc = this->forces_t(xv_, t, ID);
				for(int j=0;j<3;j++){ //simplest, not sympletic
					// xv_[j] += (xv_[3+j] + 0.5*dt_sameUnit*isfw*acc[j])*dt_sameUnit;
					// xv_[3+j] += dt_sameUnit*isfw*acc[j];
					xv_[j] += (xv_[3+j] )*dt_sameUnit*isfw;
					xv_[3+j] += dt_sameUnit*acc[j]*isfw;
				}
			}
		}
		// printf("TOI ");
		return xv_;
	}
	//=================================================



	//=================================================
	//====pesudo period orbit data //into a class??
	//// orabits and direct integration action //to set size: when initialize or resize() each
	vector<vector<vector<VecDoub>>> orbitApproxPeriod_xv;
	vector<vector<vector<VecDoub>>> orbitApproxPeriod_tp;
	vector<vector<vector<double>>> orbitApproxPeriod_time;
	vector<vector<MatrixXd>> Coefs_spline3;
	vector<vector<vector<double>>> tau_limits;
	vector<vector<vector<double>>> time_limits;
	vector<vector<int>> countSamples;
	int ID_orb = -255;
	double t0_orb = -255.;
	int badOrbitType = -255; 	//0: well orbit; 1 and 2: unbound orbit; 
								//3: irregular orbit (too fast to captured, oscilates at v=0, or strange shape).
	double dt_reintegrate;
	VecDoub dt_reintegrate_smallPeriod; //only record for small period

	bool is_doubleleaf = false;
	void set_leaf(bool is_doubleleaf1){
		is_doubleleaf = is_doubleleaf1;
	}

	std::unique_ptr<ConfocalEllipsoidalCoordSys> CS; /* from TACT code */
	void set_ConfocalEllipsoidalCoordSys_focus(double a, double b){
		CS->newalpha(a);
		CS->newbeta(b);
	}

	void initialize_orbdata(int ID, double t0)
	{
		//Cartesian xv
		this->orbitApproxPeriod_xv.resize(3); //switch: lambda, mu, nu
		for(int swit=0;swit<3;swit++){
			this->orbitApproxPeriod_xv[swit].resize(2); //leaf: orb up, orb down
		}

		//ellip tp
		this->orbitApproxPeriod_tp.resize(3);
		for(int swit=0;swit<3;swit++){
			this->orbitApproxPeriod_tp[swit].resize(2);
		}

		//period time sequence
		this->orbitApproxPeriod_time.resize(3);
		for(int swit=0;swit<3;swit++){
			this->orbitApproxPeriod_time[swit].resize(2);
		}

		//interpolation coefs
		this->Coefs_spline3.resize(3);
		for(int swit=0;swit<3;swit++){
			this->Coefs_spline3[swit].resize(2);
		}

		//limits //vectors that donot use push_back()
		this->tau_limits.resize(3);
		for(int swit=0;swit<3;swit++){
			this->tau_limits[swit].resize(2);
			this->tau_limits[swit][0].resize(2);
			this->tau_limits[swit][1].resize(2);
		}

		this->time_limits.resize(3);
		for(int swit=0;swit<3;swit++){
			this->time_limits[swit].resize(2);
			this->time_limits[swit][0].resize(2);
			this->time_limits[swit][1].resize(2);
		}

		this->countSamples.resize(3);
		for(int swit=0;swit<3;swit++){
			this->countSamples[swit].resize(2);
		}

		//some other variables
		this->ID_orb = ID;
		this->t0_orb = t0;
		this->dt_reintegrate = 0.;
		this->dt_reintegrate_smallPeriod.resize(3, 0.);

		this->badOrbitType = 0; //the default type is 0 (good)
	}

	void reset_orbitdata(){
		if(PT!=nullptr){
			printf("reset_orbitdata(): now free PT\n");
			free(PT);
			PT = nullptr;
		}
		vector<vector<vector<VecDoub>>>().swap(this->orbitApproxPeriod_xv);
		vector<vector<vector<VecDoub>>>().swap(this->orbitApproxPeriod_tp);
		vector<vector<vector<double>>>().swap(this->orbitApproxPeriod_time);
		vector<vector<MatrixXd>>().swap(this->Coefs_spline3);

		vector<vector<vector<double>>>().swap(this->tau_limits);
		vector<vector<vector<double>>>().swap(this->time_limits);
		vector<vector<int>>().swap(this->countSamples);

		this->dt_reintegrate = 0.;
		VecDoub().swap(this->dt_reintegrate_smallPeriod);

		// particle_ID = -255; //it is not reset here
		ID_orb = -255; //not set
		t0_orb = -255.; //not set
		badOrbitType = -255; //not set
		is_doubleleaf = false;

		CS->newalpha(TACT_semiaxis_Alpha);
		CS->newbeta(TACT_semiaxis_Beta);
	}

	VecDoub func_ptau_t(const VecDoub& xv, int is_fixOtherCoordinate=0){
		//set semi-axis lengths in CS before
		return this->CS->xv2tp(xv);
	}
	VecDoub func_ptau_t(const int& sp, int is_fixOtherCoordinate=0){
		//set semi-axis lengths in CS before
		VecDoub xv = {PT[sp].Pos[0], PT[sp].Pos[1], PT[sp].Pos[2], 
					PT[sp].Vel[0], PT[sp].Vel[1], PT[sp].Vel[2]};
		return this->CS->xv2tp(xv);
	}

	int estimate_ptau_t_root(double t0){
		////preparation
		int s0 = t_to_snapshot(t0, dt_step, 0., 0.);

		////the target time is out of range
		if( !(Initial_snapshot<=s0 && s0<=Final_snapshot) ){ //no <=
			printf("Wrong range of snapshots data, please check! "
				"Now return bad actions result. \n");
			badOrbitType = -3; //wrong settings
			// exit(0);
			return badOrbitType;
		}

		////unbound orbit
		VecDoub xv0 = {PT[s0].Pos[0], PT[s0].Pos[1], PT[s0].Pos[2], 
					PT[s0].Vel[0], PT[s0].Vel[1], PT[s0].Vel[2]};
		double E0 = 0.5*(xv0[3]*xv0[3]+xv0[4]*xv0[4]+xv0[5]*xv0[5]) 
			+ this->potential_t(xv0, t0, PT[s0].Id);
		// DEBUG_PRINT_V1d(1, xv0, "PT");
		if(E0>=0){
			printf("The temperory engergy is higher than zero; seen as unbound orbit.\n");
			this->badOrbitType = 3;
			return badOrbitType;
		}

		auto tp0 = func_ptau_t(xv0, 0);
		for(int i=3;i<6;i++){ //avoid zero to judge sign changing
			if(abs(tp0[i])<Err){tp0[i] = Err*Sign_(tp0[i]);}
		}
		auto xv = xv0;
		auto tp = tp0;



		////judge whether too small period
DEBUG_PRINT_I(51);
		vector<vector<bool>> is_endlimits; //only record for small period
		is_endlimits.resize(3); //coordinates
		for(int i=0;i<3;i++){
			is_endlimits[i].resize(2, false);
		} //each down and up

		int updown = 0;
		int count_points = 1, s = s0;
		double t = t0, t_min = t0, t_max = t0;
		xv = xv0, tp = tp0;
		dt_reintegrate = dt_step;
		do{
			s--;
			t -= dt_reintegrate; //??dl_ds
			if(s>=Initial_snapshot){
				xv = {PT[s].Pos[0], PT[s].Pos[1], PT[s].Pos[2], 
					PT[s].Vel[0], PT[s].Vel[1], PT[s].Vel[2]};
			}else{
				xv = this->orbitIntegrating(xv, t, PT[s].Id, dt_reintegrate, 1, false, 0);
			}
			tp = func_ptau_t(xv, 0);
			for(int swit=0;swit<3;swit++){
				if(is_endlimits[swit][updown]==false && tp[swit]*tp0[swit]<=0){
					is_endlimits[swit][updown] = true;
					time_limits[swit][0][updown] = t;
				}
			}
			count_points++;
		}while(!is_ChangedSign_vecAny(tp, tp0) 
			&& abs(s-s0)+1<CountSmallTimeStep && count_points<CountSmallTimeStep);
		t_min = t;

		updown = 1;
		s = s0, t = t0;
		xv = xv0, tp = tp0;
		do{
			s++;
			t += dt_reintegrate;
			if(s<=Final_snapshot){
				xv = {PT[s].Pos[0], PT[s].Pos[1], PT[s].Pos[2], 
					PT[s].Vel[0], PT[s].Vel[1], PT[s].Vel[2]};
			}else{
				xv = this->orbitIntegrating(xv, t, PT[s].Id, dt_reintegrate, 1, true, 0);
			}
			tp = func_ptau_t(xv, 0);
			for(int swit=0;swit<3;swit++){
				if(is_endlimits[swit][updown]==false && tp[swit]*tp0[swit]<=0){
					is_endlimits[swit][updown] = true;
					time_limits[swit][0][updown] = t;
				}
			}
			count_points++;
		}while(!is_ChangedSign_vecAny(tp, tp0) 
			&& abs(s-s0)+1<CountSmallTimeStep && count_points<CountSmallTimeStep);
		t_max = t;
		//:: One donot use this because ptau_+ should be calculate troublesomely, 
		//\\ the time step is only refered by the period of the current leaf.
		// if(is_doubleleaf){
		// 	;
		// }
		// DEBUG_PRINT_V0d(10, count_points, "count_points small");
// DEBUG_PRINT_I(52);

		if(count_points<=CountSmallTimeStep){
			// dt_reintegrate = (t_max-t_min)/CountLargeTimeStep;
			for(int swit=0;swit<3;swit++){
				if(is_endlimits[swit][0]==true && is_endlimits[swit][1]==true){
					dt_reintegrate_smallPeriod[swit] = (time_limits[swit][0][1] 
						- time_limits[swit][0][0])/CountLargeTimeStep;
				}
				DEBUG_PRINT_V0d(1, dt_reintegrate_smallPeriod[swit], "dt_reintegrate_smallPeriod[swit]");
			}
			badOrbitType = -1;
			return badOrbitType;
			// if(count_points<=3){
			// 	badOrbitType = -1; //too less sample point
			// 	return badOrbitType;
			// }else if(count_points<=3){
			// 	//() smoothing signal
			// 	badOrbitType = -2; //bad leap orbit signal
			// 	return badOrbitType;
			// }
		}
		


		////judge whether the period is in the range
DEBUG_PRINT_I(53);
		bool is_outOfRange_left = false, is_outOfRange_right = false;
		count_points = 0;
		t_min = t0, t_max = t0;
		s = s0, t = t0;
		xv = xv0, tp = tp0;
		dt_reintegrate = dt_step;
		do{
// DEBUG_PRINT_I(531);
			// s -= count_points; //from inner to outer, not uniform stepbut power-2
			s -= 1;
			// s -= CountSmallTimeStep;		
			// t -= dt_load*count_points;
			t -= dt_reintegrate;
			if(s>=Initial_snapshot){
				xv = {PT[s].Pos[0], PT[s].Pos[1], PT[s].Pos[2], 
					PT[s].Vel[0], PT[s].Vel[1], PT[s].Vel[2]};
			}else{
				is_outOfRange_left = true;
				break;
			}
			tp = func_ptau_t(xv, 0);
			count_points++;
		}while(!is_ChangedSign_vecAny(tp, tp0));
		t_min = t;
		// DEBUG_PRINT_V0d(10, count_points, "count_points middle left");

		s = s0, t = t0;
		xv = xv0, tp = tp0;
		do{
// DEBUG_PRINT_I(532);
			// s += count_points;
			s += 1;
			// s += CountSmallTimeStep;		
			// t += dt_load*count_points;
			t += dt_reintegrate;
			if(s<=Final_snapshot){
				xv = {PT[s].Pos[0], PT[s].Pos[1], PT[s].Pos[2], 
					PT[s].Vel[0], PT[s].Vel[1], PT[s].Vel[2]};
			}else{
				is_outOfRange_right = true;
				break;
			}
			tp = func_ptau_t(xv, 0);
			count_points++;
			// DEBUG_PRINT_V1d(1, tp, "tp");
			// DEBUG_PRINT_V1d(1, tp0, "tp0");
		}while(!is_ChangedSign_vecAny(tp, tp0));
		t_max = t;

		printf("Is out of range, left: %d, right: %d; points count in the range: %d.\n", 
			(int)is_outOfRange_left, (int)is_outOfRange_right, count_points);
		if( !(is_outOfRange_left || is_outOfRange_right) ){
			dt_reintegrate = dt_step; //??
			badOrbitType = 0; //at least normal period, in the range
			return badOrbitType;
		}



		////then the period is out of range
DEBUG_PRINT_I(56);
		//same timestep for long period
		// is_endlimits.resize(3); //coordinates
		// for(int i=0;i<3;i++){
		// 	is_endlimits[i].resize(2, false);
		// } //each down and up
		// updown = 0;
		// updown = 1;

		bool is_tooLongTime_left = false, is_tooLongTime_right = false;
		dt_reintegrate = dt_step*CountSmallTimeStep;
		int count_reintegration = 0;
		if(is_outOfRange_left){
			s = Initial_snapshot, t = PT[s].Time; //start
			xv = {PT[s].Pos[0], PT[s].Pos[1], PT[s].Pos[2], 
				PT[s].Vel[0], PT[s].Vel[1], PT[s].Vel[2]};
			do{
DEBUG_PRINT_I(561);
				if(count_reintegration>CountLargeTimeStep*CountSmallTimeStep/2){
					is_tooLongTime_left = true;
					break;
				}
				// s -= 1; //no use
				t -= dt_reintegrate;
				// DEBUG_PRINT_V0d(1, t, "tl");
				// if(s==Initial_snapshot-1){DEBUG_PRINT_V1d(1, xv, "xv");}
				xv = this->orbitIntegrating(xv, t, PT[s0].Id, dt_reintegrate, 1, false, 0);
				// if(s==Initial_snapshot-1){DEBUG_PRINT_V1d(1, xv, "xv");}
				tp = func_ptau_t(xv, 0);
				count_points++;
				count_reintegration++;
			}while(!is_ChangedSign_vecAny(tp, tp0));
			t_min = t;
		}

		count_reintegration = 0;
		if(is_outOfRange_right){
			s = Final_snapshot, t = PT[s].Time; //start
			xv = {PT[s].Pos[0], PT[s].Pos[1], PT[s].Pos[2], 
				PT[s].Vel[0], PT[s].Vel[1], PT[s].Vel[2]};
			do{
DEBUG_PRINT_I(562);
				if(count_reintegration>CountLargeTimeStep*CountSmallTimeStep/2){
					is_tooLongTime_right = true;
					break;
				}
				// s += 1; //no use
				t += dt_reintegrate;
				// DEBUG_PRINT_V0d(1, t, "tr");
				xv = this->orbitIntegrating(xv, t, PT[s0].Id, dt_reintegrate, 1, true, 0);
				tp = func_ptau_t(xv, 0);
				count_points++;
				count_reintegration++;
			}while(!is_ChangedSign_vecAny(tp, tp0));
			t_max = t;
		}

		if(1){ //out of range
			if( !(is_tooLongTime_left || is_tooLongTime_right) ){
				dt_reintegrate = dt_step;
			}else{
				double dt0, dt1, dt2;
				dt0 = PT[Final_snapshot].Time-PT[Initial_snapshot].Time;
				dt1 = selectMin(PT[Initial_snapshot].Time-t_min, 0.);
				dt2 = selectMin(t_max-PT[Final_snapshot].Time, 0.);
				dt_reintegrate = selectMax( dt_step, (dt1+dt2)/CountLargeTimeStep*CountLargeTimeStep );
			}
			DEBUG_PRINT_V0d(1, dt_reintegrate, "dt_reintegrate out of range");
			badOrbitType = 1;
			return badOrbitType;
		}
	}

	int search_orbitPesudoPeriod(const int& ID, const double& t0, 
		double Alpha, double Beta, int count_maxTimeStep=10000)
	{
DEBUG_PRINT_I(1200);
		//initialization
		int s0 = t_to_snapshot(t0, dt_step, 0., 0.);
		double t_sys = t0;
		this->initialize_orbdata(ID, t0);
		this->set_particle_data_time(ID, t0, count_maxTimeStep);
DEBUG_PRINT_I(1201);
		this->set_ConfocalEllipsoidalCoordSys_focus(Alpha, Beta); //it has been changed here
		this->estimate_ptau_t_root(t0);
		if(badOrbitType==-3 || badOrbitType==3){
			return badOrbitType;
		}
DEBUG_PRINT_V0d(1, Alpha, "Alpha_select");
		//debug period, wrong period, other period
		//simpletic integrator and time inverse integrator
		// //fit many surfaces
		// //fit two ksi, ratio but not ratate, refs
		// //fit many profile character indexes
		// // //tell and write
		// // //see refs and diffusion/oscallision/macrosizeparticle/multiscalesubsystem
		// // //tasks0
		// // //tasks1
		// // //supper



		//try to search orbit data
		printf("Try to search orbit data ...\n");
		VecDoub xv0 = {PT[s0].Pos[0], PT[s0].Pos[1], PT[s0].Pos[2], 
			PT[s0].Vel[0], PT[s0].Vel[1], PT[s0].Vel[2]};
// DEBUG_PRINT_I(1202);
		VecDoub xv1 = xv0;
		auto td0 = this->CS->xv2tau(xv0);
		auto tp0 = this->CS->tau2tp(td0);
DEBUG_PRINT_I(1203);
		for(int i=3;i<6;i++){ //avoid zero to judge sign changing
			if(abs(tp0[i])<Err){tp0[i] = Err*Sign_(tp0[i]);}
		}
		auto td1 = td0;
		auto tp1 = tp0;

		int ss = s0;
		int count_reintegration = 0;
		int change_direction = 0;
		for(int swit=0;swit<3;swit++){
			////forward
			bool is_forward = false;
			do{
				ss--;
				if(count_reintegration>CountLargeTimeStep*CountLargeTimeStep){
					printf("Ellipcoor_%d: the arroximate period of the orbit is still too long (%d); "
						"it is seen as unbound orbit.\n", swit, abs(ss-s0));
					this->badOrbitType = 2;
					break;
					// this boundary or fudge
				}
				if(ss>=Initial_snapshot){
					xv1 = {PT[ss].Pos[0], PT[ss].Pos[1], PT[ss].Pos[2], 
						PT[ss].Vel[0], PT[ss].Vel[1], PT[ss].Vel[2]};
					t_sys -= dt_step;
				}else{ //if t out of range, use the snapshot at range
					double Dt;
					if(this->badOrbitType==-1){
						Dt = dt_reintegrate_smallPeriod[swit];
					}else{
						Dt = dt_reintegrate; //adaptive step around this timestep is needed??
					}
					xv1 = this->orbitIntegrating(xv1, t_sys, ID, Dt, 1, is_forward, 0);
					// DEBUG_PRINT_V1d(1, xv1, "xv1");
					t_sys -= Dt;
					count_reintegration++;
				}
				// DEBUG_PRINT_V0d(1, ss, "ss");
				// DEBUG_PRINT_V0d(1, s0, "s0");
				// DEBUG_PRINT_V0d(1, Initial_snapshot, "Initial_snapshot");
				// DEBUG_PRINT_V0d(1, t_sys, "t_sys");
				// DEBUG_PRINT_V0d(1, dt_reintegrate, "dt_reintegrate");
				// DEBUG_PRINT_V0d(10, count_reintegration, "count_reintegration");

				td1 = this->CS->xv2tau(xv1);
				tp1 = this->CS->tau2tp(td1);
				int leaf = change_direction%2; //0
				this->orbitApproxPeriod_xv[swit][leaf].insert(this->orbitApproxPeriod_xv[swit][leaf].begin(), xv1);
				this->orbitApproxPeriod_tp[swit][leaf].insert(this->orbitApproxPeriod_tp[swit][leaf].begin(), tp1);
				this->orbitApproxPeriod_time[swit][leaf].insert(this->orbitApproxPeriod_time[swit][leaf].begin(), t_sys);

// DEBUG_PRINT_I(121);
				if(change_direction==0){ //0: same sign
					if(tp1[swit+3]*tp0[swit+3]<=0){
						change_direction++;
						time_limits[swit][0][0]=t_sys; //leaf0 downlimit
					}
				}
				// DEBUG_PRINT_V0d(1, ss, "ss");
				// DEBUG_PRINT_V0d(1, t_sys, "t_sys");
				// DEBUG_PRINT_V0d(1, is_forward, "is_forward");
				// DEBUG_PRINT_V0d(1, tp1[swit+3], "tp1[swit+3]");
				// DEBUG_PRINT_V0d(1, tp0[swit+3], "tp0[swit+3]");
				// DEBUG_PRINT_V0d(10, change_direction, "change_direction");
			}while( !(change_direction==1) );
			// //or
			// printf("In swit %d, size of leaf0 before s0: %d.\n", swit, this->orbitApproxPeriod_tp[swit][0].size());
			// if(this->badOrbitType==2){
			// 	// Dt *= ratio_expand;
			// 	// if(this->is_repeat){
			// 	// 	this->reset_orbitdata();
			// 	// 	this->search_orbitPesudoPeriod(ID, t0, Alpha, Beta, count_maxTimeStep);
			// 	// }
			// 	// this->is_repeat = false;
			// 	continue;
			// }

			////backward
			is_forward = true;
			t_sys = t0;
			ss = s0;
			count_reintegration = 0;
			change_direction = 0;
			// DEBUG_PRINT_V0d(1, swit, "swit");
			// DEBUG_PRINT_V0d(10, is_forward, "is_forward\n\n\n\n\n");
			do{
// DEBUG_PRINT_I(1221);
				ss++;
				if(count_reintegration>CountLargeTimeStep*CountLargeTimeStep){
					printf("Ellipcoor_%d: the arroximate period of the orbit is still too long (%d); "
						"it is seen as unbound orbit.\n", swit, abs(ss-s0));
					this->badOrbitType = 2;
					if(change_direction==0){ //when in leaf0
						for(int k=0;k<CountSmallTimeStep;k++){ //make the leaf1 with zeros and then break
							VecDoub Z(6, 0.);
							this->orbitApproxPeriod_xv[swit][1].push_back(Z);
							this->orbitApproxPeriod_tp[swit][1].push_back(Z);
							this->orbitApproxPeriod_time[swit][1].push_back(t_sys);
						}
					}
					break;
				}
				if(ss<=Final_snapshot){
// DEBUG_PRINT_I(12213);
					xv1 = {PT[ss].Pos[0], PT[ss].Pos[1], PT[ss].Pos[2], 
						PT[ss].Vel[0], PT[ss].Vel[1], PT[ss].Vel[2]};
					t_sys += dt_step;
					// //debug_print
					// for(int k=0;k<6;k++){printf("(ss xv1: %d, %le) ", k, xv1[k]);}
				}else{
					double Dt;
					if(this->badOrbitType==-1){
						Dt = dt_reintegrate_smallPeriod[swit];
					}else{
						Dt = dt_reintegrate;
					}
					xv1 = this->orbitIntegrating(xv1, t_sys, ID, Dt, 1, is_forward, 0);
					t_sys += Dt;
					count_reintegration++;
				}
				// DEBUG_PRINT_V0d(1, s0, "s0");
				// DEBUG_PRINT_V0d(1, Final_snapshot, "Initial_snapshot");
				// DEBUG_PRINT_V0d(1, dt_reintegrate, "dt_reintegrate");
				// DEBUG_PRINT_V0d(10, count_reintegration, "count_reintegration");

				td1 = this->CS->xv2tau(xv1);
				tp1 = this->CS->tau2tp(td1);
				int leaf = change_direction%2; //0
				this->orbitApproxPeriod_xv[swit][leaf].push_back(xv1);
				this->orbitApproxPeriod_tp[swit][leaf].push_back(tp1);
				this->orbitApproxPeriod_time[swit][leaf].push_back(t_sys);
				//:: there is no copy in leaf1
				//:: copy boundary for orbit data //??

				if(change_direction==0){ //0: same sign
					if(tp1[swit+3]*tp0[swit+3]<=0){ //a new changing of sign
						change_direction++;
						time_limits[swit][0][1] = t_sys; //leaf0 uplimit
						time_limits[swit][1][0] = t_sys; //leaf1 downlimit
						// //debug_print
						// for(int k=0;k<6;k++){printf("(xv1: %d, %le) ", k, xv1[k]);}
						// for(int k=0;k<6;k++){printf("(tp1: %d, %le) ", k, tp1[k]);}
					}
				}else if(change_direction==1){ //1: different sign
					if(tp1[swit+3]*tp0[swit+3]>0){ //a new changing of sign
						change_direction++;
						time_limits[swit][1][1] = t_sys; //leaf1 uplimit
						// //debug_print
						// for(int k=0;k<6;k++){printf("(xv1: %d, %le) ", k, xv1[k]);}
						// for(int k=0;k<6;k++){printf("(tp1: %d, %le) ", k, tp1[k]);}
					}
				}
				// DEBUG_PRINT_V0d(1, ss, "ss");
				// DEBUG_PRINT_V0d(1, t_sys, "t_sys");
				// DEBUG_PRINT_V0d(1, is_forward, "is_forward");
				// DEBUG_PRINT_V0d(1, tp1[swit+3], "tp1[swit+3]");
				// DEBUG_PRINT_V0d(1, tp0[swit+3], "tp0[swit+3]");
				// DEBUG_PRINT_V0d(10, change_direction, "change_direction");
			}while( !( change_direction==2) );
			t_sys = t0;
			ss = s0;
			count_reintegration = 0;
			change_direction = 0;
			// if(this->badOrbitType==2){
			// 	// Dt *= ratio_expand;
			// 	// //reintegrate the orbit
			// 	// //run this func
			// 	continue;
			// }

// DEBUG_PRINT_I(123);
			////still out of range or too less points
			//() out of range
			int size_leaf0 = orbitApproxPeriod_tp[swit][0].size(), size_leaf1 = orbitApproxPeriod_tp[swit][1].size();
			for(int leaf=0;leaf<2;leaf++){
				if( orbitApproxPeriod_tp[swit][leaf].size()<4 ){
					printf("Ellipcoor_%d, leaf_%d: the sample points in a approxxmate period is still too less "
						"size_%d.\n", swit, leaf, orbitApproxPeriod_tp[swit][leaf].size());
					this->badOrbitType = -2;					
					//:: reset orbit data in this leaf as zeros
					VecDoub Z(6, 0.);
					for(int k=0;k<CountSmallTimeStep/2;k++){
						this->orbitApproxPeriod_xv[swit][leaf].push_back(Z);
						this->orbitApproxPeriod_tp[swit][leaf].push_back(Z);
						this->orbitApproxPeriod_time[swit][leaf].push_back(0.);
					}
					for(int k=0;k<CountSmallTimeStep/2;k++){
						this->orbitApproxPeriod_xv[swit][leaf].insert(this->orbitApproxPeriod_xv[swit][leaf].begin(), Z);
						this->orbitApproxPeriod_tp[swit][leaf].insert(this->orbitApproxPeriod_tp[swit][leaf].begin(), Z);
						this->orbitApproxPeriod_time[swit][leaf].insert(this->orbitApproxPeriod_time[swit][leaf].begin(), 0.);
					}
				}
			}
// DEBUG_PRINT_I(124);



			// ////[1]Boundary conditions, switch interp:
			// //add approximate roots of +- with 3-points (invert) parabola function f, 
			// //y = f(x), where y~tau, x~ptau, three points are {(x1,y1), (x2,y2), (x3,y3)}, 
			// //such that the intersection(as root) of the poly2 function and y-axis are (0, y(x=0)).
			// //Changed head node of up leaf and tail node of down leaf as (ptau=0, tau=root1), 
			// //and add a joint node for both leafs as (ptau=0, tau=root21) and (ptau=0, tau=root22).
			// //After this and sorting, we interpolation the tau~ptau realtion by spline3 at these range.
			// //However, there are some problems that the root by spline3 and parabola are not same.
			// double xr=0.,yr, x1,y1, x2,y2, x3,y3;
			// double other_coordinate_value = 0.; //not calculated, we set it as 0..
			// VecDoub rootpoint(6), rootpoint_xv(6, other_coordinate_value), points(6);
			// //add joint node of up leaf (from end) 
			// //and joint node of down leaf (from begin):
			// x1 = (orbitApproxPeriod_tp[swit][0].at(size_leaf0-1))[swit+3], 
			// y1 = (orbitApproxPeriod_tp[swit][0].at(size_leaf0-1))[swit]; //joint of leaf0: invert order
			// x2 = orbitApproxPeriod_tp[swit][1][0][swit+3], 
			// y2 = orbitApproxPeriod_tp[swit][1][0][swit]; //joint of leaf0: default order
			// if((orbitApproxPeriod_tp[swit][0].at(size_leaf0-2))[swit] 
			//	< orbitApproxPeriod_tp[swit][1][1][swit]){ //select the more near one from the two second nodes
			// 	x3 = (orbitApproxPeriod_tp[swit][0].at(size_leaf0-2))[swit+3], 
			//	y3 = (orbitApproxPeriod_tp[swit][0].at(size_leaf0-2))[swit];
			// }else{
			// 	x3 = orbitApproxPeriod_tp[swit][1][1][swit+3], 
			//	y3 = orbitApproxPeriod_tp[swit][1][1][swit];
			// }
			// points[0] = x1, points[1] = y1, points[2] = x2, points[3] = y2, points[4] = x3, points[5] = y3;
			// yr = parabola_2d_3points_y_x(xr, points);
			// DEBUG_PRINT_V0d(1, yr, "joint node");
			// standardize_ellipcoor_range_with_axislength(yr, swit, this->CS->alpha(), this->CS->beta(), this->CS->gamma());
			// rootpoint = {yr,other_coordinate_value,other_coordinate_value, xr,other_coordinate_value,other_coordinate_value};
			// orbitApproxPeriod_tp[swit][0].push_back(rootpoint); //boundary
			// orbitApproxPeriod_xv[swit][0].push_back(rootpoint_xv);
			// orbitApproxPeriod_tp[swit][1].insert(orbitApproxPeriod_tp[swit][1].begin(), rootpoint); //boundary
			// orbitApproxPeriod_xv[swit][1].insert(orbitApproxPeriod_xv[swit][1].begin(), rootpoint_xv);
			// size_leaf0 = orbitApproxPeriod_tp[swit][0].size(), size_leaf1 = orbitApproxPeriod_tp[swit][1].size();

			// //change tail node of down leaf (from end):
			// x1 = (orbitApproxPeriod_tp[swit][1].at(size_leaf1-1))[swit+3], y1 = (orbitApproxPeriod_tp[swit][1].at(size_leaf1-1))[swit];
			// x2 = (orbitApproxPeriod_tp[swit][1].at(size_leaf1-2))[swit+3], y2 = (orbitApproxPeriod_tp[swit][1].at(size_leaf1-2))[swit];
			// x3 = (orbitApproxPeriod_tp[swit][1].at(size_leaf1-3))[swit+3], y3 = (orbitApproxPeriod_tp[swit][1].at(size_leaf1-3))[swit];
			// points[0] = x1, points[1] = y1, points[2] = x2, points[3] = y2, points[4] = x3, points[5] = y3;
			// yr = parabola_2d_3points_y_x(xr, points);
			// // DEBUG_PRINT_V0d(1, yr, "tail node");
			// standardize_ellipcoor_range_with_axislength(yr, swit, this->CS->alpha(), this->CS->beta(), this->CS->gamma());
			// rootpoint = {yr,other_coordinate_value,other_coordinate_value, xr,other_coordinate_value,other_coordinate_value};
			// // orbitApproxPeriod_tp[swit][1][size_leaf1-1] = (rootpoint); //boundary
			// // orbitApproxPeriod_xv[swit][1][size_leaf1-1] = (rootpoint_xv);
			// orbitApproxPeriod_tp[swit][1].push_back(rootpoint); //boundary
			// orbitApproxPeriod_xv[swit][1].push_back(rootpoint_xv);
			// //change head node of up leaf (from begin):
			// x1 = (orbitApproxPeriod_tp[swit][0][0])[swit+3], y1 = (orbitApproxPeriod_tp[swit][0][0])[swit];
			// x2 = (orbitApproxPeriod_tp[swit][0][1])[swit+3], y2 = (orbitApproxPeriod_tp[swit][0][1])[swit];
			// x3 = (orbitApproxPeriod_tp[swit][0][2])[swit+3], y3 = (orbitApproxPeriod_tp[swit][0][2])[swit];
			// points[0] = x1, points[1] = y1, points[2] = x2, points[3] = y2, points[4] = x3, points[5] = y3;
			// yr = parabola_2d_3points_y_x(xr, points);
			// DEBUG_PRINT_V0d(1, yr, "joint node");
			// standardize_ellipcoor_range_with_axislength(yr, swit, this->CS->alpha(), this->CS->beta(), this->CS->gamma());
			// rootpoint = {yr,other_coordinate_value,other_coordinate_value, xr,other_coordinate_value,other_coordinate_value};
			// // orbitApproxPeriod_tp[swit][0][0] = (rootpoint); //boundary
			// // orbitApproxPeriod_xv[swit][0][0] = (rootpoint_xv);
			// orbitApproxPeriod_tp[swit][0].insert(orbitApproxPeriod_tp[swit][0].begin(), rootpoint); //boundary
			// orbitApproxPeriod_xv[swit][0].insert(orbitApproxPeriod_xv[swit][0].begin(), rootpoint_xv);

			////[1]Boundary conditions, switch just erase
// DEBUG_PRINT_I(125);
			orbitApproxPeriod_xv[swit][0].erase(orbitApproxPeriod_xv[swit][0].begin());
			orbitApproxPeriod_xv[swit][0].erase(orbitApproxPeriod_xv[swit][0].end()-1); //??
			orbitApproxPeriod_xv[swit][1].erase(orbitApproxPeriod_xv[swit][1].end()-1);
			
			orbitApproxPeriod_tp[swit][0].erase(orbitApproxPeriod_tp[swit][0].begin());
			orbitApproxPeriod_tp[swit][0].erase(orbitApproxPeriod_tp[swit][0].end()-1);
			orbitApproxPeriod_tp[swit][1].erase(orbitApproxPeriod_tp[swit][1].end()-1);

// VecDoub tl = orbitApproxPeriod_time[swit][0];
// DEBUG_PRINT_V1d(1, tl, "tl");
			orbitApproxPeriod_time[swit][0].erase(orbitApproxPeriod_time[swit][0].begin());
// tl = orbitApproxPeriod_time[swit][0];
// DEBUG_PRINT_V1d(1, tl, "tl after1");
			orbitApproxPeriod_time[swit][0].erase(orbitApproxPeriod_time[swit][0].end()-1);
// tl = orbitApproxPeriod_time[swit][0];
// DEBUG_PRINT_V1d(1, tl, "tl after2");
			orbitApproxPeriod_time[swit][1].erase(orbitApproxPeriod_time[swit][1].end()-1);

			// vector<int> sizes2 = {
			// 	orbitApproxPeriod_xv[swit][0].size(), 
			// 	orbitApproxPeriod_xv[swit][1].size(), 
			// 	orbitApproxPeriod_tp[swit][0].size(), 
			// 	orbitApproxPeriod_tp[swit][1].size(), 
			// 	orbitApproxPeriod_time[swit][0].size(), 
			// 	orbitApproxPeriod_time[swit][1].size()
			// };
			// DEBUG_PRINT_V1d(1, sizes2, "sizes2");

			// size_leaf0 = orbitApproxPeriod_tp[swit][0].size();
			// size_leaf1 = orbitApproxPeriod_tp[swit][1].size();
			// DEBUG_PRINT_V0d(1, swit, "swit");
			// DEBUG_PRINT_V0d(1, size_leaf0, "sizeofleaf0 after doing bc");
			// DEBUG_PRINT_V0d(1, size_leaf1, "sizeofleaf1 after doing bc");
// DEBUG_PRINT_I(126);



			//bubble sort and swap the values from min to max
			for(int leaf=0;leaf<2;leaf++){
				int n = orbitApproxPeriod_tp[swit][leaf].size();
				for(int i=0;i<n-1;i++){
					for(int j=0;j<n-i-1;j++){
						if(orbitApproxPeriod_tp[swit][leaf][j][swit] 
							> orbitApproxPeriod_tp[swit][leaf][j+1][swit]){ //to be careful: compare ...[swit]
							orbitApproxPeriod_xv[swit][leaf][j].swap(orbitApproxPeriod_xv[swit][leaf][j+1]);
							orbitApproxPeriod_tp[swit][leaf][j].swap(orbitApproxPeriod_tp[swit][leaf][j+1]);
							swap(orbitApproxPeriod_time[swit][leaf][j], orbitApproxPeriod_time[swit][leaf][j+1]); //to be careful: no double.swap()
						}
					}
				}
			}
		}

		//// statistic the limits
		this->get_taulimits_estimatedfrom_orbitApproxPeriod();
		this->get_timelimits_estimatedfrom_orbitApproxPeriod();
		this->get_countSamples_orbitApproxPeriod();
		printf("End to search orbit data.\n");
		return 0;
	}

	int get_badOrbitType(){
		return this->badOrbitType;
	}

	int calculate_Coefs_spline3(){
		//for this->ptau_interp()
		if(this->badOrbitType!=0){
			return 1;
		}
		for(int swit=0;swit<3;swit++){
			for(int leaf=0;leaf<2;leaf++){
				int N_nodes = this->orbitApproxPeriod_tp[swit][leaf].size(); //count of nodes
				int n = N_nodes-1; //count of intervals and count of each coefs
				for(int j=0;j<N_nodes;j++){
					// DEBUG_PRINT_V0d(1, orbitApproxPeriod_tp[swit][leaf][j][swit], swit);
					// DEBUG_PRINT_V0d(2, orbitApproxPeriod_tp[swit][leaf][j][swit+3], swit+3);
					// orbitApproxPeriod_tp[swit][leaf][j][swit] = j; //j+10.;
					// // orbitApproxPeriod_tp[swit][leaf][j][swit+3] = 0.1*j*j;
					// // orbitApproxPeriod_tp[swit][leaf][j][swit+3] = sin(0.1*j*pi_8);
					// orbitApproxPeriod_tp[swit][leaf][j][swit+3] = sin(0.1*j*pi_8)+0.01*j*j*j;
				}

				VectorXd X(n+1); X = VectorXd::Zero(n+1); //x
				VectorXd Y(n+1); Y = VectorXd::Zero(n+1); //y
				VectorXd H(n); H = VectorXd::Zero(n); //dx
				VectorXd G(n); G = VectorXd::Zero(n); //dy
				MatrixXd HH(n+1,n+1); HH = MatrixXd::Zero(n+1,n+1); //matrix, width
				VectorXd M(n+1); M = VectorXd::Zero(n+1); //to solve, second serive at nods
				VectorXd V(n+1); V = VectorXd::Zero(n+1); //right hand side
				for(int i=0;i<n+1;i++){
					X(i) = orbitApproxPeriod_tp[swit][leaf][i][swit];
					Y(i) = orbitApproxPeriod_tp[swit][leaf][i][swit+3];
				}
				for(int i=0;i<n;i++){
					H(i) = X(i+1)-X(i);
					G(i) = Y(i+1)-Y(i);
					if(abs(H(i))<Err){H(i)=Err;} //notenote: to avoid same-x points and bad orfer points
				}
				for(int i=1;i<=n-1;i++){
					HH(i,i-1) = H(i-1), HH(i,i) = (H(i-1)+H(i))*2, HH(i,i+1) = H(i-1); 
					//notenote: ()*2 instead of ()/2, be careful
					V(i) = G(i)/H(i)-G(i-1)/H(i-1);
				}
				//::boundary condition 1: ''=0;
				HH(0,0) = 1., HH(n,n) = 1.;
				//::boundary condition 2: '=C;
				// HH(0,0) = 2*H(0), HH(0,1) = H(0), HH(n-1,n) = H(n-1), HH(n,n) = 2*H(n-1);
				//::boundary condition 3: '''=0;
				// HH(0,0) = -H(1), HH(0,1) = H(0)+H(1), HH(0,2) = -H(0);
				// HH(n,n-2) = -H(n-1), HH(n,n-1) = H(n-2)+H(n-1), HH(n,n) = -H(n-2);
				M = HH.lu().solve(V*6); //notenote: lu() is better than ldlt(), the later may not have M(0)==0
				// M = HH.llt().solve(V*6);
				// M = HH.ldlt().solve(V*6);
				// M = HH.colPivHouseholderQr().solve(V*6);
				// M = HH.svd().solve(V*6);

				//dbg:
				// // std::cout<<HH<<"\n";
				// std::cout<<N_nodes<<"\n";
				// std::cout<<"X: "<<X<<"\n";
				// std::cout<<"Y: "<<Y<<"\n";
				// std::cout<<"H: "<<H<<"\n";
				// // std::cout<<"G: "<<G<<"\n";
				// std::cout<<"M: "<<M<<"\n";
				// // std::cout<<"V: "<<V<<"\n";
				// // exit(0); //to display

				// MatrixXd Coef(n, 4);
				MatrixXd Coef(4, n);
				VectorXd A(n); A = VectorXd::Zero(n);
				VectorXd B(n); B = VectorXd::Zero(n);
				VectorXd C(n); C = VectorXd::Zero(n);
				VectorXd D(n); D = VectorXd::Zero(n);
				for(int i=0;i<n;i++){
					A(i) = Y(i);
					B(i) = (Y(i+1)-Y(i))/H(i) -M(i)*H(i)/2 -(M(i+1)-M(i))*H(i)/6;
					C(i) = M(i)/2;
					D(i) = (M(i+1)-M(i))/H(i)/6;
					Coef(0,i) = A(i), Coef(1,i) = B(i), Coef(2,i) = C(i), Coef(3,i) = D(i);
				}
				this->Coefs_spline3[swit][leaf] = Coef;
			}
		}
		return 0;
	}

	/*	When there are some leapings in the orbit (this->badOrbitType==-2), 
		this function will get the actions and frequencies result by other methods, then 
		reintegrate the action in that period.
	*/
	VecDoub other_period(int ID, int actionmethod){ //??
		//reset orbit data
		//fudge judge
		//re search orbit data
		return {0., 0., 0.};
	}

	/*	To get the counts of orbit samples, totally 6, the order is about 
		{swit0leaf0, swit0leaf1, swit1leaf0, swit1leaf1, swit2leaf0, swit2leaf1}.
	*/
	vector<vector<int>> get_countSamples_orbitApproxPeriod(){
		if(badOrbitType==-255 || badOrbitType==-3 || badOrbitType==3){
			vector<vector<int>> tl0;
			tl0.resize(3);
			for(int swit=0;swit<3;swit++){
				tl0[swit].resize(2, 0);
			}
			printf("Bad orbit type, %d. Now return zeros.\n", badOrbitType);			
			return tl0;
		}else{
			for(int swit=0;swit<3;swit++){
				for(int leaf=0;leaf<2;leaf++){
					this->countSamples[swit][leaf] = this->orbitApproxPeriod_tp[swit][leaf].size();
				}
			}
			return this->countSamples;
		}
	}

	/*	To get the limits of orbit samples, totally 12, the order is about 
		{swit0leaf0limit0, swit0leaf0limit1, swit0leaf1limit0, swit0leaf1limit1, ...}.
	*/
	vector<vector<vector<double>>> get_taulimits_estimatedfrom_orbitApproxPeriod(){ //??
		if(badOrbitType==-255 || badOrbitType==-3 || badOrbitType==3){
			vector<vector<vector<double>>> tl0;
			tl0.resize(3);
			for(int swit=0;swit<3;swit++){
				tl0[swit].resize(2);
				for(int leaf=0;leaf<2;leaf++){
					tl0[swit][leaf].resize(2,0.);
				}
			}
			printf("Bad orbit type, %d. Now return zeros.\n", badOrbitType);			
			return tl0;
		}else{
			for(int swit=0;swit<3;swit++){
				for(int leaf=0;leaf<2;leaf++){
					double taumin = std::numeric_limits<double>::infinity();
					double taumax = -std::numeric_limits<double>::infinity();
					for(auto i:this->orbitApproxPeriod_tp[swit][leaf]){
						//auto amin = *min_element(a.begin(),a.end()); //but not the last dim
						taumin = selectMin(taumin, i[swit]);
						taumax = selectMax(taumax, i[swit]);
					}
					this->tau_limits[swit][leaf][0] = taumin;
					this->tau_limits[swit][leaf][1] = taumax;
				}
			}
			return this->tau_limits;
		}
	}

	vector<vector<vector<double>>> get_timelimits_estimatedfrom_orbitApproxPeriod(){
		if(badOrbitType==-255 || badOrbitType==-3 || badOrbitType==3){
			vector<vector<vector<double>>> tl0;
			tl0.resize(3);
			for(int swit=0;swit<3;swit++){
				tl0[swit].resize(2);
				for(int leaf=0;leaf<2;leaf++){
					tl0[swit][leaf].resize(2,0.);
				}
			}
			printf("Bad orbit type, %d. Now return zeros.\n", badOrbitType);			
			return tl0;
		}else{
			return this->time_limits; //it is record in this->search()
		}
	}

	int write_orbitApproxPeriod(){ //should search data before
		#define FPENNL fprintf(fp, "\n");
		if(this->badOrbitType==-255){ //not calculated
			printf("The orbit has not been searched.\n");
			return -2;
		}
		//:: filename
		int ID = this->ID_orb; //[1]
		double t0 = this->t0_orb; //[1]
	    char wt_fname[MaxCharactersInString];
		sprintf(wt_fname, "%sorbit/orbit_particle_%d_time_%.6le.txt", path_gm.data(), ID, t0);
		FILE *fp = nullptr;
		fp = fopen(wt_fname, "w");
		if(fp==nullptr){
			printf("Cannot open file \"%s\".\n", wt_fname);
			return -1;
		}
		// this->get_taulimits_estimatedfrom_orbitApproxPeriod();
		// this->get_timelimits_estimatedfrom_orbitApproxPeriod();
		// this->get_countSamples_orbitApproxPeriod();

		//:: write contents
		fprintf(fp, "# #00 Write down sample points in multilevel paragraphs #Begin paragraph_level_0\n");
		fprintf(fp, "# %d #particle_ID\n", ID);
		fprintf(fp, "# %.5e #initial_time\n", t0);
		fprintf(fp, "# %d #badOrbitType\n", badOrbitType);
		fprintf(fp, "# %d #is_doubleleaf in action integration\n", (int)is_doubleleaf);
		auto af = integrate_actions(is_doubleleaf); //[6]
		fprintf(fp, "# %e %e %e     %e %e %e #actions[3] and frequencies[3]", af[0], af[1], af[2],     af[3], af[4], af[5]);
		FPENNL;
		for(int swit=0;swit<3;swit++){
			fprintf(fp, "# %d #10 spacial coordinate index #Begin paragraph_level_1\n", swit);
			for(int leaf=0;leaf<2;leaf++){
				fprintf(fp, "# %d #20 part index of orbit period #Begin paragraph_level_2\n", leaf);
				int N = orbitApproxPeriod_tp[swit][leaf].size(); //[1]
				fprintf(fp, "# %d %le #count of sample points of this part\n", N);
				vector<double> taul = tau_limits[swit][leaf]; //[2]
				fprintf(fp, "# %e %e #tau_limits of this part\n", taul[0], taul[1]);
				vector<double> timel = time_limits[swit][leaf]; //[2]
				fprintf(fp, "# %e %e #time_range of this part\n", timel[0], timel[1]);
				for(int n=0;n<N;n++){
					auto xv = orbitApproxPeriod_xv[swit][leaf][n]; //[6]
					auto time = orbitApproxPeriod_time[swit][leaf][n]; //[1]
					VecDoub abc2M = {CS->alpha(), CS->beta(), -1}; //[3] //but when change??
					auto tp = orbitApproxPeriod_tp[swit][leaf][n]; //[6]
					auto potential = potential_t(xv, time, ID); //[1]
					double energy = potential + kineticEnergy(xv); //[1]
					VecDoub angularMoment = angularMomentCartesian(xv); //[3]
					fprintf(fp, "%e %e %e %e %e %e    %e     %e %e %e     %e %e %e %e %e %e     %e %e %e %e %e \n", 
						xv[0], xv[1], xv[2], xv[3], xv[4], xv[5],     time, 
						abc2M[0], abc2M[1], abc2M[2], tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], 
						potential, energy, angularMoment[0], angularMoment[1], angularMoment[2] 
					); //xv t abc tp P E Lxyz
				}
				fprintf(fp, "# #21 End paragraph_level_2\n");
				FPENNL;
			}
			fprintf(fp, "# #11 End paragraph_level_1\n");
			FPENNL;
		}
		fprintf(fp, "# #01 End paragraph_level_0\n");
		FPENNL;

		//:: end
		fclose(fp);
		printf("Write file \"%s\" ... done.\n", wt_fname);
		return 0;
	}

	double ptau_interp(double tau, int swit, int leaf){
		if(badOrbitType==-255 || badOrbitType==-3 || badOrbitType==3){
			printf("Bad orbit type, %d. Now return zeros.\n", badOrbitType);
			return 0.;
		}
		int N_nodes = this->orbitApproxPeriod_tp[swit][leaf].size();
		int n = N_nodes-1;
		int i = 0;
		double ptau = 0.;
		double taumin = this->orbitApproxPeriod_tp[swit][leaf][0][swit], 
		taumax = this->orbitApproxPeriod_tp[swit][leaf][n][swit];
		if(!(taumin<=tau && tau<=taumax )){ //swit and leaf index //sort before
			printf("Out of range of inner interpolation, please check! Now return 0..\n");
			printf("tau_target = %e, tau_min = %e, tau_max = %e\n", tau, taumin, taumax);
			return 0.;
		}
		while( i<n-1 && !(this->orbitApproxPeriod_tp[swit][leaf][i][swit]<=tau 
			&& tau<=this->orbitApproxPeriod_tp[swit][leaf][i+1][swit]) ){
			i++;
		}
		double dx = tau-this->orbitApproxPeriod_tp[swit][leaf][i][swit];
		ptau = Coefs_spline3[swit][leaf](0,i) +Coefs_spline3[swit][leaf](1,i)*dx 
			+Coefs_spline3[swit][leaf](2,i)*dx*dx +Coefs_spline3[swit][leaf](3,i)*dx*dx*dx;
		//dbg
		// DEBUG_PRINT_V0d(1, i, "i");
		// DEBUG_PRINT_V0d(1, N_nodes, "N_nodes");
		// DEBUG_PRINT_V0d(1, dx, "dx");
		// DEBUG_PRINT_V0d(1, Coefs_spline3[swit][leaf](0,i), "Coef()");
		// DEBUG_PRINT_V0d(1, Coefs_spline3[swit][leaf](1,i), "Coef()");
		// DEBUG_PRINT_V0d(1, Coefs_spline3[swit][leaf](2,i), "Coef()");
		// DEBUG_PRINT_V0d(1, Coefs_spline3[swit][leaf](3,i), "Coef()");
		// DEBUG_PRINT_V0d(1, ptau, "ptau");
		return ptau;
	}

	VecDoub integrate_actions(bool is_doubleleaf1=false, int algorithm=0){
		VecDoub A(3, 0.);
		// if(this->badOrbitType==3 || this->badOrbitType==2){
		if(this->badOrbitType==3){
			A.resize(3, std::numeric_limits<double>::infinity());
			return A;
		}else if(this->badOrbitType==-3 || this->badOrbitType==-2){
			A.resize(3, 0.);
			return A;
		}
		switch(algorithm){
			default: { //0: trapezoidal integration
				if(is_doubleleaf1){
					for(int swit=0;swit<3;swit++){
						//:: actions
						for(int leaf=0;leaf<2;leaf++){
							int size = orbitApproxPeriod_tp[swit][leaf].size();
							for(int i=0;i<size-1;i++){ //-1
								A[swit] += ( abs(orbitApproxPeriod_tp[swit][leaf][i][swit+3]+orbitApproxPeriod_tp[swit][leaf][i+1][swit+3])
									* abs(orbitApproxPeriod_tp[swit][leaf][i][swit]-orbitApproxPeriod_tp[swit][leaf][i+1][swit]) /2. );
							}
						}
						A[swit] /= pi_8;
					}
					//no angles now
					//:: frequencies
					for(int swit=0;swit<3;swit++){
						double pp = (time_limits[swit][0][1]-time_limits[swit][0][0]) 
							+ (time_limits[swit][1][1]-time_limits[swit][1][0]);
						pp = 2.*pi_8/(pp*UnitConvert_time);
						A.push_back(pp);
					}
				}else{
					for(int swit=0;swit<3;swit++){
						//:: actions
						for(int leaf=0;leaf<1;leaf++){
							int size = orbitApproxPeriod_tp[swit][leaf].size();
							for(int i=0;i<size-1;i++){
								A[swit] += ( abs(orbitApproxPeriod_tp[swit][leaf][i][swit+3]+orbitApproxPeriod_tp[swit][leaf][i+1][swit+3])
									* abs(orbitApproxPeriod_tp[swit][leaf][i][swit]-orbitApproxPeriod_tp[swit][leaf][i+1][swit]) /2. );
							}
						}
						A[swit] *= 2;
						A[swit] /= pi_8;
					}
					//:: frequencies
					for(int swit=0;swit<3;swit++){
						double pp = (time_limits[swit][0][1]-time_limits[swit][0][0]) * 2;
						pp = 2.*pi_8/(pp*UnitConvert_time);
						A.push_back(pp);
					}
				}
				break;
			}
			case 1: { //1: spline3 integration
				//;
				break;
			}
			case 3: { //others
				//;
				break;
			}
		}

		return A;
	}

	////other funcs
	//
	//
	//
	//=================================================
};



//// some small functions and structs
struct write_angleaction{
    //particle basic info
    double particle_xv0[2*Dim];
    int particle_ID;
	double mass;
    double particle_otherInfo[6]; //particle_Type, particle_xv0_FPot, particle_xv0_DPot, semi-axis when triaxial ellipsoidal coordinate

    //acitions, angles and frequencies under formula potential
    double Value_Actions_SS_FP[LENGTH_ACTIONS_WRITE], Value_AnglesFrequencies_SS_FP[LENGTH_ANGLESFREQUENCIES_WRITE];
	//$\{ J_r,L_z,J_z,0 \}$, $\{ \theta_r,\theta_\theta,\theta_\phi, \Omega_r,Omega_\theta,\Omega_\phi \}$ //??
    double Value_Actions_AF_FP[LENGTH_ACTIONS_WRITE], Value_AnglesFrequencies_AF_FP[LENGTH_ANGLESFREQUENCIES_WRITE];
	//$\{ J_R,L_z,J_z,0 \}$, $\{ \theta_R,\theta_\phi,\theta_z, \Omega_R,Omega_\phi,\Omega_z \}$
    double Value_Actions_TF_FP[LENGTH_ACTIONS_WRITE], Value_AnglesFrequencies_TF_FP[LENGTH_ANGLESFREQUENCIES_WRITE];
	//$\{ J_\lambda,J_\mu,J_\nu, \mbox{orbit type} \}$ ...
    //, $\{ \theta_\lambda,\theta_\mu,\theta_\nu, \Omega_\lambda,Omega_\mu,\Omega_\nu \}$ //do not record \mbox{other 5 judgements}
	double Value_Actions_TEPPOD_FP[LENGTH_ACTIONS_WRITE], Value_AnglesFrequencies_TEPPOD_FP[LENGTH_ANGLESFREQUENCIES_WRITE];
	//$\{ J_\lambda,J_\mu,J_\nu, \mbox{orbit type} \}$ ...
    //, $\{ \theta_\lambda,\theta_\mu,\theta_\nu, \Omega_\lambda,Omega_\mu,\Omega_\nu \}$
	double Value_Actions_GF_FP[LENGTH_ACTIONS_WRITE], Value_AnglesFrequencies_GF_FP[LENGTH_ANGLESFREQUENCIES_WRITE];
	//$\{ J_\lambda,J_\mu,J_\nu, \mbox{orbit type} \}$ ...
    //, $\{ \theta_\lambda,\theta_\mu,\theta_\nu, \Omega_\lambda,Omega_\mu,\Omega_\nu \}$
	
    //acitions, angles and frequencies under data potential
    double Value_Actions_SS_DP[LENGTH_ACTIONS_WRITE], Value_AnglesFrequencies_SS_DP[LENGTH_ANGLESFREQUENCIES_WRITE];
	//$\{ J_r,L_z,J_z,0 \}$, $\{ \theta_r,\theta_\theta,\theta_\phi, \Omega_r,Omega_\theta,\Omega_\phi \}$ //??
    double Value_Actions_AF_DP[LENGTH_ACTIONS_WRITE], Value_AnglesFrequencies_AF_DP[LENGTH_ANGLESFREQUENCIES_WRITE];
	//$\{ J_R,L_z,J_z,0 \}$, $\{ \theta_R,\theta_\phi,\theta_z, \Omega_R,Omega_\phi,\Omega_z \}$
    double Value_Actions_TF_DP[LENGTH_ACTIONS_WRITE], Value_AnglesFrequencies_TF_DP[LENGTH_ANGLESFREQUENCIES_WRITE];
	//$\{ J_\lambda,J_\mu,J_\nu, \mbox{orbit type} \}$ ...
    //, $\{ \theta_\lambda,\theta_\mu,\theta_\nu, \Omega_\lambda,Omega_\mu,\Omega_\nu \}$ //do not record \mbox{other 5 judgements}
	double Value_Actions_TEPPOD_DP[LENGTH_ACTIONS_WRITE], Value_AnglesFrequencies_TEPPOD_DP[LENGTH_ANGLESFREQUENCIES_WRITE];
	//$\{ J_\lambda,J_\mu,J_\nu, \mbox{orbit type} \}$ ...
    //, $\{ \theta_\lambda,\theta_\mu,\theta_\nu, \Omega_\lambda,Omega_\mu,\Omega_\nu \}$
	double Value_Actions_GF_DP[LENGTH_ACTIONS_WRITE], Value_AnglesFrequencies_GF_DP[LENGTH_ANGLESFREQUENCIES_WRITE];
	//$\{ J_\lambda,J_\mu,J_\nu, \mbox{orbit type} \}$ ...
    //, $\{ \theta_\lambda,\theta_\mu,\theta_\nu, \Omega_\lambda,Omega_\mu,\Omega_\nu \}$
};
extern struct write_angleaction* Write_aa, * Write_action_samples;

/* information of a particle after simulation output */
extern struct write_firsthand{
	//:: target
	double QP[2*Dim]; //the target general coordinates Q and momentum P, such as Cartesian: x-v， angele-action: \Omega(\Theta)-J
	double Theta[Dim]; //angle-action
	double xv[2*Dim]; //6D Cartesian coors for storage, maybe redundant

	//:: Eulerian space mode
	//:: nothing because this is a particle information

	//:: Largragian particle mode
	int ID = 0; //particle ID, start from 1; be 0 when not particle; default 0
	int type; //particle type;
	double m; //particle mass
	double phi; //gravitational potential
	double type_actionJudje;
	double other; //about electromagnetics or more complicated physics ??

}* wtfh;

/* information on a grid after data process */
extern struct write_secondhand{
	//:: target
	double QP[2*Dim]; //the target general coordinates Q and momentum P, such as Cartesian: x-v， angele-action: \Omega(\Theta)-J
	double Theta[Dim] = {0., 0., 0.}; //angle of angle-action variables; default 0.
	double xv[2*Dim]; //6D Cartesian coors for storage, maybe redundant

	//:: Eulerian space mode
	double bar_PvQ[Dim]; //average P at local Q
	double bar_QvP[Dim];
	double sig_PvQ[Dim]; //diagonal(Dim) of dispersions of P at local Q
	double sig_QvP[Dim];
	double DF_Q = 0., DF_P = 0., DF_QP = 0., DF_other = 0.; 
		//probability density disribution function(DF) of Q, P, QP, and some other DF of Q

	//:: Largragian particle mode
	//:: something are mixed loosely
	int ID = 0; //particle ID, start from 1; be 0 when not particle
	double phi; //gravitational potential
	double H; //halmitonion without mass //Gadget potential??
	double S = 0.; //entropy ??
	double H_OJ = 0.; //halmitonion calculated bu angele-action
	double I[Dim]; //integrals of motion
	double other = 0.; //about mass, type; and about electromagnetics or more complicated physics ??

}* wtsh;
//:: when alloc
// Write_aa = (struct write_angleaction *) malloc(sizeof(struct write_angleaction)*(N_targets+comm_sz));
// free(Write_aa);

class AA_integrating_data{
public:
	int swit;
	double theta;
	double alpha;
	double beta;
	double gamma;
	double tau;
	double phichi;
	double Ints[Dim];
	double ptau_root;
	double ptau;
	double ptau_return;
};

class AA_motion_data{
public:
	double xv[Dim*2];
	double phixyz;
	double ABC[Dim];
	double tau[6];
	double Ints[9];
	double limits[3];
	double actions[4];
	double angles[11];
};

struct data_debug{
	vector<double> xv;
	vector<double> value_double;
	vector<double> info;
	// vector<int> value_int;
	data_debug(){
		xv.resize(Dim*2, 0.);
		vector<double>().swap(value_double);
		info.resize(Dim*2, 0.);
		// vector<double>().swap(value_int);
	}
};

#endif
