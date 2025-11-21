///////////////////////////////////////////////////////////
//// Some useful tools.
///////////////////////////////////////////////////////////

#ifndef _UTILITIES_
#define _UTILITIES_

#include<math.h>
#include<stdio.h>
#include<time.h>
#include<iostream>
#include<fstream>
#include<string>
#include"string.h"
#include<vector>
#include<algorithm>
#include"assert.h"
using namespace std;

#include <Eigen/Eigen>
using namespace Eigen;

namespace UTILITIES{

typedef vector<vector<int>> vecvecint;

#define Dim 3
#define MaxCharactersInString 200
#define pi_8 3.1415926
#define Err 1.e-18
#define SKIP_UNTILENDL(fp) fscanf((fp), "%*[^\n]%*c");

#define isSign(a) 		( (a)>=0? 1 : 0 )
#define Sign_(a) 		( (a)>0? 1 : ((a)<0? -1 : 0) )
#define selectMin(a,b) 	( (a)<=(b)? (a) : (b) )
#define selectMax(a,b) 	( (a)<=(b)? (b) : (a) )



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



//// file io
int write_data_debug(const struct data_debug_sendrecv* vdd, 
	const int N_element, const int N_eachline, string filename, 
	bool is_write_pure=true
);
int write_data_debug(const vector<struct data_debug>& vdd, 
	string filename, 
	bool is_write_pure=true
);

struct data_debug_sendrecv{
	double xv[12];
	double value_double[20];
	double info[10];
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

int copy_file(const char *SourceFile, const char *NewFile);



//// useful funcs
/* 	An assistant class for MPI task ID distribution.
	It is to obtain the task IDs of each rank from the loop index of 
	each rank. So this is an abnormal order and it might be foolish assistant ...
	These methods has the same count of distribution for ranks.
*/
class MPI_distribution_assistant{
private:
	int N_task; //count of global tasks
	int N_rank; //comm_size, count of ranks or sources
	// int total_count; //to check, should it = N_task
    int major_interval;
	int rest_interval;
    int remainder;
	int distribution_method;
	vecvecint tasks_IDs;
	//\ each vector<int> is the task IDs of each rank, integers within range [0, N_task)

public:
	MPI_distribution_assistant(int ntask, int nrank){
		N_task = ntask;
		N_rank = nrank;
		tasks_IDs.resize(N_rank);
		if(N_task<1 || N_rank<1){
			std::cout<<"Wrong input value of task count and rank count. Please check. Exit.";
			exit(0);
		}
	}

	~MPI_distribution_assistant(){
    	vecvecint().swap(tasks_IDs);
	}

	vecvecint get_tasks_IDs(){ //the rank 0 is main rank
		return tasks_IDs;
	}

	vector<int> get_distribution_info(){ //note the below output order
		vector<int> di;
		di.resize(4);
		di[0] = major_interval;
		di[1] = rest_interval;
		di[2] = remainder;
		di[3] = distribution_method;
		return di;
	}

	/* 	Method: Direct cut the global tasks list.
	*/
	void distribute_by_direct_cutting(){ //to choose
		distribution_method = 0;

    	major_interval = (int)round(ceil(N_task*1./N_rank));
		rest_interval = N_task-major_interval*(N_rank-1);
    	remainder = N_task%(N_rank);

		for(int irank=1;irank<N_rank;irank++){
			tasks_IDs[irank].resize(major_interval);
			for(int i=0;i<major_interval;i++){ //itask
				tasks_IDs[irank][i] = major_interval*(irank-1)+i;
			}
		}

		// for irank==0 begin
			tasks_IDs[0].resize(rest_interval);
			for(int i=0;i<rest_interval;i++){ //itask
				tasks_IDs[0][i] = major_interval*(N_rank-1)+i;
			}
		// for irank==0 end
	}

	/* 	Method: Give tasks from the global tasks list to each rank in turn. 
		Especially for the condition when the later task has more run time. 
		Note this destory the original order of global tasks.
	*/
	void distribute_by_give_in_turn(){ //to choose
		distribution_method = 1;

    	major_interval = (int)round(ceil(N_task*1./N_rank));
		rest_interval = N_task-major_interval*(N_rank-1);
    	remainder = N_task%(N_rank);

		//: wrong
		// for(int irank=1;irank<N_rank;irank++){
		// 	tasks_IDs[irank].resize(major_interval);
		// 	for(int i=0;i<major_interval;i++){ //itask
		// 		tasks_IDs[irank][i] = i*N_rank+irank-1;
		// 	}
		// }
		// // for irank==0 begin
		// 	tasks_IDs[0].resize(rest_interval);
		// 	for(int i=0;i<rest_interval;i++){ //itask
		// 		tasks_IDs[0][i] = i*N_rank+N_rank-1;
		// 	}
		// // for irank==0 end

		//: distribute count of each rank
		for(int irank=1;irank<N_rank;irank++){
			tasks_IDs[irank].resize(major_interval, -1);
		}
		tasks_IDs[0].resize(rest_interval, -1);

		//: distribute tasks IDs of each rank, from large ID to little ID for run time
		int n_player = N_rank;
		int i_turn = 0;
		int itask = N_task-1;
		while(itask>=0){
			for(int ipl=N_rank-n_player;ipl<N_rank;ipl++){
				tasks_IDs[ipl][i_turn] = itask;
				itask--;
			}
			if(i_turn==rest_interval-1){
				n_player -= 1;
			}
			i_turn++;
		};

		//: sort tasks IDs of each rank
		for(int irank=0;irank<N_rank;irank++){
			sort(tasks_IDs[irank].begin(), tasks_IDs[irank].end());
		}
	}

	int data_struct_local_from(int my_rank){
		if(my_rank==0){return major_interval*(N_rank-1);}
		else{return major_interval*(my_rank-1);}
	}

	int data_struct_local_count(int my_rank){
		if(my_rank==0){return rest_interval;}
		else{return major_interval;}
	}

	int data_struct_malloc_count(int my_rank){
		if(my_rank==0){return N_task;}
		else{return major_interval;}
	}

	/* To get the index of data struct if each rank has its data with distributed size. */
	int data_struct_tmp_index_locally(int loop_index, int my_rank){
		if(my_rank==0){return data_struct_local_from(0)+loop_index;}
		else{return 0+loop_index;}
	}

	/* To get the index of data struct if each rank has its data with global size. */
	int data_struct_tmp_index_everyone(int loop_index, int my_rank){
		return data_struct_local_from(my_rank)+loop_index;
	}

	void print_tasks_IDs(){
		auto di = get_distribution_info();
		DEBUG_PRINT_V1d(1, di, "major_interval, rest_interval, remainder, method");

		for(int irank=0;irank<N_rank;irank++){
			std::cout<<"rank = "<<irank<<", tasks_count_this_rank = "<<tasks_IDs[irank].size()<<":\n";
			DEBUG_PRINT_V1d(1, tasks_IDs[irank], "tasks_IDs");
		}
		std::cout<<"print_tasks_IDs(), done.\n\n";
	}

    void print_somerank_info(int my_rank){
		std::cout<<"rank = "<<my_rank<<", tasks_count_this_rank = "<<tasks_IDs[my_rank].size()<<"\n";
	}
};



//// calculation
template<typename TYPEN>
const vector<TYPEN>* const Substance_of_VecDoub_as_nullptr = nullptr;

template<typename TYPEN>
TYPEN l2norm_real(const vector<TYPEN>& a)
{
	TYPEN sum = (TYPEN)0;
	size_t n = a.size();
	for(int i=0;i<n;i++){
		sum += abs(a[i]*a[i]);
	}
	return sqrt(sum);
}

//overload operator for data class "+" ??
template<typename TYPEN>
// typename std::enable_if<std::is_same<std::nullptr_t, T>::value || std::is_pointer<T>::value>::type
TYPEN lpnorm_real_with_distance_and_coef(const vector<TYPEN>* pa, const vector<TYPEN>* pb=nullptr, 
	const vector<TYPEN>* pc=nullptr, double pp=2)
{
	TYPEN sum = (TYPEN)0;
	vector<TYPEN> a = *pa, b, c;
	size_t n = a.size();
	if(pp==0){ //to return the count of elements that is none zero
		int idx = 0;
		for(auto i:a){
			if(abs(i)<Err){
				idx++;
			}
		}
		return (TYPEN)idx;
	}else{
		if(pp==std::numeric_limits<TYPEN>::infinity()){ //to return the max element
			return *max_element(a.begin(), a.end());
		}
	}

	if(pb==nullptr){
		b.resize(n, (TYPEN)0);
	}else{
		b = *pb;
	}
	if(pc==nullptr){
		c.resize(n, (TYPEN)1);
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
	return (TYPEN)pow(sum, 1./pp);
}

template<typename TYPEN, int dim>
vector<TYPEN> calculate_mass_center_unitmass(const vector<vector<TYPEN>>& x){
	assert((x[0]).size()==dim);
	vector<TYPEN> c;
	c.resize(dim, (TYPEN)0.);
	for(auto ia : x){ //"x" should be shape like ["count of points"][dim]
		for(int i=0;i<dim;++i){
			c[i] += ia[i];
		}
	}
	for(int i=0;i<dim;++i){
		c[i] /= x.size();
	}
	return c;
}

template<typename TYPEN, int dim>
void translate_vector(vector<vector<TYPEN>>& x, const vector<TYPEN>& c){
	assert((x[0]).size()==dim);
	assert(c.size()==dim);
	int n = x.size();
	for(int j=0;j<n;++j){
		for(int i=0;i<dim;++i){
			x[j][i] -= c[i];
		}
	}
}

//from euler angle to SO3 matrix element
//ang is Euler angle
//?? only for dim==3
template<typename TYPEN, int dim>
MatrixXd SO3_angle_to_matrix(const vector<TYPEN>& ang){
	assert(ang.size()==dim);
	double c0 = cos(ang[0]), s0 = sin(ang[0]), 
	c1 = cos(ang[1]), s1 = sin(ang[1]), 
	c2 = cos(ang[2]), s2 = sin(ang[2]);
	MatrixXd mat(dim, dim);
	mat(0,0) = c1*c2;
	mat(0,1) = -c0*s2+s0*s1*c2;
	mat(0,2) = s0*s2+c0*s1*c2;
	mat(1,0) = c1*s2;
	mat(1,1) = c0*c2+s0*s1*s2;
	mat(1,2) = -s0*c2+c0*s1*s2;
	mat(2,0) = -s1;
	mat(2,1) = s0*c1;
	mat(2,2) = c0*c1;
	return mat;
}

template<typename TYPEN, int dim>
vector<vector<TYPEN>> SO3_vector(const vector<vector<TYPEN>>& x, const MatrixXd& mat){
	assert((x[0]).size()==dim);
	assert(mat.size()==dim*dim);
	int n = x.size();
	auto x_ = x;
	MatrixXd x1(1, dim), x2; //??
	for(int j=0;j<n;++j){
		for(int i=0;i<dim;++i){
			x1(i) = x[j][i];
		}
		x2 = x1*mat;
		for(int i=0;i<dim;++i){
			x_[j][i] = x2(i);
		}
	}
	return x_;
}

template<typename TYPEN, int dim>
vector<vector<TYPEN>> SO3_vector(const vector<vector<TYPEN>>& x, const vector<TYPEN>& ang){
	auto mat = SO3_angle_to_matrix<TYPEN, dim>(ang);
	return SO3_vector<TYPEN, dim>(x, mat);
}

//from Sanders TACT/general/utils.h
template<class c>
std::vector<std::vector<c>> transpose(const std::vector<std::vector<c>> &a) {
    std::vector<std::vector<c>> result(a[0].size(),std::vector<c>(a.size()));
    for (unsigned int i = 0; i < a[0].size(); i++)
        for (unsigned int j = 0; j < a.size(); j++) {
            result[i][j] = a[j][i];
        }
    return result;
}



}

#endif