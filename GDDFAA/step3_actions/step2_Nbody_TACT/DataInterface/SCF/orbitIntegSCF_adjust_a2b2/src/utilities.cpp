#include"utilities.h"

namespace UTILITIES{

int write_data_debug(
    const struct data_debug_sendrecv* vdd, 
    const int N_element, const int N_eachline, string filename, bool is_write_pure
){
    FILE* fp = fopen(filename.data(), "w");
    if(fp==nullptr){
        printf("Cannot open file %s. Now exit.\n", filename.data());
        exit(0);
    }
    cout<<"vdd element size: "<<N_element<<"\n";
    if(is_write_pure){
        for(int i=0;i<N_element;i++){
            for(int j=0;j<N_eachline;j++){
                fprintf(fp, "%le ", vdd[i].value_double[j]);
            } //main data
            fprintf(fp, "\n");
        }
    }else{
        string data_descreption = "##write: column: xv(double[6]) PoPotential1 Forces11 12 13 "
            "Potential2 ... Potential3 ... Potential3 ... "
            "#not used: tau_and_metric(double[6]) tau_and_dottau(double[6]) "
            "tau_and_ptau(double[6]) tau_and_ptau_debug(double[6]) potential_and_energy.";
        fprintf(fp, "%s\n", data_descreption.data());
        // for(auto d : vdd){
        for(int i=0;i<N_element;i++){
            auto d = vdd[i];
            for(auto a:d.xv){fprintf(fp, "%le ", a);} //count 6
            fprintf(fp, "    ");
            for(auto a:d.value_double){fprintf(fp, "%le ", a);} //main data
            fprintf(fp, "    ");
            for(auto a:d.info){fprintf(fp, "%le ", a);} //count 6 default
            // fprintf(fp, "    ");
            // for(auto a:d.value_int){fprintf(fp, "%d ", a);} //count 0 default
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
    printf("Write debug file %s ... done.\n", filename.data());
    return 0;
}

int write_data_debug(const vector<struct data_debug>& vdd, string filename, bool is_write_pure)
{
    FILE* fp = fopen(filename.data(), "w");
    if(fp==nullptr){
        printf("Cannot open file %s. Now exit.\n", filename.data());
        exit(0);
    }
    cout<<"vdd size: "<<vdd.size()<<"\n";
    if(is_write_pure){
        for(auto d : vdd){
            for(auto a:d.value_double){fprintf(fp, "%le ", a);} //main data
            fprintf(fp, "\n");
        }
    }else{
        string data_descreption = "##write: column: xv(double[6]) PoPotential1 Forces11 12 13 "
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
    }
    fclose(fp);
    printf("Write debug file %s ... done.\n", filename.data());
    return 0;
}

int copy_file(const char *SourceFile, const char *NewFile){
	ifstream in;
	ofstream out;
	in.open(SourceFile, ios::binary); //open original file
	if(in.fail()){
		std::cout<<"Error 1: Fail to open the source file."<<std::endl;
		in.close();
		out.close();
		return 0;
	}
	out.open(NewFile,ios::binary); //create target file
	if(out.fail()){
		std::cout<<"Error 2: Fail to create the new file."<<std::endl;
		out.close();
		in.close();
		return 0;
	}
	else{ //copy file
		out<<in.rdbuf();
		out.close();
		in.close();
		std::cout<<"Copy file `"<<NewFile<<"` Done."<<std::endl;
		return 1;
	}
}

}
