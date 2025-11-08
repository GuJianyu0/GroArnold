/* 
    ================================================================
    To read from .txt (simple particles data or whole particles data) 
    and write to .g1; or as versa.
    ===================================================================
*/

#include "../../../step3_actions/step2_Nbody_TACT/DataInterface/Gadget2FormatData_io.h"

int main(int argc, char* argv[]){

    int tag_read_write = atoi(argv[1]); //1: read; other: write ...
    string path_snapshot = (argv[2]); //the name of target file
    RW_Snapshot RWSS;
    if(tag_read_write==1){ //from .txt
        // string path_snapshot1 = "galaxy_general1.g1";
        // RWSS.allocate_memory();
        // RWSS.load(path_snapshot1);
        // RWSS.NumPart = 100000; //??
        // for(int n=0; n<RWSS.NumPart; n++){
        //     RWSS.Id[n] = n;
        //     std::cout<<RWSS.Id[n]<<" ";
        // }
        // RWSS.print_header_info();
        // int id = 1;
        // cout<<RWSS.NumPartTot<<" "<<RWSS.NumPart<<" "
        // <<RWSS.header1.npart[1]<<" "<<RWSS.header1.npartTotal[1]<<" Here_11\n";

        std::cout<<RWSS.num<<" "<<RWSS.NumPart<<" "<<RWSS.NumPartTot<<", tot1\n";
        RWSS.is_reorder = 0;
        RWSS.allocate_memory(); //?? reordering
        RWSS.set_header();
        std::cout<<RWSS.num<<" "<<RWSS.NumPart<<" "<<RWSS.NumPartTot<<", tot2\n";
        RWSS.print_header_info("123");
        // RWSS.read_PD_txt(path_snapshot, 0); //cold IC //one does not use it unless adding a argv[3]
        RWSS.read_PD_txt(path_snapshot, 1); //modified IC
        std::cout<<RWSS.num<<" "<<RWSS.NumPart<<" "<<RWSS.NumPartTot<<", tot3\n";
        RWSS.set_header();
        RWSS.print_header_info("456");
        RWSS.write_gadget_ics_known(path_snapshot);
        // RWSS.print_header_info("789");
        // RWSS.write_PD_toSCF(path_snapshot, ".SCF");

        // // RWSS.allocate_memory();
        // path_snapshot += ".g1";
        // RWSS.load(path_snapshot);
        // RWSS.write_PD_txt(path_snapshot, ".txt");
    }else{ //from .g1
        RWSS.is_reorder = 0;
        RWSS.allocate_memory();
        RWSS.load(path_snapshot); //??
        RWSS.write_PD_txt(path_snapshot, ".txt");
        // RWSS.write_PD_toSCF(path_snapshot, ".SCF"); //?? it is writen in python

        // // RWSS.allocate_memory();
        // path_snapshot += ".txt";
        // RWSS.read_PD_txt(path_snapshot, 1);
        // RWSS.set_header();
        // RWSS.write_gadget_ics_known(path_snapshot);
    }

    return 0;
}