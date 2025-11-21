/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
//// This prog is to read and write initial conditions N-body galaxy data in Gadget2-Format. 
//// The code is modified from read_snapshot.c provided by Volker Springel with the Gadget2  
//// source code and dice_io.c by Valentin Perret with the DICE source code. 
/////////////////////////////////////////////////////////////////////////////////////////////

#include "Gadget2FormatData_io.h"

struct GlobalVars AllVars;

int RW_Snapshot::allocate_memory(void){

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

static bool try_read_block_with_size(FILE *fd, int expected_bytes) {
    long pos = ftell(fd);
    int blocksize = 0;
    size_t n = fread(&blocksize, sizeof(blocksize), 1, fd);
    if (n != 1) { // EOF or error
        fseek(fd, pos, SEEK_SET);
        return false;
    }
    if (blocksize != expected_bytes) {
        // Not the block we expect – rewind
        fseek(fd, pos, SEEK_SET);
        return false;
    }
    // Caller will now read the payload and then consume the trailing SKIP
    return true;
}

int RW_Snapshot::load_snapshot(char *fname, int files){

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
	        printf("Can't open file `%s`.\n", buf);
	        exit(0); //gjy add
            return 1;
	    }

        printf("reading `%s' ...\n", buf);
        fflush(stdout);

        fread(&dummy, sizeof(dummy), 1, fd);
        fread(&header1, sizeof(header1), 1, fd);
// cout<<fd<<" "<<sizeof(header1)<<"\n\n\n\n\n\n";
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

        // ---- optional blocks ----
        // Before reading optional blocks, zero-init the optional fields to be safe.
        for (int ii = 1; ii <= NumPart; ++ii) {
            P[ii].Pot  = 0.0f;
            P[ii].Acc[0] = P[ii].Acc[1] = P[ii].Acc[2] = 0.0f;
            P[ii].dAdt = 0.0f;
            P[ii].Age  = 0.0f;
            P[ii].Metal= 0.0f;
        }

        // Handy counts for this file (NOT total across files)
        int N_thisfile = 0;
        for (k = 0; k < 6; ++k) N_thisfile += header1.npart[k];
        int Ngas_this   = header1.npart[0];
        int Nstar_this  = header1.npart[4];

        // ========== Potential ==========
        {
            const int expected_bytes = (int)(sizeof(float) * N_thisfile);
            if (N_thisfile > 0 && try_read_block_with_size(fd, expected_bytes)) {
                // read potentials by type-contiguous order
                for (k = 0, pc_new = pc; k < 6; k++) {
                    for (n = 0; n < header1.npart[k]; n++) {
                        if (fread(&P[pc_new].Pot, sizeof(float), 1, fd) != 1) {
                            fprintf(stderr, "[ERROR] Failed reading Potential block.\n");
                            exit(2);
                        }
                        pc_new++;
                    }
                }
                SKIP; // trailing blocksize
            }
            // else: block absent -> keep zeros
        }

        // ========== Accelerations ==========
        {
            const int expected_bytes = (int)(3 * sizeof(float) * N_thisfile);
            if (N_thisfile > 0 && try_read_block_with_size(fd, expected_bytes)) {
                for (k = 0, pc_new = pc; k < 6; k++) {
                    for (n = 0; n < header1.npart[k]; n++) {
                        if (fread(&P[pc_new].Acc[0], sizeof(float), 3, fd) != 3) {
                            fprintf(stderr, "[ERROR] Failed reading Acceleration block.\n");
                            exit(2);
                        }
                        pc_new++;
                    }
                }
                SKIP;
            }
            // else: block absent -> keep zeros
        }

        // ========== dA/dt (gas-only, IO_DTENTR) ==========
        if (Ngas_this > 0) {
            const int expected_bytes = (int)(sizeof(float) * Ngas_this);
            if (try_read_block_with_size(fd, expected_bytes)) {
                for (n = 0, pc_sph = pc; n < Ngas_this; n++) {
                    if (fread(&P[pc_sph].dAdt, sizeof(float), 1, fd) != 1) {
                        fprintf(stderr, "[ERROR] Failed reading dAdt block.\n");
                        exit(2);
                    }
                    pc_sph++;
                }
                SKIP;
            }
            // else: block absent -> keep zeros
        }

        // ========== Stellar Age (usually stars only, type=4) ==========
        if (header1.flag_stellarage) {
            if (Nstar_this > 0) {
                // Typical layout: age for stars only
                const int expected_star_bytes = (int)(sizeof(float) * Nstar_this);
                bool read_done = false;

                if (try_read_block_with_size(fd, expected_star_bytes)) {
                    // locate start index of stars in this file (type order 0..5)
                    int start_star = pc
                        + header1.npart[0] + header1.npart[1]
                        + header1.npart[2] + header1.npart[3];
                    for (n = 0, pc_new = start_star; n < Nstar_this; n++) {
                        if (fread(&P[pc_new].Age, sizeof(float), 1, fd) != 1) {
                            fprintf(stderr, "[ERROR] Failed reading Age block (stars-only).\n");
                            exit(2);
                        }
                        pc_new++;
                    }
                    SKIP;
                    read_done = true;
                } else if (try_read_block_with_size(fd, (int)(sizeof(float) * N_thisfile))) {
                    // Fallback layout: age for all particles
                    for (k = 0, pc_new = pc; k < 6; k++) {
                        for (n = 0; n < header1.npart[k]; n++) {
                            if (fread(&P[pc_new].Age, sizeof(float), 1, fd) != 1) {
                                fprintf(stderr, "[ERROR] Failed reading Age block (all types).\n");
                                exit(2);
                            }
                            pc_new++;
                        }
                    }
                    SKIP;
                    read_done = true;
                }

                // If neither matched, age is absent; keep zeros.
                (void)read_done;
            }
        }

        // ========== Metals ==========
        if (header1.flag_metals) {
            int NgasStar_this = Ngas_this + Nstar_this;
            bool read_done = false;

            // Common layout: gas + stars only
            if (NgasStar_this > 0 &&
                try_read_block_with_size(fd, (int)(sizeof(float) * NgasStar_this))) {

                // gas first
                if (Ngas_this > 0) {
                    for (n = 0, pc_sph = pc; n < Ngas_this; n++) {
                        if (fread(&P[pc_sph].Metal, sizeof(float), 1, fd) != 1) {
                            fprintf(stderr, "[ERROR] Failed reading Metal block (gas).\n");
                            exit(2);
                        }
                        pc_sph++;
                    }
                }
                // then stars
                if (Nstar_this > 0) {
                    int start_star = pc
                        + header1.npart[0] + header1.npart[1]
                        + header1.npart[2] + header1.npart[3];
                    for (n = 0, pc_new = start_star; n < Nstar_this; n++) {
                        if (fread(&P[pc_new].Metal, sizeof(float), 1, fd) != 1) {
                            fprintf(stderr, "[ERROR] Failed reading Metal block (stars).\n");
                            exit(2);
                        }
                        pc_new++;
                    }
                }
                SKIP;
                read_done = true;
            }
            // Alternate layout: metals for all particles
            else if (N_thisfile > 0 &&
                    try_read_block_with_size(fd, (int)(sizeof(float) * N_thisfile))) {
                for (k = 0, pc_new = pc; k < 6; k++) {
                    for (n = 0; n < header1.npart[k]; n++) {
                        if (fread(&P[pc_new].Metal, sizeof(float), 1, fd) != 1) {
                            fprintf(stderr, "[ERROR] Failed reading Metal block (all types).\n");
                            exit(2);
                        }
                        pc_new++;
                    }
                }
                SKIP;
                read_done = true;
            }

            // else: no metals block -> keep zeros
            (void)read_done;
        }

    }

    fclose(fd);

    Time = header1.time;
    Redshift = header1.redshift;
    return 0;
}

int RW_Snapshot::reordering(void){

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
    Id = nullptr;

    printf("space for particle ID freed\n");
    return 0;
}



// int RW_Snapshot::write_gadget_ics_known1(string path_snapshot)
// {
//     const char* ofname = (path_snapshot+".g1").data();
//     FILE *fp;
//     int temp;
//     int i, j, n, pc_sph;

//     if (!(fp = fopen(ofname, "w"))){
//         printf("cannot create init file\n");
//         exit(1);
//     }

//     temp = sizeof(struct io_header_1);
//     fwrite(&temp, sizeof(temp), 1, fp);
//     fwrite(&header1, sizeof(struct io_header_1), 1, fp);
//     fwrite(&temp, sizeof(temp), 1, fp);

//     //position block
//     temp = sizeof(float) * 3 * NumPart;
//     fwrite(&temp, sizeof(temp), 1, fp);
//     for (i = 1; i <= NumPart; i++)
//     {
//         for (j = 0; j < 3; j++)
//         {
//             fwrite(&(P[i].Pos[j]), sizeof(float), 1, fp);
//         }
//     }
//     fwrite(&temp, sizeof(temp), 1, fp);

//     //velocity block
//     temp = sizeof(float) * 3 * NumPart;
//     fwrite(&temp, sizeof(temp), 1, fp);
//     for (i = 1; i <= NumPart; i++)
//     {
//         for (j = 0; j < 3; j++)
//         {
//             fwrite(&(P[i].Vel[j]), sizeof(float), 1, fp);
//         }
//     }
//     fwrite(&temp, sizeof(temp), 1, fp);

//     //ID block
//     temp = sizeof(unsigned int) * NumPart;
//     fwrite(&temp, sizeof(temp), 1, fp);
//     for (i = 1; i <= NumPart; i++)
//     {
//         fwrite(&(P[i].Id), sizeof(int), 1, fp);
//     }
//     fwrite(&temp, sizeof(temp), 1, fp);

//     //Mass block
//     temp = sizeof(float) * NumPart;
//     fwrite(&temp, sizeof(temp), 1, fp);
//     for (i = 1; i <= NumPart; i++)
//     {
//         fwrite(&(P[i].Mass), sizeof(float), 1, fp);
//     }
//     fwrite(&temp, sizeof(temp), 1, fp);

//     if (header1.npart[0] > 0)
//     {
//         temp = sizeof(float) * Ngas;
//         fwrite(&temp, sizeof(temp), 1, fp);
//         for (n = 0, pc_sph = 1; n < header1.npart[0]; n++)
//         {
//             fwrite(&P[pc_sph].U, sizeof(float), 1, fp);
//             pc_sph++;
//         }
//         fwrite(&temp, sizeof(temp), 1, fp);

//         temp = sizeof(float) * Ngas;
//         fwrite(&temp, sizeof(temp), 1, fp);
//         for (n = 0, pc_sph = 1; n < header1.npart[0]; n++)
//         {
//             fwrite(&P[pc_sph].Rho, sizeof(float), 1, fp);
//             pc_sph++;
//         }
//         fwrite(&temp, sizeof(temp), 1, fp);

//         if (header1.flag_cooling)
//         {
//             temp = sizeof(float) * Ngas;
//             fwrite(&temp, sizeof(temp), 1, fp);
//             for (n = 0, pc_sph = 1; n < header1.npart[0]; n++)
//             {
//                 fwrite(&P[pc_sph].Ne, sizeof(float), 1, fp);
//                 pc_sph++;
//             }
//             fwrite(&temp, sizeof(temp), 1, fp);
//         }
//     }
//     fwrite(left, Left, 1, fp);
//     fclose(fp);
// }// end of write_snapshot()

int RW_Snapshot::write_gadget_ics_known1(string path_snapshot)
{
    const char* ofname = (path_snapshot+".g1").data();
    FILE *fp;
    int temp;
    int i, j, n, pc_sph;

    if (!(fp = fopen(ofname, "w"))){
        printf("cannot create init file\n");
        exit(1);
    }

    temp = sizeof(struct io_header_1);
    fwrite(&temp, sizeof(temp), 1, fp);
    fwrite(&header1, sizeof(struct io_header_1), 1, fp);
    fwrite(&temp, sizeof(temp), 1, fp);

    //position block
    temp = sizeof(float) * 3 * NumPart;
    fwrite(&temp, sizeof(temp), 1, fp);
    for (i = 1; i <= NumPart; i++)
    {
        for (j = 0; j < 3; j++)
        {
            fwrite(&(P[i].Pos[j]), sizeof(float), 1, fp);
        }
    }
    fwrite(&temp, sizeof(temp), 1, fp);

    //velocity block
    temp = sizeof(float) * 3 * NumPart;
    fwrite(&temp, sizeof(temp), 1, fp);
    for (i = 1; i <= NumPart; i++)
    {
        for (j = 0; j < 3; j++)
        {
            fwrite(&(P[i].Vel[j]), sizeof(float), 1, fp);
        }
    }
    fwrite(&temp, sizeof(temp), 1, fp);

    //ID block
    temp = sizeof(unsigned int) * NumPart;
    fwrite(&temp, sizeof(temp), 1, fp);
    for (i = 1; i <= NumPart; i++)
    {
        fwrite(&(P[i].Id), sizeof(int), 1, fp);
    }
    fwrite(&temp, sizeof(temp), 1, fp);

    //Mass block
    temp = sizeof(float) * NumPart;
    fwrite(&temp, sizeof(temp), 1, fp);
    for (i = 1; i <= NumPart; i++)
    {
        fwrite(&(P[i].Mass), sizeof(float), 1, fp);
    }
    fwrite(&temp, sizeof(temp), 1, fp);

    if (header1.npart[0] > 0)
    {
        temp = sizeof(float) * Ngas;
        fwrite(&temp, sizeof(temp), 1, fp);
        for (n = 0, pc_sph = 1; n < header1.npart[0]; n++)
        {
            fwrite(&P[pc_sph].U, sizeof(float), 1, fp);
            pc_sph++;
        }
        fwrite(&temp, sizeof(temp), 1, fp);

        temp = sizeof(float) * Ngas;
        fwrite(&temp, sizeof(temp), 1, fp);
        for (n = 0, pc_sph = 1; n < header1.npart[0]; n++)
        {
            fwrite(&P[pc_sph].Rho, sizeof(float), 1, fp);
            pc_sph++;
        }
        fwrite(&temp, sizeof(temp), 1, fp);

        if (header1.flag_cooling)
        {
            temp = sizeof(float) * Ngas;
            fwrite(&temp, sizeof(temp), 1, fp);
            for (n = 0, pc_sph = 1; n < header1.npart[0]; n++)
            {
                fwrite(&P[pc_sph].Ne, sizeof(float), 1, fp);
                pc_sph++;
            }
            fwrite(&temp, sizeof(temp), 1, fp);
        }
    }
    fwrite(left, Left, 1, fp);
    fclose(fp);
}// end of write_snapshot()

// int RW_Snapshot::write_gadget_ics_known(string path_snapshot){

//     char fname[MaxCharactersInString];
//     // if(pathload!=nullptr) sprintf(fname, "%s%s_%03d", pathload, bname, snap);
//     // else sprintf(fname, "%s", this->path, bname, snap);
//     sprintf(fname, "%s", path_snapshot.data());

//     FILE *fp1;
//     int dummy, ntot_withmasses, ptype;
//     unsigned long int i, j, k;
//     int t, n, off, pc, pc_new, pc_sph;
//     int files = 1;
//     char buf[MaxCharactersInString];
//     #define SKIP2 fwrite(&dummy, sizeof(dummy), 1, fp1); //record

//     /* hearder */
//     //// We have read data to hearder and *P.
//     //// change particle data, set galaxy* only for number, write.
//     // int ID_change_one = ID_change;
//     // // DEBUG_PRINT_V0d(1, this->P[ID_change_one].Mass, "");
//     // // DEBUG_PRINT_V0d(1, rate, "");
//     // this->P[ID_change_one].Mass *= rate;
//     // // DEBUG_PRINT_V0d(1, this->P[ID_change_one].Mass, "");

//     ////write all pd to IC file
//     int N_tot_part = this->NumPartTot;
//     // int N_tot_part = N_all;
//     int N_tot_part_stars = 0;
//     // for(int i_comp=0;i_comp<components;i_comp++){
//     //     if(type_comp[i_comp]>=2){ //particle with type (>=2) are stars
//     //         N_tot_part_stars += N_comp[i_comp];
//     //     }
//     // }

//     for (i = 0, pc = 1; i < files; i++, pc = pc_new) {
//         if (files > 1) //gjy changed
//             sprintf(buf, "%s.g1_file%d", fname, (int)i);
//         else
//             sprintf(buf, "%s.g1", fname);

//         if (!(fp1 = fopen(buf, "w"))) {
//             fprintf(stderr, "Can't open file `%s`.\n", buf);
//             exit(0);
//         }

//         fflush(stdout);
//         // Header
//         dummy = sizeof(header1);
// // std::cout<<dummy<<"\n";
// // exit(0);
//         fwrite(&dummy, sizeof(dummy), 1, fp1);
//         fwrite(&header1, sizeof(header1), 1, fp1);
//         fwrite(&dummy, sizeof(dummy), 1, fp1);

//         for (k = 0, ntot_withmasses = 0; k < 6; k++) {
//             if (header1.mass[k] == 0)
//                 ntot_withmasses += header1.npart[k];
//         }

//         // Positions
//         dummy = 3 * sizeof(float) * N_tot_part;
//         SKIP2;
//         for (k = 0, pc_new = pc; k < 6; k++) {
//             for (n = 0; n < header1.npart[k]; n++) {
//                 fwrite(&P[pc_new].Pos[0], sizeof(float), 3, fp1);
//                 pc_new++;
//             }
//         }
//         SKIP2;

//         // Velocities
//         SKIP2;
//         for (k = 0, pc_new = pc; k < 6; k++) {
//             for (n = 0; n < header1.npart[k]; n++) {
//                 fwrite(&P[pc_new].Vel[0], sizeof(float), 3, fp1);
//                 pc_new++;
//             }
//         }
//         SKIP2;

//         // Identifiers
//         dummy = sizeof(int) * N_tot_part;
//         SKIP2;
//         for (k = 0, pc_new = pc; k < 6; k++) {
//             for (n = 0; n < header1.npart[k]; n++) {
//                 fwrite(&P[pc_new].Id, sizeof(int), 1, fp1);
//                 pc_new++;
//             }
//         }
//         SKIP2;

//         // Mass
//         if (ntot_withmasses > 0) {
//             dummy = sizeof(float) * N_tot_part;
//             SKIP2;
//         }
//         for (k = 0, pc_new = pc; k < 6; k++) {
//             for (n = 0; n < header1.npart[k]; n++) {
//                 if (header1.mass[k] == 0)
//                     fwrite(&P[pc_new].Mass, sizeof(float), 1, fp1);
//                 else
//                     P[pc_new].Mass = header1.mass[k];
//                 pc_new++;
//             }
//         }
//         if (ntot_withmasses > 0) {
//             SKIP2;
//         }

//         //// Gas specific datablocks
//         if (header1.npart[0] > 0) {
//             // Internal energy
//             dummy = sizeof(float) * header1.npart[0];
//             SKIP2;
//             for (n = 0, pc_sph = pc; n < header1.npart[0]; n++) {
//                 fwrite(&P[pc_sph].U, sizeof(float), 1, fp1);
//                 pc_sph++;
//             }
//             SKIP2;
//             // Density
//             SKIP2;
//             for (n = 0, pc_sph = pc; n < header1.npart[0]; n++) {
//                 fwrite(&P[pc_sph].Rho, sizeof(float), 1, fp1);
//                 pc_sph++;
//             }
//             SKIP2;
//         }

//         //// added, two
//         //Potential
//         dummy = sizeof(float) * N_tot_part;
//         SKIP2;
//         for (k = 0, pc_new = pc; k < 6; k++) {
//             for (n = 0; n < header1.npart[k]; n++) {
//                 fwrite(&P[pc_new].Pot, sizeof(float), 1, fp1);
//                 pc_new++;
//             }
//         }
//         SKIP2;

//         //Accelerations
//         dummy = 3 * sizeof(float) * N_tot_part;
//         SKIP2;
//         for (k = 0, pc_new = pc; k < 6; k++) {
//             for (n = 0; n < header1.npart[k]; n++) {
//                 fwrite(&P[pc_new].Acc[0], sizeof(float), 3, fp1);
//                 pc_new++;
//             }
//         }
//         SKIP2;

//         ////others no use
//         // Metallicity
//         dummy = sizeof(int) * N_tot_part; //gjy note: int??
//         SKIP2;
//         for (k = 0, pc_new = pc; k < 6; k++) {
//             for (n = 0; n < header1.npart[k]; n++) {
//                 fwrite(&P[pc_new].Metal, sizeof(float), 1, fp1);
//                 pc_new++;
//             }
//         }
//         SKIP2;

//         // Age for star particles
//         if (N_tot_part_stars > 0) {
//             dummy = sizeof(int) * (N_tot_part_stars);
//             SKIP2;
//             for (k = 2, pc_new = header1.npart[0] + header1.npart[1] + 1; k < 6; k++) {
//                 for (n = 0; n < header1.npart[k]; n++) {
//                     fwrite(&P[pc_new].Age, sizeof(float), 1, fp1);
//                     pc_new++;
//                 }
//             }
//             SKIP2;
//         }

//     }

//     fclose(fp1);
//     printf("Wrote %s ... done.\n", buf);
//     return 0;
// }

int RW_Snapshot::write_gadget_ics_known(string path_snapshot){
    char fname[MaxCharactersInString];
    sprintf(fname, "%s", path_snapshot.data());

    // Make sure NumPart/NumPartTot are consistent
    if(NumPartTot <= 0){
        fprintf(stderr, "[ERROR] NumPartTot=%d. Nothing to write.\n", NumPartTot);
        exit(2);
    }
    NumPart = NumPartTot;

    // 1) Reorder by type (validates Type in [0..5])
    reorder_particles_by_type_or_die();

    // 2) Build header from P[].Type and sanity-check counts
    set_header();

    //The below is same with old version. It already writes in 6 contiguous type-blocks using header1.npart[k].)
    FILE *fp1;
    int dummy, ntot_withmasses;
    unsigned long int i, j, k;
    int n, pc, pc_new, pc_sph;
    int files = 1;
    char buf[MaxCharactersInString];
    #define SKIP2 fwrite(&dummy, sizeof(dummy), 1, fp1);

    for (i = 0, pc = 1; i < (unsigned long)files; i++, pc = pc_new) {
        if (files > 1) sprintf(buf, "%s.g1_file%lu", fname, i);
        else           sprintf(buf, "%s.g1", fname);

        if (!(fp1 = fopen(buf, "w"))) {
            fprintf(stderr, "Can't open file `%s`.\n", buf);
            exit(1);
        }

        // ----- header -----
        dummy = sizeof(header1);
        fwrite(&dummy, sizeof(dummy), 1, fp1);
        fwrite(&header1, sizeof(header1), 1, fp1);
        fwrite(&dummy, sizeof(dummy), 1, fp1);

        for (k = 0, ntot_withmasses = 0; k < 6; k++)
            if (header1.mass[k] == 0)
                ntot_withmasses += header1.npart[k];

        const int N_tot_part = NumPartTot; // already synced
        // ----- positions -----
        dummy = 3 * sizeof(float) * N_tot_part; SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++){
            for (n = 0; n < header1.npart[k]; n++){
                fwrite(&P[pc_new].Pos[0], sizeof(float), 3, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // ----- velocities -----
        SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++){
            for (n = 0; n < header1.npart[k]; n++){
                fwrite(&P[pc_new].Vel[0], sizeof(float), 3, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // ----- IDs -----
        dummy = sizeof(int) * N_tot_part; SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++){
            for (n = 0; n < header1.npart[k]; n++){
                fwrite(&P[pc_new].Id, sizeof(int), 1, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // ----- masses (per-particle if header1.mass[k]==0) -----
        if (ntot_withmasses > 0){ dummy = sizeof(float) * N_tot_part; SKIP2; }
        for (k = 0, pc_new = pc; k < 6; k++){
            for (n = 0; n < header1.npart[k]; n++){
                if (header1.mass[k] == 0)
                    fwrite(&P[pc_new].Mass, sizeof(float), 1, fp1);
                else
                    /* fixed-mass types (not used here) */ P[pc_new].Mass = header1.mass[k];
                pc_new++;
            }
        }
        if (ntot_withmasses > 0) SKIP2;

        // ----- gas blocks (if any) -----
        if (header1.npart[0] > 0) {
            // U
            dummy = sizeof(float) * header1.npart[0]; SKIP2;
            for (n = 0, pc_sph = pc; n < header1.npart[0]; n++){ fwrite(&P[pc_sph].U,   sizeof(float), 1, fp1); pc_sph++; }
            SKIP2;
            // Rho
            dummy = sizeof(float) * header1.npart[0]; SKIP2;
            for (n = 0, pc_sph = pc; n < header1.npart[0]; n++){ fwrite(&P[pc_sph].Rho, sizeof(float), 1, fp1); pc_sph++; }
            SKIP2;
        }

        // ----- Potential -----
        dummy = sizeof(float) * N_tot_part; SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++){
            for (n = 0; n < header1.npart[k]; n++){
                fwrite(&P[pc_new].Pot, sizeof(float), 1, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // ----- Acc -----
        dummy = 3 * sizeof(float) * N_tot_part; SKIP2;
        for (k = 0, pc_new = pc; k < 6; k++){
            for (n = 0; n < header1.npart[k]; n++){
                fwrite(&P[pc_new].Acc[0], sizeof(float), 3, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // ----- Metal (uses float per particle) -----
        dummy = sizeof(int) * N_tot_part; SKIP2; //record size
        for (k = 0, pc_new = pc; k < 6; k++){
            for (n = 0; n < header1.npart[k]; n++){
                fwrite(&P[pc_new].Metal, sizeof(float), 1, fp1);
                pc_new++;
            }
        }
        SKIP2;

        // ----- Age for star particles (types >=2) -----
        int N_tot_part_stars = 0;
        for (int tk=2; tk<6; ++tk) N_tot_part_stars += header1.npart[tk];
        if (N_tot_part_stars > 0) {
            dummy = sizeof(int) * N_tot_part_stars; SKIP2;
            for (k = 2, pc_new = header1.npart[0] + header1.npart[1] + 1; k < 6; k++){
                for (n = 0; n < header1.npart[k]; n++){
                    fwrite(&P[pc_new].Age, sizeof(float), 1, fp1);
                    pc_new++;
                }
            }
            SKIP2;
        }

        fclose(fp1);
        printf("Wrote %s ... done.\n", buf);
    }

    return 0;
}

int RW_Snapshot::reorder_particles_by_type_or_die(){
    if(P == nullptr || NumPartTot <= 0){
        fprintf(stderr, "[ERROR] No particles to reorder (P=null or NumPartTot<=0).\n");
        exit(2);
    }

    const int N = NumPartTot;
    int counts[6] = {0,0,0,0,0,0};
    for(int i=1;i<=N;i++){
        int t = P[i].Type;
        if(t < 0 || t > 5){
            fprintf(stderr, "[ERROR] P[%d].Type=%d out of [0..5].\n", i, t);
            exit(2);
        }
        counts[t]++;
    }

    // Compute start offsets (1-based arrays)
    int start[7]; start[0] = 1;
    for(int k=0;k<6;k++) start[k+1] = start[k] + counts[k];

    int cursor[6];
    for(int k=0;k<6;k++) cursor[k] = start[k];

    std::vector<particle_data> tmp(N+1); // index 1..N used
    for(int i=1;i<=N;i++){
        int t = P[i].Type;
        int dst = cursor[t]++;
        tmp[dst] = P[i];
    }

    // Copy back
    for(int i=1;i<=N;i++) P[i] = tmp[i];

    // Optional: ensure IDs exist (fail fast or normalize).
    for(int i=1;i<=N;i++){
        if(P[i].Id <= 0) P[i].Id = i;
    }

    return 0;
}

////to debug
int main0001(){

    abort();
    return 0;
}
