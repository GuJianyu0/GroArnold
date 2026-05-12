#!/usr/bin/env python
# -*- coding:utf-8 -*-

# ============================================================================================
# Description: To generate IC files with changed settings from the manucarft IC file for simulation.
# ============================================================================================

import os
import sys
import numpy as np
import analysis_data_distribution as ads

## basic informations; the two kind of new files will be write to "totalfolder+"GDDFAA/step1_galaxy_IC_preprocess/step1_set_IC_DDFA/"
totalfolder = "../../../"
manucraft = totalfolder+"install_and_run/IC_DICE_manucraft.params"
user_settings_multi_file = totalfolder+"install_and_run/user_settings_multi.txt"
setting_file = totalfolder+"install_and_run/IC_setting_list.txt" #name in fileA for example: lambda to spin_L
galaxy_general_location_path = totalfolder+"GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"



## io before
def read_IC_settings(filename):

    Mass_vir = 0.
    frac_mass_comp1 = 0.
    mass_comp1 = 0.
    N_comp1 = 0
    v_sigma1 = 30.0 #??
    cold_alpha1   = 1.0
    cold_alphamax1 = 3.0
    scale_length_comp1 = 19.6
    seed1 = 487543

    file_handle = open(filename, mode="r")
    st = file_handle.readlines()
    # print(st)
    # print(st[-1])
    print(np.shape(st))

    for s in st:
        if ("Mass_vir" in s) and s[0]!="#":
            n = float(s.split()[1])
            print(n)
            Mass_vir = n
        if ("frac_mass_comp1" in s) and s[0]!="#":
            n = float(s.split()[1])
            print(n)
            frac_mass_comp1 = n
        if ("N_comp1" in s) and s[0]!="#":
            n = int(s.split()[1])
            print(n)
            N_comp1 = n
        if ("v_sigma1" in s) and s[0]!="#":
            n = float(s.split()[1])
            print(n)
            v_sigma1 = n
        if ("cold_alpha1" in s) and s[0]!="#":
            n = float(s.split()[1])
            print(n)
            cold_alpha1 = n
        if ("cold_alphamax1" in s) and s[0]!="#":
            n = float(s.split()[1])
            print(n)
            cold_alphamax1 = n
        if ("scale_length_comp1" in s) and s[0]!="#":
            n = float(s.split()[1])
            print(n)
            scale_length_comp1 = n
        if ("seed1" in s) and s[0]!="#":
            n = int(s.split()[1])
            print(n)
            seed1 = n

    mass1 = Mass_vir*frac_mass_comp1/N_comp1
    file_handle.close()

    return mass1, N_comp1, v_sigma1, cold_alpha1, cold_alphamax1, scale_length_comp1, seed1

class Galaxy_params_by_reading:
    def __init__(self, which_varing=0):
        self.which_varing = which_varing
        self.manucraft_st = None

        self.comment1 = "## This file is params for a galaxy initial condition (IC).\n"\
            +"## the name of the galaxy is galaxy_general.XXX when calculating; then, they are rename to galaxy_general_123.XXX to store.\n"\
            +"## When reading, vacum line are dismissed. The sign of notes: '#' or '/''/' or '%' or '!'.\n"\
            +"\n"\
            +"# ##an example of param\n"\
            +"# example_abcd 			            100	200	#the first is name, the second is value, the left are no use.\n"\
            +"example_efgh			            110.		#no use example\n"\
            +"\n"\
            +"####------------------------------------------------------------------\n"\
            +"####about the galaxy model\n"
        
        #?? some params value can be changed here
        #paramsetting
        self.comment2 = "##default semiaxis of elliptical coordinate when calculation angle-actions in Sanders TACT Fudge Method\n"\
            +"TACT_semiaxis_Alpha			    -3.		#$-alpha^2$, where alpha is length of the longset semiaxis\n"\
            +"TACT_semiaxis_Beta			    -2.		#$-bata^2$\n"\
            +"TACT_semiaxis_Gamma			    -1.		#$-gamma^2$, while this is default -1. in that prog\n"\
            +"\n"\
            +"##softening of particle type\n"\
            +"softening_type_gas	        	5.e-2  		#0: gas\n"\
            +"softening_type_halo	        	5.e-2  		#1: halo\n"\
            +"softening_type_disk	        	2.e-2  		#2: disk\n"\
            +"softening_type_bulge	    	2.e-2  		#3: bulge\n"\
            +"softening_type_stars	    	2.e-2  		#4: stars\n"\
            +"softening_type_bndry	    	2.e-2  		#5: bndry\n"\
            +"\n"\
            +"#softening_type_gas	        	5.e-1  		#0: gas\n"\
            +"#softening_type_halo	        	5.e-1  		#1: halo\n"\
            +"#softening_type_disk	        	2.e-1  		#2: disk\n"\
            +"#softening_type_bulge	    		2.e-1  		#3: bulge\n"\
            +"#softening_type_stars	    		2.e-1  		#4: stars\n"\
            +"#softening_type_bndry	    		2.e-1  		#5: bndry\n"\
            +"\n"\
            +"#softening_type_gas	        	    5.e+0  		#0: gas\n"\
            +"#softening_type_halo	        	5.e+0  		#1: halo\n"\
            +"#softening_type_disk	        	2.e+0  		#2: disk\n"\
            +"#softening_type_bulge	    		2.e+0  		#3: bulge\n"\
            +"#softening_type_stars	    		2.e+0  		#4: stars\n"\
            +"#softening_type_bndry	    		2.e+0  		#5: bndry\n"\
            +"\n"\
            +"#softening_type_gas	        	    5.e-0  		#0: gas\n"\
            +"#softening_type_halo	        	5.e-0  		#1: halo\n"\
            +"#softening_type_disk	        	2.e-0  		#2: disk\n"\
            +"#softening_type_bulge	    		2.e-0  		#3: bulge\n"\
            +"#softening_type_stars	    		2.e-0  		#4: stars\n"\
            +"#softening_type_bndry	    		2.e-0  		#5: bndry\n"\
            +"\n"\
            +"##instructions:\n"\
            +"# $T_\mathrm{relaxCmpl} = N/ln(\lambda N)T_\mathrm{dyn}, \n"\
            +"# T_\mathrm{dyn} = \sqrt{R_s^3/(GM)}$;\n"\
            +"# $R_s/N^(1/2) <= \epsilon_\mathrm{softeningShould} <= R_s/N^(1/3)$.\n"\
            +"\n"\
            +"\n"\
            +"\n"\
            +"####-------------------------------------------------------------------\n"\
            +"####about a component of the galaxy\n"\
            +"##total number of component1 (type1, halo) particles; N_comp1\n"\
            +"# N_comp1		                10000		#N\n"\
            +"%s		                %d #1000000		#N\n"\
            +"##total number of component2 (type0, gas) particles; N_comp2\n"\
            +"#N_comp2		            2000		#N\n"\
            +"\n"\
            +"####type of a component; type_comp1\n"\
            +"%s		            %d #1		#N\n"\
            +"# type_comp2		            2		#N\n"\
            +"\n"\
            +"##mass fraction of a component; frac_mass_comp1\n"\
            +"%s	            %e #1. #0.9	#frac\n"\
            +"# frac_mass_comp2	        0.06		#frac\n"\
            +"\n"\
            +"####expected parameters; unused\n"\
            +"##softening of a component; unused\n"\
            +"softening_comp1	            0.05		#soft\n"\
            +"# softening_comp2	        0.02		#soft\n"\
            +"\n"\
            +"##scale length of a component; scale_length_comp1; unused\n"\
            +"%s	        %e #19.6		#sl\n"\
            +"# scale_length_comp2	    0.		#sl\n"\
            +"\n"\
            +"## falttening of a component; flatx_comp1, flaty_comp1, flatz_comp1; unused\n"\
            +"%s		            %e #1.\n"\
            +"%s		            %e #0.6\n"\
            +"%s		            %e #0.3\n"\
            +"\n"\
            +"##setted power of one powerlaw or EinasoUsual profile of a component; powerS1; unused\n"\
            +"%s		                %e #1.7\n"\
            +"##setted powers of halo profile of a component; powerA1, powerB1; unused but recorded\n"\
            +"%s		                %e #1.\n"\
            +"%s		                %e #3.\n"\
            +"\n"\
            +"##angular moment and spin; spin_L\n"\
            +"%s				%e #0.\n"\
            +"\n"\
            +"\n"\
            +"\n"\
            +"##cold down method to generate\n"\
            +"mass1					    0.\n"\
            +"cold_alpha1				    1.0\n"\
            +"cold_alphamax1				3.0\n"\
            +"v_sigma_cold1				    30.0\n"\
            +"seed1					    876543\n"\
            +"\n"

        self.comment3 = ""

        self.modelId = 1 #default
        self.modelInfo = "_type-halo_theExpectedValues_values_End" #default
        self.modelPath = "galaxy_general/" #default
        
        self.Mass_vir = None
        self.Mass_vir_name1 = "m200"
        self.Mass_vir_name2 = "Mass_vir"

        self.components = None
        self.components_name1 = None
        self.components_name2 = "components"

        self.N_comp1 = None
        self.N_comp1_name1 = "npart1"
        self.N_comp1_name2 = "N_comp1"

        self.type_comp1 = None
        self.type_comp1_name1 = None
        self.type_comp1_name2 = "type_comp1"

        self.frac_mass_comp1 = None
        self.frac_mass_comp1_name1 = "mass_frac1"
        self.frac_mass_comp1_name2 = "frac_mass_comp1"

        self.scale_length_comp1 = None
        self.scale_length_comp1_name1 = "scale_length1"
        self.scale_length_comp1_name2 = "scale_length_comp1"

        self.flatx_comp1 = None
        self.flatx_comp1_name1 = "flatx1"
        self.flatx_comp1_name2 = "flatx_comp1"

        self.flaty_comp1 = None
        self.flaty_comp1_name1 = "flaty1"
        self.flaty_comp1_name2 = "flaty_comp1"

        self.flatz_comp1 = None
        self.flatz_comp1_name1 = "flatz1"
        self.flatz_comp1_name2 = "flatz_comp1"

        self.powerS1 = None
        self.powerS1_name1 = None
        self.powerS1_name2 = "powerS1"

        self.powerA1 = None
        self.powerA1_name1 = "alpha_struct1"
        self.powerA1_name2 = "powerA1"

        self.powerB1 = None
        self.powerB1_name1 = "beta_struct1"
        self.powerB1_name2 = "powerB1"

        self.spin_L = None
        self.spin_L_name1 = "lambda"
        self.spin_L_name2 = "spin_L"

        # self.paramsetting = None
        # self.paramsetting_name1 = ""
        # self.paramsetting_name2 = "paramsetting"

        return None

def read_params_to_class(file=manucraft):
    file_handle = open(file, mode="r")
    st = file_handle.readlines()
    # print("Lines count of param fileA: ", np.shape(st))
    file_handle.close()

    GP = Galaxy_params_by_reading()
    GP.manucraft_st = st
    for s in st:
        if (GP.Mass_vir_name1 in s) and s[0]!="#":
            n = float(s.split()[1])
            GP.Mass_vir = n

        GP.components = 1

        if (GP.N_comp1_name1 in s) and s[0]!="#":
            n = int(s.split()[1])
            GP.N_comp1 = n

        GP.type_comp1 = 1

        if (GP.frac_mass_comp1_name1 in s) and s[0]!="#":
            n = float(s.split()[1])
            GP.frac_mass_comp1 = n

        if (GP.scale_length_comp1_name1 in s) and s[0]!="#":
            n = float(s.split()[1])
            GP.scale_length_comp1 = n

        if (GP.flatx_comp1_name1 in s) and s[0]!="#":
            n = float(s.split()[1])
            GP.flatx_comp1 = n

        if (GP.flaty_comp1_name1 in s) and s[0]!="#":
            n = float(s.split()[1])
            GP.flaty_comp1 = n

        if (GP.flatz_comp1_name1 in s) and s[0]!="#":
            n = float(s.split()[1])
            GP.flatz_comp1 = n

        GP.powerS1 = 1.7 #default

        GP.powerA1 = 1. #default
        if (GP.powerA1_name1 in s) and s[0]!="#":
            n = float(s.split()[1])
            GP.powerA1 = n #changed

        GP.powerB1 = 3. #default
        if (GP.powerB1_name1 in s) and s[0]!="#":
            n = float(s.split()[1])
            GP.powerA1 = n #changed


        if (GP.spin_L_name1 in s) and s[0]!="#":
            n = float(s.split()[1])
            GP.spin_L = n

        # if (GP.paramsetting_name1 in s) and s[0]!="#":
        #     n = float(s.split()[1])
        #     GP.paramsetting = n

    return GP

def read_IC_setting_list():
    file_handle = open(setting_file, mode="r")
    gn = file_handle.readline()[:-1].split() #name of the init galaxy model, e.g. DPL_spin
    sn = file_handle.readline()[:-1].split() #name of the changed param in param fileA, e.g. lambda
    sv = file_handle.readline()[:-1].split() #value list of the changed param, e.g. 0.0 0.1 0.2
    gn = gn[0]
    sn = sn[0]
    sv = np.array(sv).astype(float)
    # ads.DEBUG_PRINT_V(0, sn)
    file_handle.close()
    return gn, sn, sv

def write_params_fileA(GP, gm_IC_fileA, sn, svi):
    mnst = GP.manucraft_st
    k = 0 #(k+1) is the indes where sn in mnst #k begin
    k1 = False
    for s in mnst:
        k += 1
        if len(s[:-1].split())>0 and s[0]!="#":
            if sn==s.split()[0]:
                print("The changed param of GP.manucraft_st: ", s.split()[0])
                k1 = True
                break
    if k1==False:
        print("It will generate the param files only if the changed setting is in the original manucarft.")
        sys.exit(0)
    #k end

    mnst[k-1] = "%s\t\t\t\t\t%e #changed\n"%(sn, svi)
    # ads.DEBUG_PRINT_V(0, mnst[k-1])
    file_handle = open(gm_IC_fileA, mode="w")
    for s in mnst:
        file_handle.write(s)
    file_handle.close()

    print("The params_fileA has been wroten.")
    return 0

def write_params_fileB(GP, gmnB):
    file_handle = open(gmnB, mode="w")
    file_handle.write(GP.comment1)
    file_handle.write("##model identity\nmodelId\t\t\t\t\t\t%d\n"%(GP.modelId))
    file_handle.write("##model infomation\nmodelInfo\t\t\t\t\t\t%s\n"%(GP.modelInfo))
    file_handle.write("##model path to load and put\nmodelPath\t\t\t\t\t\t%s\n"%(GP.modelPath))
    file_handle.write("\n####about whole galaxy\n")
    
    if GP.Mass_vir is not None:
        file_handle.write("##virial mass, 1e10*M_\Odot\n%s\t\t\t\t\t\t%e\n"%(GP.Mass_vir_name2, GP.Mass_vir))
    
    if GP.components is not None:
        file_handle.write("##number of components\n%s\t\t\t\t\t\t%d\n"%(GP.components_name2, GP.components))
    
    if 1:
        file_handle.write(GP.comment2 %(
            GP.N_comp1_name2, GP.N_comp1, 
            GP.type_comp1_name2, GP.type_comp1, 
            GP.frac_mass_comp1_name2, GP.frac_mass_comp1, 
            GP.scale_length_comp1_name2, GP.scale_length_comp1, 
            GP.flatx_comp1_name2, GP.flatx_comp1, 
            GP.flaty_comp1_name2, GP.flaty_comp1, 
            GP.flatz_comp1_name2, GP.flatz_comp1, 
            GP.powerS1_name2, GP.powerS1, 
            GP.powerA1_name2, GP.powerA1, 
            GP.powerB1_name2, GP.powerB1, 
            GP.spin_L_name2, GP.spin_L
            # GP.paramsetting_name2, GP.paramsetting
        ))

    file_handle.write(GP.comment3)
    file_handle.close()
    print("The params_fileB has been wroten.")
    return 0

def drive_multi_lists():
    ## use shell to dormain
    return 0

def read_by_first_lineword_from_text(keyword, stringlist, value_type=float):
    value_after = None
    for s in stringlist:
        if (keyword in s) and s[0]!="#":
            v = value_type(s.split()[1])
            print("read keyword:", s, ", value: ", v)
            value_after = v
    return value_after



## io after
def read_output_folder_name( #do not use
    filename="../../../install_and_run/output_folder_name.txt"
)->str:
    fh = open(filename)
    date_str = fh.readlines()[0][:-1]
    print("The output folder is: ", date_str)
    fh.close()
    return date_str

def read_output_folder_name_from_user_settings( #use
    is_only_first=False, 
    filename=user_settings_multi_file
)->str:
    file_handle = open(filename)
    date_str = file_handle.readlines()
    file_handle.close()
    mgs_name = date_str[6].split()
    if is_only_first:
        return mgs_name[0]
    else:
        return mgs_name

def read_fitmassdensity_tag_from_IC_setting_list(name_MG):
    if name_MG=="default":
        filename="../../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/init/IC_setting_list.txt"
        fh = open(filename)
        st = fh.readlines()
        fh.close()
        s = st[3].split() #a line without "\n"
        tag1 = int(s[0]) #the first word of line
        tag2 = int(s[1])
        tag3 = int(s[2])
    else:
        print("Do not read fitmassdensity_tag.")
        tag1 = -1
        tag2 = -1
        tag3 = -1
    return tag1, tag2, tag3

class Galaxy_params_to_record:
    def __init__(self, which_varing=0):
        self.which_varing = which_varing
        self.manucraft_st = None

        #work11_paramsresult
        self.comment0 = "## When reading, vacum line are dismissed. The sign of notes: '#' or '/''/' or '%' or '!'.\n"\
            +"## The value of -1 is usually unknown value.\n"\
            +"\n"\
            +"# ##an example of param\n"\
            +"# example_abcd 			            100	200	#the first is name, the second is value, the left are no use.\n"\
            +"example_efgh			            110.		#no use example\n"\
            +"\n"\
            +"####------------------------------------------------------------------\n"\
            +"####about the global settings\n"\
            +"##softening of particle type\n"\
            +"softening_type_gas	        	5.e-2  		#0: gas\n"\
            +"softening_type_halo	        	5.e-2  		#1: halo\n"\
            +"softening_type_disk	        	2.e-2  		#2: disk\n"\
            +"softening_type_bulge	    	2.e-2  		#3: bulge\n"\
            +"softening_type_stars	    	2.e-2  		#4: stars\n"\
            +"softening_type_bndry	    	2.e-2  		#5: bndry\n"\
            +"\n"\
            +"\n"\
            +"\n"\
            +"## This file records galaxy params by direct calculation and fit.\n"\
            +"####-------------------------------------------------------------------\n"\
            +"####about a component of the galaxy\n"\
            +"##tag name of calculated galaxy snapshot; gm_name\n"\
            +"%s		                %s\n"\
            +"\n"\
            +"##number id of calculated galaxy snapshot; gm_num\n"\
            +"%s		                %d\n"\
            +"\n"\
            +"##path of dealed with snapshot txt; path_snapshot\n"\
            +"%s		                %s\n"\
            +"\n"\
            +"##total number of component1 (type1, halo) particles; N_comp1\n"\
            +"%s		                %d #1000000		#N\n"\
            +"\n"\
            +"####type of a component; type_comp1\n"\
            +"%s		            %d #1		#N\n"\
            +"\n"\
            +"##mass fraction of a component; frac_mass_comp1\n"\
            +"%s	            %e #1. #0.9	#frac\n"\
            +"\n"\
            +"##virial mass of a component; Mass_virial\n"\
            +"%s	        %e #137.0		#sm\n"\
            +"\n"
        self.comment1 = "####Cartesian parameters on xv space; by direct calculation and fitting. "\
            +"The order of params are in mess but the value is corresponded.\n"\
            +"##median scale length of a component; scale_length_comp_median1\n"\
            +"%s	        %e #10.0\n"\
            +"\n"\
            +"##detected flattening of a component; flatx_comp1, flaty_comp1, flatz_comp1\n"\
            +"%s		            %e #1.\n"\
            +"%s		            %e #0.6\n"\
            +"%s		            %e #0.3\n"\
            +"\n"\
            +"##angular moment and spin; spin_L\n"\
            +"%s				    %e #0.\n"\
            +"\n"\
            +"##v_sigma and v_radical; beta_sigma_total\n"\
            +"%s				%e #1.\n"\
            +"\n"\
            +"##detected scale length of a component; scale_length_comp_fit1\n"\
            +"%s	        %e #19.6\n"\
            +"\n"\
            +"##detected scale density of a component; scale_density_comp_fit1\n"\
            +"%s	        %e #1e-3\n"\
            +"\n"\
            +"##xv fit model of a component; fitmodel_name_xv\n"\
            +"%s	        %s #DFXV_doublepowerlaw_exp\n"\
            +"\n"\
            +"##power of powerlaw1 or Einaso_usual of a component; powerS1\n"\
            +"%s		                %e #1.7\n"\
            +"##power of powerlaw2, A, B  of a component; powerA1, powerB1, powerA2, powerB2\n"\
            +"%s		                %e #1.\n"\
            +"%s		                %e #3.\n"\
            +"%s		                %e #3.\n"\
            +"%s		                %e #3.\n"\
            +"\n"
        self.comment2 = "\n"\
            +"\n"\
            +"####Action parameters on AA space; the specific meaning is determined by formula\n"\
            +"The order of params are in mess but the value is corresponded.\n"\
            +"##actions median; Jsum_median, Jl_median, Jm_median, Jn_median\n"\
            +"%s    			%e #10000.0\n"\
            +"%s    			%e #10000.0\n"\
            +"%s    			%e #1000.0\n"\
            +"%s    			%e #1000.0\n"\
            +"##frequencies median; Ol_median, Om_median, On_median\n"\
            +"%s    			%e #1.0\n"\
            +"%s    			%e #1.0\n"\
            +"%s    			%e #1.0\n"\
            +"##flatennings in actions directly; actions_ratio_direct_m, actions_ratio_direct_n\n"\
            +"%s		                %e #0.1\n"\
            +"%s		                %e #0.1\n"\
            +"\n"\
            +"##spin of the processed (use the E, P and L after centerization and triaxialization "\
            +"processing); spin_L_processed_direct\n"\
            +"%s		                %e #0.1\n"\
            +"\n"\
            +"##AA fit model of a component; fitmodel_name_AA\n"\
            +"%s	            %s #DFAA_doublepowerlaw_exp\n"\
            +"\n"\
            +"##actions scales; J1_scale_fit, J2_scale_fit, J3_scale_fit, J4_scale_fit\n"\
            +"%s    			%e #1.0\n"\
            +"%s    			%e #10.0\n"\
            +"%s    			%e #10000.0\n"\
            +"%s    			%e #10000.0\n"\
            +"##log poly coef; poly_coeff_k1_fit, poly_coeff_k2_fit\n"\
            +"%s    			%e #1.0\n"\
            +"%s    			%e #1.0\n"\
            +"##flatennings in actions by fit, to replace frequencies; actions_coef_free_m, actions_coef_free_n\n"\
            +"%s		                %e #0.1\n"\
            +"%s		                %e #0.1\n"\
            +"##power of exponents; powerAA_E1\n"\
            +"%s		                %e #1.\n"\
            +"##power of powerlaw2, A, B  of a component; powerAA_P1_fit, powerAA_P2_fit, powerAA_P3_fit, powerAA_P4_fit\n"\
            +"%s		                %e #1.\n"\
            +"%s		                %e #3.\n"\
            +"%s		                %e #3.\n"\
            +"%s		                %e #3.\n"\
            +"##some other variables by direct calculation, here are spin_L_parameter rotate_sig_parameter "\
            +"beta_parameter and beta_z_parameter before preprocess (P1) and after preprocess (P2)\n"\
            +"%s		                %e #1.\n"\
            +"%s		                %e #1.\n"\
            +"%s		                %e #1.\n"\
            +"%s		                %e #1.\n"\
            +"%s		                %e #1.\n"\
            +"%s		                %e #1.\n"\
            +"%s		                %e #1.\n"\
            +"%s		                %e #1.\n"\
            +"\n"

        self.tag_xv_or_AA = 0
        # self.tag_is_assigned = False #each this class ~ each half fit part ~ each param result file ~ half galaxy
        self.content_assigned1 = None
        self.content_assigned2 = None
        self.paramsresult1 = None
        self.paramsresult2 = None

    # def assign_value_from_MG_to_GP_xv(self, MG, fitresult):
    #     N_fp = np.len(fitresult[0]) #[name_write, value_write], [value_fit, name_fit]
    #     for i in np.arange(N_fp):
    #         for je in self.paramsresult_1:
    #             if je[1]==fitresult[1][i] and je[2]!=2:
    #                 je[0] = fitresult[0][i]
    #     # self.tag_is_assigned = True
    #     return 0

    # def assign_value_to_centent(self, fitmodel, tag_xv_or_AA1):
    #     # if self.tag_is_assigned==tag_xv_or_AA1:
    #     #     print("This class of output galaxy has been assigned value. Exit.")
    #     #     sys.exit(0)
    #     self.content_assigned1 = self.comment1 %(
    #         "a", "b", 
    #         1., 2.
    #     )
    #     self.content_assigned2 = self.comment2 %(
    #         "a", "b", 
    #         1., 2.
    #     )
    #     # self.tag_is_assigned = True
    #     return 0

    def reset_tag_xv_or_AA(self, tag):
        self.tag_xv_or_AA = tag

    def record_gm_params_result_txt(self, path, paramsresult, xvOJ_tag):
        if xvOJ_tag == 1:
            self.paramsresult1 = paramsresult
            # ads.DEBUG_PRINT_V(0, paramsresult["spin_L_old_direct"])
            self.content_assigned1 = self.comment1 %(
                "scale_length_comp1_median_direct", paramsresult["scale_length_comp1_median_direct"], 
                "flatx_comp1_direct", paramsresult["flatx_comp1_direct"], 
                "flaty_comp1_direct", paramsresult["flaty_comp1_direct"], 
                "flatz_comp1_direct", paramsresult["flatz_comp1_direct"], 
                "spin_L_old_direct", paramsresult["spin_L_old_direct"], 
                "beta_sigma_total_direct", paramsresult["beta_sigma_total_direct"], 
                "scale_length_comp1_fit", paramsresult["scale_length_comp1_fit"], 
                "scale_density_comp1_fit", paramsresult["scale_density_comp1_fit"], 
                "fitmodel_name_xv_fit", paramsresult["fitmodel_name_xv_fit"], 
                "powerS1_fit", paramsresult["powerS1_fit"], 
                "powerA1_fit", paramsresult["powerA1_fit"], 
                "powerB1_fit", paramsresult["powerB1_fit"], 
                "powerA2_fit", paramsresult["powerA2_fit"], 
                "powerB2_fit", paramsresult["powerB2_fit"]
            )
            file_handle = open(path, mode="w")
            file_handle.write(self.content_assigned1)
            file_handle.close()
        elif xvOJ_tag == 2:
            self.paramsresult2 = paramsresult
            self.content_assigned2 = self.comment2 %(
                "Jsum_median_direct", paramsresult["Jsum_median_direct"], 
                "Jl_median_direct", paramsresult["Jl_median_direct"], 
                "Jm_median_direct", paramsresult["Jm_median_direct"], 
                "Jn_median_direct", paramsresult["Jn_median_direct"], 
                "Ol_median_direct", paramsresult["Ol_median_direct"], 
                "Om_median_direct", paramsresult["Om_median_direct"], 
                "On_median_direct", paramsresult["On_median_direct"], 
                "actions_ratio_m_direct", paramsresult["actions_ratio_m_direct"], 
                "actions_ratio_n_direct", paramsresult["actions_ratio_n_direct"], 
                "spin_L_processed_direct", paramsresult["spin_L_processed_direct"], 
                "fitmodel_name_AA_fit", paramsresult["fitmodel_name_AA_fit"], 
                "J1_scale_fit", paramsresult["J1_scale_fit"], 
                "J2_scale_fit", paramsresult["J2_scale_fit"], 
                "J3_scale_fit", paramsresult["J3_scale_fit"], 
                "J4_scale_fit", paramsresult["J4_scale_fit"], 
                "poly_coeff_k1_fit", paramsresult["poly_coeff_k1_fit"], 
                "poly_coeff_k2_fit", paramsresult["poly_coeff_k2_fit"], 
                "actions_coef_free_m_fit", paramsresult["actions_coef_free_m_fit"], 
                "actions_coef_free_m_fit", paramsresult["actions_coef_free_m_fit"], 
                "powerAA_E1_fit", paramsresult["powerAA_E1_fit"], 
                "powerAA_P1_fit", paramsresult["powerAA_P1_fit"], 
                "powerAA_P2_fit", paramsresult["powerAA_P2_fit"], 
                "powerAA_P3_fit", paramsresult["powerAA_P3_fit"], 
                "powerAA_P4_fit", paramsresult["powerAA_P4_fit"], 
                "spin_L_parameter_P1_direct", paramsresult["spin_L_parameter_P1_direct"], 
                "rotate_sig_parameter_P1_direct", paramsresult["rotate_sig_parameter_P1_direct"], 
                "beta_parameter_P1_direct", paramsresult["beta_parameter_P1_direct"], 
                "beta_z_parameter_P1_direct", paramsresult["beta_z_parameter_P1_direct"], 
                "spin_L_parameter_P2_direct", paramsresult["spin_L_parameter_P2_direct"], 
                "rotate_sig_parameter_P2_direct", paramsresult["rotate_sig_parameter_P2_direct"], 
                "beta_parameter_P2_direct", paramsresult["beta_parameter_P2_direct"], 
                "beta_z_parameter_P2_direct", paramsresult["beta_z_parameter_P2_direct"]
            )
            file_handle = open(path, mode="a")
            file_handle.write(self.content_assigned2)
            file_handle.close()
        else:
            print("No such tag. Do nothing.")
        return 0

def write_analysis_folder_for_gm_name_tag(gm_name_tag):
    path_gm_data = ""
    path_gm_analys = "what"+"savefig/"+gm_name_tag
    os.makedirs(path_gm_analys, exist_ok=True)
    return path_gm_analys



## settings alternative
def setting_list_scale_length():
    return 0

def setting_list_scale_density(): #not
    return 0

def setting_list_powerlaw1():
    return 0

def setting_list_powerlaw2(): #not
    return 0

def setting_list_power3(): #not
    return 0

def setting_list_axis_ratio_1(): #by comparing simulated snapshots along time
    return 0

def setting_list_axis_ratio_2(): #not
    return 0

def setting_list_spin(): #?? total angular moment and total energy
    return 0

def setting_list_rate_vsigma(): #not
    return 0

def setting_list_total_energy(): #not
    return 0

def setting_list_particles_counts(): #more changings when calculating actions
    return 0

def setting_list_other_profiles(): #not
    return 0



if __name__ == '__main__':

    GP = read_params_to_class()
    gn, sn, sv = read_IC_setting_list()
    if gn=="manucraft":
        print("Reffuse to write the manucraft IC file by this prog. Exit.")
        sys.exit(0)
    N_sv = len(sv)

    for i_file in range(N_sv):
        gm_IC_fileA = totalfolder+"GDDFAA/step1_galaxy_IC_preprocess/step1_set_IC_DDFA/IC_DICE_"+gn+"%d"%(i_file)+".params"
        write_params_fileA(GP, gm_IC_fileA, sn, sv[i_file])

        gm_IC_fileB = totalfolder+"GDDFAA/step1_galaxy_IC_preprocess/step1_set_IC_DDFA/IC_param_"+gn+"%d"%(i_file)+".txt"
        GP_changed = read_params_to_class(gm_IC_fileA)
        write_params_fileB(GP_changed, gm_IC_fileB)



    # # calculate param0
    # # param1 name in fitmodel
    # fitresult1 = {} #here set param name1 and assign value of param0 and param1
    # # assign value
    # # write to comment
    # # run all of all
