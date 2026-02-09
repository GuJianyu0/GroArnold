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

MaxGalaxyComponents = 10



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
            +"##instructions:\n"\
            +"# $T_\mathrm{relaxCmpl} = N/ln(\lambda N)T_\mathrm{dyn}, \n"\
            +"# T_\mathrm{dyn} = \sqrt{R_s^3/(GM)}$;\n"\
            +"# $R_s/N^(1/2) <= \epsilon_\mathrm{softeningShould} <= R_s/N^(1/3)$.\n"\
            +"\n"\
            +"\n"\
            +"\n"\
            +"####-------------------------------------------------------------------\n"\
            +"####about a component of the galaxy\n"
            # +"##total number of component1 (type1, halo) particles; N_comp1\n"\
            # +"# N_comp1		                10000		#N\n"\
            # +"%s		                %d #1000000		#N\n"\
            # +"##total number of component2 (type0, gas) particles; N_comp2\n"\
            # +"#N_comp2		            2000		#N\n"\
            # +"\n"\
            # +"####type of a component; type_comp1\n"\
            # +"%s		            %d #1		#N\n"\
            # +"# type_comp2		            2		#N\n"\
            # +"\n"\
            # +"##mass fraction of a component; frac_mass_comp1\n"\
            # +"%s	            %e #1. #0.9	#frac\n"\
            # +"# frac_mass_comp2	        0.06		#frac\n"\
            # +"\n"\
            # +"####expected parameters; unused\n"\
            # +"##softening of a component; unused\n"\
            # +"softening_comp1	            0.05		#soft\n"\
            # +"# softening_comp2	        0.02		#soft\n"\
            # +"\n"\
            # +"##scale length of a component; scale_length_comp1; unused\n"\
            # +"%s	        %e #19.6		#sl\n"\
            # +"# scale_length_comp2	    0.		#sl\n"\
            # +"\n"\
            # +"## falttening of a component; flatx_comp1, flaty_comp1, flatz_comp1; unused\n"\
            # +"%s		            %e #1.\n"\
            # +"%s		            %e #0.6\n"\
            # +"%s		            %e #0.3\n"\
            # +"\n"\
            # +"##setted power of one powerlaw or EinasoUsual profile of a component; powerS1; unused\n"\
            # +"%s		                %e #1.7\n"\
            # +"##setted powers of halo profile of a component; powerA1, powerB1; unused but recorded\n"\
            # +"%s		                %e #1.\n"\
            # +"%s		                %e #3.\n"\
            # +"\n"\
            # +"##angular moment and spin; spin_L\n"\
            # +"%s				%e #0.\n"\
            # +"\n"\
            # +"\n"\
            # +"\n"\
            # +"##cold down method to generate\n"\
            # +"mass1					    0.\n"\
            # +"cold_alpha1				    1.0\n"\
            # +"cold_alphamax1				3.0\n"\
            # +"v_sigma_cold1				    30.0\n"\
            # +"seed1					    876543\n"\
            # +"\n"

        self.comment3 = ""

        self.modelId = 1 #default
        self.modelInfo = "_type-halo_theExpectedValues_values_End" #default
        self.modelPath = "galaxy_general/" #default
        
        # --- softening of particle type (Gadget types 0..5) ---
        # If present in source params, they should override the defaults in comment2.
        # Keep None if not found (so the existing default comment2 remains unchanged).
        self.softening_type_gas   = None  # type 0
        self.softening_type_halo  = None  # type 1
        self.softening_type_disk  = None  # type 2
        self.softening_type_bulge = None  # type 3
        self.softening_type_stars = None  # type 4
        self.softening_type_bndry = None  # type 5

        self.Mass_vir = None
        self.Mass_vir_name1 = "m200"
        self.Mass_vir_name2 = "Mass_vir"

        self.components = None
        self.components_name1 = None
        self.components_name2 = "components"

        self.spin_L = None
        self.spin_L_name1 = "lambda"
        self.spin_L_name2 = "spin_L"

        #arrays for up to MaxGalaxyComponents components
        self.profile_model = [None]*MaxGalaxyComponents
        self.N_comp = [None]*MaxGalaxyComponents
        self.type_comp = [None]*MaxGalaxyComponents
        self.frac_mass_comp = [None]*MaxGalaxyComponents
        self.scale_length_comp = [None]*MaxGalaxyComponents
        self.flatx_comp = [None]*MaxGalaxyComponents
        self.flaty_comp = [None]*MaxGalaxyComponents
        self.flatz_comp = [None]*MaxGalaxyComponents
        self.alpha_struct = [None]*MaxGalaxyComponents
        self.beta_struct  = [None]*MaxGalaxyComponents
        self.paramsetting = [None]*MaxGalaxyComponents

        # self.N_comp1 = None
        # self.N_comp1_name1 = "npart1"
        # self.N_comp1_name2 = "N_comp1"

        # self.type_comp1 = None
        # self.type_comp1_name1 = None
        # self.type_comp1_name2 = "type_comp1"

        # self.frac_mass_comp1 = None
        # self.frac_mass_comp1_name1 = "mass_frac1"
        # self.frac_mass_comp1_name2 = "frac_mass_comp1"

        # self.scale_length_comp1 = None
        # self.scale_length_comp1_name1 = "scale_length1"
        # self.scale_length_comp1_name2 = "scale_length_comp1"

        # self.flatx_comp1 = None
        # self.flatx_comp1_name1 = "flatx1"
        # self.flatx_comp1_name2 = "flatx_comp1"

        # self.flaty_comp1 = None
        # self.flaty_comp1_name1 = "flaty1"
        # self.flaty_comp1_name2 = "flaty_comp1"

        # self.flatz_comp1 = None
        # self.flatz_comp1_name1 = "flatz1"
        # self.flatz_comp1_name2 = "flatz_comp1"

        # self.powerS1 = None
        # self.powerS1_name1 = None
        # self.powerS1_name2 = "powerS1"

        # self.powerA1 = None
        # self.powerA1_name1 = "alpha_struct1"
        # self.powerA1_name2 = "powerA1"

        # self.powerB1 = None
        # self.powerB1_name1 = "beta_struct1"
        # self.powerB1_name2 = "powerB1"

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

    # for s in st:
    #     if (GP.Mass_vir_name1 in s) and s[0]!="#":
    #         n = float(s.split()[1])
    #         GP.Mass_vir = n

    #     GP.components = 1

    #     if (GP.N_comp1_name1 in s) and s[0]!="#":
    #         n = int(s.split()[1])
    #         GP.N_comp1 = n

    #     GP.type_comp1 = 1

    #     if (GP.frac_mass_comp1_name1 in s) and s[0]!="#":
    #         n = float(s.split()[1])
    #         GP.frac_mass_comp1 = n

    #     if (GP.scale_length_comp1_name1 in s) and s[0]!="#":
    #         n = float(s.split()[1])
    #         GP.scale_length_comp1 = n

    #     if (GP.flatx_comp1_name1 in s) and s[0]!="#":
    #         n = float(s.split()[1])
    #         GP.flatx_comp1 = n

    #     if (GP.flaty_comp1_name1 in s) and s[0]!="#":
    #         n = float(s.split()[1])
    #         GP.flaty_comp1 = n

    #     if (GP.flatz_comp1_name1 in s) and s[0]!="#":
    #         n = float(s.split()[1])
    #         GP.flatz_comp1 = n

    #     GP.powerS1 = 1.7 #default

    #     GP.powerA1 = 1. #default
    #     if (GP.powerA1_name1 in s) and s[0]!="#":
    #         n = float(s.split()[1])
    #         GP.powerA1 = n #changed

    #     GP.powerB1 = 3. #default
    #     if (GP.powerB1_name1 in s) and s[0]!="#":
    #         n = float(s.split()[1])
    #         GP.powerA1 = n #changed

    #     if (GP.spin_L_name1 in s) and s[0]!="#":
    #         n = float(s.split()[1])
    #         GP.spin_L = n

    #     if (GP.paramsetting_name1 in s) and s[0]!="#":
    #         n = float(s.split()[1])
    #         GP.paramsetting = n

    # helpers
    def _get_val(name, cast=float):
        for s in st:
            if s.startswith("#"):
                continue
            tok = s.split()
            if len(tok) >= 2 and tok[0] == name:
                try:
                    return cast(tok[1])
                except:
                    return None
        return None

    def _set_softening_by_type(ptype, val):
        # ptype: Gadget particle type 0..5
        if val is None:
            return
        if ptype == 0 and GP.softening_type_gas is None:
            GP.softening_type_gas = val
        elif ptype == 1 and GP.softening_type_halo is None:
            GP.softening_type_halo = val
        elif ptype == 2 and GP.softening_type_disk is None:
            GP.softening_type_disk = val
        elif ptype == 3 and GP.softening_type_bulge is None:
            GP.softening_type_bulge = val
        elif ptype == 4 and GP.softening_type_stars is None:
            GP.softening_type_stars = val
        elif ptype == 5 and GP.softening_type_bndry is None:
            GP.softening_type_bndry = val

    # Global: Mass_vir (from DICE m200 if present)
    GP.Mass_vir = _get_val(GP.Mass_vir_name1, float)
    if GP.Mass_vir is None:
        GP.Mass_vir = _get_val(GP.Mass_vir_name2, float)

    # spin (lambda) if present
    GP.spin_L = _get_val(GP.spin_L_name1, float)

    # ---------------------------------------------------------------------
    # Softening: read from file if present
    #  (A) explicit per-type keys: softening_type_*
    #  (B) fallback: per-component softening{i} mapped by type{i} (0..5)
    # ---------------------------------------------------------------------
    GP.softening_type_gas   = _get_val("softening_type_gas", float)
    GP.softening_type_halo  = _get_val("softening_type_halo", float)
    GP.softening_type_disk  = _get_val("softening_type_disk", float)
    GP.softening_type_bulge = _get_val("softening_type_bulge", float)
    GP.softening_type_stars = _get_val("softening_type_stars", float)
    GP.softening_type_bndry = _get_val("softening_type_bndry", float)

    # Detect components (MaxGalaxyComponents) by presence of nparti or mass_fraci
    comp_count = 0
    for i in range(1, MaxGalaxyComponents+1):
        ni = _get_val(f"npart{i}", int)
        fi = _get_val(f"mass_frac{i}", float)
        ti = _get_val(f"type{i}", int)
        if ni is None and fi is None and ti is None:
            continue  # not defined -> skip
        idx = i-1
        
        GP.N_comp[idx] = ni if ni is not None else 0
        GP.frac_mass_comp[idx] = fi if fi is not None else 0.0
        GP.type_comp[idx] = ti if ti is not None else 1  # default halo

        # fallback softening: "softening{i}" (e.g. softening1) mapped by type{i}
        si = _get_val(f"softening{i}", float)
        if si is not None:
            _set_softening_by_type(GP.type_comp[idx], si)

        #optional geometric / profile parameters if present
        GP.scale_length_comp[idx] = _get_val(f"scale_length{i}", float)
        GP.flatx_comp[idx] = _get_val(f"flatx{i}", float)
        GP.flaty_comp[idx] = _get_val(f"flaty{i}", float)
        GP.flatz_comp[idx] = _get_val(f"flatz{i}", float)
        GP.alpha_struct[idx] = _get_val(f"alpha_struct{i}", float)
        GP.beta_struct[idx]  = _get_val(f"beta_struct{i}", float)
        GP.profile_model[idx]        = _get_val(f"profile_model{i}", int)
        
        comp_count += 1

    GP.components = comp_count if comp_count>0 else None
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
    # ads.DEBUG_PRINT_V(0, mnst)
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
        file_handle.write("##virial mass, 1e10*M_\\Odot\n%s\t\t\t\t\t\t%e\n"%(GP.Mass_vir_name2, GP.Mass_vir))
    
    if GP.components is not None:
        file_handle.write("##number of components\n%s\t\t\t\t\t\t%d\n"%(GP.components_name2, GP.components))

    # if 1:
    #     file_handle.write(GP.comment2 %(
    #         GP.N_comp1_name2, GP.N_comp1, 
    #         GP.type_comp1_name2, GP.type_comp1, 
    #         GP.frac_mass_comp1_name2, GP.frac_mass_comp1, 
    #         GP.scale_length_comp1_name2, GP.scale_length_comp1, 
    #         GP.flatx_comp1_name2, GP.flatx_comp1, 
    #         GP.flaty_comp1_name2, GP.flaty_comp1, 
    #         GP.flatz_comp1_name2, GP.flatz_comp1, 
    #         GP.powerS1_name2, GP.powerS1, 
    #         GP.powerA1_name2, GP.powerA1, 
    #         GP.powerB1_name2, GP.powerB1, 
    #         GP.spin_L_name2, GP.spin_L
    #         # GP.paramsetting_name2, GP.paramsetting
    #     ))

    #softening_type and Sanders TACT is in GP.comment2
    # file_handle.write(GP.comment2)

    # Replace ONLY the softening_type_* values in GP.comment2 if we have parsed values.
    # Leave other lines untouched.
    comment2_lines = GP.comment2.splitlines(True)  # keep line breaks
    for ii, line in enumerate(comment2_lines):
        if line.lstrip().startswith("softening_type_gas") and GP.softening_type_gas is not None:
            comment2_lines[ii] = "softening_type_gas\t        \t%e  \t\t#0: gas\n" % (GP.softening_type_gas)
        elif line.lstrip().startswith("softening_type_halo") and GP.softening_type_halo is not None:
            comment2_lines[ii] = "softening_type_halo\t        \t%e  \t\t#1: halo\n" % (GP.softening_type_halo)
        elif line.lstrip().startswith("softening_type_disk") and GP.softening_type_disk is not None:
            comment2_lines[ii] = "softening_type_disk\t        \t%e  \t\t#2: disk\n" % (GP.softening_type_disk)
        elif line.lstrip().startswith("softening_type_bulge") and GP.softening_type_bulge is not None:
            comment2_lines[ii] = "softening_type_bulge\t    \t%e  \t\t#3: bulge\n" % (GP.softening_type_bulge)
        elif line.lstrip().startswith("softening_type_stars") and GP.softening_type_stars is not None:
            comment2_lines[ii] = "softening_type_stars\t    \t%e  \t\t#4: stars\n" % (GP.softening_type_stars)
        elif line.lstrip().startswith("softening_type_bndry") and GP.softening_type_bndry is not None:
            comment2_lines[ii] = "softening_type_bndry\t    \t%e  \t\t#5: bndry\n" % (GP.softening_type_bndry)
    file_handle.write("".join(comment2_lines))

    #per-component block
    file_handle.write("####-------------------------------------------------------------------\n")
    file_handle.write("####components (looped)\n")
    if GP.components is None:
        file_handle.write("# No components detected in source params.\n")
    else:
        for i in range(GP.components):
            ii = i+1
            file_handle.write(f"\n## Component {ii}\n")
            if GP.N_comp[i] is not None:
                file_handle.write(f"N_comp{ii}\t\t\t\t\t\t{int(GP.N_comp[i])}\n")
            if GP.type_comp[i] is not None:
                file_handle.write(f"type_comp{ii}\t\t\t\t\t{int(GP.type_comp[i])}\n")
            if GP.frac_mass_comp[i] is not None:
                file_handle.write(f"frac_mass_comp{ii}\t\t\t\t{GP.frac_mass_comp[i]:.8e}\n")
            if GP.scale_length_comp[i] is not None:
                file_handle.write(f"scale_length_comp{ii}\t\t\t{GP.scale_length_comp[i]:.8e}\n")
            if GP.flatx_comp[i] is not None:
                file_handle.write(f"flatx_comp{ii}\t\t\t\t{GP.flatx_comp[i]:.8e}\n")
            if GP.flaty_comp[i] is not None:
                file_handle.write(f"flaty_comp{ii}\t\t\t\t{GP.flaty_comp[i]:.8e}\n")
            if GP.flatz_comp[i] is not None:
                file_handle.write(f"flatz_comp{ii}\t\t\t\t{GP.flatz_comp[i]:.8e}\n")
            if GP.profile_model[i] is not None:
                file_handle.write(f"profile_model{ii}\t\t\t\t\t\t{int(GP.profile_model[i])}\n")
            if GP.alpha_struct[i] is not None:
                file_handle.write(f"alpha_struct{ii}\t\t\t\t{GP.alpha_struct[i]:.8e}\n")
            if GP.beta_struct[i] is not None:
                file_handle.write(f"beta_struct{ii}\t\t\t\t{GP.beta_struct[i]:.8e}\n")

    if GP.spin_L is not None:
        file_handle.write(f"\n{GP.spin_L_name2}\t\t\t\t\t\t{GP.spin_L:.8e}\n")

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
    # is_only_first=False, 
    filename=user_settings_multi_file
)->str:
    file_handle = open(filename)
    date_str = file_handle.readlines()
    file_handle.close()
    mgs_name = date_str[6].split() #a list
    # if is_only_first:
    #     return mgs_name[0]
    # else:
    #     return mgs_name
    return mgs_name

def read_fitmassdensity_tag(name_MG):
    ads.DEBUG_PRINT_V(1, name_MG, "name_MG")
    if name_MG=="default":
        filename="../../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/init/IC_setting_list.txt"
        fh = open(filename)
        st = fh.readlines()
        fh.close()
        s = st[3].split() #a line without "\n"
        ads.DEBUG_PRINT_V(1, s, "st[3].split()")
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
            +"##xv fit model of a component; fitmodelfunc_name_xv\n"\
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
            +"##AA fit model of a component; fitmodelfunc_name_AA\n"\
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

    # def assign_value_to_centent(self, fitmodelfunc, tag_xv_or_AA1):
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
                "fitmodelfunc_name_xv_fit", paramsresult["fitmodelfunc_name_xv_fit"], 
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
                "fitmodelfunc_name_AA_fit", paramsresult["fitmodelfunc_name_AA_fit"], 
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
    # # param1 name in fitmodelfunc
    # fitresult1 = {} #here set param name1 and assign value of param0 and param1
    # # assign value
    # # write to comment
    # # run all of all
