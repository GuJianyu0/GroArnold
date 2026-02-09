#!/usr/bin/env python
# -*- coding:utf-8 -*-

# from fileinput import filename
import numpy as np

import analysis_data_distribution as add
import transformation_some as ts
# import triaxialize_galaxy as tg
# import galaxy_models as gm
# import fit_rho_fJ as fff



class Read_galaxy_data:
    '''
    To read galaxy data (in subject of classical galatic dynamics).
    '''

    def __init__(self, path_file):
        self.path_file = path_file
        self.data = np.loadtxt(self.path_file) #float array
        print("Read from `%s` ... Done."%(self.path_file))

        self.system_total_mass = None
        self.system_particles_count = len(self.data)
        self.system_space_dimension = 3

        self.particle_x_coordinates = None #Cartesian
        self.particle_v_velocities = None #Cartesian
        self.particle_current_time = None
        self.particle_IDs = None
        self.particle_type = None
        self.particle_mass = None
        self.particle_energy = None
        self.particle_potential = None
        self.particle_ellipsoidal_ABC = None
        self.particle_otherInfo = None #not limited

        self.particle_orbit_sequence = None #if the file is about a particle in differnet space-time
        self.particle_action_coordinate_type = None
        self.particle_angles = None
        self.particle_actions = None
        self.particle_angleFrequences = None

        self.space_mean_velocities = None #Cartesian #if the file is about space coordinates instead of particles, similarly hereinafter
        self.space_dispersion_velocities = None #Cartesian
        self.space_DF_x = None #Cartesian
        self.space_DF_actions = None #Cartesian
        self.space_other_fields = None #Cartesian #not limited

    def AAAA_read_by_strings(self):
        file_handle = open(self.path_file, mode="r")
        self.data = file_handle.readlines() #string array
        print("Not provided now. Exit.")
        exit(0)

    def AAAA_set_particle_variables(self, 
        col_particle_x_coordinates = None, 
        col_particle_v_velocities = None, 
        col_particle_current_time = None, 
        col_particle_IDs = None, 
        col_particle_type = None, 
        col_particle_mass = None, 
        col_particle_energy = None, 
        col_particle_potential = None, 
        col_particle_ellipsoidal_ABC = None, 
        col_particle_otherInfo = None, 

        col_particle_orbit_sequence = None, 
        col_particle_action_coordinate_type = None, 
        col_particle_angles = None, 
        col_particle_actions = None, 
        col_particle_angleFrequences = None, 

        col_space_mean_velocities = None, 
        col_space_dispersion_velocities = None, 
        col_space_DF_x = None, 
        col_space_DF_actions = None, 
        col_space_other_fields = None
    ):

        if col_particle_x_coordinates is not None:
            self.particle_x_coordinates = self.data[:, col_particle_x_coordinates:col_particle_x_coordinates+self.system_space_dimension]
        if col_particle_v_velocities is not None:
            self.particle_v_velocities = self.data[:, col_particle_v_velocities:col_particle_v_velocities+self.system_space_dimension]
        if col_particle_current_time is not None:
            self.particle_current_time = self.data[:, col_particle_current_time]
        if col_particle_IDs is not None:
            self.particle_IDs = self.data[:, col_particle_IDs]
        if col_particle_type is not None:
            self.particle_type = self.data[:, col_particle_type]
        if col_particle_mass is not None:
            self.particle_mass = self.data[:, col_particle_mass]
        if col_particle_current_time is not None:
            self.particle_current_time = self.data[:, col_particle_current_time]
        if col_particle_energy is not None:
            self.particle_energy = self.data[:, col_particle_energy]
        if col_particle_potential is not None:
            self.particle_potential = self.data[:, col_particle_potential]
        if col_particle_ellipsoidal_ABC is not None:
            self.particle_ellipsoidal_ABC = self.data[:, col_particle_ellipsoidal_ABC:col_particle_ellipsoidal_ABC+self.system_space_dimension]
        if col_particle_otherInfo is not None:
            self.particle_otherInfo = self.data[:, col_particle_otherInfo:col_particle_otherInfo+6] #free length

        if col_particle_orbit_sequence is not None:
            self.particle_orbit_sequence = self.data[:, col_particle_orbit_sequence:col_particle_orbit_sequence+(1+self.system_space_dimension)*2] #t,x(dim), H,v(dim)
        if col_particle_action_coordinate_type is not None:
            self.particle_action_coordinate_type = self.data[:, col_particle_action_coordinate_type+1] #cannonical coordinate type, orbit type by actions
        if col_particle_actions is not None:
            self.particle_actions = self.data[:, col_particle_actions:col_particle_actions+self.system_space_dimension]
        if col_particle_angles is not None:
            self.particle_angles = self.data[:, col_particle_angles:col_particle_angles+self.system_space_dimension]
        if col_particle_angleFrequences is not None:
            self.particle_angleFrequences = self.data[:, col_particle_angleFrequences:col_particle_angleFrequences+self.system_space_dimension]
            
        if col_space_mean_velocities is not None:
            self.space_mean_velocities = self.data[:, col_space_mean_velocities:col_space_mean_velocities+self.system_space_dimension]
        if col_space_dispersion_velocities is not None:
            self.space_dispersion_velocities = self.data[:, col_space_dispersion_velocities:col_space_dispersion_velocities+self.system_space_dimension]
        if col_space_DF_x is not None:
            self.space_DF_x = self.data[:, col_space_DF_x]
        if col_space_DF_actions is not None:
            self.space_DF_actions = self.data[:, col_space_DF_actions]
        if col_space_other_fields is not None:
            self.space_other_fields = self.data[:, col_space_other_fields:col_space_other_fields+3] #free length

        return 

    def write_numpy_savetxt(self, path_write, data_write, notes=None):
        if notes is None:
            np.savetxt(path_write, data_write)
        else: #??
            file_handle = open(path_write, mode="w")
            file_handle.write("## "+notes[-1]+"\n") #the last note is put at the first line
            N_dw = len(data_write)
            N_notes = len(notes)
            for i in np.arange(N_dw):
                note = ""
                if i<N_notes-1:
                    note = notes[i]
                file_handle.write("%s"%(data_write[i])) #bad "[]"
                file_handle.write(" # "+note+"\n") #each note is put in a line
            file_handle.close()
        print("Write to `%s` ... Done."%(path_write))



class Read_data_galaxy: #old class
    def __init__(self, m, gb="./", gmn="galaxy_general/", wc=0):
        self.model = m
        self.whatcannonical = wc #bad setting
        self.galaxybox_name = gb
        self.galaxymodel_name = gmn
        # self.dataread = 0.
        # self.xdata_eff = 0.
        # self.ydata_eff = 0.
        # self.xdata_eff_input = 0.
        # self.ydata_eff_input = 0.
        if self.whatcannonical == 2:  # mass density
            self.col_x1 = 0  # general coordinate part1
            self.col_x2 = 3  # general coordinate part2
            self.col_y1 = 16  # DF
            self.col_y2 = -1  # other
            self.col_I = 20
        elif self.whatcannonical == 5:  # action probability density
            self.col_x1 = 6  # general coordinate or frequency
            self.col_x2 = 9  # general momentum
            self.col_y1 = 17  # DF
            self.col_y2 = -1  # other
            self.col_I = 20
        else:
            print("class read_data_galaxy: No such type of data provided!")
            exit(0)

    def data_original_simulationoOutput(self, snapshot_id):
        self.middle_name = "txt/"
        self.datafile_name = "snapshot_%03d.txt" % (snapshot_id)
        # self.col_x1 = 0  # general coordinate part1, x
        # self.col_x2 = 3  # general coordinate part2, v
        # self.col_y1 = 0  # should be set
        # self.col_y2 = 0  # should be set
        filename = self.galaxybox_name+self.galaxymodel_name + \
            self.middle_name+self.datafile_name
        dataread = np.array(np.loadtxt(filename, dtype=float))
        # dataread = np.array(np.where(np.isfinite(dataread), dataread, 0.)) #set value who is not finite to 0.
        print("Read %s ... Done." %(self.datafile_name))
        return dataread

    def data_original_AAOutput(self, snapshot_id, method_and_tags=""):
        self.middle_name = "aa/"
        self.datafile_name = "snapshot_%03d.action.method_%s.txt" % (snapshot_id, method_and_tags)
        # self.col_x1 = 0  # general coordinate part1, x
        # self.col_x2 = 3  # general coordinate part2, v
        # self.col_y1 = 0  # should be set
        # self.col_y2 = 0  # should be set
        filename = self.galaxybox_name+self.galaxymodel_name + \
            self.middle_name+self.datafile_name
        dataread = np.array(np.loadtxt(filename, dtype=float))
        # dataread = np.array(np.where(np.isfinite(dataread), dataread, 0.)) #set value who is not finite to 0.
        print("Read %s ... Done." %(self.datafile_name))
        return dataread

    def data_original_DF(self, snapshot_id, method_and_tags=""):
        self.middle_name = "aa/"
        self.datafile_name = "snapshot_%03d.secondhand_%s.txt" % (snapshot_id, method_and_tags)
        filename = self.galaxybox_name+self.galaxymodel_name + \
            self.middle_name+self.datafile_name
        dataread = np.array(np.loadtxt(filename, dtype=float))
        # dataread = np.array(np.where(np.isfinite(dataread), dataread, 0.)) #set value who is not finite to 0.
        print("Read %s ... Done." %(self.datafile_name))
        return dataread

    def data_original_NDFA(self, snapshot_id, method_and_tags=""):
        self.middle_name = "aa/"
        self.datafile_name = "snapshot_%03d.secondhand_%s.txt" % (snapshot_id, method_and_tags)
        filename = self.galaxybox_name+self.galaxymodel_name + \
            self.middle_name+self.datafile_name
        dataread = np.loadtxt(filename, dtype=float)
        # dataread = np.array(np.where(np.isfinite(dataread), dataread, 0.)) #set value who is not finite to 0.
        print("Read %s ... Done." %(self.datafile_name))
        return dataread

    def data_original_AAOutput_pseudoPeriod(self, snapshot_id):
        self.middle_name = "aa/"
        self.datafile_name = "snapshot_%03d.action.method_directorbit.txt" % (snapshot_id)
        # self.datafile_name = "snapshot_%03d.action.directorb_bak.txt" % (snapshot_id)
        # self.col_x1 = 0  # general coordinate part1, x
        # self.col_x2 = 3  # general coordinate part2, v
        # self.col_y1 = 0  # should be set
        # self.col_y2 = 0  # should be set
        filename = self.galaxybox_name+self.galaxymodel_name + \
            self.middle_name+self.datafile_name
        dataread = np.array(np.loadtxt(filename, dtype=float))
        # dataread = np.array(np.where(np.isfinite(dataread), dataread, 0.)) #set value who is not finite to 0.
        print("Read %s ... Done." %(self.datafile_name))
        return dataread

    def data_secondhand_snapshot(self, snapshot_id, wc=0):
        self.whatcannonical = wc
        if self.whatcannonical == 2:  # mass density
            self.col_x1 = 0  # general coordinate part1
            self.col_x2 = 3  # general coordinate part2
            self.col_y1 = 16  # DF
            self.col_y2 = -1  # other
            self.col_I = 20
        elif self.whatcannonical == 5:  # action probability density
            self.col_x1 = 6  # general coordinate or frequency
            self.col_x2 = 9  # general momentum
            self.col_y1 = 17  # DF
            self.col_y2 = -1  # other
            self.col_I = 20
        else:
            print("class read_data_galaxy: No such type of data provided!")
            exit(0)

        self.middle_name = "aa/"
        self.datafile_name = "snapshot_%03d.secondhand_%03d.txt" % (
            snapshot_id, self.whatcannonical)
        filename = self.galaxybox_name+self.galaxymodel_name + \
            self.middle_name+self.datafile_name
        # add.DEBUG_PRINT_V(1,"\n\n\n\n",filename)
        dataread = np.array(np.loadtxt(filename, dtype=float))
        # dataread = np.array(np.where(np.isfinite(dataread), dataread, 0.)) #set value who is not finite to 0.
        return dataread

    def data_sample_screen(self, d, x_down=0., x_up=np.inf,
                           is_abs=False, is_scaled=False, is_logx=False, is_logy=False, 
                           wc=2, col_xCF=6,col_xM=9,col_y=-4
    ):
        xdata_eff = None
        ydata_eff = None
        if self.whatcannonical == 2:  # mass density
            ## process: read
            xdata = d[:, self.col_x1:self.col_x1+6]  # hstack vvv
            ydata = d[:, self.col_y1]

            ## process: range
            # x0 = self.model.params_dict['length_scale'][0]
            # y0 = self.model.params_dict['density_scale'][0]
            # x_down = x0*scalerate_down
            # x_up = x0*scalerate_up
            # xdata_eff, xdata_listCondition, xdata_listConditionNot = add.screen_boundary(
            #     abs(xdata), x_down, x_up)
            # ydata_eff = ydata[xdata_listCondition]
            # print("xdata_listConditionNot: ", len(xdata_listConditionNot))
            # print("After screening without other optional process, mean and median of abs x, ... of abs y: ", \
            #     np.mean(abs(xdata_eff), axis=0), np.median(abs(xdata_eff), axis=0),
            #     np.mean(abs(ydata_eff), axis=0), np.median(abs(ydata_eff), axis=0))
            xdata_eff = xdata
            ydata_eff = ydata

            ## process: optional
            if is_abs == True:
                xdata_eff = abs(xdata_eff)
                # ydata_eff = abs(ydata_eff)
            # if is_scaled == True:
            #     xdata_eff /= x0
            #     ydata_eff /= y0
            if is_logx == True:
                xdata_eff = np.log(xdata_eff)/np.log(10) #f/Uint
            if is_logy == True:
                ydata_eff = np.log(ydata_eff)/np.log(10)

        elif self.whatcannonical == 5:  # action probability density
            ## process: read
            xdata = np.hstack(
                (d[:, self.col_x1:self.col_x1+3], d[:, self.col_x2:self.col_x2+3]))
            ydata = d[:, self.col_y1]

            ## process: range
            xdata_eff = xdata
            ydata_eff = ydata
            # x0 = self.model.params_dict['action_scale'][0]
            # y0 = x0**-3
            # x_down = x0*scalerate_down
            # x_up = x0*scalerate_up
            # xdata_eff, xdata_listCondition, xdata_listConditionNot = add.screen_boundary(
            #     abs(xdata), x_down, x_up)
            # ydata_eff = ydata[xdata_listCondition]
            # print("xdata_listConditionNot: ", len(xdata_listConditionNot))
            # print("After screening without other optional process, mean and median of x, ... of y: ", np.mean(xdata_eff, axis=0), np.median(xdata_eff, axis=0),
            #       np.mean(ydata_eff, axis=0), np.median(ydata_eff, axis=0))

            ## process: optional
            if is_abs == True:
                xdata_eff = abs(xdata_eff)
                # ydata_eff = abs(ydata_eff)
            # if is_scaled == True:
            #     xdata_eff /= x0
            #     ydata_eff /= y0
            if is_logx == True:
                xdata_eff = np.log(xdata_eff)/np.log(10)
            if is_logy == True:
                ydata_eff = np.log(ydata_eff)/np.log(10)

        else:
            print("class read_data_galaxy: No such type of data provided!")
            exit(0)

        if is_logx==False:
            xdata_J, cl, cnl = add.screen_boundary_PM(xdata_eff[:,3:6], x_down, x_up, is_abs=True)
            add.DEBUG_PRINT_V(1, xdata_eff[0], ydata[0])
            # add.DEBUG_PRINT_V(1, xdata_eff.shape, ydata_eff.shape)
        return xdata_eff[cl], ydata_eff[cl]

    def data_sample_combination(self, xd, yd, comb, *p):
        x = 0.
        xp = 0.
        y = yd
        if self.whatcannonical == 2:
            x = xd[:, 0:3]
            xp = xd[:, 3:6]
        elif self.whatcannonical == 5:
            x = xd[:, 3:6]
            xp = xd[:, 0:3]
        else:
            print("class read_data_galaxy: No such type of data provided!")
            exit(0)
            return 0., 0.
        s0, s1 = x.shape
        if comb == "nothing":
            pass
        elif comb == "radius":
            xdata_eff_input = add.norm_l(x, 0, axis=1, l=2)  # r(x,y,z)
            # add.DEBUG_PRINT_V(0,xdata_eff_input.shape)
        elif comb == "radius_ratio":
            x_ = np.zeros(s0)
            for i in range(s1):
                x_ += (x[:, i]*p[i])**2
            xdata_eff_input = x_**0.5  # norm_2(x*qx,y*yy,z*qz)
        elif comb == "norm_l_inf":
            xdata_eff_input = np.sum(x, axis=1)  # x+y+z
        elif comb == "oscillator_OJ":
            x_ = np.zeros(s0)
            for i in range(s1):
                x_ += (x[:, i]*xp[:, i])
            xdata_eff_input = x_/xp[:, 0]  # OJ
        ydata_eff_input = y
        return xdata_eff_input, ydata_eff_input

def io_to_galaxy_data(filename, cols):
    data = np.loadtxt(filename)
    return data[:, cols]

def read_actions(filename, bd=np.inf, bd_min=None, is_angles=False, actionmethod="AA_TF_DP"):
    print("reading ...")
    RG = Read_galaxy_data(filename)
    RG.AAAA_set_particle_variables(
        col_particle_IDs=7-1, col_particle_mass=8-1
    )
    print("reading ... done.")

    data = RG.data
    mass = RG.particle_mass
    IDs = RG.particle_IDs
    Dim = 3
    iast = 28
    adur = 10
    AA_TF_FP = data[:, iast+adur*0:iast+adur*0+adur]
    AA_OD_FP = data[:, iast+adur*1:iast+adur*1+adur] #none
    AA_GF_FP = data[:, iast+adur*2:iast+adur*2+adur] #none
    iast += adur*5 # = 78
    AA_TF_DP = data[:, iast+adur*0:iast+adur*0+adur]
    AA_OD_DP = data[:, iast+adur*1:iast+adur*1+adur]
    AA_GF_DP = data[:, iast+adur*2:iast+adur*2+adur] #none

    AA_method = None
    if actionmethod=="AA_TF_DP":
        AA_method = AA_TF_DP
    else:
        print("Not provided. Tmp.")
        exit(0)
    Act = AA_method[:, 0:3]
    Ang = AA_method[:, 3+1:7]
    Fre = AA_method[:, 7:10]
    add.DEBUG_PRINT_V(1, AA_TF_DP.shape, Act.shape, Fre.shape)
    AA = None
    if not is_angles:
        AA = np.hstack((Act, Fre))
    else:
        AA = np.hstack((Act, Fre, Ang))
    cols = [0,1,2]
    bd_min_use = 1./bd
    if bd_min is not None:
        bd_min_use = bd_min
    AA_cl, cl, cln = add.screen_boundary_some_cols(AA, cols, bd_min_use, bd, value_discard=bd*1e4)
    xv = data[cl,0:6]
    add.DEBUG_PRINT_V(1, np.shape(xv), np.shape(AA_cl), "datashape")
    return xv, AA_cl, mass



# download AGAMA, GALA, galpy, about orbit fractional
# python files to wrap: 
# RW data of CMGD, models of CMGD, deal with snapshot, simple transformations, 
# simple statistics, simple algorithm such as KDTree and interpolation, 
# fit, plot and compare, calculate small categories about CMGD

if __name__ == '__main__':

    path_file = "../../step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
        +"galaxy_general_4_EinastoUsual_triaxial_soft5.0_count1e4/txt/snapshot_5000.txt"
    RG = Read_galaxy_data(path_file)
    RG.AAAA_set_particle_variables(col_particle_x_coordinates=0, 
        col_particle_v_velocities=3, col_particle_mass=8)
    x = RG.particle_x_coordinates
    v = RG.particle_v_velocities
    m = RG.particle_mass
    N_dim = RG.system_space_dimension
    N_ptc = RG.system_particles_count
    xv = np.hstack((x, v))

    add.DEBUG_PRINT_V(1, np.shape(x), np.shape(m), np.shape(N_ptc))
