#!/usr/bin/env python
# -*- coding:utf-8 -*-

import h5py  
import numpy as np
import tables
import io, sys
reload(sys)
sys.setdefaultencoding('utf-8')



#### basic algrithm functions
def heap_adjust(parent,heap): #to updata nodes and ajust
    child=2*parent+1
    while len(heap)>child:
        if child+1<len(heap) and heap[child+1]<heap[child]:
            child+=1
        if heap[parent]<=heap[child]:
            break
        heap[parent],heap[child]=heap[child],heap[parent]
        parent,child=child,child*2+1
 
def find_median(nums):
    k=len(nums)//2 +1
    heap=nums[:k]
    for i in range(k,-1,-1): #前n/2个元素建堆
        heap_adjust(i,heap)
    for j in range(k,len(nums)):
        if nums[j]>heap[0]:
            heap[0]=nums[j]
            heap_adjust(0,heap)
    #奇数时是最中间的数，偶数时是最中间两数的均值
    return heap[0] if len(nums)%2==1 else float(heap[0]+min(heap[1],heap[2]))/2

def process_adjust_centre(f2): #discard

    A = np.loadtxt(f2)
    B = np.loadtxt(f2)

    for j in [3,4,5]:
        v_c = 0
        for i in A[:,j]: #A[3:6] are 3 velocities
            v_c += i

        v_c = v_c/len(A)
        # print(len(A), v_c)
        A[:,j] -= v_c

    for j in [0,1,2]:
        x_m = 0
        # print(A[0,j], x_m)
        x_m = find_median(B[:,j]) #A[0:3,:] are 3 Carticein coords; this func has changed its para A, so use a new one B
        # print(A[0,j], x_m)
        A[:,j] -= x_m

    return A

def process_adjust_centre_data(A):
    for j in [3,4,5]:
        v_c = np.mean(A[:,j])
        print(len(A), v_c)
        A[:,j] -= v_c
    for j in [0,1,2]:
        x_m = np.median(A[:,j])
        print(A[0,j], x_m)
        A[:,j] -= x_m
    return A

#### data of one snapshot, rw functions
def readdata_bysequeance(snap_id, ff, opt1=1):

    #### read data:
    snap_id_=str(snap_id).rjust(3,'0')
    fname = ff+"/snaps/snapshot_"+snap_id_+".hdf5"
    print('read %s' % fname)
    h5 = tables.open_file(fname)
    # halo = h5.root.PartType1 #1halo 2disk
    # baryons = h5.root.PartType2
    ##halo has: ['Acceleration', 'Coordinates', 'ParticleIDs', 'Potential', ...
    ## 'TimeStep', 'Velocities', '_AttributeSet', ...]
    # f = h5py.File(fname,'r')
    # f.visit(0)
    # halo = h5.keys()

    idx = 0
    N_tot = 0
    gas = h5.root.PartType0
    halo = h5.root.PartType1
    disk = h5.root.PartType2
    bulge = h5.root.PartType3
    #star = h5.root.PartType4
    #print(h5.root._g_check_has_child("halo"))
    type_all = [gas, halo, disk, bulge]
    compname = ["gas", "halo", "disk", "bulge"]

    for comp in type_all: #??
        N_comp = len(comp.ParticleIDs)
        data=np.zeros((N_comp,18))

        data[:,0:3] = comp.Coordinates #x_{1,2,3}
        data[:,3:6] = comp.Velocities #v_{1,2,3}

        data[:,6] = comp.ParticleIDs #particalIDs
        data[:,7] = 123 #comp.Type #none in hdf5
        data[:,8] = comp.Masses #Mass

        data[:,9] = 123 #comp.Density #none
        data[:,10] = 123 #comp.InternalEnergy #none
        data[:,11] = 123 #comp.SmoothingLength #none

        data[:,12]=comp.Potential #Potential
        data[:,13:16]=comp.Acceleration #acceleration

        data[:,16] = 123 #comp.RateOfChangeOfEntropy #none
        data[:,17]=comp.TimeStep #TimeSteps of particals in one snap are not same??

        print(N_comp)

    # case IO_POS:
    #   strcpy(buf, "Coordinates");
    #   break;
    # case IO_VEL:
    #   strcpy(buf, "Velocities");
    #   break;
    # case IO_ID:
    #   strcpy(buf, "ParticleIDs");
    #   break;
    # case IO_MASS:
    #   strcpy(buf, "Masses");
    #   break;
    # case IO_U:
    #   strcpy(buf, "InternalEnergy");
    #   break;
    # case IO_RHO:
    #   strcpy(buf, "Density");
    #   break;
    # case IO_HSML:
    #   strcpy(buf, "SmoothingLength");
    #   break;
    # case IO_POT:
    #   strcpy(buf, "Potential");
    #   break;
    # case IO_ACCEL:
    #   strcpy(buf, "Acceleration"); //gjy note: these are name chars
    #   break;
    # case IO_DTENTR:
    #   strcpy(buf, "RateOfChangeOfEntropy");
    #   break;
    # case IO_TSTP:
    #   strcpy(buf, "TimeStep");
    #   break;

        if opt1 != 0:
            #data = process_adjust_centre_data(data)
            aa = 0
        
        #### save data:
        filename = (ff+"/txt/snap_%d."+compname[idx]+".txt") % snap_id
        # filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_diceMW/txt/snap_%d.txt" % snap_id
        np.savetxt(filename, data, fmt="%f")
        idx += 1
        N_tot += N_comp
        print('%s has been wroten.\n' % filename)

    h5.close()
    return 0

def readdata_byname(f0, f1):
    h5 = tables.open_file(f0)
    # halo = h5.root.PartType1
    disk = h5.root.PartType2
    
    # N_tot=len(halo.ParticleIDs)
    # data=np.zeros((N_tot,12)) # 12 = 3+3+1 +1+1 +3
    # data[:,0:3]=halo.Coordinates #x_{1,2,3}
    # data[:,3:6]=halo.Velocities #v_{1,2,3}
    # data[:,6]=halo.ParticleIDs #particalIDs
    # data[:,7]=halo.Potential #Potential
    # data[:,8:11]=halo.Acceleration #acceleration
    # data[:,11]=halo.TimeStep #TimeSteps of particals in one snap are same.
    # print(sum(data[:,7]))
    # print(N_tot)
    N_tot=len(disk.ParticleIDs)
    data=np.zeros((N_tot,12)) # 12 = 3+3+1 +1+1 +3
    data[:,0:3]=disk.Coordinates #x_{1,2,3}
    data[:,3:6]=disk.Velocities #v_{1,2,3}
    data[:,6]=disk.ParticleIDs #particalIDs
    data[:,7]=disk.Potential #Potential
    data[:,8:11]=disk.Acceleration #acceleration
    data[:,11]=disk.TimeStep #TimeSteps of particals in one snap are same.
    print(sum(data[:,7]))
    print(N_tot)
    h5.close()

    np.savetxt(f1, data, fmt="%f")
    print('corresponding txt has been wroten.\n')
    return data

def readdata_justread(f0):
    h5 = tables.open_file(f0)
    halo = h5.root.PartType1
    # baryon = h5.root.PartType2

    N_tot = len(halo.ParticleIDs)
    data = np.zeros((N_tot,12)) # 12 = 3+3+1 +1+1 +3
    data[:,0:3]  = halo.Coordinates #x_{1,2,3}
    data[:,3:6]  = halo.Velocities #v_{1,2,3}
    data[:,6]    = halo.ParticleIDs #particalIDs
    data[:,7]    = halo.Potential #Potential
    data[:,8:11] = halo.Acceleration #acceleration
    data[:,11]   = halo.TimeStep #TimeSteps of particals in one snap are same.

    print(N_tot)
    print(sum(data[:,7]))
    print(max(data[:,0]), max(data[:,1]), max(data[:,2]))

    h5.close()
    print("data has been read.\n")
    return data

def rw_opt_fname(opt0, opt1=0, snap_id=0, file_ori='a.txt', file_tgt='b.txt'):
    '''if opt0==0: read snaps in /?? from snap_0.hdf5 to snap_<parameter snap_id>.hdf5;
    else: read a certein snapfile by name given by <parameter file_ori>.
    
    if opt1==0, it will not adjust xv to median and centre;
    else: adjust it.'''

    if opt0 == 0:
        readdata_bysequeance(snap_id, file_ori, 1) #暂未提供
        return 0

    else:
        data = readdata_byname(file_ori, file_tgt)
        if opt1 != 0:
            data = process_adjust_centre(file_tgt)
        
            np.savetxt(file_tgt, data, fmt="%f")
            print('done.\n')

        return 1






#### main()
if __name__ == '__main__':

    # ff = a=sys.argv[1]
    ff = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_Bovy13"
    for j in range(2): #10
        for i in range(0+j*10,j*10+10): #10
            readdata_bysequeance(i,ff,1)

#### end.



#################discard:
# def process__v_to_v_centre(data0_name):

#     A = np.loadtxt(data0_name)

#     for j in [3,4,5]:
#     #for j in enumerate([3,4,5]): enumerate() return (i, enum_i), not only enum_i
#         v_c = 0
#         for i in A[:,j]: #A[3:6] are 3 velocities
#             v_c += i

#         v_c = v_c/len(A)
#         print(len(A), v_c)
#         A[:,j] -= v_c

#     data_new_name='data_new.txt'
#     np.savetxt(data_new_name, A, fmt="%f")
#     print('v_centre has been processed.\n')
#     return 0

# def process__x_to_x_median(data0_name):

#     A = np.loadtxt(data0_name)
#     B = np.loadtxt(data0_name)

#     for j in [0,1,2]:
#         x_m = 0
#         # for i in A[:,j]: #A[0:3,:] are 3 Carticein coords
#         #     x_m += i
#         print(A[0,j], x_m)
#         x_m = find_median(B[:,j]) #this func has changed its para A ...
#         print(A[0,j], x_m)
#         A[:,j] -= x_m

#     data_new_name='data_new.txt'
#     np.savetxt(data_new_name, A, fmt="%f")
#     print('x_median has been processed.\n')
#     return 0

#if __name__ == '__main__':
    #print("snap_id is", sys.argv[1])
    #fname='hdf5_snaps/snapshot_%d.hdf5' % snap_id
    #fname='hdf5_snaps/snapshot_'+sys.argv[1]+'.hdf5'
    
    #print(a, 'has been wroten', '\n')
    #print('snifghsig=%f\ngmosrtj\'dsn' % a)



    # fname='../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy4_dice_plummer/snaps/snapshot_001.hdf5'
    # readdata_byname(fname)

    # data0_name='../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy3/0test/snap_999_0.txt'
    # process__v_to_v_centre(data0_name)
    # process__x_to_x_median(data0_name)
