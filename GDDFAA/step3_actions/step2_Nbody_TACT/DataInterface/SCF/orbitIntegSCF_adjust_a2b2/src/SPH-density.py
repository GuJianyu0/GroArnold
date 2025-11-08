import numpy as np
from sklearn.neighbors import KDTree
import os


# 只计算30 kpc以内的粒子的MOI
rlimit = 30.0

def calc_MOI(x):
    MOI=np.zeros(9)
    MOI = MOI.reshape(3,3)
    mass = 1.37e-2
    r = np.sqrt(x[:,0]**2.0 + x[:,1]**2.0+x[:,2]**2.0 )
    xsub = x[r<=rlimit]
    nrow, ncol = xsub.shape
    print("使用r<%5.1f kpc的%d个粒子计算MOI。" %(rlimit,nrow))
    for i in range(3):
        for j in range(3):
            for k in range(nrow):
                MOI[i,j]+=mass*xsub[k,i]*xsub[k,j]
    return MOI


def xyz_2_PA(x):
    nrow,itmp = x.shape
    x_MOI = np.zeros_like(x)
    eigenvalues, eigenvectors = np.linalg.eigh(calc_MOI(x))
    vl = eigenvectors[:,2]
    vm = eigenvectors[:,1]
    vn = eigenvectors[:,0]
    print("坐标变换至MOI主轴系")
    for ii in range(nrow):
        vec = x[ii,0:3]
        x_MOI[ii,0] = np.matmul(vec,vl)
        x_MOI[ii,1] = np.matmul(vec,vm)
        x_MOI[ii,2] = np.matmul(vec,vn)
    return x_MOI

def calc_density(xyz):
    tree = KDTree(xyz, leaf_size=40)
    ntot,itmp=xyz.shape
    mp=1.37e-2
    output=[]
    xCOD=np.zeros(3)
    cumu=0.0
    k=20
    for ii in range(ntot):
        target=[xyz[ii]]  #query要求传入2-D数组，因此有两个方括号
        distances, indices = tree.query(target, k=k) #两个返回值都是2-D数组
        rmax=distances[0][k-1]
        r = np.sqrt(xyz[ii][0]**2.+xyz[ii][1]**2.+xyz[ii][2]**2.)
        vol=(4./3.)*np.pi*rmax**3.0
        mass=float(k)*mp
        dens=mass/vol
        xCOD += xyz[ii,:]*dens
        cumu += dens
        output=np.append(output,xyz[ii])
        #output=np.append(output,r)
        output=np.append(output,dens)
    
    output=output.reshape(ntot,4)
    listt = output[:,3]
    sorted_idx = np.argsort(listt)
    sorted_output = output[sorted_idx]
    xCOD /= cumu
    sorted_output[:,0:3] -= xCOD
    return sorted_output

##############################################################
#正式开始计算

filename = 'halo.out'
print("处理文件：",filename)
tmp = np.array(np.loadtxt(filename, dtype=float))

x=tmp[:,0:3]

#坐标变换
#x = xyz_2_PA(x)
print("计算SPH型密度，密度中心并修正坐标")
dens = calc_density(x)
ofname = 'halo-dens.out'

print("写入文件 %s" %ofname)
np.savetxt(ofname,dens,fmt="%.3e")

#f = open(ofname,"w")
#nline,ntmp=x.shape
#f.write("0000\n" )
#f.write("%d\n" %nline )
#f.write("0.0\n" )
#for j in range(nline):
#    f.write("%.5e  %.5e  %.5e   %.5e\n" %(dens[j,0],dens[j,1],dens[j,2],dens[j,3]) )
#f.close()






