# 对输入文件进行密度中心坐标、漂移速度修正
import numpy as np
from sklearn.neighbors import KDTree

# 只计算30 kpc以内的粒子的MOI
rlimit = 10.0

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


def xyz_2_PA(x,v):
    nrow,itmp = x.shape
    x_MOI = np.zeros_like(x)
    v_MOI = np.zeros_like(x)
    eigenvalues, eigenvectors = np.linalg.eigh(calc_MOI(x))
    vl = eigenvectors[:,2]
    vm = eigenvectors[:,1]
    vn = eigenvectors[:,0]
    print("坐标，速度矢量变换至MOI主轴系")
    for ii in range(nrow):
        vec = x[ii,0:3]
        x_MOI[ii,0] = np.matmul(vec,vl)
        x_MOI[ii,1] = np.matmul(vec,vm)
        x_MOI[ii,2] = np.matmul(vec,vn)
    for ii in range(nrow):
        vec = v[ii,0:3]
        v_MOI[ii,0] = np.matmul(vec,vl)
        v_MOI[ii,1] = np.matmul(vec,vm)
        v_MOI[ii,2] = np.matmul(vec,vn)
    return x_MOI,v_MOI


def DensityCenter(xyz,v):
    tree = KDTree(xyz, leaf_size=40)
    ntot,itmp=xyz.shape
    mp=1.37e-2
    xCOD=np.zeros(3)
    vCOD=np.zeros(3)
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
        vCOD +=   v[ii,:]*dens
        cumu += dens
    
    return xCOD/cumu, vCOD/cumu


########################################################
# 读入文件，该文件直接从snapshot转换来，未做任何修正
# raw = np.loadtxt("halo+vel+pot.out")
# m = raw[:,0]
# x = raw[:,1:4]
# v = raw[:,4:7]
# pot = raw[:,7]

raw = np.loadtxt("snapshot_5000.txt") #gjy changed
m = raw[:,8]
x = raw[:,0:3]
v = raw[:,3:6]
pot = raw[:,14]

nline, ncol = raw.shape


# 计算这些束缚粒子的密度中心的坐标和速度
print("计算密度中心的位置和速度。")
xCOD, vCOD = DensityCenter(x,v)

# 修正束缚粒子的坐标和速度
print("密度中心坐标：", xCOD)
print("密度中心速度：", vCOD)
x[:,0:3] -= xCOD[0:3]
v[:,0:3] -= vCOD[0:3]

x,v = xyz_2_PA(x,v)
MOI = calc_MOI(x)
print(MOI)
print("输出到文件 gadget.out，已做密度中心的位置和速度修正。以及MOI主轴系修正。")

#print("输出到文件 gadget.out，已做密度中心的位置和速度修正。")
f = open("gadget.out","w")
f.write("0000\n")
f.write("%06d\n" %nline)
f.write("0.0\n")

for i in range(nline):
    f.write("%13.5e  %13.5e  %13.5e  %13.5e  %13.5e  %13.5e  %13.5e  %13.5e\n"
      %(m[i], x[i,0], x[i,1], x[i,2], v[i,0], v[i,1], v[i,2], pot[i])  )

f.close()




