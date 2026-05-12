# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#######################################

tmp = np.loadtxt("a2b2c2.txt")
a2 = tmp[0]
b2 = tmp[1]
c2 = tmp[2]

t_max = 10.0

theta_ell=np.linspace(0,2.0*np.pi,60)
theta_hyp1=np.linspace(-0.499*np.pi,0.499*np.pi,60)
theta_hyp2=np.linspace( 0.501*np.pi,1.499*np.pi,60)

bnd=10.0

n_la = 10
n_mu = 10
n_nu = 10

# 焦点位置, +/- Delta_2
f_bc = [np.sqrt(b2-c2),-np.sqrt(b2-c2)]

# 焦点位置, +/- Delta_1
f_ab = [np.sqrt(a2-b2),-np.sqrt(a2-b2)]

# 焦点位置, +/- sqrt(Delta_2^2+Delta_1^2)
f_ac = [np.sqrt(a2-c2),-np.sqrt(a2-c2)]

print("Plotting: a2 = %.2f b2 = %.2f c2 = %.2f" %(a2,b2,c2))

######################################## 函数

def Equi_la_on_XY(la,theta):
    x = np.sqrt(la-a2)*np.cos(theta)
    y = np.sqrt(la-b2)*np.sin(theta)
    return x,y

def Equi_mu_on_XY(mu,theta):
    x = np.sqrt(a2-mu)*np.tan(theta)
    y = np.sqrt(mu-b2)/np.cos(theta)
    return x,y

#########
def Equi_la_on_XZ(la,theta):
    x = np.sqrt(la-a2)*np.cos(theta)
    z = np.sqrt(la-c2)*np.sin(theta)
    return x,z


def Equi_mu_on_XZ(mu,theta):
    x = np.sqrt(a2-mu)*np.tan(theta)
    z = np.sqrt(mu-c2)/np.cos(theta)
    return x,z


def Equi_nu_on_XZ(nu,theta):
    x = np.sqrt(a2-nu)*np.tan(theta)
    z = np.sqrt(nu-c2)/np.cos(theta)
    return x,z

#########
def Equi_la_on_YZ(la,theta):
    y = np.sqrt(la-b2)*np.cos(theta)
    z = np.sqrt(la-c2)*np.sin(theta)
    return y,z

def Equi_mu_on_YZ(mu,theta):
    y = np.sqrt(mu-b2)*np.cos(theta)
    z = np.sqrt(mu-c2)*np.sin(theta)
    return y,z

def Equi_nu_on_YZ(nu,theta):
    y = np.sqrt(b2-nu)*np.tan(theta)
    z = np.sqrt(nu-c2)/np.cos(theta)
    return y,z


#######################################

#######################################


'''
#        画图：y轴= 3个度规系数, x轴= x
'''
fig = plt.figure(figsize=(12,12))  #创建图像对象，尺寸24*24英寸
#fig = plt.figure(figsize=(24,13.5))  #创建图像对象，尺寸24*13.5英寸(4:3)
#plt.title(r"$a^2$ = %.2f $b^2$ = %.2f $c^2$ = %.2f" %(a2,b2,c2), fontsize=18)
'''
#################################### 图1 : XY
fig2 = fig.add_subplot(1,3,1)
plt.xlabel('X [kpc]',fontsize=18)
plt.ylabel('Y [kpc]',fontsize=18)

raw=np.loadtxt("CP-XY.dat")
t=raw[:,0]
x=raw[:,1]
y=raw[:,2]

if (x.max()>y.max()):
    bnd = x.max()*1.1
else:
    bnd = y.max()*1.1

plt.xlim([-bnd,bnd])
plt.ylim([-bnd,bnd])

plt.scatter(x,y,c=t,s=10,marker='o',cmap='rainbow')

#XY平面上
#等la线为椭圆
beg=np.log10(a2)
end=np.log10(10*a2)
for la in np.logspace(beg,end,n_la):
    x,y=Equi_la_on_XY(la,theta_ell)
    plt.plot(x,y,c='red',linestyle='--',linewidth=1)

#等mu线双曲线
beg=np.log10(b2)
end=np.log10(a2)
for mu in np.logspace(beg,end,n_mu):
    x,y=Equi_mu_on_XY(mu,theta_hyp1)
    plt.plot(x,y,c='blue',linestyle='--',linewidth=1)
    x,y=Equi_mu_on_XY(mu,theta_hyp2)
    plt.plot(x,y,c='blue',linestyle='--',linewidth=1)

# 焦点位置, +/- Delta_1
xf = np.zeros_like(f_ab)
yf = f_ab
plt.scatter(xf,yf,marker='o',s=100,c='black')

################################ 图2 : XZ
fig2 = fig.add_subplot(1,3,2)
plt.xlabel('X [kpc]',fontsize=18)
plt.ylabel('Z [kpc]',fontsize=18)

raw=np.loadtxt("CP-XZ.dat")
t=raw[:,0]
x=raw[:,1]
z=raw[:,2]

if (x.max()>z.max()):
    bnd = x.max()*1.1
else:
    bnd = z.max()*1.1

plt.xlim([-bnd,bnd])
plt.ylim([-bnd,bnd])

plt.scatter(x,z,c=t,s=10,marker='o',cmap='rainbow')

#XZ平面上
#等 la 线为椭圆
beg=np.log10(a2)
end=np.log10(10*a2)
for la in np.logspace(beg,end,n_la):
    x,z=Equi_la_on_XZ(la,theta_ell)
    plt.plot(x,z,c='red',linestyle='--',linewidth=1)

#等mu 线双曲线
beg=np.log10(b2)
end=np.log10(a2)
for mu in np.logspace(beg,end,n_mu):
    x,z=Equi_mu_on_XZ(mu,theta_hyp1)
    plt.plot(x,z,c='blue',linestyle='--',linewidth=1)
    x,z=Equi_mu_on_XZ(mu,theta_hyp2)
    plt.plot(x,z,c='blue',linestyle='--',linewidth=1)

#等nu 线双曲线
beg=np.log10(c2)
end=np.log10(b2)
for nu in np.logspace(beg,end,n_nu):
    x,z=Equi_nu_on_XZ(nu,theta_hyp1)
    plt.plot(x,z,c='black',linestyle='--',linewidth=1)
    x,z=Equi_nu_on_XZ(nu,theta_hyp2)
    plt.plot(x,z,c='black',linestyle='--',linewidth=1)

# 焦点位置
xf = np.zeros_like(f_bc)
zf = f_bc
plt.scatter(xf,zf,marker='o',s=100,c='blue')
xf = np.zeros_like(f_ac)
zf = f_ac
plt.scatter(xf,zf,marker='o',s=100,c='red')
'''

#################################### 图3 : YZ
fig2 = fig.add_subplot(1,1,1)
plt.xlabel('Y [kpc]',fontsize=18)
plt.ylabel('Z [kpc]',fontsize=18)

raw=np.loadtxt("Orbit.dat")
t=raw[:,0]
y=raw[:,2]
z=raw[:,3]

if (y.max()>z.max()):
    bnd = y.max()*1.1
else:
    bnd = z.max()*1.1

bnd=5
plt.xlim([-bnd,bnd])
plt.ylim([-bnd,bnd])

plt.scatter(y,z,c=t,s=5,marker='o',cmap='rainbow')

#YZ平面上
#等 la 线为椭圆
beg=np.log10(a2)
end=np.log10(10*a2)
for la in np.logspace(beg,end,n_la):
    y,z=Equi_la_on_YZ(la,theta_ell)
    plt.plot(y,z,c='red',linestyle='--',linewidth=1)

#等mu 线为椭圆
beg=np.log10(b2)
end=np.log10(a2)
for mu in np.logspace(beg,end,n_mu):
    y,z=Equi_mu_on_YZ(mu,theta_ell)
    plt.plot(y,z,c='blue',linestyle='--',linewidth=1)

#等nu 线双曲线
beg=np.log10(c2)
end=np.log10(b2)
for nu in np.logspace(beg,end,n_nu):
    y,z=Equi_nu_on_YZ(nu,theta_hyp1)
    plt.plot(y,z,c='black',linestyle='--',linewidth=1)
    y,z=Equi_nu_on_YZ(nu,theta_hyp2)
    plt.plot(y,z,c='black',linestyle='--',linewidth=1)

# 焦点位置
yf = np.zeros_like(f_bc)
zf = f_bc
plt.scatter(yf,zf,marker='o',s=100,c='blue')
yf = np.zeros_like(f_ac)
zf = f_ac
plt.scatter(yf,zf,marker='o',s=100,c='red')
yf = f_ab
zf = np.zeros_like(f_ac)
plt.scatter(yf,zf,marker='o',s=100,c='black')


####################################
plt.savefig('YZ-plane.png')
exit()
