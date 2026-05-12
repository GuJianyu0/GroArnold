# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

####################################### 常数

delta_i = np.loadtxt("delta_i.txt")
delta_1 = delta_i[0]
delta_2 = delta_i[1]

c2=1.0
b2=1.0+delta_2*delta_2
a2=b2 +delta_1*delta_1

print(a2,b2,c2)

theta_ell=np.linspace(0,2.0*np.pi,60)
theta_hyp1=np.linspace(-0.499*np.pi,0.499*np.pi,60)
theta_hyp2=np.linspace( 0.501*np.pi,1.499*np.pi,60)

bnd=10.0

f_bc = [np.sqrt(b2-c2),-np.sqrt(b2-c2)]
f_ab = [np.sqrt(a2-b2),-np.sqrt(a2-b2)]
f_ac = [np.sqrt(a2-c2),-np.sqrt(a2-c2)]

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
    return x,z


#######################################

'''
#        画图：y轴= 3个度规系数, x轴= x
'''
fig = plt.figure(figsize=(36,12))  #创建图像对象，尺寸24*24英寸
#plt.title(r"$a^2$ = %.2f $b^2$ = %.2f $c^2$ = %.2f" %(a2,b2,c2), fontsize=18)

#################################### 图1 : XY平面
fig2 = fig.add_subplot(1,3,1)
plt.xlabel('x [kpc]',fontsize=18)
plt.ylabel('y [kpc]',fontsize=18)

plt.xlim([-bnd,bnd])
plt.ylim([-bnd,bnd])

#XY平面上
#等la线为椭圆
beg=np.log10(a2)
end=np.log10(10*a2)
for la in np.logspace(beg,end,5):
    x,y=Equi_la_on_XY(la,theta_ell)
    plt.plot(x,y,c='red',linestyle='--',linewidth=1)

#等mu线双曲线
beg=np.log10(b2)
end=np.log10(a2)
for mu in np.logspace(beg,end,6):
    x,y=Equi_mu_on_XY(mu,theta_hyp1)
    plt.plot(x,y,c='blue',linestyle='--',linewidth=1)
    x,y=Equi_mu_on_XY(mu,theta_hyp2)
    plt.plot(x,y,c='blue',linestyle='--',linewidth=1)

# 焦点位置
xf = np.zeros_like(f_ab)
yf = f_ab
plt.scatter(xf,yf,marker='o',s=100,c='black')

#################################### 图2 : XZ 平面
fig2 = fig.add_subplot(1,3,2)
plt.xlabel('x [kpc]',fontsize=18)
plt.ylabel('z [kpc]',fontsize=18)

plt.xlim([-bnd,bnd])
plt.ylim([-bnd,bnd])

#XZ平面上
#等 la 线为椭圆
beg=np.log10(a2)
end=np.log10(10*a2)
for la in np.logspace(beg,end,5):
    x,z=Equi_la_on_XZ(la,theta_ell)
    plt.plot(x,z,c='red',linestyle='--',linewidth=1)

#等mu 线双曲线
beg=np.log10(b2)
end=np.log10(a2)
for mu in np.logspace(beg,end,5):
    x,z=Equi_mu_on_XZ(mu,theta_hyp1)
    plt.plot(x,z,c='blue',linestyle='--',linewidth=1)
    x,z=Equi_mu_on_XZ(mu,theta_hyp2)
    plt.plot(x,z,c='blue',linestyle='--',linewidth=1)

#等nu 线双曲线
beg=np.log10(c2)
end=np.log10(b2)
for nu in np.logspace(beg,end,4):
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


#################################### 图3 : YZ 平面
fig2 = fig.add_subplot(1,3,3)
plt.xlabel('y [kpc]',fontsize=18)
plt.ylabel('z [kpc]',fontsize=18)

plt.xlim([-bnd,bnd])
plt.ylim([-bnd,bnd])

#XZ平面上
#等 la 线为椭圆
beg=np.log10(a2)
end=np.log10(10*a2)
for la in np.logspace(beg,end,5):
    y,z=Equi_la_on_YZ(la,theta_ell)
    plt.plot(y,z,c='red',linestyle='--',linewidth=1)

#等mu 线为椭圆
beg=np.log10(b2)
end=np.log10(a2)
for mu in np.logspace(beg,end,5):
    y,z=Equi_mu_on_YZ(mu,theta_ell)
    plt.plot(y,z,c='blue',linestyle='--',linewidth=1)

#等nu 线双曲线
beg=np.log10(c2)
end=np.log10(b2)
for nu in np.logspace(beg,end,4):
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


plt.savefig('Equal-lmn.png')

exit()



#################################### 图4 : x-y
fig2 = fig.add_subplot(3,3,4)
plt.xlabel('x [kpc]',fontsize=18)
plt.ylabel('y [kpc]',fontsize=18)

if (x.max()>y.max()):
    bnd = x.max()*1.1
else:
    bnd = y.max()*1.1

plt.xlim([-bnd,bnd])
plt.ylim([-bnd,bnd])

plt.scatter(x,y,c=t,s=1,label=r'$time$',marker='o',cmap='rainbow')


#################################### 图5 : x-z
fig2 = fig.add_subplot(3,3,5)
plt.xlabel('x [kpc]',fontsize=18)
plt.ylabel('z [kpc]',fontsize=18)

if (x.max()>z.max()):
    bnd = x.max()*1.1
else:
    bnd = z.max()*1.1

plt.xlim([-bnd,bnd])
plt.ylim([-bnd,bnd])

plt.scatter(x,z,c=t,s=1,label=r'$time$',marker='o',cmap='rainbow')

#################################### 图6 : x-z
fig2 = fig.add_subplot(3,3,6)
plt.xlabel('y [kpc]',fontsize=18)
plt.ylabel('z [kpc]',fontsize=18)
if (y.max()>z.max()):
    bnd = y.max()*1.1
else:
    bnd = z.max()*1.1
plt.xlim([-bnd,bnd])
plt.ylim([-bnd,bnd])
plt.scatter(y,z,c=t,s=1,label=r'$time$',marker='o',cmap='rainbow')

#################################### 图6 : R-z - NOT IN USE
#fig2 = fig.add_subplot(3,3,6)
#plt.xlabel('R [kpc]',fontsize=18)
#plt.ylabel('z [kpc]',fontsize=18)
#R=np.sqrt(x*x+y*y)
#if (R.max()>z.max()):
#    bnd = R.max()*1.1
#else:
#    bnd = z.max()*1.1
#plt.xlim([0,2*bnd])
#plt.ylim([-bnd,bnd])
#plt.scatter(R,z,c=t,s=1,label=r'$time$',marker='o',cmap='rainbow')



#################################### 图7 : p_lambda - lambda
fig2 = fig.add_subplot(3,3,7)
plt.xlabel(r'$\lambda$',fontsize=18)
plt.ylabel(r'$p_\lambda$',fontsize=18)
#plt.yscale('log')

dt = np.diff(t)
dt = np.append(dt,[1.0])

d_la = np.diff(la)
d_la = np.append(d_la,[0.0])
p_la = d_la * P2 / dt

# 设置x轴显示范围
plt.xlim([a2,la.max()*1.1])

# 设置y轴显示范围，使用百分位数作为基准，再乘以一个比例系数
ymin, ymax = np.percentile(p_la,(5,95),method='midpoint')
plt.ylim([ymin*2.0,ymax*2.0])
plt.scatter(la,p_la,c=t,s=0.3,label=r'$time$',marker='o',cmap='rainbow')


#################################### 图8 : p_mu - mu
fig2 = fig.add_subplot(3,3,8)
plt.xlabel(r'$\mu$',fontsize=18)
plt.ylabel(r'$p_\mu$',fontsize=18)

d_mu = np.diff(mu)
d_mu = np.append(d_mu,[0.0])
p_mu = d_mu * Q2 / dt

plt.xlim([b2,a2])

ymin, ymax = np.percentile(p_mu,(5,95),method='midpoint')
plt.ylim([ymin*2.0,ymax*2.0])
plt.scatter(mu,p_mu,c=t,s=0.3,label=r'$time$',marker='o',cmap='rainbow')

#################################### 图9 : p_nu - nu
fig2 = fig.add_subplot(3,3,9)
plt.xlabel(r'$\nu$',fontsize=18)
plt.ylabel(r'$p_\nu$',fontsize=18)

d_nu = np.diff(nu)
d_nu = np.append(d_nu,[0.0])
p_nu = d_nu * R2 / dt

plt.xlim([c2,b2])

ymin, ymax = np.percentile(p_nu,(5,95),method='midpoint')
plt.ylim([ymin*2.0,ymax*2.0])
plt.scatter(nu,p_nu,c=t,s=0.3,label=r'$time$',marker='o',cmap='rainbow')





#fig.colorbar(cax)

#plt.legend(fontsize=18)

plt.savefig('Metrics-3x3.png')

#plt.show()

