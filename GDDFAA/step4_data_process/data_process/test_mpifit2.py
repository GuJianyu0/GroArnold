# import numpy as np
# import mpfit

# def myfunct(p, fjac=None, x=None, y=None, err=None):
# 	status = 0
# 	model = p[0] + p[1]*x + p[2]*x**2 + p[3]*np.sqrt(x) + p[4]*np.log(x)
# 	return [status, (y-model)/(err)]

# x = np.arange(100)*1.+0.1
# p0 = [5., 2., 5., 1.5, 2.]
# p = [1., -1., 1., -1., 1.]
# y = p[0] + p[1]*x + p[2]*x**2 + p[3]*np.sqrt(x) + p[4]*np.log(x)
# err = np.ones(x.shape)*0.1
# fa = {'x':x, 'y':y, 'err':err}

# m = mpfit.mpfit(myfunct, p0, functkw=fa)
# print('status = ', m.status)
# if (m.status <= 0):
# 	print('error message = ', m.errmsg)
# print('parameters = ', m.params)



import numpy as np
import matplotlib.pyplot as plt
from mpfit import mpfit

#定义拟合的x,y,和误差
x = np.arange(6,7,0.01)
random = np.random.rand(len(x))* ((0.6*np.exp(2*x + 1.6)))
y = (0.6*np.exp(2*x + 1.6)) + random


#p为parameter
def fit_function(x,p):
    a = p[0]
    b = p[1]
    c = p[2]
    function = a*np.exp(b*x+c)
    return function

def myfunc(p, fjac=None, x=None, y=None, err=None):
    model=fit_function(x,p)
    status=0
    return [status,(y-model)/err]

#设置参数的初始值
a = 0.7
b = 2.1
c = 1.8
p = np.array([a,b,c])

#这里是对参数的一些设置
parinfo=[]
for i in range(len(p)):
    parinfo.append({'value':0.,'fixed':0,'limited':[0,0],'limits':[0.,0.],'step':0,'mpside':2,'mpmaxstep':0})
    parinfo[i]['value']=p[i]
parinfo[0]['limited']=[1,1]
parinfo[0]['limits']=[0,2]
parinfo[0]['mpmaxstep']=0.01
parinfo[1]['limited']=[1,1]
parinfo[1]['limits']=[0,10]
parinfo[1]['mpmaxstep']=0.01
parinfo[2]['limited']=[1,1]
parinfo[2]['limits']=[1,5]
parinfo[2]['mpmaxstep']=0.01
    
#设置误差
ey = random
#开始迭代,可以调整迭代次数
fkw={'x':x,'y':y,'err':ey}
m=mpfit(myfunc,p,parinfo=parinfo,functkw=fkw,maxiter=300)
print('fit_parameters:',m.params)
print('fit_parameters_error:',m.perror)
