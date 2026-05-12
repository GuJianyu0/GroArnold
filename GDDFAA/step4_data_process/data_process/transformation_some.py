#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def norm_l(a,l=2):
    x=0
    for e in a:
        x += e**l
    return x**(1./l)

class Transform_data: #not used
    '''
    Transform_data: x -> 
    |x|, log(|x|), log(|x|+1)*sign(x), restrict range, 
    affine group transformation, Fourier transformation, 
    ...
    '''

    def __init__(self, data):
        self.data = data
        return 

def func(x, a=1.,b=1.,c=0.):
    return a*np.exp(-b*x)+c -1.
    #return x*1.

def SO3(x, theta):
    '''
    This is a rotation group SO3 for a given point.
    @param x: the initial point in Carteisain coordinate, 1-dim array;
    @param theta: the three Euler angles along x,y,z-axis, 1-dim array; Note the angles' rotation wise is counterclockwise.
    '''
    theta0,theta1,theta2 = theta[0], theta[1], theta[2]
    # if len(x)!=3:
    #     print(x,theta)
    #     print("Incorrect vector dim! 3-dim please.\n")
    #     return np.array([0.,0.,0.])
    # rotate Eular angle
    # SO3_0 = np.array([  [   np.cos(theta0) , -np.sin(theta0), 0               ],
    #                     [   np.sin(theta0) , np.cos(theta0) , 0               ],
    #                     [   0              , 0              , 1               ] ]) #rotate along x-axis

    # SO3_1 = np.array([  [   1              , 0              , 0               ],
    #                     [   0              , np.cos(theta1) , -np.sin(theta1) ],
    #                     [   0              , np.sin(theta1) , np.cos(theta1)  ] ]) #rotate along y-axis

    # SO3_2 = np.array([  [   np.cos(theta2) , -np.sin(theta2), 0               ],
    #                     [   np.sin(theta2) , np.cos(theta2) , 0               ],
    #                     [   0              , 0              , 1               ] ]) #rotate along z-axis

    SO3_0 = np.array([  [ 1.              , 0.              , 0.              ],
                        [ 0.              , np.cos(theta0)  , np.sin(theta0)  ],
                        [ 0.              , -np.sin(theta0) , np.cos(theta0)  ] ]) #rotate along x-axis

    SO3_1 = np.array([  [ np.cos(theta1)  , 0.              , np.sin(theta1)  ],
                        [ 0.              , 1.              , 0.              ],
                        [ -np.sin(theta1) , 0.              , np.cos(theta1)  ] ]) #rotate along y-axis

    SO3_2 = np.array([  [ np.cos(theta2)  , np.sin(theta2)  , 0.              ],
                        [ -np.sin(theta2) , np.cos(theta2)  , 0.              ],
                        [ 0.              , 0.              , 1.              ] ]) #rotate along z-axis

    #SO3_123 = np.array([SO3_0, SO3_1, SO3_2])
    x_ = x[0:3]
    x_ = np.dot(x_,SO3_0)
    x_ = np.dot(x_,SO3_1)
    x_ = np.dot(x_,SO3_2)
    return x_

def SO3_vec_be(begin,end, theta):
    '''
    This is to rotate a vector with a begin point (like velocity with a coordinate), and return the new vetor without the begin point.
    This is more like a rotetor for a line rigid body. The variables b_ and e_ below are the new begin point and end point of the line rigid body.
    '''
    b_ = SO3(begin, theta)
    e_ = SO3(end, theta)
    return e_-b_

def stretching3(x, k0,k1,k2):
    return np.array([x[0]*k0, x[1]*k1, x[2]*k2])

def translation3(x, x0,x1,x2):
    return np.array([x[0]+x0, x[1]+x1, x[2]+x2])



if __name__ == '__main__':

    # ##1. an example of a set of points
    # l = 100
    # x0 = np.linspace(0,2,l)
    # y0 = func(x0)
    # z0 = np.zeros(l)
    # s = x0.shape
    # x1 = np.zeros(s)
    # y1 = np.zeros(s)
    # z1 = np.zeros(s)
    # for i in np.arange(s[0]):
    #     xx = x0[i]
    #     yy = y0[i]
    #     zz = z0[i]
    #     xy = SO3([xx,func(xx),0.], 1./1*np.pi,0,0) #for 2d data + z-axis
    #     xy = stretching3(xy, 1,1,1)
    #     xy = translation3(xy, 0,0,0)
    #     x1[i] = xy[0]
    #     y1[i] = xy[1]
    #     z1[i] = xy[2]
    # fig = plt.figure(figsize=(16, 12), dpi=80, facecolor=(0.0, 0.0, 0.0))
    # ax = Axes3D(fig)
    # ax.grid(True)
    # ax.scatter3D(x0,y0,z0, color="red")
    # ax.scatter3D(x1,y1,z1, color="blue")
    # plt.show()



    ##4. vector
    # begin = np.array([0.,1.,0.])
    # be = np.array([-10.,0.,0.])
    begin = np.array([0.6,0.8,0.])
    be = np.array([8.,-6.,0.])
    end = be+begin #be = end-begin #dimentionless

    # theta = [1./6*np.pi, 0., 0.] #pi/6 along x-axis
    theta = [0., 1./6*np.pi, 0.] #pi/6 along y-axis
    # theta = [0., 0., 1./6*np.pi] #pi/6 along z-axis
    begin2 = SO3(begin, theta)
    end2 = SO3(end, theta)
    # be2 = end2-begin2
    be2 = SO3_vec_be(begin,end, theta)

    coor_0 = begin
    coor_2 = begin2
    vel_0 = be
    vel_2 = be2
    print(begin, begin2)
    print(end, end2)
    print(be, be2)
    print(coor_0, vel_0)
    print(coor_2, vel_2)

    x0 = [begin[0],end[0]]
    y0 = [begin[1],end[1]]
    z0 = [begin[2],end[2]]
    x2 = [begin2[0],end2[0]]
    y2 = [begin2[1],end2[1]]
    z2 = [begin2[2],end2[2]]
    fig = plt.figure(figsize=(16, 12), dpi=80, facecolor=(0.0, 0.0, 0.0))
    ax = Axes3D(fig)
    ax.grid(True)
    ax.scatter3D([0.],[0.],[0.], color="black")
    ax.plot3D(x0,y0,z0, color="red")
    ax.plot3D(x2,y2,z2, color="blue")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    # plt.axis('scaled')
    #import matplotlib.ticker as mticker
    #plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))
    plt.show()



    # ##2. ellips 
    # ##a fool implicit function plot way??
    # a,b,c = 4.0, 6.0, 2.0
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x0 = a * np.outer(np.cos(u), np.sin(v))
    # y0 = b * np.outer(np.sin(u), np.sin(v))
    # z0 = c * np.outer(np.ones(np.size(u)), np.cos(v))

    # ##affine transformation example
    # s = x0.shape
    # x1 = np.zeros(s)
    # y1 = np.zeros(s)
    # z1 = np.zeros(s)
    # for i in np.arange(s[0]):
    #     for j in np.arange(s[1]):
    #         xx = x0[i,j]
    #         yy = y0[i,j]
    #         zz = z0[i,j]
    #         xy = SO3([xx,yy,zz], 1./2*np.pi,1./4*np.pi,1./4*np.pi)
    #         xy = stretching3(xy, 1,1,1)
    #         xy = translation3(xy, 0,0,0)
    #         #print(xy)
    #         x1[i,j] = xy[0]
    #         y1[i,j] = xy[1]
    #         z1[i,j] = xy[2]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ##the outer surface of ellips
    # from matplotlib import cm
    # ax.plot_surface(x0, y0, z0, color='r',cmap=cm.coolwarm)
    # ax.plot_surface(x1, y1, z1, color='b',cmap=cm.coolwarm)
    # ##the isoheight shadow
    # # cset = ax.contourf(x0, y0, z0, zdir='x', offset=-1.6*a, cmap=cm.coolwarm)
    # # cset = ax.contourf(x0, y0, z0, zdir='y', offset=-1.6*b, cmap=cm.coolwarm)
    # # cset = ax.contourf(x0, y0, z0, zdir='z', offset=-1.6*c, cmap=cm.coolwarm)
    # cset = ax.contourf(x1, y1, z1, zdir='x', offset=-2*a, cmap=cm.coolwarm)
    # cset = ax.contourf(x1, y1, z1, zdir='y', offset=-2*b, cmap=cm.coolwarm)
    # cset = ax.contourf(x1, y1, z1, zdir='z', offset=-2*c, cmap=cm.coolwarm)
    # ##axis modification
    # abc = norm_l([a,b,c])/1.2
    # ax.set_xlabel('x-axis')
    # ax.set_xlim(-2*abc, 2*abc)
    # ax.set_ylabel('y-axis')
    # ax.set_ylim(-2*abc, 2*abc)
    # ax.set_zlabel('z-axis')
    # ax.set_zlim(-2*abc, 2*abc)
    # # plt.axis('scaled')
    # plt.show()



    # ##3. another method to plot the elllips before
    # from mayavi import mlab
    # mlab.clf()
    # x, y, z = np.mgrid[-3:3:50j, -3:3:50j, -3:3:50j]
    # # Plot a sphere of radius 1
    # values = x*x + y*y + z*z - np.sqrt(3)
    # mlab.contour3d(x, y, z, values, contours=[0])
    # mlab.axes()
    # # Plot a torus
    # R = 2
    # r = 1
    # values = (R - np.sqrt(x**2 + y**2))**2 + z**2 - r**2
    # mlab.figure()
    # mlab.contour3d(x, y, z, values, contours=[0])
    # mlab.axes()
    # # Plot a Scherk's second surface
    # x, y, z = np.mgrid[-4:4:100j, -4:4:100j, -8:8:100j]
    # values = np.sin(z) - np.sinh(x)*np.sinh(y)
    # mlab.figure()
    # mlab.contour3d(x, y, z, values, contours=[0])
    # mlab.axes()
    # mlab.show()