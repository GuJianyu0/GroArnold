#!/usr/bin/env python
# -*- coding:utf-8 -*-

#In[1]
import sys
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#In[2]
from galpy.potential import MWPotential2014
from galpy.orbit import Orbit
o= Orbit([1.,0.1,1.1,0.,0.05])
ts=numpy.linspace(0.,100.,1001)
o.integrate(ts,MWPotential2014)

from galpy.actionAngle import estimateDeltaStaeckel
estimateDeltaStaeckel(MWPotential2014,o.R(ts),o.z(ts))
# 0.40272708556203662

#In[3]
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.potential import LogarithmicHaloPotential
lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
aAIA= actionAngleIsochroneApprox(pot=lp,b=0.8)

from galpy.actionAngle import estimateBIsochrone
from galpy.orbit import Orbit
obs= numpy.array([1.56148083,0.35081535,-1.15481504,0.88719443,-0.47713334,0.12019596]) #orbit similar to GD-1
o= Orbit(obs)
ts= numpy.linspace(0.,100.,1001)
o.integrate(ts,lp)
estimateBIsochrone(lp,o.R(ts),o.z(ts))
# (0.78065062339131952, 1.2265541473461612, 1.4899326335155412) #bmin,bmedian,bmax over the orbit

aAIA= actionAngleIsochroneApprox(pot=lp,b=0.8)
aAIA.plot(*obs,type='jr')
aAIA.actionsFreqsAngles(*obs)
aAIA(*obs,nonaxi=True)

#In[4]
import pynbody
s= pynbody.new(star=1)
s['mass']= 1.
s['eps']= 0.
from galpy.potential import SnapshotRZPotential
sp= SnapshotRZPotential(s,num_threads=1)
# s = pynbody.load('testdata/g15784.lr.01024.gz')
h = s.halos()
h1 = h[1]

# pynbody.analysis.halo.center(h1,mode='hyb')
# pynbody.analysis.angmom.faceon(h1, cen=(0,0,0),mode='ssc')
# s.physical_units()
# from galpy.util.conversion import _G
# g= pynbody.array.SimArray(_G/1000.)
# g.units= 'kpc Msol**-1 km**2 s**-2 G**-1'
# s._arrays['mass']= s._arrays['mass']*g

#In[5]
from galpy.potential import SnapshotPotential #.InterpSnapshotPotential

from galpy.potential import InterpSnapshotRZPotential
spi= InterpSnapshotRZPotential(h1,rgrid=(numpy.log(0.01),numpy.log(20.),101),logR=True,zgrid=(0.,10.,101),interpPot=True,zsym=True)
spi.normalize(R0=10.)

sc = pynbody.load('Repos/pynbody-testdata/g15784.lr.01024.gz'); hc = sc.halos(); hc1= hc[1]; pynbody.analysis.halo.center(hc1,mode='hyb'); pynbody.analysis.angmom.faceon(hc1, cen=(0,0,0),mode='ssc'); sc.physical_units()
sn= pynbody.filt.BandPass('rxy','7 kpc','9 kpc')
R,vR,vT,z,vz = [numpy.ascontiguousarray(hc1.s[sn][x]) for x in ('rxy','vr','vt','z','vz')]
ro, vo= 10., 294.62723076942245
R/= ro
z/= ro
vR/= vo
vT/= vo
vz/= vo
from galpy.orbit import Orbit
numpy.random.seed(1)
ii= numpy.random.permutation(len(R))[0]
o= Orbit([R[ii],vR[ii],vT[ii],z[ii],vz[ii]])
ts= numpy.linspace(0.,100.,1001)
o.integrate(ts,spi)
o.plot()

from galpy.actionAngle import actionAngleStaeckel
aAS= actionAngleStaeckel(pot=spi,delta=0.45,c=True)
jr,lz,jz= aAS(R,vR,vT,z,vz)
from galpy.util import plot as galpy_plot
galpy_plot.scatterplot(lz,jr,'k.',xlabel=r'$J_\phi$',ylabel=r'$J_R$',xrange=[0.,1.3],yrange=[0.,.6])


# %%
