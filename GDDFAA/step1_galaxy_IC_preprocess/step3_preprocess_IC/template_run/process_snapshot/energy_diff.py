import numpy as np


raw = np.loadtxt("energy.txt")

Ep = raw[:,2]
Ek = raw[:,3]
time = raw[:,0]

nline, ncol = raw.shape

Ebeg = Ep[0]+Ek[0]
Efin = Ep[nline-1]+Ek[nline-1]
print("T=%5.3f Gyr时的能量相对T=0 Gyr时的变化率, %.3e" %(time[nline-1],(Efin-Ebeg)/np.abs(Ebeg)))


dEp = np.diff(Ep)
dEk = np.diff(Ek)

dEtot = dEp + dEk
dEtot = np.append(dEtot,0.0)

dEtot_over_Etot = dEtot / np.abs(Ep+Ek)

E0 = np.ones(nline)*Ebeg
Etot_over_E0 =  (Ep+Ek-E0) / np.abs(E0)


out =[]
for i in range(nline):
    out=np.append(out,time[i])
    out=np.append(out,dEtot_over_Etot[i])
    out=np.append(out,Etot_over_E0[i])

out=out.reshape(nline,3)

np.savetxt("energy_diff.txt",out,fmt="%.6e")