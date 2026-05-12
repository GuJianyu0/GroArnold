#!/usr/bin/env python
# -*- coding:utf-8 -*-
#In[] modules
import numpy as np
import matplotlib.pyplot as plt
import analysis_data_distribution as ads

#In[] settings
Dim = 3
col_actions = 78
col_frequencies = col_actions+7
# galaxy_name = sys.argv[1]
# snapshot_ID = int(sys.argv[2])
snapshot_ID = 10 #fixed

G = 43007.1         # Gravitational constant (kpc, km/s, 1 M_sun)
N_MW = 1e11         # stars count
M_MW = 1.37e12*0.05 # total mass of stars (1 M_sun)
Rs_MW = 20.0        # scale length (kpc)
ma = M_MW/N_MW      # Mass (arbitrary units)
v_size_Sun = 230.0  # Initial velocity (km/s)
t_during_Sun = 0.25             # Time (Gyr)
pos_Sun_in_SkyCoor = [0., 0., 4.84e-6] # RA (deg), Dec (deg), Disance (pc), with reference point being the Earth
pos_Sun_in_GC_Cartesian = [-8.12197337, 0., 0.0208] # with reference point being Galactic Center
pos_Obs_in_SkyCoor = [0., 0., 0.]
pos_Obs_in_GC_Cartesian = [-8.12197337, 0., 0.020779]
pos_GC_in_SkyCoor = [266., -29., 8.122e3]
pos_GC_in_GC_Cartesian = [0., 0., 0.]

colnames_6D_Gaia = ["ra", "dec", "dist", "pmra", "pmdec", "radial_velocity"]
colnames_6D_unit_Gaia = None

data_labels_Angus22 = [
    "Seq", "KIC", "Gaia", "RAdeg", "e_RAdeg", "DEdeg", "e_DEdeg", 
    "plx", "e_plx", "Dist", "b_Dist", "B_Dist", 
    "pmRA", "e_pmRA", "pmDE", "e_pmDE", 
    "RVel-DR2", "e_RVel-DR2", "RVel-apo", "e_RVel-apo", 
    "RVel-lam", "e_RVel-lam", 
    "vx-calc", "vx-inf", "e_vx-inf", 
    "vy-calc", "vy-inf", "e_vy-inf", 
    "vz-calc", "vz-inf", "e_vz-inf", 
    "vxvy", "vxvz", "vxlnd", "vyvz", "vylnd", "vzlnd"
]

data_description_Angus22 = '''
----------------------------------------------------------------------------
 Bytes   Format Units    Label    Explanations
----------------------------------------------------------------------------
   1-  6 I6     ---      Seq      [0/148589] Sequential number identifier
   8- 15 I8     ---      KIC      Kepler Input catalog identifier
  17- 35 I19    ---      Gaia     Gaia EDR3 identifier
  37- 43 F7.3   deg      RAdeg    [279/302] Gaia EDR3 right ascension (ICRS)
  45- 49 F5.3   arcsec e_RAdeg    [0.006/0.1] Uncertainty in RAdeg
  51- 56 F6.3   deg      DEdeg    [36/53] Gaia EDR3 declination (ICRS)
  58- 62 F5.3   arcsec e_DEdeg    [0.007/0.1] Uncertainty in DEdeg
  64- 69 F6.3   mas      plx      [0.09/18.6] Gaia EDR3 parallax
  71- 75 F5.3   mas    e_plx      [0.007/0.1] Uncertainty in plx
  77- 83 F7.1   pc       Dist     [53/18256]? Distance;
                                   Bailer-Jones+, 2021, I/352
  85- 91 F7.1   pc     b_Dist     [53/14207]? Lower bound on Dist
  93- 99 F7.1   pc     B_Dist     [53/23570]? Upper bound on Dist
 101-107 F7.3   mas/yr   pmRA     [-37.7/40.3] Gaia EDR3 proper motion in RA
 109-115 F7.3   mas/yr e_pmRA     [-38/41] Uncertainty in pmRA
 117-123 F7.3   mas/yr   pmDE     [-39.2/38.7] Gaia EDR3 proper motion in DE
 125-131 F7.3   mas/yr e_pmDE     [-40/39] Uncertainty in pmDE
 133-138 F6.1   km/s     RVel-DR2 [-393/107]? Gaia DR2 radial velocity
 140-143 F4.1   km/s   e_RVel-DR2 [0.1/20]? Uncertainty in RVel-DR2
 145-153 F9.4   km/s     RVel-apo [-589/71]? APOGEE DR16 radial velocity
 155-160 F6.4   km/s   e_RVel-apo [0.0005/2]? Uncertainty in RVel-apo
 162-169 F8.3   km/s     RVel-lam [-403.0/94.0]? LAMOST DR5 radial velocity
 171-174 F4.1   km/s   e_RVel-lam [1/89]? Uncertainty in RVel-lam
 176-182 F7.2   km/s     vx-calc  [-838/1421]? vx velocity calculated using RVel
 184-189 F6.1   km/s     vx-inf   [-307/387] vx velocity sample, inferred
                                   without RVel
 191-194 F4.1   km/s   e_vx-inf   [2/18] Standard deviation of vx
 196-202 F7.2   km/s     vy-calc  [-312/621]? vy velocity calculated using RVel
 204-208 F5.1   km/s     vy-inf   [115/289] vy velocity sample, inferred without
                                   RVel
 210-213 F4.1   km/s   e_vy-inf   [20/26] Standard deviation of vy
 215-221 F7.2   km/s     vz-calc  [-714/1090]? zx velocity calculated using RVel
 223-228 F6.1   km/s     vz-inf   [-192/194] vz velocity sample, inferred
                                   without RVel
 230-233 F4.1   km/s   e_vz-inf   [2/11] Standard deviation of vz
 235-240 F6.2   ---      vxvy     [28/225] covariance between vx and vy samples
 242-247 F6.2   ---      vxvz     [-25/97] covariance between vx and vz samples
 249-253 F5.2   ---      vxlnd    [-0.6/2] covariance between vx and
                                   ln(distance) samples
 255-260 F6.2   ---      vyvz     [54.3/206] covariance between vy & vz samples
 262-266 F5.2   ---      vylnd    [-0.35/0.16] covariance between vy and
                                   ln(distance) samples
 268-272 F5.2   ---      vzlnd    [-0.38/0.35] covariance between vz and
                                   ln(distance) samples
https://cdsarc.cds.unistra.fr/ftp/J/AJ/164/25/ReadMe
'''

#In[] functions
def plot_data_of_observed_stars_3d(xv, suffix="suffix"):
    '''
    @param xv: an (N,6) array or an (N,3) array.
    '''
    Cartesians = ["x", "y", "z", "vx", "vy", "vz"]
    proj = [0, 1, 2]
    fontsize = 20.
    pointsize = 0.2
    figsize = 20, 15 #for 3, 3
    dpi = 400
    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = "3d"
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")
    ax1.grid(True)
    ax1.scatter(xv[:, 0+proj[0]], xv[:, 0+proj[1]], xv[:, 0+proj[2]], c='blue', s=2, alpha=0.5)
    # ax1.scatter([pos_Obs_in_GC_Cartesian[0+proj[0]]], [pos_Obs_in_GC_Cartesian[0+proj[1]]], c='green', s=10, alpha=1.0)
    # ax1.scatter([pos_Sun_in_GC_Cartesian[0+proj[0]]], [pos_Sun_in_GC_Cartesian[0+proj[1]]], c='orange', s=10, alpha=1.0)
    # ax1.scatter([pos_GC_in_GC_Cartesian[0+proj[0]]], [pos_GC_in_GC_Cartesian[0+proj[1]]], c='k', s=10, alpha=1.0)
    ax1.set_title("Cartesian positions of stars, kpc")
    ax1.set_xlabel("{}".format(Cartesians[0+proj[0]]))
    ax1.set_ylabel("{}".format(Cartesians[0+proj[1]]))
    ax1.set_zlabel("{}".format(Cartesians[0+proj[2]]))
    # ax1.view_init(elev=0., azim=0.) #view default
    # ax1.view_init(elev=-125., azim=-125.) #view mean original pos
    ax1.view_init(elev=140., azim=110.) #view along obs line
    # ax1.view_init(elev=130., azim=110.) #view from behind side
    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/obsstar_xyz_"+suffix+".png", format="png", bbox_inches='tight')
    return 0

#In[] main
if __name__ == "__main__":

    ## obs data
    # positions = np.loadtxt("../data/table1_Angus22_xv_Cartesian.txt")
    # positions = positions[np.where(positions[:,2]>0.6)[0]]

    positions = np.loadtxt("../data/stellar_data_pos.txt")
    x = positions
    # x_center = np.array([pos_Obs_in_GC_Cartesian])
    x_center = np.mean(positions, axis=0)
    # x_center = (np.array([pos_Obs_in_GC_Cartesian])+np.mean(positions, axis=0))/2.
    radius = 0.3
    r = ads.norm_l(x-x_center, axis=1)
    mask = (r<=radius)
    indices = np.where(mask)[0]
    positions = x[indices]

    print(len(positions))
    plot_data_of_observed_stars_3d(positions, suffix="pos_Gaia_xyz")
