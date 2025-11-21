#!/usr/bin/env python
# -*- coding:utf-8 -*-
#In[] modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric
# from astroquery.gaia import Gaia # This may print notice
# from astroquery.vizier import Vizier

import analysis_data_distribution as ads

#In[] settings
Dim = 3
q_pers_see = [0., 10., 50., 90., 100.]
q_pers_see_more = [0., 1., 10., 50., 90., 99., 100.]
col_actions = 78
col_frequencies = col_actions+7
# galaxy_name = sys.argv[1]
# snapshot_ID = int(sys.argv[2])
snapshot_ID = 10 #fixed

path_observed = "../data/samples_observed/"
file_path_new_download = path_observed+"stellar_data_gaia_new_download.csv"
file_path_gaiavizier = path_observed+"stellar_data_download_cone_20p_m10p.csv"
# file_path_gaiavizier = path_observed+"stellar_data_download_cone_200p_m15p.csv"
# file_path_gaiavizier = path_observed+"stellar_data_download_dist_500p.csv"
file_path_gaiaquery = path_observed+"stellar_data_download_cone_0p_0p.csv"
# file_path_original = path_observed+"stellar_data_download_selected.csv"
# file_path_original = "path_observed+"stellar_data_download_cone_0p_0p.csv"
# file_path_original = path_observed+"stellar_data_download_obj_0p_0p.csv"
# file_path_pos = path_observed+"stellar_data_pos.csv"
file_path_6D_SkyCoor = path_observed+"stellar_data_6D_SkyCoor.csv" #remove nan pos
file_path_6D_Cartesian = path_observed+"stellar_data_6D_Cartesian.csv" #might screen pos

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
colnames_6D_Vizier = ['RAJ2000', 'DEJ2000', 'Plx', 'pmRA', 'pmDE', 'RV']
colnames_6D_Cartesian = ["x", "y", "z", "vx", "vy", "vz"]

introduction_labels_Gaia = """
https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html?utm_source=chatgpt.com
ra (Right Ascension): degrees
dec (Declination): degrees
dist (Distance): ?? parsecs
pmra (Proper Motion in Right Ascension): milliarcseconds per year (mas/yr)
pmdec (Proper Motion in Declination): milliarcseconds per year (mas/yr)
radial_velocity: kilometers per second (km/s)
"""

introduction_labels_Vizier = """
The list of columns specified (['RAJ2000', 'DEJ2000', 'Plx', 'pmRA', 'pmDE', 'RV']) corresponds to different parameters related to the observed stars:
"RAJ2000": Right Ascension (RA) in the J2000 equatorial coordinate system, with units in degrees. It specifies the celestial longitude of the star.
"DEJ2000": Declination (Dec) in the J2000 equatorial coordinate system, also in degrees. It specifies the celestial latitude of the star.
"Plx": ?? Parallax, in milliarcseconds (mas). It is used to calculate the distance to the star, typically given by D=1/p in parsecs (pc).
"pmRA": Proper motion in Right Ascension, in milliarcseconds/year (mas/yr). It describes how fast the star appears to move across the sky along the RA axis.
"pmDE": Proper motion in Declination, in milliarcseconds/year (mas/yr). It describes how fast the star appears to move across the sky along the Dec axis.
"RV": Radial velocity, in km/s. It represents the speed of the star along the line of sight, indicating whether the star is moving towards or away from the observer.
These variables correspond as follows:
RAJ2000 corresponds to ra.
DEJ2000 corresponds to dec.
Plx (parallax) can be used to determine dist (distance).
pmRA and pmDE correspond to pmra and pmdec, respectively.
RV corresponds to radial_velocity.
"""

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
# Function to io data
def screen_by_SkyCoor_range(data, Sky_down=None, Sky_up=None):
    '''
    Screen positions by range in Sky coordinate.

    @param data: an (N,3) array.
    @param Sky_down: an (3,) list.
    @param Sky_up: an (3,) list.
    @param Sky_down, @param Sky_up and each raw of @param data is [ra (deg), dec (deg), dist (pc)].
    '''
    sd = [0., -90., 0.]
    su = [360., 90., np.inf]
    if Sky_down is None:
        Sky_down = sd
    else:
        if Sky_down[0] is None:
            Sky_down[0] = sd[0]
        if Sky_down[1] is None:
            Sky_down[1] = sd[1]
        if Sky_down[2] is None:
            Sky_down[2] = sd[2]
    if Sky_up is None:
        Sky_up = su
    else:
        if Sky_up[0] is None:
            Sky_up[0] = su[0]
        if Sky_up[1] is None:
            Sky_up[1] = su[1]
        if Sky_up[2] is None:
            Sky_up[2] = su[2]

    mask = np.ones(len(data)).astype(bool)
    for i in range(3):
        mask &= (data[:,i]>=Sky_down[i]) & (data[:,i]<=Sky_up[i])
        # ads.DEBUG_PRINT_V(1, mask, len(mask.astype(int)))
    indices = np.where(mask)[0]
    return data[indices], indices

def save_csv_6D_Cartesian(xv_SkyCoor, file_path_save, is_only_pos=True):
    # xv_Cartesian = xv_SkyCoor #note: not copy
    xv_Cartesian = xv_SkyCoor.copy()
    if is_only_pos:
        col = list(range(6))
        col[0:3] = colnames_6D_Cartesian[0:3]
        col[3:6] = colnames_6D_Gaia[3:6]
        xv_Cartesian.columns = col
    else:
        xv_Cartesian.columns = colnames_6D_Cartesian #note: columns[3:6] is not processed
    positions = (xv_SkyCoor.to_numpy())[:, 0:3]
    positions = convert_coordiante_SkyCoord_to_Cartesian(positions)
    xv_Cartesian[colnames_6D_Cartesian[0:3]] = positions
    xv_Cartesian.to_csv(file_path_save, index=False)
    ads.DEBUG_PRINT_V(1, xv_Cartesian.loc[0:10], "xv_Cartesian")
    return 0

def save_csv_gaia_vizier_to_colnames_6D_Gaia(data, file_path_save, is_pc_to_kpc=False):
    data_cols = data[colnames_6D_Vizier] #note: not copy
    # data_cols = data[colnames_6D_Vizier].copy()
    ads.DEBUG_PRINT_V(1, data_cols.columns, np.shape(data_cols), "shape initial")
    data_cols.loc[:, "Plx"] = convert_parallax_to_distance(data_cols.to_numpy()[:,2])
    data_cols = data_cols.dropna(subset=colnames_6D_Vizier[0:3])
    data_cols.columns = colnames_6D_Gaia
    # data_cols = pd.DataFrame(data_cols, columns=colnames_6D_Gaia) #??
    # ads.DEBUG_PRINT_V(0, np.percentile(data_cols.to_numpy()[:,2], q=q_pers_see))
    if is_pc_to_kpc:
        data_cols.loc[:, "dist"] = data_cols.to_numpy()[:,2]*1e3 #kpc to pc
    ads.DEBUG_PRINT_V(1, data_cols.columns, np.shape(data_cols), "shape new")
    data_cols.to_csv(file_path_save, index=False)
    print("Save csv file, done.")
    return 0

def save_csv_gaia_gaiaquery_to_colnames_6D_Gaia(data, file_path_save, is_pc_to_kpc=False):
    data_cols = data[colnames_6D_Gaia] #note: not copy
    # data_cols = data[colnames_6D_Gaia].copy()
    if is_pc_to_kpc:
        data_cols.loc[:, "dist"] = data_cols.to_numpy()[:,2]*1e3 #kpc to pc
    data_cols = pd.DataFrame(data_cols, columns=colnames_6D_Gaia)
    data_cols.to_csv(file_path_save, index=False)
    print("Save csv file, done.")
    return 0

def save_csv_with_cols_Gaia(data, file_path_save, colnames=None, if_max_seq_items=False):
    if if_max_seq_items:
        pd.options.display.max_seq_items = None #display complete columns names
    data.info()
    ads.DEBUG_PRINT_V(1, data.columns, colnames, "cols cols")
    data_cols = None
    if colnames is not None:
        data_cols = data[colnames]
    else:
        data_cols = data
    # ads.DEBUG_PRINT_V(1, data.to_numpy()[0,:], data_cols.to_numpy()[0,:], "savecols")
    data_cols.to_csv(file_path_save, index=False)
    return 0

def load_data_pd(file_path, names=None):
    """
    Load data from a file with possible missing values.

    Parameters:
    - file_path (str): Path to the data file.

    Returns:
    - data (pd.DataFrame): DataFrame with loaded data and missing values handled.
    """
    # Reading the data file using pandas with fixed-width formatting
    try:
        data = pd.read_fwf(
            file_path, header=None, names=names, #["Col{}".format(i) for i in range(37)], 
            na_values=['', ' ', 'nan'], keep_default_na=True
        )
        print(data.head()) #display the names and the first several lines
    except Exception as e:
        print(f"Error occurred while loading data: {e}")
        return None
    return data

def save_6D_data_Angus22(file_path, file_path_xv):
    '''
    Save positions and velocities data from catalog to txt.
    '''
    data = load_data_pd(file_path, names=data_labels_Angus22)
    column_names = ["Dist", "RAdeg", "DEdeg", "vx-inf", "vy-inf", "vz-inf"]
    selected_array = data[column_names].to_numpy()
    np.savetxt(file_path_xv, selected_array)
    return selected_array

# Function to download stellar data within a certain radius around the Sun
def download_stellar_data_within_radius(radius_pc):
    """
    Download stellar data near the Sun within a given radius.

    Parameters:
    - radius_pc (float): Radius in parsecs within which to query stellar data.

    Returns:
    - result (Table): A table containing the stellar data (RA, Dec, Distance).
    """
    
    # query2
    # Convert radius to equivalent parallax in milliarcseconds
    parallax_limit = 1000 / radius_pc  # in milliarcseconds

    # Construct the ADQL query to select stars within the given radius
    query2 = f"""
        SELECT TOP 10000
            source_id, ra, dec, parallax
        FROM gaiadr3.gaia_source
        WHERE parallax >= {parallax_limit}
    """

    # # query 1
    # # Define the search radius around the Sun in parsecs
    # coord = SkyCoord(ra=0, dec=0, distance=radius_pc * u.pc, frame='galactocentric')
    # radius = radius_pc * u.pc
    # dist_limit = radius_pc

    # # Construct the ADQL query to select stars within the given radius
    # query1 = f"""
    #     SELECT TOP 10000
    #         source_id, ra, dec, parallax
    #     FROM gaiadr3.gaia_source
    #     WHERE parallax >= {1 / radius_pc}
    # """

    # Execute the query
    job = Gaia.launch_job_async(query2)
    result = job.get_results()
    
    return result

def download_Gaia_data_object_async(width=0.1, height=0.1, ra=0., dec=0.):
    """
    Download stellar data near the Sun within a given radius using a cone search.
    """
    Gaia.ROW_LIMIT = -1  # Ensure the default row limit.
    # Gaia.ROW_LIMIT = 50  # Ensure the default row limit.
    # Gaia.ROW_LIMIT = 1000000  # Ensure the default row limit.
    Vizier.ROW_LIMIT = -1  # No limit on the number of rows returned

    # Define the search center as the Sun's position (RA=0, Dec=0) and radius
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    width = u.Quantity(0.1, u.deg)
    height = u.Quantity(0.1, u.deg)
    print("Try to download data ...")
    result = Gaia.query_object_async(coordinate=coord, width=width, height=height)
    result.pprint(max_lines=10, max_width=130)
    print("Download data, done.")
    return result

def download_gaia_data_query_distance(distance_pc=500.):
    """
    Search and download Gaia stellar sources data whose distance (corresponding parallax) is around 500 pc.

    Parameters:
    - distance_pc (float): Distance to the Earth in parsecs.

    Returns:
    - result (DataFrame): A DataFrame containing the stellar data.
    """
    # Calculate the parallax value corresponding to the given distance in parsecs
    plx_value = 1000.0 / distance_pc  # Parallax in milliarcseconds

    # Set up Vizier query with Gaia DR3
    Gaia.ROW_LIMIT = -1  # Ensure the default row limit.
    # Vizier.ROW_LIMIT = -1  # No limit on the number of rows returned
    # catalog = 'I/355/gaiadr3'  # Gaia DR3 catalog

    # Query by parallax
    print("Try to download data ...")
    job = Gaia.launch_job_async("""
        SELECT * 
        FROM gaiadr3.gaia_source 
        WHERE parallax >= 2.0
    """) #??
    result = job.get_results()
    print("Download data, done.")

    print(result)
    return result

def download_gaia_data_vizier_distance(distance_pc=500.):
    """
    Search and download Gaia stellar sources data whose distance (corresponding parallax) is around 500 pc.

    Parameters:
    - distance_pc (float): Distance to the Earth in parsecs.

    Returns:
    - result (DataFrame): A DataFrame containing the stellar data.
    """
    # Calculate the parallax value corresponding to the given distance in parsecs
    plx_value = 1000.0 / distance_pc  # Parallax in milliarcseconds

    # Set up Vizier query with Gaia DR3
    Gaia.ROW_LIMIT = -1  # Ensure the default row limit.
    Vizier.ROW_LIMIT = -1  # No limit on the number of rows returned
    catalog = 'I/355/gaiadr3'  # Gaia DR3 catalog

    # Query by parallax
    print("Try to download data ...")
    v = Vizier(columns=['RAJ2000', 'DEJ2000', 'Plx', 'pmRA', 'pmDE', 'RV'])
    # result = v.query_constraints(catalog=catalog, Plx=f'>{plx_value - 0.1} & <{plx_value + 0.1}')
    # result = v.query_constraints(catalog=catalog, Plx=f'<{plx_value}')
    result = v.query_constraints(catalog=catalog, Plx=f'>{plx_value}')
    print("Download data, done.")
    print(result)
    return result[0] if result else None

def download_Gaia_data_vizier_radec(ra=0., dec=0., radius_deg=1.):
    """
    Download specific columns of Gaia data from Vizier using a cone search-like approach.

    Parameters:
    - ra (float): Right ascension of the search center in degrees.
    - dec (float): Declination of the search center in degrees.
    - radius_deg (float): Radius in degrees within which to query Gaia data.

    Returns:
    - result (Table): A table containing the specified stellar data columns (RA, Dec, Distance, pmRA, pmDec, radial velocity).
    """
    # Define Vizier object with the desired columns
    v = Vizier(columns=colnames_6D_Vizier)
    v.ROW_LIMIT = -1  # No row limit

    # Define the search coordinates and radius
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    radius = u.Quantity(radius_deg, u.deg)

    # Perform the query
    print("Try to download data ...")
    result = v.query_region(coord, radius=radius, catalog='I/355/gaiadr3')
    print("Download data, done.")
    print(result)
    return result[0] if result else None

def download_Gaia_data_cone_search(radius_deg=1., ra=0., dec=0.):
    """
    Download stellar data near the Sun within a given radius using a cone search.

    Parameters:
    - radius_deg (float): Radius in deg within which to query stellar data.

    Returns:
    - result (Table): A table containing the stellar data (RA, Dec, Distance).
    """
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, default
    Gaia.ROW_LIMIT = -1  # Ensure the default row limit.
    # Gaia.ROW_LIMIT = 1000000  # Ensure the default row limit.

    # Define the search center as the Sun's position (RA=0, Dec=0) and radius
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    radius = u.Quantity(radius_deg, u.deg)
    print("Try to download data ...")
    job = Gaia.cone_search_async(coord, radius=radius)
    result = job.get_results()
    print("Download data, done.")
    print(result)
    return result

def download_Gaia_data_cone_search_only_6D(radius_deg=1., ra=0., dec=0.):
    """
    Download stellar data near the given RA/Dec within a given radius using a cone search.

    Parameters:
    - radius_deg (float): Radius in degrees within which to query stellar data.
    - ra (float): Right Ascension of the cone center in degrees.
    - dec (float): Declination of the cone center in degrees.

    Returns:
    - result (Table): A table containing the selected stellar data (RA, Dec, Distance, proper motions, radial velocity).
    """
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, default
    Gaia.ROW_LIMIT = -1  # Ensure no row limit.

    # Define the search center as the given RA/Dec and radius
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    radius = u.Quantity(radius_deg, u.deg)

    # Define the columns to retrieve
    columns = colnames_6D_Gaia
    query = f"""SELECT 
            {', '.join(columns)} 
        FROM gaiadr3.gaia_source 
        WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius.to(u.deg).value}))=1
    """

    # Perform a cone search using Gaia archive
    print("Try to download data ...")
    job = Gaia.launch_job_async(query)
    result = job.get_results()
    
    print("Download data, done.")
    print(result)
    return result

def download_Gaia_data_dist_range_only_6D(radius_deg=1., ra=0., dec=0., dist_down_pc=0., dist_up_pc=np.inf):
    """
    Download stellar data near the given RA/Dec within a given radius using a cone search.

    Parameters:
    - radius_deg (float): Radius in degrees within which to query stellar data.
    - ra (float): Right Ascension of the cone center in degrees.
    - dec (float): Declination of the cone center in degrees.

    Returns:
    - result (Table): A table containing the selected stellar data (RA, Dec, Distance, proper motions, radial velocity).
    """
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, default
    Gaia.ROW_LIMIT = -1  # Ensure no row limit.

    # Define the search center as the given RA/Dec and radius
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    radius = u.Quantity(radius_deg, u.deg)

    # Define the columns to retrieve
    columns = colnames_6D_Gaia
    query = f"""SELECT 
            {', '.join(columns)} 
        FROM gaiadr3.gaia_source 
        WHERE dist <= {dist_up_pc}
    """
    # WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius.to(u.deg).value}))=1

    # Perform a cone search using Gaia archive
    print("Try to download data ...")
    job = Gaia.launch_job_async(query)
    result = job.get_results()
    
    print("Download data, done.")
    print(result)
    return result

# Convert coordinates
def convert_coordiante_SkyCoord_to_Cartesian(positions_SkyCoord):
    """
    https://docs.astropy.org/en/stable/coordinates/galactocentric.html
    Convert coordinates from (Distance, RA, Dec) to Cartesian (x, y, z) with origin at the Galactic Center.

    Parameters:
    - ra_deg (float or array): Right ascension in degrees.
    - dec_deg (float or array): Declination in degrees.
    - distance_pc (float or array): Distance in parsecs.

    Returns:
    - x, y, z (arrays): Cartesian coordinates in kiloparsecs.
    """
    # Create SkyCoord object with the given coordinates
    ra_deg, dec_deg, distance_pc = positions_SkyCoord[:, 0], positions_SkyCoord[:, 1], positions_SkyCoord[:, 2]
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, distance=distance_pc * u.pc, frame='icrs')
    
    # Transform to Galactocentric frame
    galactocentric = c.transform_to(Galactocentric())
    
    # Extract Cartesian coordinates in kpc
    x = galactocentric.x.to(u.kpc).value
    y = galactocentric.y.to(u.kpc).value
    z = galactocentric.z.to(u.kpc).value
    positions_Cartesian = np.vstack((x, y, z)).T
    return positions_Cartesian

def convert_coordinate_SkyCoor_to_GC(xv):
    '''
    https://astroquery.readthedocs.io/en/latest/gaia/gaia.html
    '''
    xv_Cartesian = xv #??
    return xv_Cartesian

def convert_parallax_to_distance(Plx):
    """
    Convert parallax in milliarcseconds to distance in parsecs.

    Parameters:
    - Plx (float or array): Parallax in milliarcseconds.

    Returns:
    - distance (float or array): Distance in parsecs.
    """
    # Avoid division by zero for very small or zero parallax
    with np.errstate(divide='ignore', invalid='ignore'):
        dist = np.where(Plx > 0, 1000.0 / Plx, np.nan)
    return dist

def percentile_each_col(data):
    n = np.shape(data)[1]
    pers = list(range(n))
    for i in range(n):
        pers[i] = np.percentile(data[:,i], q=q_pers_see)
    return pers

def plot_funcs_1d(datalist, xlabel="xlabel", ylabel="ylabel", zlabel="zlabel", suffix="suffix"):
    figsize = 20, 15 #for 3, 3
    fontsize = 40.
    pointsize = 32.
    # dpi = None
    dpi = 200
    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = None
    # projection = "3d"
    ax1 = fig.add_subplot(1, 1, 1, projection=projection)
    ax1.grid(True)
    for dl in datalist:
        x = dl[0]
        y = dl[1]
        label = dl[2]
        ax1.scatter(x, y, s=pointsize, label=label)
        ax1.legend(fontsize=fontsize)
        ax1.set_xlabel("{}".format(xlabel), fontsize=fontsize)
        ax1.set_ylabel("{}".format(ylabel), fontsize=fontsize)
        # ax1.set_zlabel("{}".format(zlabel), fontsize=fontsize)
        # ax1.set_title("Cartesian positions of stars, kpc")
        # ax1.view_init(elev=130., azim=110.) #view from behind side
    plt.tick_params(which='major', length=0, labelsize=fontsize*0.8) #size of the number characters
    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/plot_funcs_"+suffix+".png", format="png", bbox_inches='tight')
    return 0

def plot_data_of_observed_stars(xv, proj=[0,1], labels_pos=None, labels_vel=None, is_plot_refer=False, suffix="suffix"):
    '''
    @param xv: an (N,6) array or an (N,3) array.
    '''
    Cartesians = ["x", "y", "z", "vx", "vy", "vz"]
    if labels_pos is None:
        labels_pos = [Cartesians[0+proj[0]], Cartesians[0+proj[1]]]
    if labels_vel is None:
        labels_vel = [Cartesians[3+proj[0]], Cartesians[3+proj[1]]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.scatter(xv[:, 0+proj[0]], xv[:, 0+proj[1]], c='blue', marker="+", s=2, alpha=0.5, label="stars")
    if is_plot_refer:
        ax1.scatter([pos_Obs_in_GC_Cartesian[0+proj[0]]], [pos_Obs_in_GC_Cartesian[0+proj[1]]], c='green', marker="+", s=20, alpha=1.0, label="Earth")
        ax1.scatter([pos_Sun_in_GC_Cartesian[0+proj[0]]], [pos_Sun_in_GC_Cartesian[0+proj[1]]], c='orange', marker="+", s=20, alpha=1.0, label="Sun")
        ax1.scatter([pos_GC_in_GC_Cartesian[0+proj[0]]], [pos_GC_in_GC_Cartesian[0+proj[1]]], c='k', marker="+", s=20, alpha=1.0, label="GalacticCenter")
    ax1.set_title("Cartesian positions of stars")
    ax1.set_xlabel("{}".format(labels_pos[0]))
    ax1.set_ylabel("{}".format(labels_pos[1]))
    ax1.grid(True)
    ax1.legend()
    if np.shape(xv)[1]>=6:
        ax2.scatter(xv[:, 3+proj[0]], xv[:, 3+proj[1]], c='red', s=2, alpha=0.5)
        ax2.set_title("Cartesian velocities of stars")
        ax1.set_xlabel("{}".format(labels_vel[0]))
        ax1.set_ylabel("{}".format(labels_vel[1]))
        ax2.grid(True)
        ax2.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/obsstar_xy_"+suffix+".png", format="png", bbox_inches='tight')
    return 0

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

    ## data download
    # radius_deg = 1. #deg #1e4
    # radius_deg = 5. #deg
    radius_deg = 10. #deg
    # radius_deg = 20. #deg
    # ra_cone, dec_cone = 0., 0. #Sun
    # ra_cone, dec_cone = 266., -29. #GC
    # ra_cone, dec_cone = 200., -15. #some
    ra_cone, dec_cone = 20., -10. #some
    # distance_pc = 5.
    # distance_pc = 50.
    distance_pc = 500.
    # distance_pc = 5000.

    # stellar_data = download_Gaia_data_vizier_radec(ra=ra_cone, dec=dec_cone, radius_deg=radius_deg)
    
    # # stellar_data = download_Gaia_data_dist_range_only_6D(dist_up_pc=distance_pc)
    # # stellar_data = download_gaia_data_vizier_distance(distance_pc=distance_pc)
    
    # # stellar_data = download_Gaia_data_cone_search_only_6D(radius_deg=radius_deg, ra=ra_cone, dec=dec_cone)
    # # stellar_data = download_Gaia_data_cone_search(radius_deg=radius_deg, ra=ra_cone, dec=dec_cone)
    
    # # width_object_async = 10.
    # # height_object_async = 10.
    # # stellar_data = download_Gaia_data_object_async(width=width_object_async, height=height_object_async, ra=ra_cone, dec=dec_cone)

    # stellar_data.write(file_path_new_download, format="csv", overwrite=True)
    # print("Save download data, done.")
    


    # # data preprocess
    # os.rename(file_path_new_download, file_path_gaiavizier)
    # # stellar_data = pd.read_csv(file_path_gaiaquery)
    # # save_csv_gaia_gaiaquery_to_colnames_6D_Gaia(stellar_data, file_path_6D_SkyCoor)
    # stellar_data = pd.read_csv(file_path_gaiavizier)
    # save_csv_gaia_vizier_to_colnames_6D_Gaia(stellar_data, file_path_6D_SkyCoor)

    # positions_SkyCoor = pd.read_csv(file_path_6D_SkyCoor).to_numpy()
    # pers = percentile_each_col(positions_SkyCoor)
    # ads.DEBUG_PRINT_V(1, pers, "pers")
    # dist_mean = np.median(positions_SkyCoor[:,2])
    # Sky_down = [None, None, dist_mean*(1.-5./57.)]
    # Sky_up = [None, None, dist_mean*(1.+5./57.)]
    # positions_SkyCoor, _ = screen_by_SkyCoor_range(positions_SkyCoor, Sky_down=Sky_down, Sky_up=Sky_up)
    # ads.DEBUG_PRINT_V(1, dist_mean, np.shape(positions_SkyCoor))
    # positions_SkyCoor = pd.DataFrame(positions_SkyCoor) #to csv
    # save_csv_6D_Cartesian(positions_SkyCoor, file_path_6D_Cartesian, is_only_pos=True)
    # positions_Cartesian = pd.read_csv(file_path_6D_Cartesian)
    # ads.DEBUG_PRINT_V(1, positions_Cartesian.columns)



    ## data plot
    positions_SkyCoor = pd.read_csv(file_path_6D_SkyCoor).to_numpy()[:,0:3]
    positions_Cartesian = pd.read_csv(file_path_6D_Cartesian).to_numpy()[:,0:3]
    pers1 = percentile_each_col(positions_SkyCoor)
    pers2 = percentile_each_col(positions_Cartesian)
    ads.DEBUG_PRINT_V(1, pers1, pers2, "pers")
    ads.DEBUG_PRINT_V(1, np.shape(positions_SkyCoor), np.shape(positions_Cartesian), "positions shapes")
    plot_data_of_observed_stars(positions_SkyCoor, proj=[0,1], labels_pos=["ra", "dec"], suffix="pos_Gaia_radec")
    plot_data_of_observed_stars(positions_Cartesian, proj=[0,1], suffix="pos_Gaia_xy")
    # plot_data_of_observed_stars_3d(positions_Cartesian, suffix="pos_Gaia_xyz")





    # ## old
    # # positions = np.loadtxt("../data/table1_Angus22_xv_Cartesian.txt")
    # # positions = positions[np.where(positions[:,2]>0.6)[0]]
    # positions_SkyCoor = pd.read_csv(file_path_6D_removenan).to_numpy()
    # positions = pd.read_csv(file_path_pos).to_numpy()

    # x = positions
    # # x_center = np.array([pos_Obs_in_GC_Cartesian])
    # x_center = np.mean(positions, axis=0)
    # # x_center = (np.array([pos_Obs_in_GC_Cartesian])+np.mean(positions, axis=0))/2.
    # radius = 0.3
    # r = ads.norm_l(x-x_center, axis=1)
    # mask = (r<=radius)
    # indices = np.where(mask)[0]
    # positions_SkyCoor = positions_SkyCoor[indices]
    # positions = positions[indices]

    # plot_data_of_observed_stars(positions, proj=[0,1], suffix="pos_Gaia_xy")
    # plot_data_of_observed_stars(positions, proj=[0,2], suffix="pos_Gaia_xz")
    # plot_data_of_observed_stars_3d(positions, suffix="pos_Gaia_xyz")

    # pS = convert_coordiante_SkyCoord_to_Cartesian(np.array([pos_Sun_in_SkyCoor]))
    # pE = convert_coordiante_SkyCoord_to_Cartesian(np.array([pos_Obs_in_SkyCoor]))
    # pG = convert_coordiante_SkyCoord_to_Cartesian(np.array([pos_GC_in_SkyCoor]))
    # ads.DEBUG_PRINT_V(1, pE, pS, pG)
    # index_sort = np.argsort(positions_SkyCoor[:,2])
    # positions_SkyCoor = positions_SkyCoor[index_sort]
    # value_pers = np.percentile(positions_SkyCoor[:,2], [0., 0.1, 1., 50., 99., 99.9, 100.])
    # ads.DEBUG_PRINT_V(1, positions_SkyCoor[0:2], value_pers)

    # ## Angus22 data
    # file_path = "../data/table1_Angus.dat"
    # file_path_xv_data = "../data/table1_Angus22_xv_data.txt"
    # # save_6D_data_Angus22(file_path, file_path_xv_data)
    # xv_data = np.loadtxt(file_path_xv_data)
    # ads.DEBUG_PRINT_V(1, xv_data[0:10], np.shape(xv_data), "np.shape(xv_data)")
    
    # file_path_xv_convert = "../data/table1_Angus22_xv_Cartesian.txt"
    # # x_Sun, y_Sun, z_Sun = convert_coordiante_SkyCoord_to_Cartesian(0., 0., 0.)
    # # ads.DEBUG_PRINT_V(1, x_Sun, y_Sun, z_Sun, "xyz Sun")
    # # x, y, z = convert_coordiante_SkyCoord_to_Cartesian(xv_data[0:10,0], xv_data[0:10,1], xv_data[0:10,2])
    # x, y, z = convert_coordiante_SkyCoord_to_Cartesian(xv_data[:,0], xv_data[:,1], xv_data[:,2])
    # xv = xv_data*1.
    # xv[:,0] = x; xv[:,1] = y; xv[:,2] = z
    # np.savetxt(xv, file_path_xv_convert)
    # ads.DEBUG_PRINT_V(1, xv[0:10], "xv")
