# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:47:42 2021

@author: Samuli Launiainen
"""
import sys
import pandas as pd
import numpy as np
#from scipy.stats import binned_statistic

#from .vegetation import (
#    naslund_heightcurve,
#    marklund,
#    crownheight,
#    crown_biomass_distr
#)


#def tree_data2(dbhfile, z, dbh_bins=4, species='pine'):
#    """
#    reads runkolukusarjat from Hyde and creates lad-profiles for pine, spruce and decid.
#    Uses 2008 data
#    Args:
#        z - grid (m)
#        quantiles - cumulative frequency thresholds for grouping trees
#    Returns:
#        res (dict)
#            species (str)
#            lad (array) - leaf-area density, integrates to 1
#            leaf_mass (array)- average leaf mass (kg) per tree
#            N (array)- trees ha-1
#            
#    """
#    dat = np.loadtxt(dbhfile, skiprows=1)
#    
#    # use year 2008 data
#    if species == 'pine':
#        data = dat[:, [0, 2]]
#    if species == 'spruce':
#        data = dat[:, [0, 4]]
#    if species == 'birch':
#        data = dat[:, [0, 6]]
#
#    # bin dbh-data to k dbh_bins. As allometry = non-linear f(dbh), the binning results to biases in biomass components as
#    # computed for midpoints of dbh-bins. Need to be re-thought.
#    
#    x = binned_statistic(data[:,0], data[:,1], statistic='sum', bins=dbh_bins)
#    dbh = x[1][0:-1] + np.diff(x[1])
#    N = np.fix(x[0])
#
#    k = len(N)
#    h = np.ones(k)*np.NaN
#    ht = np.ones(k)*np.NaN
#    leaf_mass = np.ones(k)*np.NaN
#    lad = np.ones((k, len(z)))*np.NaN
#    
#    #
#    for m in range(k):
#        h[m] = naslund_heightcurve(dbh[m], species)
#        ht[m] = crownheight(h[m], species)
#        l, _ = crown_biomass_distr(species,z,htop=h[m],hbase=ht[m])
#        lad[m,:] = l
#        bm, _ = marklund(dbh[m], species)
#        leaf_mass[m] = bm[3];
#        del bm
#
#    return {'species': species, 'h': h, 'lad': lad,
#            'leaf_mass': leaf_mass, 'N': N
#           }
    
def read_forcing(forc_fp, start_time, end_time,
                 dt=1800.0, na_values='NaN', sep=';'):
    """
    Reads forcing data from to dataframe
    Args:
        forc_fp (str): forcing file name
        start_time (str): starting time [yyyy-mm-dd], if None first date in
            file used
        end_time (str): ending time [yyyy-mm-dd], if None last date
            in file used
        dt (float): time step [s], if given checks
            that dt in file is equal to this
        na_values (str/float): nan value representation in file
        sep (str): field separator
    Returns:
        Forc (dataframe): dataframe with datetime as index and cols read from file
    """

    # filepath
    dat = pd.read_csv(forc_fp, header='infer', na_values=na_values, sep=sep)

    # set to dataframe index
    tvec = pd.to_datetime(dat[['year', 'month', 'day', 'hour', 'minute']])
    tvec = pd.DatetimeIndex(tvec)
    dat.index = tvec

    dat = dat[(dat.index >= start_time) & (dat.index <= end_time)]

    # convert: H2O mmol / mol --> mol / mol; Prec kg m-2 in dt --> kg m-2 s-1
    dat['H2O'] = 1e-3 * dat['H2O']
    dat['Prec'] = dat['Prec'] / dt

    cols = ['doy', 'Prec', 'P', 'Tair', 'Tdaily', 'U', 'Ustar', 'H2O', 'CO2', 'Zen',
            'LWin', 'diffPar', 'dirPar', 'diffNir', 'dirNir']
    # these for phenology model initialization
    if 'X' in dat:
        cols.append('X')
    if 'DDsum' in dat:
        cols.append('DDsum')

    # for case for bypassing soil computations
    if 'Tsh' in dat:
        cols.append('Tsh')
    if 'Wh' in dat:
        cols.append('Wh')
    if 'Tsa' in dat:
        cols.append('Tsa')
    if 'Ws' in dat:
        cols.append('Ws')
    if 'Rew' in dat:
        cols.append('Rew')        
    # Forc dataframe from specified columns
    Forc = dat[cols].copy()

    # Check time step if specified
    if len(set(Forc.index[1:]-Forc.index[:-1])) > 1:
        sys.exit("Forcing file does not have constant time step")
    if (Forc.index[1] - Forc.index[0]).total_seconds() != dt:
        sys.exit("Forcing file time step differs from dt given")

    return Forc

def read_data(ffile, start_time=None, end_time=None, na_values='NaN', sep=';'):

    dat = pd.read_csv(ffile, header='infer', na_values=na_values, sep=sep)
    # set to dataframe index
    tvec = pd.to_datetime(dat[['year', 'month', 'day', 'hour', 'minute']])
    tvec = pd.DatetimeIndex(tvec)
    dat.index = tvec

    # select time period
    if start_time is None:
        start_time = dat.index[0]
    if end_time is None:
        end_time = dat.index[-1]

    dat = dat[(dat.index >= start_time) & (dat.index <= end_time)]

    return dat

def e_sat(T):
    """
    Computes saturation vapor pressure (Pa), slope of vapor pressure curve
    [Pa K-1]  and psychrometric constant [Pa K-1]
    IN:
        T - air temperature (degC)
    OUT:
        esa - saturation vapor pressure in Pa
        s - slope of saturation vapor pressure curve (Pa K-1)
    SOURCE:
        Campbell & Norman, 1998. Introduction to Environmental Biophysics. (p.41)
    """

    esa = 611.0 * np.exp((17.502 * T) / (T + 240.97))  # Pa
    s = 17.502 * 240.97 * esa / ((240.97 + T)**2)

    return esa, s    

def tridiag(a, b, C, D):
    """
    tridiagonal matrix algorithm
    a=subdiag, b=diag, C=superdiag, D=rhs
    """
    n = len(a)
    V = np.zeros(n)
    G = np.zeros(n)
    U = np.zeros(n)
    x = np.zeros(n)

    V[0] = b[0].copy()
    G[0] = C[0] / V[0]
    U[0] = D[0] / V[0]

    for i in range(1, n):  # nr of nodes
        V[i] = b[i] - a[i] * G[i - 1]
        U[i] = (D[i] - a[i] * U[i - 1]) / V[i]
        G[i] = C[i] / V[i]

    x[-1] = U[-1]
    inn = n - 2
    for i in range(inn, -1, -1):
        x[i] = U[i] - G[i] * x[i + 1]
    return x

def smooth(a, WSZ):
    """
    smooth a by taking WSZ point moving average.
    NOTE: even WSZ is converted to next odd number.
    """
    WSZ = int(np.ceil(WSZ) // 2 * 2 + 1)
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    x = np.concatenate((start, out0, stop))
    return x

def lad_weibul(z, LAI, h, hb=0.0, b=None, c=None, species=None):
    """
    Generates leaf-area density profile from Weibull-distribution
    Args:
        z: height array (m), monotonic and constant steps
        LAI: leaf-area index (m2m-2)
        h: canopy height (m), scalar
        hb: crown base height (m), scalar
        b: Weibull shape parameter 1, scalar
        c: Weibull shape parameter 2, scalar
        species: 'pine', 'spruce', 'birch' to use table values
    Returns:
        LAD: leaf-area density (m2m-3), array \n
    SOURCE:
        Teske, M.E., and H.W. Thistle, 2004, A library of forest canopy structure for 
        use in interception modeling. Forest Ecology and Management, 198, 341-350. 
        Note: their formula is missing brackets for the scale param.
        Here their profiles are used between hb and h
    AUTHOR:
        Gabriel Katul, 2009. Coverted to Python 16.4.2014 / Samuli Launiainen
    """
    
    para = {'pine': [0.906, 2.145], 'spruce': [2.375, 1.289], 'birch': [0.557, 1.914]} 
    
    if (max(z) <= h) | (h <= hb):
        raise ValueError("h must be lower than uppermost gridpoint")
        
    if b is None or c is None:
        b, c = para[species]
    
    z = np.array(z)
    dz = abs(z[1]-z[0])
    N = np.size(z)
    LAD = np.zeros(N)

    a = np.zeros(N)

    # dummy variables
    ix = np.where( (z > hb) & (z <= h)) [0]
    x = np.linspace(0, 1, len(ix)) # normalized within-crown height

    # weibul-distribution within crown
    cc = -(c / b)*(((1.0 - x) / b)**(c - 1.0))*(np.exp(-((1.0 - x) / b)**c)) \
            / (1.0 - np.exp(-(1.0 / b)**c))

    a[ix] = cc
    a = np.abs(a / sum(a*dz))    

    LAD = LAI * a

    # plt.figure(1)
    # plt.plot(LAD,z,'r-')      
    return LAD

def lad_constant(z, LAI, h, hb=0.0):
    """
    creates constant leaf-area density distribution from ground to h.

    INPUT:
        z: height array (m), monotonic and constant steps
        LAI: leaf-area index (m2m-2)
        h: canopy height (m), scalar
        hb: crown base height (m), scalar
     OUTPUT:
        LAD: leaf-area density (m2m-3), array
    Note: LAD must cover at least node 1
    """
    if max(z) <= h:
        raise ValueError("h must be lower than uppermost gridpoint")

    z = np.array(z)
    dz = abs(z[1]-z[0])
    N = np.size(z)
    
#    # dummy variables
#    a = np.zeros(N)
#    x = z[z <= h] / h  # normalized heigth
#    n = np.size(x)
#
#    if n == 1: n = 2
#    a[1:n] = 1.0
    
    # dummy variables
    a = np.zeros(N)
    ix = np.where( (z > hb) & (z <= h)) [0]
    if ix.size == 0:
        ix = [1]

    a[ix] = 1.0
    a = a / sum(a*dz)
    LAD = LAI * a
    return LAD
