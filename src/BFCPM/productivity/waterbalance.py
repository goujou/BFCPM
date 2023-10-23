# -*- coding: utf-8 -*-
"""Water balance module."""
import matplotlib.pyplot as plt
import numpy as np

from .constants import EPS, WATER_DENSITY

# for testing __init__
# params = {'interception': {'LAI': 4.0, 'wmax': 0.2, 'alpha': 1.28},
#          'snowpack': {'Tm': 0.0, 'Km': 2.9e-5, 'Kf': 5.8e-6, 'R': 0.05,
#                       'Tmax': 0.5, 'Tmin': -0.5, 'Sliq': 0.0, 'Sice': 0.0},
#          'organic_layer': {'DM': 0.2, 'Wmax': 10.0, 'Wmin': 0.1, 'Wcrit': 4.0,
#                            'alpha': 1.28, 'Wliq': 10.0},
#          'soil': {'depth': 0.5, 'Ksat': 1e-5,
#                     'pF': {'ThetaS': 0.50, 'ThetaR': 0.03, 'alpha': 0.06, 'n': 1.35},# Hyytiala A-horizon pF curve
#                     'MaxPond': 0.0, 'Wliq': 0.4}
#          }


class Canopywaterbudget(object):
    """
    Combines Interception, Snowpack, Organiclayer
    """

    def __init__(self, params):

        self.interception = Interception(params["interception"])
        self.snow = Snowpack(params["snowpack"])
        self.bottomlayer = OrganicLayer(params["organic_layer"])

    def run(self, dt, forcing):
        """
        solves waterbudgets aboveground
        """
        Rnc = forcing["Rnc"]  # canopy layer
        Rng = forcing["Rng"]  # ground layer
        T = forcing["Tair"]
        Prec = forcing["Prec"]

        # interception
        Prec, Interc, Ecan, imbe = self.interception.run(dt, T, Prec, Rnc)

        # snowpack
        Prec, mbe = self.snow.run(dt, T, Prec)

        # bottom layer
        Prec, Ebl = self.bottomlayer.run(dt, Prec, T, Rng, Snowcover=self.snow.SWE)

        return Prec, Ecan, Ebl


class Interception:
    """
    Big-leaf model for rainfall interception in canopy
    Evaporation follows Priestley-Taylor equilibrium evaporation
    """

    def __init__(self, p):
        self.wmax = p["wmax"]  # max storage (mm/LAI)
        self.LAI = p["LAI"]
        self.alpha = p["alpha"]  # priestley-taylor alpha (-)

        self.Wmax = self.wmax * self.LAI
        self.W = 0.0  # storage mm

    def run(self, dt, T, Prec, AE):
        """
        Calculates canopy rainfall water interception and canopy water storage changes
        Args:
            self - object
            T - air temperature (degC)
            Prec - precipitation rate (mm s-1 = kg m-2 s-1)
            AE - available energy (Wm-2)
        Returns:
            self - updated state W
            Trfall - thoughfall rate to forest floor (mm s-1)
            Interc - interception rate (mm s-1)
            Evap - evaporation from canopy store (mm s-1)
            MBE - mass balance error (mm)
        """
        Wo = self.W  # initial storage
        Prec = Prec * dt  # mm

        # interception (mm)
        Interc = (self.Wmax - self.W) * (1.0 - np.exp(-(1.0 / self.Wmax) * Prec))
        # new canopy storage, mm
        self.W = self.W + Interc
        # Throughfall to field layer mm
        Trfall = Prec - Interc

        # evaporate from canopy storage mm s-1
        if Prec > 0:
            erate = 0.0  # zero during precipitation events
        else:
            erate, L = eq_evap(AE, T)
            erate = np.maximum(0.0, self.alpha * erate / L)

        Evap = np.minimum(erate * dt, self.W)  # mm

        self.W = self.W - Evap  # new storage

        # canopy storage mass-balance error
        MBE = (self.W - Wo) - (Prec - Evap - Trfall)

        return Trfall / dt, Interc / dt, Evap / dt, MBE


class Snowpack(object):
    """
    degree-day snow model
    """

    def __init__(self, p):
        """
        Args:
            p (dict)
        """

        self.Tm = p["Tm"]  # melt temp, degC
        self.Km = p["Km"]  # melt coeff, mm d-1 degC-1
        self.Kf = p["Kf"]  # freeze coeff
        self.R = p["R"]  # max fraction of liquid water in snow
        self.Tmin = p["Tmin"]  # below all Prec is snow
        self.Tmax = p["Tmax"]  # above all Prec is liquid

        # water storages (liquid and ice) kgm-2 = mm
        self.liq = p["Sliq"]
        self.ice = p["Sice"]
        self.SWE = self.liq + self.ice

    def run(self, dt, T, Prec):
        """
        run snowpack
        Args:
            dt - [s]
            T - air temp [degC]
            Prec - precipitation rate [kg m-2 s-1 = mm s-1]
        """
        Prec = Prec * dt
        SWE0 = self.SWE

        # state of Prec depends on T
        fliq = np.maximum(
            0.0, np.minimum(1.0, (T - self.Tmin) / (self.Tmax - self.Tmin))
        )

        if T >= self.Tm:
            melt = np.minimum(self.ice, self.Km * dt * (T - self.Tm))  # mm
            freeze = 0.0
        else:
            melt = 0.0
            freeze = np.minimum(self.liq, self.Kf * dt * (self.Tm - T))  # mm

        # amount of water as ice and liquid in snowpack
        ice = np.maximum(0.0, self.ice + (1 - fliq) * Prec + freeze - melt)
        liq = np.maximum(0.0, self.liq + fliq * Prec - freeze + melt)

        trfall = np.maximum(0.0, liq - ice * self.R)  # mm

        # new state
        self.liq = np.maximum(0.0, liq - trfall)  # mm, liquid water in snow
        self.ice = ice
        self.SWE = self.liq + self.ice

        # mass-balance error mm
        mbe = (self.SWE - SWE0) - (Prec - trfall)

        return trfall / dt, mbe


class OrganicLayer(object):
    """
    Primitive model for organic layer moisture budget
    """

    def __init__(self, p):
        self.DM = p["DM"]  # kg DMm-2
        self.Wmax = p["Wmax"]  # gH2O/gDM, this equals 'field capacity'
        self.Wmin = p["Wmin"]
        self.Wcrit = p["Wcrit"]  # evaporation rate starts to decr. from here
        self.alpha = p["alpha"]  # priestley-taylor alpha

        # initial state
        self.Wliq = self.Wmax * p["Wliq"]  # g g-1
        self.WatSto = self.Wliq * self.DM  # kg m-2 = mm

    def run(self, dt, Prec, T, Rnet, Snowcover=0.0):
        """
        Args:
            dt (s)
            Prec (kg m-2 s-1)
            T (degC)
            Rnet (Wm-2)
            Snowcover - snow water equivalent, >0 sets evap to zero
        Returns:
            evap (kg m-2 s-1)
            trall (kg m-2 s-1)
        """

        # rates
        if Snowcover > 0 or Prec > 0:
            evap = 0.0
        else:
            f = np.minimum(1.0, self.Wliq / self.Wcrit)
            Eo, L = eq_evap(Rnet, T)  # Wm-2
            evap = f * self.alpha * Eo / L  # kg m-2 s-1
            evap = np.minimum((self.Wliq - self.Wmin) * self.DM / dt, evap)

        interc = np.minimum(Prec, (self.Wmax - self.Wliq) * self.DM / dt)

        trfall = Prec - interc

        self.WatSto += (interc - evap) * dt
        self.Wliq = self.WatSto / self.DM

        return trfall, evap


class Bucket(object):
    """
    Single-layer soil water bucket model (loosely following Guswa et al, 2002 WRR)
    """

    def __init__(self, p):
        """
        Args
            D - depth [m]
            Ksat - hydr. cond. [m/s]
            poros - porosity [vol/vol]
            pF - vanGenuchten water retention parameters {dict}
            MaxPond - maximum ponding depth above bucket [m]
            Wliq0 - initial water content [vol/vol]
        """

        self.D = p["depth"]
        self.Ksat = p["Ksat"]
        self.pF = p["pF"]

        self.Fc = psi_theta(self.pF, x=-1.0)
        self.Wp = psi_theta(self.pF, x=-150.0)

        self.poros = self.pF["ThetaS"]  # porosity

        # water storages
        self.MaxSto = self.poros * self.D
        self.MaxPond = p["MaxPond"]

        # initial state
        self.SurfSto = 0.0
        self.WatSto = self.D * p["Wliq"]
        self.Wliq = p["Wliq"]
        self.Wair = self.poros - self.Wliq
        self.Sat = self.Wliq / self.poros
        self.h = theta_psi(self.pF, self.Wliq)  # water potential [m]
        self.Kh = self.Ksat * hydraulic_conductivity(self.pF, self.h)

        # relatively extractable water
        self.REW = np.minimum((self.Wliq - self.Wp) / (self.Fc - self.Wp + EPS), 1.0)

    def update_state(self, dWat, dSur=0):
        """
        updates state by computed dSto [m] and dSurf [m]
        """
        self.WatSto += dWat
        self.SurfSto += dSur

        self.Wliq = self.poros * self.WatSto / self.MaxSto
        self.Wair = self.poros - self.Wliq
        self.Sat = self.Wliq / self.poros
        self.h = theta_psi(self.pF, self.Wliq)
        self.Kh = self.Ksat * hydraulic_conductivity(self.pF, self.h)
        self.REW = np.minimum((self.Wliq - self.Wp) / (self.Fc - self.Wp + EPS), 1.0)

    def run(self, dt, rr=0, et=0, latflow=0):
        """Bucket model water balance
        Args:
            dt [s]
            rr = potential infiltration [mm s-1 = kg m-2 s-1]
            et [mm s-1]
            latflow [mm s-1]
        Returns:
            infil [mm] - infiltration [m]
            Roff [mm] - surface runoff
            drain [mm] - percolation /leakage
            roff [mm] - surface runoff
            et [mm] - evapotranspiration
            mbe [mm] - mass balance error
        """
        # fluxes
        Qin = (rr + latflow) * dt / WATER_DENSITY  # m, potential inputs
        et = et * dt / WATER_DENSITY

        # free drainage from profile
        drain = min(self.Kh * dt, max(0, (self.Wliq - self.Fc)) * self.D)  # m

        # infiltr is restricted by availability, Ksat or available pore space
        infil = min(Qin, self.Ksat * dt, (self.MaxSto - self.WatSto + drain + et))

        # change in surface and soil water store
        SurfSto0 = self.SurfSto
        dSto = infil - drain - et

        # in case of Qin excess, update SurfSto and route Roff
        q = Qin - infil
        if q > 0:
            dSur = min(self.MaxPond - self.SurfSto, q)
            roff = q - dSur  # runoff, m
        else:
            dSur = 0.0
            roff = 0.0

        # update state variables
        self.update_state(dSto, dSur)

        # mass balance error
        mbe = dSto + (self.SurfSto - SurfSto0) - (rr + latflow - et - drain - roff)

        return (
            WATER_DENSITY * infil,
            WATER_DENSITY * roff,
            WATER_DENSITY * drain,
            WATER_DENSITY * mbe,
        )


def theta_psi(pF, x):
    # converts water content (m3m-3) to potential (m)

    ts = pF["ThetaS"]
    tr = pF["ThetaR"]
    a = pF["alpha"]
    n = pF["n"]
    m = 1.0 - np.divide(1.0, n)

    x = np.minimum(x, ts)
    x = np.maximum(x, tr)  # checks limits
    s = (ts - tr) / ((x - tr) + EPS)

    s = np.maximum(s, 1.0)  # Holger

    Psi = -1e-2 / a * (s ** (1.0 / m) - 1.0) ** (1.0 / n)  # m

    return Psi


def psi_theta(pF, x):
    # converts water potential (m) to water content (m3m-3)
    x = 100 * np.minimum(x, 0)  # cm
    ts = pF["ThetaS"]
    tr = pF["ThetaR"]
    a = pF["alpha"]
    n = pF["n"]
    m = 1.0 - np.divide(1.0, n)

    Th = tr + (ts - tr) / (1.0 + abs(a * x) ** n) ** m
    return Th


def hydraulic_conductivity(pF, x, Ksat=1.0):
    # Hydraulic conductivity (vanGenuchten-Mualem)
    # x = water potential [m]
    x = 100 * np.minimum(x, 0)  # cm
    a = pF["alpha"]
    n = pF["n"]
    m = 1.0 - np.divide(1.0, n)

    def relcond(x):
        Seff = 1.0 / (1.0 + abs(a * x) ** n) ** m
        r = Seff**0.5 * (1.0 - (1.0 - Seff ** (1 / m)) ** m) ** 2.0
        return r

    return Ksat * relcond(x)


def eq_evap(AE, T, P=101300.0):
    """
    Calculates the equilibrium evaporation according to McNaughton & Spriggs,\
    1986.
    INPUT:
        AE - Available energy (Wm-2)
        T - air temperature (degC)
        P - pressure (Pa)
    OUTPUT:
        equilibrium evaporation rate (Wm-2)
        lat. heat of vaporization (J kg-1)
    """
    NT = 273.15
    cp = 1004.67  # J/kg/K

    # L lat. heat of vaporization (J/kg), esa = saturation vapor pressure (Pa),
    # slope of esa (Pa K-1), psychrometric constant g (Pa K-1)
    L = 1e3 * (3147.5 - 2.37 * (T + NT))  # lat heat of vapor [J/kg]
    esa = 1e3 * (0.6112 * np.exp((17.67 * T) / (T + 273.16 - 29.66)))  # Pa

    s = 17.502 * 240.97 * esa / ((240.97 + T) ** 2)
    g = P * cp / (0.622 * L)

    # equilibrium evaporation # Wm-2 = Js-1m-2
    x = np.divide((AE * s), (s + g))

    x = np.maximum(x, 0.0)
    return x, L
