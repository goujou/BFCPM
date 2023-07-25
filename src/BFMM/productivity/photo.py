# -*- coding: utf-8 -*-
"""Leaf-scale functions for photosynthesis and stomatal control."""

import matplotlib.pyplot as plt
import numpy as np
from frozendict import frozendict

from .constants import (AIR_VISCOSITY, DEG_TO_KELVIN, EPS, GAS_CONSTANT,
                        GRAVITY, MOLECULAR_DIFFUSIVITY_CO2,
                        MOLECULAR_DIFFUSIVITY_H2O, O2_IN_AIR,
                        THERMAL_DIFFUSIVITY_AIR)

H2O_CO2_RATIO = 1.6  # H2O to CO2 diffusivity ratio [-]
TN = 25.0 + DEG_TO_KELVIN  # reference temperature [K]


def leaf_boundary_layer_conductance(u, d, Ta, dT, P=101300.0):
    """
    Computes 2-sided leaf boundary layer conductance assuming mixed forced and free
    convection form two parallel pathways for transport through leaf boundary layer.
    INPUT: u - mean velocity (m/s)
           d - characteristic dimension of the leaf (m)
           Ta - ambient temperature (degC)
           dT - leaf-air temperature difference (degC)
           P - pressure(Pa)
    OUTPUT: boundary-layer conductances (mol m-2 s-1)
        gb_h - heat (mol m-2 s-1)
        gb_c- CO2 (mol m-2 s-1)
        gb_v - H2O (mol m-2 s-1)
        r - ratio of free/forced convection
    Reference: Campbell, S.C., and J.M. Norman (1998),
    An introduction to Environmental Biophysics, Springer, 2nd edition, Ch. 7
    Gaby Katul & Samuli Launiainen
    Note: the factor of 1.4 is adopted for outdoor environment, see Campbell and Norman, 1998, p. 89, 101.
    """

    u = np.maximum(u, EPS)

    # print('U', u, 'd', d, 'Ta', Ta, 'P', P)
    factor1 = 1.4 * 2  # forced conv. both sides, 1.4 is correction for turbulent flow
    factor2 = 1.5  # free conv.; 0.5 comes from cooler surface up or warmer down

    # -- Adjust diffusivity, viscosity, and air density to pressure/temp.
    t_adj = (101300.0 / P) * ((Ta + 273.15) / 293.16) ** 1.75
    Da_v = MOLECULAR_DIFFUSIVITY_H2O * t_adj
    Da_c = MOLECULAR_DIFFUSIVITY_CO2 * t_adj
    Da_T = THERMAL_DIFFUSIVITY_AIR * t_adj
    va = AIR_VISCOSITY * t_adj
    rho_air = 44.6 * (P / 101300.0) * (273.15 / (Ta + 273.13))  # [mol/m3]

    # ----- Compute the leaf-level dimensionless groups
    Re = u * d / va  # Reynolds number
    Sc_v = va / Da_v  # Schmid numbers for water
    Sc_c = va / Da_c  # Schmid numbers for CO2
    Pr = va / Da_T  # Prandtl number
    Gr = GRAVITY * (d**3) * abs(dT) / (Ta + 273.15) / (va**2)  # Grashoff number

    # ----- aerodynamic conductance for "forced convection"
    gb_T = (0.664 * rho_air * Da_T * Re**0.5 * (Pr) ** 0.33) / d  # [mol/m2/s]
    gb_c = (0.664 * rho_air * Da_c * Re**0.5 * (Sc_c) ** 0.33) / d  # [mol/m2/s]
    gb_v = (0.664 * rho_air * Da_v * Re**0.5 * (Sc_v) ** 0.33) / d  # [mol/m2/s]

    # ----- Compute the aerodynamic conductance for "free convection"
    gbf_T = (0.54 * rho_air * Da_T * (Gr * Pr) ** 0.25) / d  # [mol/m2/s]
    gbf_c = 0.75 * gbf_T  # [mol/m2/s]
    gbf_v = 1.09 * gbf_T  # [mol/m2/s]

    # --- aerodynamic conductance: "forced convection"+"free convection"
    gb_h = factor1 * gb_T + factor2 * gbf_T
    gb_c = factor1 * gb_c + factor2 * gbf_c
    gb_v = factor1 * gb_v + factor2 * gbf_v
    # gb_o3=factor1*gb_o3+factor2*gbf_o3

    # r = Gr / (Re**2)  # ratio of free/forced convection

    return gb_h, gb_c, gb_v  # , r


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
    s = 17.502 * 240.97 * esa / ((240.97 + T) ** 2)

    return esa, s


""" ---- alternative A-gs models ----- """


def photo_c3_analytical(photop, Qp, T, VPD, ca, gb_c, gb_v):
    """
    Leaf photosynthesis and gas-exchange by co-limitation optimality model of
    Vico et al. 2013 AFM
    IN:
        photop - parameter dict with keys: Vcmax, Jmax, Rd, alpha, theta, La, tresp
           can be scalars or arrays.
           tresp - dictionary with keys: Vcmax, Jmax, Rd: temperature sensitivity
           parameters. OMIT key if no temperature adjustments for photoparameters.
        Qp - incident PAR at leaves (umolm-2s-1)
        Tleaf - leaf temperature (degC)
        VPD - leaf-air vapor pressure difference(mol/mol)
        ca - ambient CO2 (ppm)
        gb_c - boundary-layer conductance for co2 (mol m-2 s-1)
        gb_v - boundary-layer conductance for h2o (mol m-2 s-1)
    OUT:
        An - net CO2 flux (umolm-2s-1)
        Rd - dark respiration (umolm-2s-1)
        fe - leaf transpiration rate (molm-2s-1)
        gs - stomatal conductance for CO2 (mol/m-2s-1)
        ci - leaf internal CO2 (ppm)
        cs - leaf surface CO2 (ppm)
    """
    Tk = T + DEG_TO_KELVIN

    MaxIter = 20

    # --- params ----
    Vcmax = photop["Vcmax"]
    Jmax = photop["Jmax"]
    Rd = photop["Rd"]
    alpha = photop["alpha"]
    theta = photop["theta"]
    La = photop["La"]
    g0 = photop["g0"]

    # From Bernacchi et al. 2001

    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))
    Ko = 2.784e5 * np.exp(36380.0 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))

    if "tresp" in photop:  # adjust parameters for temperature
        tresp = photop["tresp"]
        Vcmax_T = tresp["Vcmax"]
        Jmax_T = tresp["Jmax"]
        Rd_T = tresp["Rd"]
        Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
            Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk
        )

    # --- model parameters k1_c, k2_c [umol/m2/s]
    Km = Kc * (1.0 + O2_IN_AIR / Ko)
    J = (
        Jmax
        + alpha * Qp
        - ((Jmax + alpha * Qp) ** 2.0 - (4 * theta * Jmax * alpha * Qp)) ** (0.5)
    ) / (2.0 * theta)

    k1_c = J / 4.0
    k2_c = (J / 4.0) * Km / Vcmax

    # --- iterative solution for cs
    err = 9999.0
    cnt = 1
    cs = ca  # leaf surface CO2
    while err > 0.01 and cnt < MaxIter:
        NUM1 = -k1_c * (k2_c - (cs - 2 * Tau_c))
        DEN1 = (k2_c + cs) ** 2
        NUM2 = np.sqrt(
            H2O_CO2_RATIO
            * VPD
            * La
            * k1_c**2
            * (cs - Tau_c)
            * (k2_c + Tau_c)
            * ((k2_c + (cs - 2.0 * H2O_CO2_RATIO * VPD * La)) ** 2)
            * (k2_c + (cs - H2O_CO2_RATIO * VPD * La))
        )

        DEN2 = (
            H2O_CO2_RATIO
            * VPD
            * La
            * ((k2_c + cs) ** 2)
            * (k2_c + (cs - H2O_CO2_RATIO * VPD * La))
        )

        gs_opt = (NUM1 / DEN1) + (NUM2 / DEN2) + EPS

        ci = (1.0 / (2 * gs_opt)) * (
            -k1_c
            - k2_c * gs_opt
            + cs * gs_opt
            + Rd
            + np.sqrt(
                (k1_c + k2_c * gs_opt - cs * gs_opt - Rd) ** 2
                - 4 * gs_opt * (-k1_c * Tau_c - k2_c * cs * gs_opt - k2_c * Rd)
            )
        )

        An = gs_opt * (cs - ci)
        An1 = np.maximum(An, 0.0)
        cs0 = cs
        cs = ca - An1 / gb_c

        err = np.nanmax(abs(cs - cs0))
        cnt = cnt + 1
        # print('err', err)

    ix = np.where(An < 0)
    gs_opt[ix] = g0

    if type(ca) is float:
        ci[ix] = ca
        cs[ix] = ca
    else:
        ci[ix] = ca[ix]
        cs[ix] = ca[ix]

    gs_v = H2O_CO2_RATIO * gs_opt

    geff = (gb_v * gs_v) / (gb_v + gs_v)  # molm-2s-1
    fe = geff * VPD  # leaf transpiration rate

    if len(An) == 1:
        return float(An), float(Rd), float(fe), float(gs_opt), float(ci), float(cs)
    else:
        return An, Rd, fe, gs_opt, ci, cs


def photo_c3_medlyn(photop, Qp, T, VPD, ca, gb_c, gb_v, P=101300.0):
    """
    Leaf gas-exchange by Farquhar-Medlyn model, where co-limitation as in
    Vico et al. 2013 AFM
    IN:
        photop - parameter dict with keys: Vcmax, Jmax, Rd, alpha, theta, La, tresp
           can be scalars or arrays.
           tresp - dictionary with keys: Vcmax, Jmax, Rd: temperature sensitivity
           parameters. OMIT key if no temperature adjustments for photoparameters.
        Qp - incident PAR at leaves (umolm-2s-1)
        Tleaf - leaf temperature (degC)
        VPD - leaf-air vapor pressure difference (mol/mol)
        ca - ambient CO2 (ppm)
        gb_c - boundary-layer conductance for co2 (mol m-2 s-1)
        gb_v - boundary-layer conductance for h2o (mol m-2 s-1)
        P - atm. pressure (Pa)
    OUT:
        An - net CO2 flux (umolm-2s-1)
        Rd - dark respiration (umolm-2s-1)
        fe - leaf transpiration rate (molm-2s-1)
        gs - stomatal conductance for CO2 (mol/m-2s-1)
        ci - leaf internal CO2 (ppm)
        cs - leaf surface CO2 (ppm)
    """
    Tk = T + DEG_TO_KELVIN
    VPD = 1e-3 * VPD * P  # kPa

    MaxIter = 50

    # --- params ----
    Vcmax = photop["Vcmax"]
    Jmax = photop["Jmax"]
    Rd = photop["Rd"]
    alpha = photop["alpha"]
    theta = photop["theta"]
    g1 = photop["g1"]  # slope parameter
    g0 = photop["g0"]

    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))
    Ko = 2.784e5 * np.exp(36380.0 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))

    if "tresp" in photop:  # adjust parameters for temperature
        tresp = photop["tresp"]
        Vcmax_T = tresp["Vcmax"]
        Jmax_T = tresp["Jmax"]
        Rd_T = tresp["Rd"]
        Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
            Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk
        )

    # --- model parameters k1_c, k2_c [umol/m2/s]
    Km = Kc * (1.0 + O2_IN_AIR / Ko)
    J = (
        Jmax
        + alpha * Qp
        - ((Jmax + alpha * Qp) ** 2.0 - (4 * theta * Jmax * alpha * Qp)) ** (0.5)
    ) / (2 * theta)
    k1_c = J / 4.0
    k2_c = J / 4.0 * Km / Vcmax

    # --- iterative solution for cs and ci
    err = 9999.0
    cnt = 1
    cs = ca  # leaf surface CO2
    ci = 0.8 * ca  # internal CO2
    while err > 0.01 and cnt < MaxIter:
        # CO2 demand (Vico eq. 1) & gs_opt (Medlyn eq. xx)
        An = k1_c * (ci - Tau_c) / (k2_c + ci) - Rd  # umolm-2s-1
        An1 = np.maximum(An, 0.0)
        gs_opt = (1.0 + g1 / (VPD**0.5)) * An1 / (cs - Tau_c)  # mol m-2s-1
        gs_opt = np.maximum(g0, gs_opt)  # g0 is the lower limit

        # CO2 supply
        cs = np.maximum(ca - An1 / gb_c, 0.5 * ca)  # through boundary layer
        ci0 = ci
        ci = np.maximum(cs - An1 / gs_opt, 0.5 * ca)  # through stomata

        err = max(abs(ci0 - ci))
        cnt += 1
    # when Rd > photo, assume stomata closed and ci == ca
    ix = np.where(An < 0)
    if type(ca) is float:
        ci[ix] = ca
        cs[ix] = ca
    else:
        ci[ix] = ca[ix]
        cs[ix] = ca[ix]
    gs_opt[ix] = g0
    gs_v = H2O_CO2_RATIO * gs_opt

    geff = (gb_v * gs_v) / (gb_v + gs_v)  # molm-2s-1
    fe = geff * VPD / (1e-3 * P)  # leaf transpiration rate

    return An, Rd, fe, gs_opt, ci, cs


def freeze_photo_c3(photop, Qp, *remaining_args, **kwargs):
    # lru, make A_gs' parameters immutable
    _photop = frozendict(
        {
            key: (
                val
                if not isinstance(val, dict)
                else frozendict(
                    {sub_key: tuple(sub_val) for sub_key, sub_val in val.items()}
                )
            )
            for key, val in photop.items()
        }
    )
    _Qp = tuple(Qp)

    return (_photop, _Qp) + remaining_args, kwargs


def unfreeze_photo_c3(_photop, _Qp, *remaining_args, **kwargs):
    # restore original values from frozendict _photop and tuple _Qp
    photop = {
        key: (
            val
            if not isinstance(val, frozendict)
            else {sub_key: np.array(sub_val) for sub_key, sub_val in val.items()}
        )
        for key, val in _photop.items()
    }
    Qp = np.array(_Qp)

    return (photop, Qp) + remaining_args, kwargs


# @make_lru_cacheable(freeze_photo_c3, unfreeze_photo_c3, maxsize=50_000)
def photo_c3_medlyn_farquhar(photop, Qp, T, VPD, ca, gb_c, gb_v, P=101300.0):
    """
    Leaf gas-exchange by Farquhar-Medlyn model, where co-limitation as in standard Farquhar-
    model
    IN:
        photop - parameter dict with keys: Vcmax, Jmax, Rd, alpha, theta, La, tresp
           can be scalars or arrays.
           tresp - dictionary with keys: Vcmax, Jmax, Rd: temperature sensitivity
           parameters. OMIT key if no temperature adjustments for photoparameters.
        Qp - incident PAR at leaves (umolm-2s-1)
        Tleaf - leaf temperature (degC)
        VPD - leaf-air vapor pressure difference (mol/mol)
        ca - ambient CO2 (ppm)
        gb_c - boundary-layer conductance for co2 (mol m-2 s-1)
        gb_v - boundary-layer conductance for h2o (mol m-2 s-1)
        P - atm. pressure (Pa)
    OUT:
        An - net CO2 flux (umolm-2s-1)
        Rd - dark respiration (umolm-2s-1)
        fe - leaf transpiration rate (molm-2s-1)
        gs - stomatal conductance for CO2 (mol/m-2s-1)
        ci - leaf internal CO2 (ppm)
        cs - leaf surface CO2 (ppm)
    """

    Tk = T + DEG_TO_KELVIN
    VPD = np.maximum(EPS, 1e-3 * VPD * P)  # kPa

    MaxIter = 50

    # --- params ----
    Vcmax = photop["Vcmax"]
    Jmax = photop["Jmax"]
    Rd = photop["Rd"]
    alpha = photop["alpha"]
    theta = photop["theta"]
    g1 = photop["g1"]  # slope parameter
    g0 = photop["g0"]
    beta = photop["beta"]
    # print beta
    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))
    Ko = 2.784e5 * np.exp(36380.0 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))

    if "tresp" in photop:  # adjust parameters for temperature
        tresp = photop["tresp"]
        Vcmax_T = tresp["Vcmax"]
        Jmax_T = tresp["Jmax"]
        Rd_T = tresp["Rd"]
        Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
            Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk
        )
    # ---
    Km = Kc * (1.0 + O2_IN_AIR / Ko)
    J = (
        Jmax
        + alpha * Qp
        - ((Jmax + alpha * Qp) ** 2.0 - (4 * theta * Jmax * alpha * Qp)) ** (0.5)
    ) / (2 * theta)

    # --- iterative solution for cs and ci
    err = 9999.0
    cnt = 1
    cs = ca  # leaf surface CO2
    ci = 0.8 * ca  # internal CO2
    while err > 0.01 and cnt < MaxIter:
        # -- rubisco -limited rate
        Av = Vcmax * (ci - Tau_c) / (ci + Km)
        # -- RuBP -regeneration limited rate
        Aj = J / 4.0 * (ci - Tau_c) / (ci + 2.0 * Tau_c)

        # An = np.minimum(Av, Aj) - Rd  # single limiting rate
        x = Av + Aj
        y = Av * Aj
        An = (x - (x**2.0 - 4.0 * beta * y) ** 0.5) / (
            2.0 * beta
        ) - Rd  # co-limitation

        An1 = np.maximum(An, 0.0)

        # stomatal conductance
        gs_opt = g0 + (1.0 + g1 / (VPD**0.5)) * An1 / cs
        gs_opt = np.maximum(g0, gs_opt)  # g0 is the lower limit

        # CO2 supply
        cs = np.maximum(ca - An1 / gb_c, 0.5 * ca)  # through boundary layer
        ci0 = ci
        ci = np.maximum(cs - An1 / gs_opt, 0.1 * ca)  # through stomata

        err = max(abs(ci0 - ci))
        cnt += 1

    # when Rd > photo, assume stomata closed and ci == ca
    ix = np.where(An < 0)
    if type(ca) is float:
        ci[ix] = ca
        cs[ix] = ca
    else:
        ci[ix] = ca[ix]
        cs[ix] = ca[ix]
    gs_opt[ix] = g0
    gs_v = H2O_CO2_RATIO * gs_opt

    geff = (gb_v * gs_v) / (gb_v + gs_v)  # molm-2s-1
    fe = geff * VPD / (1e-3 * P)  # leaf transpiration rate

    return An, Rd, fe, gs_opt, ci, cs


def photo_c3_bwb(photop, Qp, T, RH, ca, gb_c, gb_v, P=101300.0):
    """
    Leaf gas-exchange by Farquhar-Ball-Woodrow-Berry model, as in standard Farquhar-
    model
    IN:
        photop - parameter dict with keys: Vcmax, Jmax, Rd, alpha, theta, La, tresp
           can be scalars or arrays.
           tresp - dictionary with keys: Vcmax, Jmax, Rd: temperature sensitivity
           parameters. OMIT key if no temperature adjustments for photoparameters.
        Qp - incident PAR at leaves (umolm-2s-1)
        Tleaf - leaf temperature (degC)
        rh - relative humidity at leaf temperature (-)
        ca - ambient CO2 (ppm)
        gb_c - boundary-layer conductance for co2 (mol m-2 s-1)
        gb_v - boundary-layer conductance for h2o (mol m-2 s-1)
        P - atm. pressure (Pa)
    OUT:
        An - net CO2 flux (umolm-2s-1)
        Rd - dark respiration (umolm-2s-1)
        fe - leaf transpiration rate (molm-2s-1)
        gs - stomatal conductance for CO2 (mol/m-2s-1)
        ci - leaf internal CO2 (ppm)
        cs - leaf surface CO2 (ppm)
    """
    Tk = T + DEG_TO_KELVIN

    MaxIter = 50

    # --- params ----
    Vcmax = photop["Vcmax"]
    Jmax = photop["Jmax"]
    Rd = photop["Rd"]
    alpha = photop["alpha"]
    theta = photop["theta"]
    g1 = photop["g1"]  # slope parameter
    g0 = photop["g0"]
    beta = photop["beta"]

    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))
    Ko = 2.784e5 * np.exp(36380.0 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))

    if "tresp" in photop:  # adjust parameters for temperature
        tresp = photop["tresp"]
        Vcmax_T = tresp["Vcmax"]
        Jmax_T = tresp["Jmax"]
        Rd_T = tresp["Rd"]
        Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
            Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk
        )

    # --- model parameters k1_c, k2_c [umol/m2/s]
    Km = Kc * (1.0 + O2_IN_AIR / Ko)
    J = (
        Jmax
        + alpha * Qp
        - ((Jmax + alpha * Qp) ** 2.0 - (4 * theta * Jmax * alpha * Qp)) ** (0.5)
    ) / (2 * theta)

    # --- iterative solution for cs and ci
    err = 9999.0
    cnt = 1
    cs = ca  # leaf surface CO2
    ci = 0.8 * ca  # internal CO2
    while err > 0.01 and cnt < MaxIter:
        # -- rubisco -limited rate
        Av = Vcmax * (ci - Tau_c) / (ci + Km)
        # -- RuBP -regeneration limited rate
        Aj = J / 4.0 * (ci - Tau_c) / (ci + 2.0 * Tau_c)

        # An = np.minimum(Av, Aj) - Rd  # single limiting rate
        # co-limitation
        x = Av + Aj
        y = Av * Aj
        An = (x - (x**2.0 - 4.0 * beta * y) ** 0.5) / (2.0 * beta) - Rd

        An1 = np.maximum(An, 0.0)
        # bwb -scheme
        gs_opt = g0 + g1 * An1 / ((cs - Tau_c)) * RH
        gs_opt = np.maximum(g0, gs_opt)  # gcut is the lower limit

        # CO2 supply
        cs = np.maximum(ca - An1 / gb_c, 0.5 * ca)  # through boundary layer
        ci0 = ci
        ci = np.maximum(cs - An1 / gs_opt, 0.1 * ca)  # through stomata

        err = max(abs(ci0 - ci))
        cnt += 1

    # when Rd > photo, assume stomata closed and ci == ca
    ix = np.where(An < 0)
    gs_opt[ix] = g0
    ci[ix] = ca[ix]
    cs[ix] = ca[ix]
    gs_v = H2O_CO2_RATIO * gs_opt

    geff = (gb_v * gs_v) / (gb_v + gs_v)  # molm-2s-1
    esat, _ = e_sat(T)
    VPD = (1.0 - RH) * esat / P  # mol mol-1
    fe = geff * VPD  # leaf transpiration rate

    return An, Rd, fe, gs_opt, ci, cs


def photo_farquhar(photop, Qp, ci, T, co_limi=False):
    """
    Calculates leaf net CO2 exchange and dark respiration rate (umol m-2 s-1).
    INPUT:
        photop - dict with keys:
            Vcmax
            Jmax
            Rd
            qeff
            alpha
            theta
            beta
        Qp - incident Par (umolm-2s-1)
        ci - leaf internal CO2 mixing ratio (ppm)
        T - leaf temperature (degC)
        co_limi - True uses co-limitation function of Vico et al., 2014.
    OUTPUT:
        An - leaf net CO2 exchange (umol m-2 leaf s-1)
        Rd - leaf dark respiration rate (umol m-2 leaf s-1)
    NOTE: original and co_limi -versions converge when beta ~ 0.8
    """
    Tk = T + DEG_TO_KELVIN  # K

    # --- params ----
    Vcmax = photop["Vcmax"]
    Jmax = photop["Jmax"]
    Rd = photop["Rd"]
    alpha = photop["alpha"]
    theta = photop["theta"]
    beta = photop["beta"]  # co-limitation parameter

    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))
    Ko = 2.784e5 * np.exp(36380.0 * (Tk - TN) / (TN * GAS_CONSTANT * Tk))

    if "tresp" in photop:  # adjust parameters for temperature
        tresp = photop["tresp"]
        Vcmax_T = tresp["Vcmax"]
        Jmax_T = tresp["Jmax"]
        Rd_T = tresp["Rd"]
        Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
            Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk
        )

    Km = Kc * (1.0 + O2_IN_AIR / Ko)
    J = (
        Jmax
        + alpha * Qp
        - ((Jmax + alpha * Qp) ** 2.0 - (4.0 * theta * Jmax * alpha * Qp)) ** 0.5
    ) / (2.0 * theta)

    if not co_limi:
        # -- rubisco -limited rate
        Av = Vcmax * (ci - Tau_c) / (ci + Km)
        # -- RuBP -regeneration limited rate
        Aj = J / 4 * (ci - Tau_c) / (ci + 2.0 * Tau_c)

        # An = np.minimum(Av, Aj) - Rd  # single limiting rate
        x = Av + Aj
        y = Av * Aj
        An = (x - (x**2 - 4 * beta * y) ** 0.5) / (2 * beta) - Rd  # co-limitation
        return An, Rd, Av, Aj
    else:  # use Vico et al. eq. 1
        k1_c = J / 4.0
        k2_c = (J / 4.0) * Km / Vcmax

        An = k1_c * (ci - Tau_c) / (k2_c + ci) - Rd
        return An, Rd, Tau_c, Kc, Ko, Km, J


def photo_temperature_response(Vcmax0, Jmax0, Rd0, Vcmax_T, Jmax_T, Rd_T, T):
    """
    Adjusts Farquhar / co-limitation optimality model parameters for temperature
    INPUT:
        Vcmax0, Jmax0, Rd0 - parameters at ref. temperature 298.15 K
        Vcmax_T, Jmax_T, Rd_T - temperature response parameter lists
        T - leaf temperature (K)
    OUTPUT: Nx1-arrays
        Vcmax, Jmax,Rd (umol m-2(leaf) s-1)
        Gamma_star - CO2 compensation point
    CALLED from Farquhar(); Opti_C3_Analytical(); Opti_C3_Numerical()
    REFERENCES:
        Medlyn et al., 2002.Plant Cell Environ. 25, 1167-1179; based on Bernacchi
        et al. 2001. Plant Cell Environ., 24, 253-260.
    Samuli Launiainen, Luke, 28.3.2017
    """

    # --- CO2 compensation point -------
    Gamma_star = 42.75 * np.exp(37830 * (T - TN) / (TN * GAS_CONSTANT * T))

    # ------  Vcmax (umol m-2(leaf)s-1) ------------
    Ha = 1e3 * Vcmax_T[0]  # J mol-1, activation energy Vcmax
    Hd = 1e3 * Vcmax_T[1]  # J mol-1, deactivation energy Vcmax
    Sd = Vcmax_T[2]  # entropy factor J mol-1 K-1

    NOM = np.exp(Ha * (T - TN) / (GAS_CONSTANT * TN * T)) * (
        1.0 + np.exp((TN * Sd - Hd) / (TN * GAS_CONSTANT))
    )
    DENOM = 1.0 + np.exp((T * Sd - Hd) / (T * GAS_CONSTANT))
    Vcmax = Vcmax0 * NOM / DENOM

    del Ha, Hd, Sd, DENOM, NOM

    # ----  Jmax (umol m-2(leaf)s-1) ------------
    Ha = 1e3 * Jmax_T[0]  # J mol-1, activation energy Vcmax
    Hd = 1e3 * Jmax_T[1]  # J mol-1, deactivation energy Vcmax
    Sd = Jmax_T[2]  # entropy factor J mol-1 K-1

    NOM = np.exp(Ha * (T - TN) / (GAS_CONSTANT * TN * T)) * (
        1.0 + np.exp((TN * Sd - Hd) / (TN * GAS_CONSTANT))
    )
    DENOM = 1.0 + np.exp((T * Sd - Hd) / (T * GAS_CONSTANT))
    Jmax = Jmax0 * NOM / DENOM

    del Ha, Hd, Sd, DENOM, NOM

    # --- Rd (umol m-2(leaf)s-1) -------
    Ha = 1e3 * Rd_T[0]  # J mol-1, activation energy dark respiration
    Rd = Rd0 * np.exp(Ha * (T - TN) / (TN * GAS_CONSTANT * T))

    return Vcmax, Jmax, Rd, Gamma_star


def apparent_photocapacity(b, psi_leaf):
    """
    computes relative photosynthetic capacity as a function of leaf water potential
    Function shape from Kellomäki & Wang, adjustments for Vcmax and Jmax
    IN:
       beta - parameters, 2x1 array
       psi - leaf water potential (MPa)
    OUT:
       f - relative value [0.2 - 1.0]
    """
    psi_leaf = np.array(np.size(psi_leaf), ndmin=1)
    f = (1.0 + np.exp(b[0] * b[1])) / (1.0 + np.exp(b[0] * (b[1] - psi_leaf)))
    f[f < 0.2] = 0.2

    return f


def topt_deltaS_conversion(xin, Ha, Hd, var_in="deltaS"):
    """
    Converts between entropy factor Sv (J mol-1) and temperature optimum
    Topt (K). Medlyn et al. 2002 PCE 25, 1167-1179 eq.19.
    INPUT:
        xin, Ha(J mol-1), Hd(J mol-1)
        input:'deltaS' [Jmol-1] or 'Topt' [K]
    OUT:
        xout - Topt or Sv
    Farquhar parameters temperature sensitivity
    """

    if var_in.lower() == "deltas":  # Sv --> Topt
        xout = Hd / (xin - GAS_CONSTANT * np.log(Ha / (Hd - Ha)))
    else:  # Topt -->Sv
        c = GAS_CONSTANT * np.log(Ha / (Hd - Ha))
        xout = (Hd + xin * c) / xin
    return xout


def photo_Toptima(T10):
    """
    computes acclimation of temperature optima of Vcmax and Jmax to 10-day mean air temperature
    Args:
        T10 - 10-day mean temperature (degC)
    Returns:
        Tv, Tj - temperature optima of Vcmax, Jmax
        rjv - ratio of Jmax25 / Vcmax25
    Reference: Lombardozzi et al., 2015 GRL, eq. 3 & 4
    """
    # --- parameters
    Hav = 72000.0  # J mol-1
    Haj = 50000.0  # J mol-1
    Hd = 200000.0  # J mol.1

    T10 = np.minimum(40.0, np.maximum(10.0, T10))  # range 10...40 degC
    # vcmax T-optima
    dSv = 668.39 - 1.07 * T10  # J mol-1
    Tv = Hd / (dSv - GAS_CONSTANT * np.log(Hav / (Hd - Hav))) - DEG_TO_KELVIN  # degC
    # jmax T-optima
    dSj = 659.70 - 0.75 * T10  # J mol-1
    Tj = Hd / (dSj - GAS_CONSTANT * np.log(Haj / (Hd - Haj))) - DEG_TO_KELVIN  # degC

    rjv = 2.59 - 0.035 * T10  # Jmax25 / Vcmax25

    return Tv, Tj, rjv


"""--- scripts for testing functions ---- """


def test_photo_temperature_response(species="pine"):
    T = np.linspace(1.0, 40.0, 79)
    Tk = T + DEG_TO_KELVIN
    if species.upper() == "PINE":
        photop = {
            "Vcmax": 55.0,
            "Jmax": 105.0,
            "Rd": 1.3,
            "tresp": {
                "Vcmax": [78.3, 200.0, 650.1],
                "Jmax": [56.0, 200.0, 647.9],
                "Rd": [33.0],
            },
        }
    if species.upper() == "SPRUCE":
        photop = {
            "Vcmax": 60.0,
            "Jmax": 114.0,
            "Rd": 1.5,
            "tresp": {
                "Vcmax": [53.2, 202.0, 640.3],  # Tarvainen et al. 2013 Oecologia
                "Jmax": [38.4, 202.0, 655.8],
                "Rd": [33.0],
            },
        }
    if species.upper() == "DECID":
        photop = {
            "Vcmax": 50.0,
            "Jmax": 95.0,
            "Rd": 1.3,
            "tresp": {
                "Vcmax": [77.0, 200.0, 636.7],  # Medlyn et al 2002.
                "Jmax": [42.8, 200.0, 637.0],
                "Rd": [33.0],
            },
        }

    Vcmax = photop["Vcmax"]
    Jmax = photop["Jmax"]
    Rd = photop["Rd"]
    tresp = photop["tresp"]
    Vcmax_T = tresp["Vcmax"]
    Jmax_T = tresp["Jmax"]
    Rd_T = tresp["Rd"]
    Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
        Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk
    )

    plt.figure(4)
    plt.subplot(311)
    plt.plot(T, Vcmax, "o")
    plt.title("Vcmax")
    plt.subplot(312)
    plt.plot(T, Jmax, "o")
    plt.title("Jmax")
    plt.subplot(313)
    plt.plot(T, Rd, "o")
    plt.title("Rd")


def Topt_to_Sd(Ha, Hd, Topt):
    Sd = Hd * 1e3 / (Topt + DEG_TO_KELVIN) + GAS_CONSTANT * np.log(Ha / (Hd - Ha))
    return Sd


def Sd_to_Topt(Ha, Hd, Sd):
    Topt = Hd * 1e3 / (Sd + GAS_CONSTANT * np.log(Ha / (Hd - Ha)))
    return Topt - DEG_TO_KELVIN
