"""
Module containing general utility functions.
"""
from __future__ import annotations

from functools import _CacheInfo, _lru_cache_wrapper, wraps
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
from frozendict import frozendict  # type: ignore[attr-defined]
from LAPM.discrete_linear_autonomous_pool_model import \
    DiscreteLinearAutonomousPoolModel as DLAPM
from matplotlib import animation
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d, interp2d

from . import Q_
from .productivity.constants import EPS, PAR_TO_UMOL
from .productivity.utils import e_sat, read_forcing

# from autograd import grad
# from scipy.optimize import root as root_


def reindent(s: str, numSpaces: int) -> str:
    """Change block indentation of string.

    Args:
        s: string oh which to change the block indentation
        numSpaces: number of spaces for new indentation

    Returns:
        string with new block indentation
    """
    s_list = s.split("\n")
    s_list = [(numSpaces * " ") + line.lstrip() for line in s_list]
    s = "\n".join(s_list)
    return s


def cached_property(cache_name: str) -> Callable:
    """
    Decorator replacing ``@property`` and returning cached value if it exists.

    If a class ``@property`` turns out to be computationally expensive and
    called often, then replace it by ``@cached_property``. At each call,
    the dictionary ``cache_name`` (needs to be initialized in ``class.__init__``)
    is checked for a precomputed value. If not found, the ``property`` code
    is executed. The result is then saved in the cache dictionary and
    returned. Delete the cache by re-initializing the cache dictionary.

    Args:
        cache_name: name of cache dictionary variable

    Returns:
        decorator function
    """

    def decorator(func):
        @wraps(func)
        def _get(self):
            cache_dict = getattr(self, cache_name)
            try:
                value = cache_dict[func.__name__]
                return value
            except KeyError:
                value = func(self)
            cache_dict[func.__name__] = value
            return value

        return property(_get, None, None)

    return decorator


def freeze(obj: Any) -> Hashable:
    """
    Recursive function which turns dictionaries into
    ``frozendict`` objects, lists into tuples, and sets
    into frozensets.
    Can also be used to turn JSON data into a hashable value.

    Args:
        obj: Object to be made hashable.

    Returns:
        Hashable version of ``obj``.

    Notes:
        https://github.com/zelaznik/frozen_dict/blob/master/freeze_recursive.py
    """
    try:
        # See if the object is hashable
        hash(obj)
        return obj
    except TypeError:
        pass

    try:
        # Try to see if this is a mapping
        try:
            obj[tuple(obj)]
        except KeyError:
            is_mapping = True
        else:
            is_mapping = False
    except (TypeError, IndexError):
        is_mapping = False

    if is_mapping:
        frz = {k: freeze(obj[k]) for k in obj}
        return frozendict(frz)

    # See if the object is a set like
    # or sequence like object
    try:
        obj[0]  # pylint: disable=pointless-statement
        cls = tuple
    except TypeError:
        cls = frozenset  # type: ignore
    except IndexError:
        cls = tuple

    try:
        iter(obj)
        is_iterable = True
    except TypeError:
        is_iterable = False

    if is_iterable:
        return cls(freeze(i) for i in obj)

    msg = "Unsupported type: %r" % type(obj).__name__
    raise TypeError(msg)


def freeze_all(*args, **kwargs) -> Tuple[Hashable, Hashable]:
    """
    Freeze all ``args`` and ``kwargs`` to make them hashable.

    Default function for ``freeze_func`` argument of :meth:`~.utils.make_lru_cacheable`.
    Can be massively slower and should be replaced by a freeze function
    tailored to the arguments of the specific function to be cached.

    Args:
        args: Positional arguments of function to be cached.
        kwargs: Keyword arguments of a function to be cached.

    Returns:
        Hashable versions of ``args`` and ``kwargs``.
    """
    frozen_args = (freeze(arg) for arg in args)
    frozen_kwargs = {key: freeze(val) for key, val in kwargs.items()}

    return frozen_args, frozen_kwargs  # type: ignore


def unfreeze_none(*frozen_args, **frozen_kwargs) -> Tuple[Hashable, Hashable]:
    """
    Do not 'unfreeze' any of ``frozen_args`` or ``frozen_kwargs``.

    Default function for ``unfreeze_func`` argument of :meth:`~.utils.make_lru_cacheable`.
    Should be replaced by a 'unfreeze' function suited to the input needs
    of the specific function ``f`` to be cached.

    Args:
        frozen_args: Hashable version of positional arguments of ``f``.
        frozen_kwargs: Hashable versions of keyword arguments of ``f``.

    Returns:
        frozen_args, frozen_kwargs
    """
    return frozen_args, frozen_kwargs  # type: ignore


def make_lru_cacheable(
    freeze_func: Callable = freeze_all,
    unfreeze_func: Callable = unfreeze_none,
    maxsize: int = None,
) -> Callable:
    """
    ``@lru_cache`` replacement for functions with unhashable arguments.

    The retunred wrapper function has two additional properties:

    - ``cache_info``: provides information on the cache's state
    - ``cache_clear()``: method to empty the cache

    Args:
        freeze_func: Function to make arguments of ``f`` hashable.
        unfreeze_func: Function to 'dehash' function arguments to make them
            match the function interface of ``f``.
        maxsize: Number of return values maximally stored in lru cache.
            Defaults to ``None`` representing an unlimited cache size.

    Returns:
        Wrapper around ``f`` that uses an ``@lru_cache`` decorator.
    """
    typed = False

    def decorating_function(user_function):
        def frozen_func(*args, **kwargs):
            """
            Version of ``user_function`` with hashable arguments and hence
            applicable to ``@lru_cache``.
            """
            unfrozen_args, unfrozen_kwargs = unfreeze_func(*args, **kwargs)
            return user_function(*unfrozen_args, **unfrozen_kwargs)

        # cached version of user_function
        cached_func = _lru_cache_wrapper(frozen_func, maxsize, typed, _CacheInfo)

        @wraps(user_function)
        def wrapper(*args, **kwargs):
            """
            Make arguments of ``user_function`` hashable and then call
            lru cached version of it.
            """
            frozen_args, frozen_kwargs = freeze_func(*args, **kwargs)
            return cached_func(*frozen_args, **frozen_kwargs)

        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear

        return wrapper

    return decorating_function


def latexify_unit(unit: str) -> str:
    """
    Convert unit to LaTeX format.

    Args:
        unit: unit to be converted

    Returns:
        converted unit
    """
    if unit:
        latex_dict = {
            "gC": r"\gC",
            "g_gluc": r"\ggluc",
            "g_dw": r"\gdw",
            r"\ggluc/\gdw": r"\frac{\ggluc}{\gdw}",
            "yr": r"\yr",
            r"1/\yr": r"\yr^{-1}",
            r"/\yr": r"\,\yr^{-1}",
            "MJ": r"\MJ",
            r"/\MJ": r"\MJ^{-1}",
            "cm": r"\centiMeter",
            "m": r"\meter",
            "centiMeter": "centimeter",
            #            r"/\meter^2": r"\meter^{-2}",
            #            r"/\meter^3": r"\meter^{-3}",
            #            r"/\meter": r"\meter^{-1}",
            r"\gdw/\meter^3": r"\frac{\gdw}{\meter^3}",
            r"\ggluc/\meter^3": r"\frac{\ggluc}{\meter^3}",
            r"\meter^3/\gdw": r"\frac{\meter^3}{\gdw}",
            r"\meter^2/k\gdw": r"\frac{\meter^2}{\kgdw}",
        }
        unit = "$" + unit + "$"
        for old, new in latex_dict.items():
            unit = unit.replace(old, new)

    return unit


def get_global_pool_nrs_from_entity_nrs(
    entity_nrs: List[int], ds: xr.Dataset
) -> np.ndarray:
    """Return array of DMR pool numbers for entity.

    Args:
        entity_nrs: entity numbers in ``ds`` whose pool numbers are considered
        ds: ``Dataset`` from
            :meth:`~.simulation.simulation.Simulation.get_stocks_and_fluxes_dataset`

    Returns:
        pool numbers of the DMR belonging to the entity
    """
    pool_nrs_list = list()
    for entity_nr in entity_nrs:
        if entity_nr < ds.nr_trees:
            start = entity_nr * ds.nr_tree_pools
            pool_nrs_list.append(np.arange(start, start + ds.nr_tree_pools, 1))
        elif entity_nr == ds.soil_entity_nr:
            start = entity_nr * ds.nr_tree_pools
            pool_nrs_list.append(np.arange(start, start + ds.nr_soil_pools, 1))
        elif entity_nr == ds.wood_product_entity_nr:
            start = (entity_nr - 1) * ds.nr_tree_pools + ds.nr_soil_pools
            pool_nrs_list.append(np.arange(start, start + ds.nr_wood_product_pools, 1))
        else:
            raise (ValueError(f"Not a valid entity number: entity_nr={entity_nr}"))

    return np.array(pool_nrs_list).flatten()


def create_dmr_from_stocks_and_fluxes(
    ds_origin: xr.Dataset, divide_by: int = 1, GPP_total_prepend: float = 0.0
) -> DMR:
    r"""Create a discrete model run from a simulation dataset containing stocks and fluxes.

    Args:
        ds_origin: dataset of stocks and fluxes as returned by
            :meth:`~.simulation.simulation.Simulation.get_stocks_and_fluxes_dataset`
        divide_by: the timestep of the data in ``ds`` might be too coarse such
            ``DMR`` raises an exception telling that a negative value occurs
            on the diagonal of a discrete compartmental matrix :math:`\mathrm{B}`
            during the reconstruction from data. In order to avoid this,
            increase this paramter. A value of 2 would decrease the timestep from
            annual to 6-monhts, for example.
        GPP_total_prepend: For global C balance check, we need to know the previous
            time step's GPP if we cut out the simulation from a larger one.

    Returns:
        discrete model run based on the simulation output dataset
    """
    ds = ds_origin.copy()

    # create tree entity-pool pairs
    entity_pool_pairs = []
    for entity in ds.coords["entity"][ds.tree_entity_nrs]:
        for pool in ds.coords["pool"][ds.tree_pool_nrs]:
            entity_pool_pairs.append((entity, pool))

    # create soil entity-pool pairs
    entity = ds.coords["entity"][ds.soil_entity_nr]
    for pool in ds.coords["pool"][ds.soil_pool_nrs]:
        entity_pool_pairs.append((entity, pool))

    # create wood product entity-pool pairs
    entity = ds.coords["entity"][ds.wood_product_entity_nr]
    for pool in ds.coords["pool"][ds.wood_product_pool_nrs]:
        entity_pool_pairs.append((entity, pool))

    nr_all_pools = (
        ds.nr_trees.data * ds.nr_tree_pools.data
        + ds.nr_soil_pools.data
        + ds.nr_wood_product_pools.data
    )
    nr_times = len(ds.coords["time"])

    # create stocks and external fluxes from and to right entities
    xs = np.nan * np.ones((nr_times, nr_all_pools))
    Us = np.nan * np.ones((nr_times, nr_all_pools))
    Rs = np.nan * np.ones((nr_times, nr_all_pools))

    newly_planted_biomass = np.zeros_like(Us)
    for nr_pool, entity_pool_pair in enumerate(entity_pool_pairs):
        entity, pool = entity_pool_pair
        xs[:, nr_pool] = ds.stocks.sel(entity=entity, pool=pool)

        entity_newly_planted_biomass = ds.newly_planted_biomass.sel(
            entity=entity, pool=pool
        )
        #        print(entity.data, entity_newly_planted_biomass.data)

        # newly planted biomass in first step is already in x0
        entity_newly_planted_biomass[0, ...] = 0.0
        # what a chance to make index errors, or to repair them
        entity_newly_planted_biomass = np.roll(
            entity_newly_planted_biomass, axis=0, shift=-1
        )

        entity_newly_planted_biomass[np.isnan(entity_newly_planted_biomass)] = 0.0
        Us[:, nr_pool] = (
            ds.input_fluxes.sel(entity_to=entity, pool_to=pool)
            + entity_newly_planted_biomass
        )
        #        print(nr_pool, Us[:, nr_pool])
        newly_planted_biomass[:, nr_pool] += entity_newly_planted_biomass
        Rs[:, nr_pool] = ds.output_fluxes.sel(entity_from=entity, pool_from=pool)

    # create internal fluxes from and to right entities
    Fs = np.nan * np.ones((nr_times, nr_all_pools, nr_all_pools))
    for nr_from, entity_pool_pair_from in enumerate(entity_pool_pairs):
        entity_from, pool_from = entity_pool_pair_from
        for nr_to, entity_pool_pair_to in enumerate(entity_pool_pairs):
            entity_to, pool_to = entity_pool_pair_to

            if nr_from == nr_to:
                Fs[:, nr_to, nr_from] = 0.0
            else:
                flux = ds.internal_fluxes.sel(
                    entity_from=entity_from,
                    pool_from=pool_from,
                    entity_to=entity_to,
                    pool_to=pool_to,
                )
                Fs[:, nr_to, nr_from] = flux

    # replace nan by 0
    Fs = np.nan_to_num(Fs, copy=False, nan=0.0, posinf=None, neginf=None)

    # interpolate between stocks
    times = np.arange(xs.shape[0])
    times_expanded = times * divide_by
    f = interp1d(times_expanded, xs, kind="linear", axis=0)
    times_divided = np.arange((xs.shape[0] - 1) * divide_by + 1)
    xs_divided = f(times_divided)

    # divide the fluxes if timestep too coarse
    Us_divided = np.nan * np.ones_like(xs_divided)
    Us_divided[:-1] = np.repeat(Us[:-1] / divide_by, divide_by, axis=0)
    Rs_divided = np.nan * np.ones_like(xs_divided)
    Rs_divided[:-1] = np.repeat(Rs[:-1] / divide_by, divide_by, axis=0)
    Fs_divided = np.nan * np.ones((len(xs_divided), nr_all_pools, nr_all_pools))
    Fs_divided[:-1] = np.repeat(Fs[:-1] / divide_by, divide_by, axis=0)

    # create discrete model run
    dmr = DMR.from_fluxes(
        xs_divided[0], times_divided, Us_divided[:-1], Fs_divided[:-1], Rs_divided[:-1]
    )

    # check consistency of stocks with dmr solution
    soln = dmr.solve()

    tree_pool_nrs = ds.tree_pool_nrs
    soil_pool_nrs = ds.soil_pool_nrs
    wood_product_pool_nrs = ds.wood_product_pool_nrs

    # trees
    for entity_nr in ds.tree_entity_nrs:
        pools = get_global_pool_nrs_from_entity_nrs([entity_nr], ds)
        #       # debugging code
        #        for k in range(10):
        #            print("entity_nr", entity_nr.data, "pool_nr", k)
        ##            print(ds.stocks.isel(entity=entity_nr)[:, tree_pool_nrs[k]].data - soln[:, pools[k]])
        ##            print(ds.stocks.isel(entity=entity_nr)[:, tree_pool_nrs[k]].data)
        ##            print(soln[:, pools[k]])
        ##            print()
        #
        #            for t in range(soln.shape[0]):
        ##                if t == 11:
        ##                    continue
        #
        ##                print(t, pools[k])
        ##                print(ds.stocks.isel(entity=entity_nr)[t, tree_pool_nrs[k]].data)
        ##                print(soln[t, pools[k]])
        #                assert_accuracy(
        #                    ds.stocks.isel(entity=entity_nr)[t, tree_pool_nrs[k]],
        #                    soln[t, pools[k]],
        #                   tol=1e-07
        #                )

        # ignore time steps in which the tree got newly planted:
        # in the simulation the planting happens at the beginning of the
        # time step, but in the DMR the newly planted biomass comes through
        # U at the end of the previous time step
        # this leads to a one-time-step missmatch in the stocks that is
        # corrected right in the next time step
        tis = np.array(np.arange(soln.shape[0]))
        d = ds.newly_planted_biomass.isel(entity=entity_nr).sum(dim="pool").data
        to_ignore = tis[d > 0]
        valid_tis = np.delete(tis, to_ignore)

        assert_accuracy(
            ds.stocks.isel(entity=entity_nr, time=valid_tis)[
                :, tree_pool_nrs.data
            ].data,
            soln[valid_tis, :][:, pools],
            tol=1e-07,
        )

    # soil
    entity_nr = ds.soil_entity_nr
    pools = get_global_pool_nrs_from_entity_nrs([entity_nr], ds)
    assert_accuracy(
        ds.stocks.isel(entity=entity_nr)[:, soil_pool_nrs].data,
        soln[:, pools],
        #        tol=1e-07
        tol=1e-03,
    )

    # wood products
    entity_nr = ds.wood_product_entity_nr
    pools = get_global_pool_nrs_from_entity_nrs([entity_nr], ds)
    assert_accuracy(
        ds.stocks.isel(entity=entity_nr)[:, wood_product_pool_nrs].data,
        soln[:, pools],
        tol=1e-08,
    )

    # check mass balance with GPP from simulation
    # in adapted dmr (vII to vI) only net_U enters the system, so outputs seem smaller, but in
    # fact the difference is captured in U-net_U
    GPPs_total = ds.GPP_total
    if GPPs_total is not None:
        GPPs = np.diff(GPPs_total[:-1], prepend=GPP_total_prepend)
        stock_unit = Q_(1, ds.stocks.attrs["units"])
        flux_unit = Q_(1, ds.input_fluxes.attrs["units"])
        Delta_t = Q_(1, "yr")

        #        print(GPP_total_prepend)
        #        for k in range(len(soln)-1):
        #            print(k)
        #            print(soln[k+1].sum()-soln[k].sum())
        ##            print(GPPs)
        #            print(GPPs[k]-dmr.acc_external_output_vector()[k].sum()\
        #                + newly_planted_biomass[k].sum())
        #
        ##        print((soln[1:].sum(1)-soln[:-1].sum(1)).shape)
        ##        print(GPPs.shape)
        ##        print(dmr.acc_external_output_vector().sum(1).shape)
        ##        print((Us_divided[:-1]-dmr.net_Us).sum(1).shape)
        ##        print(newly_planted_biomass[:-1].sum(1))

        assert_accuracy(
            Q_(soln[1:].sum(1) - soln[:-1].sum(1), stock_unit),
            Q_(GPPs - dmr.acc_external_output_vector().sum(1), flux_unit) * Delta_t
            + Q_(newly_planted_biomass[:-1].sum(1), stock_unit),
            tol=1e-06,
        )

    # add pool number descriptions to dmr
    dmr.tree_pool_nrs = get_global_pool_nrs_from_entity_nrs(ds.tree_entity_nrs.data, ds)

    dmr.soil_pool_nrs = get_global_pool_nrs_from_entity_nrs(
        [ds.soil_entity_nr.data], ds
    )

    dmr.wood_product_pool_nrs = get_global_pool_nrs_from_entity_nrs(
        [ds.wood_product_entity_nr.data], ds
    )

    return dmr


def create_dmr_inputs_only(dmr: DMR) -> DMR:
    """Return a ``DMR`` instance that coniders only the external input.

    Start values are zero, compartmental matrices ``Bs`` will be used from ``dmr``.
    """
    start_values = np.zeros(dmr.nr_pools)
    times = np.arange(len(dmr.Bs) + 1)
    dmr_inputs_only = DMR.from_Bs_and_net_Us(start_values, times, dmr.Bs, dmr.net_Us)

    # add pool number descriptions to dmr
    dmr_inputs_only.tree_pool_nrs = dmr.tree_pool_nrs
    dmr_inputs_only.soil_pool_nrs = dmr.soil_pool_nrs
    dmr_inputs_only.wood_product_pool_nrs = dmr.wood_product_pool_nrs

    return dmr_inputs_only


# def load_data(
#    dpath: Path,
#    start_date: str,
#    end_date: str
# ) -> pd.DataFrame:
#    """Load data from file.
#
#    Args:
#        dpath: data file
#        start_date: start date of planned simulation
#        end_date: end date of planned simulation
#
#    Returns:
#        data
#    """
#    # read data
#    start_time = start_date + " 00:00:00"
#    end_time = end_date + " 23:30:00"
#    data = read_data(dpath, start_time, end_time)
#
#    return data


def load_forcing(fpath: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Load forcing from file.

    Args:
        fpath: forcing file
        start_date: start date of planned simulation
        end_date: end date of planned simulation

    Returns:
        forcing
    """
    # read forcing
    start_time = start_date + " 00:00:00"
    end_time = end_date + " 23:30:00"
    forcing = read_forcing(
        fpath, start_time, end_time, dt=1800.0, na_values="NaN", sep=";"
    )

    # convert units, add VPD
    forcing["dirPar"] *= PAR_TO_UMOL
    forcing["diffPar"] *= PAR_TO_UMOL

    es, _ = e_sat(forcing["Tair"])
    forcing["D"] = np.maximum(EPS, es / forcing["P"] - forcing["H2O"])  # vpd (mol/mol)

    return forcing


# def load_data_and_forcing(
#    dpath: Path,
#    fpath: Path,
#    start_date: str,
#    end_date: str
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#    """Load data and forcing from files.
#
#    Args:
#        dpath: data file
#        fpath: forcing file
#        start_date: start date of planned simulation
#        end_date: end date of planned simulation
#
#    Returns:
#        data, forcing
#    """
#    # read forcing data
#    data = load_data(dpath, start_date, end_date)
#    forcing = load_data(fpath, start_date, end_date)
#
#    return data, forcing


def detrend_variable(df: pd.DataFrame, variable_name: str) -> pd.Series:
    """Remove linear trend from a variable.

    Compute annual mean of first and last year and from them a linear trend.
    Then remove the linear trend from the variable.

    Args:
        df: ``DataFrame`` that contains the variable.
        variable_name: Name of the variable.

    Returns:
        detrended variable
    """
    x = df.index
    s = df[variable_name]
    y1 = s.loc[s.index.year == x[0].year].mean()
    y2 = s.loc[s.index.year == x[-1].year].mean()
    annual_mean_trend = (y2 - y1) * ((x - x[0]) / (x[-1] - x[0]))

    return s - annual_mean_trend


def detrend_forcing(
    df: pd.DataFrame, nr_copies: int, variable_names: list
) -> pd.DataFrame:
    """Detrend variables from a ``DataFrame`` and copy the ``DataFrame``.

    First detrend the variables and then copy the data frame ``nr_copies`` times.

    Args:
        df: ``DataFrame`` to detrend variables from and copy subsequently.
        nr_copies: How many copies to make (5: return ``DataFrame`` is
            5 times as long as original ``DataFrame``).
        variable_names: Names of variables to detrend.
            Other variables will just be copied.

    Returns:
        Extended ``DataFrame`` with detrended variables updated and copied.
    """
    t0 = df.index[0]
    dt = df.index[1] - df.index[0]

    df_detrended = df.copy()
    for variable_name in variable_names:
        df_detrended[variable_name] = detrend_variable(df, variable_name)

    # create forcing copy (multiple times) and adapt the new index
    df_detrended = pd.concat([df_detrended] * nr_copies)
    new_index = t0 + dt * np.arange(0, len(df) * nr_copies, 1)
    df_detrended.set_index([new_index], inplace=True)

    return df_detrended


def extend_forcing(
    df: pd.DataFrame, nr_copies: int, variable_names: list
) -> pd.DataFrame:
    """Extend variables from a ``DataFrame`` and copy the ``DataFrame``.

    Extrapolate trend of the variables and then copy the data frame
    ``nr_copies`` times.

    Args:
        df: ``DataFrame`` to extemd variables from and copy subsequently.
        nr_copies: How many copies to make (5: return ``DataFrame`` is
            5 times as long as original ``DataFrame``).
        variable_names: Names of variables to extend the trend.
            Other variables will just be copied.

    Returns:
        Extended ``DataFrame`` with trend-extended variables updated and copied.
    """
    t0 = df.index[0]
    tend = df.index[-1]
    dt = df.index[1] - df.index[0]

    # create forcing copy (multiple times) and adapt the new index
    df_extended = pd.concat([df] * nr_copies)
    new_index = t0 + dt * np.arange(len(df_extended))
    df_extended.set_index([new_index], inplace=True)

    x = df_extended.index
    for variable_name in variable_names:
        s = df[variable_name]
        y1 = s.loc[s.index.year == t0.year].mean()
        y2 = s.loc[s.index.year == tend.year].mean()

        annual_mean_trend = (y2 - y1) * ((x - x[0]) / (s.index[-1] - x[0]))

        s_detrended = detrend_variable(df, variable_name)
        s_detrended_extended = pd.concat([s_detrended] * nr_copies, ignore_index=True)
        s_extended = s_detrended_extended + annual_mean_trend
        s_extended.index = df_extended.index
        df_extended[variable_name] = s_extended

    return df_extended


def assert_accuracy(x: Q_, y: Q_, tol: float):
    """Check if ``x`` and ``y`` are close.

    Args:
        x, y: quantities to compare
        tol: tolerance

    Raises:
        AssertException: if both absolute and relative
            error are greater than or equal to ``tol``
    """
    abs_err = np.abs(x - y)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_err = abs_err / np.abs(x) * 100
        rel_err = np.nan_to_num(rel_err)

    if isinstance(abs_err, Q_):
        abs_err = abs_err.magnitude

    assert np.all(np.less(abs_err, tol, where=~np.isnan(abs_err))) or np.all(
        np.less(rel_err, tol, where=~np.isnan(rel_err))
    )


def array_to_slice(a: np.ndarray) -> slice:
    """Make a sequence of subsequent numbers to a ``slice``."""
    return slice(a[0], a[-1] + 1, 1)


def add_stocks_from_dmr_to_ds(ds: xr.Dataset, dmr: DMR) -> xr.Dataset:
    """Add ``DMR`` solution to simulation dataset as ``solution``."""
    nr_entities = len(ds.coords["entity"])
    nr_times = len(ds.coords["time"])
    nr_pools = len(ds.coords["pool"])

    data_vars = dict()
    data = np.nan * np.ones((nr_entities, nr_times, nr_pools))
    soln = dmr.solve()

    # trees
    for entity_nr in ds.tree_entity_nrs:
        pools = get_global_pool_nrs_from_entity_nrs([entity_nr], ds)
        data[entity_nr, :, array_to_slice(ds.tree_pool_nrs.data)] = soln[:, pools]

    # soil
    entity_nr = ds.soil_entity_nr
    pools = get_global_pool_nrs_from_entity_nrs([entity_nr], ds)
    data[entity_nr, :, array_to_slice(ds.soil_pool_nrs.data)] = soln[:, pools]

    # wood products
    entity_nr = ds.wood_product_entity_nr
    pools = get_global_pool_nrs_from_entity_nrs([entity_nr], ds)
    data[entity_nr, :, array_to_slice(ds.wood_product_pool_nrs.data)] = soln[:, pools]

    data_vars["solution"] = xr.DataArray(
        data=data,
        dims=["entity", "time", "pool"],
        attrs={"units": ds.stocks.attrs["units"]},
    )

    return ds.assign(data_vars)  # type: ignore[arg-type]


# create start age moments and distributions
def load_start_age_data_from_eq_for_soil_and_wood_products(
    dmr: DMR, dmr_eq: DLAPM, up_to_order: int
) -> Dict[str, Union[np.ndarray, Callable]]:
    """Create/load start age moments and densities.

    Overwrite all age values associated to trees with zeros.

    Args:
        dmr: discrete model run of current simulation
        dmr_eq: fake equilibrium DMR from spinup
        up_to_order: moments of interest

    Returns:
        dictionary

        - "variable_name": data or function

    """
    start_values = dmr.start_values.copy()

    dmr_active_pool_nrs = np.append(dmr.soil_pool_nrs, dmr.wood_product_pool_nrs)
    dmr_eq_active_pool_nrs = np.append(
        dmr_eq.soil_pool_nrs, dmr_eq.wood_product_pool_nrs
    )

    # use dmr.start values only for soil and wood product model pools
    # everything associated to trees set to zero
    index_set = np.ones_like(start_values, dtype=bool)
    index_set[dmr_active_pool_nrs] = False
    start_values[index_set] = 0.0

    start_age_data: Dict[str, Union[np.ndarray, Callable]] = dict()

    # compile start age moments, also for inputs_only system
    for k in range(1, up_to_order + 1, 1):
        var_name = f"start_age_moments_{k}"
        data = np.zeros((k, dmr.nr_pools))
        data[:, dmr_active_pool_nrs] = dmr_eq.age_moment_vector_up_to(k)[
            :, dmr_eq_active_pool_nrs
        ]
        data = (start_values[np.newaxis, :] > 0) * data
        start_age_data[var_name] = data

        var_name = f"start_age_moments_{k}_inputs_only"
        start_age_data[var_name] = 0 * data

    # create initial mass functions
    p0_eq = dmr_eq.age_masses_func()

    def p0_no_trees(ai):
        data = np.zeros(dmr.nr_pools)
        data[dmr_active_pool_nrs] = p0_eq(ai)[dmr_eq_active_pool_nrs]
        return data

    def p0(ai):
        data = p0_no_trees(ai)
        if ai == 0:
            data[dmr.tree_pool_nrs] = dmr.start_values[dmr.tree_pool_nrs]
        return data

    P0_eq = dmr_eq.cumulative_age_masses_func()

    def P0_no_trees(ai):
        data = np.zeros(dmr.nr_pools)
        data[dmr_active_pool_nrs] = P0_eq(ai)[dmr_eq_active_pool_nrs]
        return data

    def P0(ai):
        data = P0_no_trees(ai)
        data[dmr.tree_pool_nrs] = dmr.start_values[dmr.tree_pool_nrs]
        return data

    start_age_data["p0"] = p0
    start_age_data["P0"] = P0

    # create inputs_only start age moments and distributions
    start_age_data["p0_inputs_only"] = lambda ai: np.zeros(dmr.nr_pools)
    start_age_data["P0_inputs_only"] = lambda ai: np.zeros(dmr.nr_pools)

    return start_age_data


# create start age moments and distributions
def load_start_age_data_from_eq(
    dmr: DMR, dmr_eq: DLAPM, up_to_order: int
) -> Dict[str, Union[np.ndarray, Callable]]:
    """Create/load start age moments and densities.

    Args:
        dmr: discrete model run of current simulation
        dmr_eq: fake equilibrium DMR from spinup
        up_to_order: moments of interest

    Returns:
        dictionary

        - "variable_name": data or function

    """
    start_values = dmr.start_values.copy()

    start_age_data: Dict[str, Union[np.ndarray, Callable]] = dict()

    # compile start age moments, also for inputs_only system
    for k in range(1, up_to_order + 1, 1):
        var_name = f"start_age_moments_{k}"
        data = np.zeros((k, dmr.nr_pools))
        data = dmr_eq.age_moment_vector_up_to(k)
        data = (start_values[np.newaxis, :] > 0) * data
        start_age_data[var_name] = data

        var_name = f"start_age_moments_{k}_inputs_only"
        start_age_data[var_name] = 0 * data

    # create initial mass functions
    p0_eq = dmr_eq.age_masses_func()
    p0 = p0_eq

    P0_eq = dmr_eq.cumulative_age_masses_func()
    P0 = P0_eq

    start_age_data["p0"] = p0
    start_age_data["P0"] = P0

    # create inputs_only start age moments and distributions
    start_age_data["p0_inputs_only"] = lambda ai: np.zeros(dmr.nr_pools)  # type: ignore[assignment]
    start_age_data["P0_inputs_only"] = lambda ai: np.zeros(dmr.nr_pools)  # type: ignore[assignment]

    return start_age_data


def create_stand_cross_section_video(
    ds: xr.Dataset, animation_filepath: Path, base_N: float, relative: bool = False
):
    """Create a video showing a stand cross section during the simulation.

    Args:
        ds: simulationd dataset
        animation_filepath: where to store the output file
        base_N: base value for x-axis (trees per m^2)
        relative: absolute or relative (percentage) values of y-axis
    """
    tree_names = ds.entity[ds.tree_entity_nrs].data
    tree_existence = ds.stocks.sel(entity=ds.tree).sum(dim="pool") > 0
    nr_trees = len(tree_names)
    N_per_m2 = ds.N_per_m2 * tree_existence
    H = ds.height.sel(tree=tree_names)
    dbh = ds.DBH.sel(tree=tree_names)
    V_T = ds.V_T_tree.sel(tree=tree_names) * 10_000

    variable_names = ["Height", "DBH", "$V_T$"]
    units = ["[m]", "[cm]", "[m$^3\,$ha$^{-1}$]"]
    ylabels = []
    yvals = [H, dbh, V_T]
    if relative:
        yvals_rel = []
        for val in yvals:
            yvals_rel.append(val.diff(dim="time") / val * 100)

        yvals = yvals_rel

        for k in range(len(variable_names)):
            ylabel = variable_names[k]
            ylabels.append("$\Delta\,$" + ylabel + " [\%]")
    else:
        for k in range(len(variable_names)):
            ylabel = variable_names[k]
            ylabels.append(ylabel + " " + units[k])

    def animate(i):
        print(i, f"{i/(len(ds.time)-1)*100:2.2f} %", end="\r")
        for ax, ylabel, yval, ylim in zip(axes, ylabels, yvals, ylims):
            ax.clear()

            Ns = N_per_m2.isel(time=i).data
            living_cols = [c for k, c in enumerate(cols) if Ns[k] > 0]
            ys = yval.isel(time=i).data
            ys = ys[Ns > 0]
            Ns = Ns[Ns > 0]
            free_space = base_N - sum(Ns)
            free_space_per_tree = free_space / (len(Ns) + 1)

            Ns_with_free_space = [free_space_per_tree]
            for N in Ns:
                Ns_with_free_space.append(N)
                Ns_with_free_space.append(free_space_per_tree)

            if sum(Ns) > 0:
                line_data = np.cumsum([0] + Ns_with_free_space)

                x = (line_data[:-1] + line_data[1:]) / 2
                x_data = np.array([val for k, val in enumerate(x) if k % 2 == 1])

                width = Ns_with_free_space
                width_data = [val for k, val in enumerate(width) if k % 2 == 1]
                ax.bar(
                    x_data,
                    height=ys,
                    width=width_data,
                    color=living_cols,
                    tick_label=None,
                    edgecolor="black",
                )

            ax.axes.xaxis.set_ticks([])

            ax.tick_params(axis="x", labelsize=40)
            ax.tick_params(axis="y", labelsize=40)

            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.set_ylabel(ylabel, fontsize=40)

        axes[0].set_title(
            f"Stand cross section - {str(ds.time[i].data)[:4]} yr", fontsize=50
        )

        custom_lines = [
            Line2D([0], [0], color=cols[nr], lw=20) for nr in range(nr_trees)
        ]
        axes[0].legend(custom_lines, tree_names, fontsize=30, loc=1)
        axes[-1].set_xlabel(r"Number of trees per ha", fontsize=50)

    #############

    fig, axes = plt.subplots(figsize=(30, 15), nrows=3)

    cmap_names = ["Set1", "Set3", "tab20"]
    for cmap_name in cmap_names:
        cols = plt.get_cmap(cmap_name).colors[:nr_trees]
        if len(cols) >= nr_trees:
            break

    xlim = [0, base_N]
    ylims = [[0, yval.max().data * 1.1] for yval in yvals]
    for ax, ylim in zip(axes, ylims):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    animate(0)
    fig.tight_layout()

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=animate(0),
        frames=np.arange(1, len(ds.time) - int(relative)),
    )

    anim.save(animation_filepath, fps=2, extra_args=["-vcodec", "libx264"])
    plt.close()


def create_simulation_video(
    ds: xr.Dataset,
    dmr_eq: DLAPM,
    density_pool_nrs: np.ndarray,
    animation_filepath: Path,
    resolution: int = 1,
    time_index_start: int = 0,
    clearcut_index: Union[int, None] = None,
    time_index_stop=-1,
    year_shift: int = 0,
    lw: float = 10,
    fontsize: float = 25,
    cache_size: int = 0,
):
    """Create a video showing a stand cross section during the simulation.

    Args:
        ds: simulationd dataset
        dmr_eq: equilibrium DMR from spinup for initial ages
        density_pool_nrs: pool numbers to aggregate in the density plot
        animation_filepath: where to store the output file
        resolution: 10 is ten times higher, the higher the smoother the video
        time_index_start: when to start the simulation
        clearcut_index: where to add the lines of a potential clearcut
        time_index_stop: when to stop the simulation
        year_shift: adapt the simulation year labels
        lw: line width
        fontsize: fontsize
        cache_size: for DMR to speed up denisty computations
    """

    def animate(ti: float):
        """Make the frame at the given time index."""
        print(
            f"{ti:2.2f}",
            f"{(ti-time_index_start)/(time_index_stop-time_index_start)*100:2.2f} %",
            end="\r",
        )

        fig.suptitle(
            f"Simulation year: {int(str(ds.time[int(ti)].data)[:4])+year_shift}\n",
            fontsize=fontsize * 1.5,
        )

        # the C panel
        ax = ax_C

        ax.clear()
        ax.set_title(
            "Carbon in trees, soil, and wood products", fontsize=fontsize * 1.5
        )

        if ti > time_index_start:
            tis_smooth_so_far = tis_smooth[tis_smooth <= ti]
            years_smooth_so_far = f_years(tis_smooth_so_far)

            var_names = ["tree_biomass", "soil_biomass", "wood_product_biomass"]
            #            variables = [ds[var] for var in var_names]

            (l,) = ax.plot(
                years_smooth_so_far + year_shift,
                f_total_biomass_q(years_smooth_so_far),
                label="total biomass",
                lw=lw,
            )
            if clearcut_index is not None:
                ax.axhline(
                    f_total_biomass_q(years[clearcut_index]),
                    color=l.get_c(),
                    alpha=0.5,
                    lw=lw,
                )

            for var_name in var_names:
                f = f_biomasses_dict[var_name]
                (l,) = ax.plot(
                    years_smooth_so_far + year_shift,
                    f(years_smooth_so_far),
                    label=f"{var_name}".replace("_", " "),
                    lw=lw,
                )
                if clearcut_index is not None:
                    ax.axhline(
                        f(years[clearcut_index]), color=l.get_c(), alpha=0.7, lw=lw
                    )

            if clearcut_index is not None:
                ax.axvline(
                    f_years(clearcut_index) + year_shift,
                    color="black",
                    alpha=0.2,
                    lw=lw,
                )

        ax.set_xlim(np.array(time_lim) + year_shift)
        ax.set_ylim([0, np.max(total_biomass).magnitude * 1.2])
        ax.set_ylabel(r"mass [kgC/m$^2$]", fontsize=fontsize)
        ax.set_xlabel("time [yr]", fontsize=fontsize)
        ax.legend(loc=8, fontsize=fontsize)  # 8 = lower center
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)

        # V_T
        ax = ax_V

        ax.clear()
        ax.set_title("Total stem volume", fontsize=fontsize * 1.5)

        if ti > time_index_start:
            tis_smooth_so_far = tis_smooth[tis_smooth <= ti]
            years_smooth_so_far = f_years(tis_smooth_so_far)

            (l,) = ax.plot(
                years_smooth_so_far + year_shift,
                f_V_T(years_smooth_so_far),
                #                label="$V_T$",
                lw=lw,
            )
            if clearcut_index is not None:
                ax.axhline(
                    f_V_T(years[clearcut_index]),
                    color=l.get_c(),
                    alpha=0.5,
                    lw=lw,
                )

            if clearcut_index is not None:
                ax.axvline(
                    f_years(clearcut_index) + year_shift,
                    color="black",
                    alpha=0.2,
                    lw=lw,
                )

        ax.set_xlim(np.array(time_lim) + year_shift)
        ax.set_ylim([0, np.max(V_T).magnitude * 1.2])
        ax.set_xlabel("time [yr]", fontsize=fontsize)
        ax.set_ylabel(r"[m$^3\,$ha$^{-1}$]", fontsize=fontsize)
        #        ax.legend(loc=1, fontsize=fontsize)  # 8 = lower center
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)

        # density
        ax = ax_density
        ax.clear()
        mean_age = f_sim_mean_age_density_pools(ti)
        #        max_age = 3 * mean_age
        ax.set_title("SOM carbon age", fontsize=fontsize * 1.5)

        #        ages_soil = np.arange(0, maxx_soil, 1, dtype=int)

        (l,) = ax.plot(
            ages_density_pools,
            [f_p_density_pools(ai, ti) / 1000 for ai in ages_density_pools],
            lw=lw,
        )
        ax.axvline(mean_age, label="mean age", c=l.get_color(), ls="--", lw=lw)

        # der graue Ausgangspunkt
        if clearcut_index is not None:
            ax.plot(
                ages_density_pools,
                [
                    f_p_density_pools(ai, clearcut_index) / 1000
                    for ai in ages_density_pools
                ],
                lw=lw,
                color="black",
                alpha=0.2,
            )
            ax.axvline(
                f_sim_mean_age_density_pools(clearcut_index),
                c="black",
                alpha=0.2,
                ls="--",
                lw=lw,
            )

        ax.set_xlabel("age [yr]", fontsize=fontsize)
        ax.set_ylabel(r"mass density [kgC/m$^2$/yr]", fontsize=fontsize)
        ax.set_ylim([0, ax.get_ylim()[-1]])
        ax.legend(fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)

        ax.set_xlim(xlim_density)
        ax.set_ylim(ylim_density)

    # load DMR drom simulation dataset
    dmr = create_dmr_from_stocks_and_fluxes(ds)
    if cache_size > 0:
        dmr.initialize_state_transition_operator_matrix_cache(cache_size)

    # load initial mean age and age distribution
    start_age_data = load_start_age_data_from_eq_for_soil_and_wood_products(
        dmr, dmr_eq, 1
    )
    p0 = start_age_data["p0"]  # initial age density vector

    # prepare functions for density plot
    start_mean_age = start_age_data["start_age_moments_1"]
    sim_mean_age_vector = dmr.age_moment_vector(1, start_mean_age)
    soln = dmr.solve()
    p = dmr.age_densities_single_value_func(p0)

    # density in selected pools
    soln_density_pools = soln[:, density_pool_nrs]

    def p_density_pools(ai, ti):
        dens_density_pools = p(ai, ti)[density_pool_nrs]
        return (
            dens_density_pools * soln_density_pools[ti, :]
        ).sum() / soln_density_pools[ti, :].sum()

    # create underlying figure and axes
    fig, axes = plt.subplots(figsize=(30, 15), nrows=3)

    # total simulation years
    years = ds.time.data
    tis = np.arange(len(years))  # avoid showing the clearcut at the end
    f_years = interp1d(tis, years)

    # mean age function, interpolated to make it smooth (via resolution parameter)
    sim_mean_age_density_pools = (
        sim_mean_age_vector[:, density_pool_nrs] * soln[:, density_pool_nrs]
    ).sum(axis=1) / soln[:, density_pool_nrs].sum(axis=1)
    f_sim_mean_age_density_pools = interp1d(tis, sim_mean_age_density_pools)

    # density function, interpolated to make it smooth (via resolution parameter)
    mean_age_density_pools = sim_mean_age_density_pools
    max_age_density_pools = 3 * mean_age_density_pools
    ages_density_pools = np.arange(0, int(np.nanmax(max_age_density_pools)), 1)
    z = np.array([p_density_pools(ai, ti) for ai in ages_density_pools for ti in tis])
    f_p_density_pools = interp2d(ages_density_pools, tis, z.transpose())

    # compute x and y axis limits for density plot
    maxx_density = 2 * np.nanmax(sim_mean_age_density_pools)
    xlim_density = [0, maxx_density]
    maxy_density = np.nanmax(
        [p_density_pools(ai, ti) for ai in ages_density_pools for ti in tis]
    )
    ylim_density = [0, maxy_density * 1.5 / 1000]

    # C plot biomass functions for smoothing
    var_names = ["tree_biomass", "soil_biomass", "wood_product_biomass"]
    variables = [ds[var] for var in var_names]
    total_biomass = sum([var for var in variables])
    total_biomass = Q_(total_biomass.data, "gC/m^2").to("kgC/m^2")  # type: ignore

    ys = total_biomass.magnitude
    f_total_biomass = interp1d(years, ys)
    f_total_biomass_q = lambda xis: Q_(f_total_biomass(xis), total_biomass.units)

    f_biomasses_dict = dict()

    def f_q_biomass_wrapper(var):
        biomass = Q_(var.data, "gC/m^2").to("kgC/m^2")
        ys = biomass.magnitude
        f_biomass = interp1d(years, ys)
        f_biomass_q = lambda xis: Q_(f_biomass(xis), biomass.units)

        return f_biomass_q

    for var_name in var_names:
        var = ds[var_name]

        f_biomass_q = f_q_biomass_wrapper(var)
        f_biomasses_dict[var_name] = f_biomass_q

    V_T = Q_(ds.V_T_tree.sum(dim="tree").data, "m^3 / m^2").to("m^3 / ha")
    f_V_T_ = interp1d(years, V_T.magnitude)
    f_V_T = lambda xis: Q_(f_V_T_(xis), V_T.units)

    fig.subplots_adjust(hspace=0.4)

    ax_C = axes[0]
    ax_V = axes[1]
    ax_density = axes[2]

    # collect rendering data
    time_lim = [years[time_index_start], years[time_index_stop]]

    tis_smooth = np.linspace(
        time_index_start,
        time_index_stop,
        (time_index_stop + 1 - time_index_start) * resolution,
    )

    animate(time_index_start + 6.5)  # just for quick eye testing
    fig.tight_layout()

    # set up animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=animate(time_index_start),
        frames=np.arange(
            time_index_start, time_index_stop + 1 / resolution, 1 / resolution
        ),
    )

    # finally animate and save the video file
    print("computing and saving the animation")
    anim.save(animation_filepath, fps=2 * resolution, extra_args=["-vcodec", "libx264"])
    del dmr._state_transition_operator_matrix_cache


# def root_with_autograd(objective_function, x0, args=tuple()):
#    """`Scipy.optimize.root` with automatic differentiation
#
#    We provide a jacobian to root, obtained by automatic differentiation.
#    It does not seem to help in my problematic cases. Maybe it happens
#    anyway already in the bakground of `root`.
#    """
#    def objective_for_root(optim_vars, *args):
#        return objective_function(optim_vars, *args)
#
#    def objective_for_grad(optim_vars, *args):
#        optim_vars_ = np.nan * np.ones_like(optim_vars)
#        for k, v in enumerate(optim_vars):
#            optim_vars_[k] = getattr(v, "_value", v)
#        return objective_function(*optim_vars_, *args)
#
#    gradient = grad(objective_for_grad)
#
#    root_res = root_(objective_for_root, x0, jac=gradient, args=args)
##    if (not root_res.success) and (np.abs(root_res.fun) > 1e-12):
##        print(root_res)
##        raise
#
#    return root_res


def get_pars_comb(pars: Dict[str, List[float]], pars_comb_nr: int) -> Dict[str, float]:
    r"""Make all possible combinations of parameter values and return one.

    Args:
        pars: {"p1": [v1, v2, ...], "p2": [v1, v2,...], ...}
        pars_comb_nr: the number of the parameter combination to return

    Returns:
        dictionary like

        - {"p1": v2, "p2": v6, ...}
    """

    def my_product(inp):
        return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

    return list(my_product(pars))[pars_comb_nr]
