"""
This module contains the abstract base class :class:`~wood_product_model_abc.WoodProductModelABC`.

All wood product model classes should derive from here in order
to ensure the commutincation with the stand.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Callable

import numpy as np
import xarray as xr
from CompartmentalSystems.helpers_reservoir import \
    numerical_function_from_expression
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel

from .. import Q_
from ..type_aliases import InputFluxes, OutputFluxesInt


class WoodProductModelABC(metaclass=ABCMeta):
    """
    Abstract base class for all wood product soil models.

    All wood product model classes should derive from here in order
    to ensure the communication with the stand.

    Args:
        srm: underlying symbolic model
        initialize_params: function to load the parameters
        stock_unit: the carbon unit to be used
        Delta_t: the time step used
        initial_stocks: if not provided, then set to zero

    Attributes:
        Delta_t (``Quantity``): the time step used
        srm (``SmoothReservoirModel``): the underlying symbolic model
    """

    def __init__(
        self,
        srm: SmoothReservoirModel,
        initialize_params: Callable,
        stock_unit: Q_[float] = Q_("1 gC/m^2"),
        Delta_t: Q_[float] = Q_("1 yr"),
        initial_stocks: Q_[np.ndarray] = None,
    ):
        self.stock_unit = stock_unit
        self.Delta_t = Delta_t
        self.flux_unit = stock_unit / Delta_t

        self.srm = srm  # import in subclass

        # load and convert model parameters
        params = initialize_params()  # import in subclass
        par_dict = {name: value.magnitude for name, value in params.items()}

        # create start_values
        sv = np.zeros(srm.nr_pools)

        if initial_stocks is not None:
            sv = initial_stocks.to(stock_unit).magnitude

        # create carbon stocks
        self._xs = [sv]

        # create U_func
        self._U_func = self._create_U_func()
        self._Us: list[np.ndarray] = []

        # prepare parameters and functions

        func_dict: dict[str, Callable] = dict()
        free_variables = tuple(srm.state_vector)

        # create internal flux functions
        internal_flux_funcs = dict()
        for (pool_from, pool_to), expr in srm.internal_fluxes.items():
            internal_flux_funcs[
                (pool_from, pool_to)
            ] = numerical_function_from_expression(
                expr, free_variables, par_dict, func_dict
            )
        self._internal_flux_funcs = internal_flux_funcs
        self._Fs: list[np.ndarray] = []

        # create R_func
        self._R_func = numerical_function_from_expression(
            srm.external_outputs, free_variables, par_dict, func_dict
        )
        self._Rs: list[np.ndarray] = []

    # properties and methods required by stand

    @property
    def stock_unit(self) -> Q_[float]:
        """Unit of carbon stocks."""
        return self._stock_unit

    @property
    def flux_unit(self) -> Q_[float]:
        """Unit of carbon fluxes."""
        return self._flux_unit

    @property
    def nr_pools(self) -> int:
        """Number of carbon pools."""
        return self.srm.nr_pools

    @property
    def xs(self) -> list[np.ndarray]:
        """List of state vectors."""
        return self._xs

    @property
    def Us(self) -> list[np.ndarray]:
        """List of external input vectors."""
        return self._Us

    @property
    def Fs(self) -> list[np.ndarray]:
        """List of internal flux matrices."""
        return self._Fs

    @property
    def Rs(self) -> list[np.ndarray]:
        """List of external output vectors."""
        return self._Rs

    @property
    def pool_names(self) -> list[str]:
        """List of pool names."""
        return [sv.name for sv in self.srm.state_vector]

    @property
    def start_vector(self) -> np.ndarray:
        """Initial stocks."""
        return self.xs[0]

    def update(self, input_fluxes: InputFluxes) -> OutputFluxesInt:
        """Update the wood-product pools.

        Args:
            input_fluxes: dictionary of fluxes into the wood-product model

            - pool_name: flux

        Returns:
            wood-product's output fluxes dictionary

            - pool_name: flux [gC yr-1]
        """
        nr_pools = self.srm.nr_pools

        U = self._create_U(input_fluxes)
        self._Us.append(U)

        # prepare function arguments
        args = tuple((float(x) for x in self.xs[-1]))

        # F, internal fluxes
        F = np.zeros((nr_pools, nr_pools))
        for (pool_from, pool_to), func in self._internal_flux_funcs.items():
            F[pool_to, pool_from] = func(*args)

        self._Fs.append(F)

        # R, external outfluxes
        R = self._R_func(*args).reshape(nr_pools)
        self._Rs.append(R)

        # update state vector
        Delta_x = U + F.sum(axis=1) - F.sum(axis=0) - R
        x = self.xs[-1] + Delta_x * self.Delta_t.magnitude
        self._xs.append(x)

        # check consistency
        self._check_consistency()

        # return outflux dictionary
        output_fluxes: OutputFluxesInt = {
            self.srm.state_vector[pool_nr].name: Q_(flux, self.flux_unit)
            for pool_nr, flux in enumerate(R)
        }

        return output_fluxes

    def get_stocks_and_fluxes_dataset(self) -> xr.Dataset:
        """Return an ``xarray.Dataset`` with all C stocks and fluxes over time.

        Returns:
            ``xarray.Dataset`` given by

            .. code-block:: python

                coords = {
                    "timestep": np.arange(nr_timesteps),
                    "pool": pool names,
                    "pool_to": pool names
                    "pool_from": pool names
                }

                data_vars = {
                    "stocks": xr.DataArray(
                        data_vars=stocks data,
                        dims=["timestep", "pool"],
                        attrs={
                            "units": stock unit [gC],
                            "cell_methods": "time: instantaneous"
                        }
                    ),

                    "input_fluxes": xr.DataArray(
                        data_vars=input fluxes data,
                            mean over time step,
                        dims=["timestep", "pool_to"],
                        attrs={
                            "units": flux unit [gC yr-1],
                            "cell_methods": "time: total"
                        }
                    ),

                    "output_fluxes": xr.DataArray(
                        data_vars=output fluxes data,
                            mean over time step,
                        dims=["timestep", "pool_from"],
                        attrs={
                            "units": flux unit [gC yr-1],
                            "cell_methods": "time: total"
                        }
                    ),

                    "internal_fluxes": xr.DataArray(
                        data_vars=internal fluxes data,
                            mean over time step,
                        dims=["timestep", "pool_to", "pool_from"],
                        attrs={
                            "units": flux unit [gC yr-1],
                            "cell_methods": "time: total"
                        }
                    )
                }
        """
        srm = self.srm
        nr_pools = srm.nr_pools

        timesteps = np.arange(len(self.xs))
        pool_names = [sv.name for sv in srm.state_vector]
        coords = {
            "timestep": timesteps,
            "pool": pool_names,
            "pool_to": pool_names,
            "pool_from": pool_names,
        }

        data_vars = dict()

        # stocks
        data_vars["stocks"] = xr.DataArray(
            data=np.array(self.xs),
            dims=["timestep", "pool"],
            attrs={
                "units": Q_.get_netcdf_unit(self.stock_unit),
                "cell_methods": "time: instantaneous",
            },
        )

        # fluxes
        flux_attrs = {
            "units": Q_.get_netcdf_unit(self.flux_unit),
            #            "cell_methods": "time: mean"
            "cell_methods": "time: total",
        }

        # input fluxes
        Us = Q_(np.array(self.Us + [np.nan * np.ones(nr_pools)]), self.stock_unit)
        #        Us_mean = Us / self.Delta_t
        data_vars["input_fluxes"] = xr.DataArray(
            #            data=Us_mean.magnitude,
            data=Us.magnitude,
            dims=["timestep", "pool_to"],
            attrs=flux_attrs,
        )

        # output fluxes
        Rs = Q_(np.array(self.Rs + [np.nan * np.ones(nr_pools)]), self.stock_unit)
        #        Rs_mean = Rs / self.Delta_t
        data_vars["output_fluxes"] = xr.DataArray(
            #           data=Rs_mean.magnitude,
            data=Rs.magnitude,
            dims=["timestep", "pool_from"],
            attrs=flux_attrs,
        )

        # internal fluxes
        Fs = Q_(
            np.array(self.Fs + [np.nan * np.ones((nr_pools, nr_pools))]),
            self.stock_unit,
        )
        #        Fs_mean = Fs / self.Delta_t
        data_vars["internal_fluxes"] = xr.DataArray(
            #            data=Fs_mean.magnitude,
            data=Fs.magnitude,
            dims=["timestep", "pool_to", "pool_from"],
            attrs=flux_attrs,
        )

        # create dataset
        ds = xr.Dataset(
            data_vars=data_vars,  # type: ignore
            coords=coords,  # type: ignore
        )

        return ds

    ###########################################################################

    @abstractmethod
    def _create_U_func(self) -> Callable:
        """Create the input flux function."""

    @abstractmethod
    def _create_U(self, input_fluxes: dict[str, Q_[float]]) -> np.ndarray:
        """Make numeric input vector from infput fluxes."""

    def _check_consistency(self):
        x = self.xs[-1]

        # check that no pool is negative
        assert np.all(x >= 0)
