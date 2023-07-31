"""Object to store a simulation (spinup) to be continued later on."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import dill
import numpy as np
import pandas as pd
import pint
import xarray as xr
from bgc_md2.notebook_helpers import write_to_logfile

from .. import Q_, ureg, zeta_gluc
from ..stand import Stand
from ..trees.single_tree_params import species_params_to_latex_table
from ..type_aliases import SimulationProfile, SpeciesParams
from . import utils as sim_utils
from .simulation import Simulation


class RecordedSimulation:
    """Results of a recorded simulation. Can be used to continue from here."""

    def __init__(
        self,
        #        forcing: pd.Dataframe,
        simulation: Simulation,
        additional_vars: Dict[str, Any],
        ds: xr.Dataset,
    ):
        #        self.forcing = forcing
        self.simulation = simulation
        #    times: List[Any]
        #    GPP_totals: List[Q_[float]]
        #    GPP_years: List[Q_[float]]
        self.additional_vars = additional_vars
        self.ds = ds

    @classmethod
    def from_simulation_run(
        cls,
        sim_name: str,
        logfile_path: Path,
        sim_profile: SimulationProfile,
        light_model: str,
        forcing: pd.DataFrame,
        custom_species_params: SpeciesParams,
        stand: Stand,
        #    final_felling: bool,
        emergency_action_str: str,
        emergency_direction: str,
        emergency_q: float,
        emergency_stand_action_str: str = "",
        recorded_spinup_simulation: RecordedSimulation = None,
    ) -> RecordedSimulation:
        """Run a simulation and record a bunch of variables.

        Args:
            sim_name: name of the simulation
            logfile_path: path to the logfile
            sim_profile: List of trees with data for the simulation
            forcing: external forcing ``DataFrame``
            stand: stand to be used for simulation
            emergency_action_str: element of ["Cut", "CutWait3AndReplant", Thin", "Die"]

                - Cut: Cuts a number of trees from above or below
                - CutWait3AndReplant: Like "Cut" but with delayed replanting.
                - Thin: Thin the entire stand equally.
                - Die: Remove the unsustaibale tree.

            emergency_direction: `above` or `below`, applies only for cutting
            emergency_q: Fraction of trees to keep on "Thin"
            emergency_stand_action_str: What to do with remaining trees in stand.
            recorded_spinup_simulation: If simulation is to be continued from here.

        Returns:
            tuple:

            - simulation, storing all data
            - additonal variables than can be added to the dataset
        """
        if recorded_spinup_simulation is None:
            # prepare recording of some tree properties
            additional_var_names = [
                "rs",
                "Hs",
                "dbhs",
                "R_Ms",
                "LAs",
                "V_Ts",
                "V_THs",
                "V_TSs",
                "LabileCs_assimilated",
                "LabileCs_respired",
                "GLs",
                "GRs",
                "GSs",
                "MLs",
                "MRs",
                "MSs",
                "f_L_times_Es",
                "f_R_times_Es",
                "f_O_times_Es",
                "f_T_times_Es",
                "f_CS_times_CSs",
                "coefficient_sums",
                "R_M_corrections",
                "rho_Ws",
                "delta_Ws",
                "SWs",
                "B_S_stars",
                "C_S_stars",
            ]
            additional_vars: Dict[
                str, Union[Dict[str, List[Q_[float]]], List[Q_[float]]]
            ] = {
                var_name: {tree.name: [] for tree in stand.trees}
                for var_name in additional_var_names
            }

            # prepare recording of some stand properties
            additional_vars["stand_basal_area"] = []
            additional_vars["dominant_tree_height"] = []
            additional_vars["mean_tree_height"] = []
        else:
            additional_vars = recorded_spinup_simulation.additional_vars.copy()

        def callback(stand: Stand) -> bool:
            """Callback function for :meth:`~..stand.Stand.run_simulation``."""
            for tree in stand.trees:
                C_only_tree = tree.C_only_tree

                additional_vars["rs"][tree.name].append(  # type: ignore
                    tree.r if tree.is_alive else np.nan
                )
                additional_vars["Hs"][tree.name].append(  # type: ignore
                    tree.H if tree.is_alive else np.nan
                )
                additional_vars["dbhs"][tree.name].append(  # type: ignore
                    tree.dbh if tree.is_alive else np.nan
                )
                additional_vars["R_Ms"][tree.name].append(  # type: ignore
                    C_only_tree.tree.R_M * zeta_gluc * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else Q_(np.nan, "gC/yr/m^2")
                )
                additional_vars["LAs"][tree.name].append(  # type: ignore
                    C_only_tree.tree.LA * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["V_Ts"][tree.name].append(  # type: ignore
                    C_only_tree.tree.V_T * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["V_THs"][tree.name].append(  # type: ignore
                    C_only_tree.tree.V_TH * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["V_TSs"][tree.name].append(  # type: ignore
                    C_only_tree.tree.V_TS * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )

                additional_vars["LabileCs_assimilated"][tree.name].append(  # type: ignore
                    Q_(tree._LabileC_assimilated, "gC") * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["LabileCs_respired"][tree.name].append(  # type: ignore
                    Q_(tree._LabileC_respired, "gC") * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )

                additional_vars["GLs"][tree.name].append(  # type: ignore
                    Q_(C_only_tree._GL, "gC/yr") * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["GRs"][tree.name].append(  # type: ignore
                    Q_(C_only_tree._GR, "gC/yr") * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["GSs"][tree.name].append(  # type: ignore
                    Q_(C_only_tree._GS, "gC/yr") * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["MLs"][tree.name].append(  # type: ignore
                    Q_(C_only_tree._ML, "gC/yr") * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["MRs"][tree.name].append(  # type: ignore
                    Q_(C_only_tree._MR, "gC/yr") * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["MSs"][tree.name].append(  # type: ignore
                    Q_(C_only_tree._MS, "gC/yr") * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )

                additional_vars["f_L_times_Es"][tree.name].append(  # type: ignore
                    C_only_tree._f_L_times_E * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["f_R_times_Es"][tree.name].append(  # type: ignore
                    C_only_tree._f_R_times_E * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["f_O_times_Es"][tree.name].append(  # type: ignore
                    C_only_tree._f_O_times_E * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["f_T_times_Es"][tree.name].append(  # type: ignore
                    C_only_tree._f_T_times_E * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["f_CS_times_CSs"][tree.name].append(  # type: ignore
                    C_only_tree._f_CS_times_CS * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )

                additional_vars["coefficient_sums"][tree.name].append(  # type: ignore
                    C_only_tree._coefficient_sum if tree.is_alive else np.nan
                )
                additional_vars["R_M_corrections"][tree.name].append(  # type: ignore
                    C_only_tree._R_M_correction * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )

                additional_vars["rho_Ws"][tree.name].append(  # type: ignore
                    C_only_tree.tree._rho_W if tree.is_alive else np.nan
                )
                additional_vars["delta_Ws"][tree.name].append(  # type: ignore
                    C_only_tree.tree._delta_W if tree.is_alive else np.nan
                )
                additional_vars["SWs"][tree.name].append(  # type: ignore
                    C_only_tree.tree.SW if tree.is_alive else np.nan
                )
                additional_vars["B_S_stars"][tree.name].append(  # type: ignore
                    C_only_tree.tree.B_S_star * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )
                additional_vars["C_S_stars"][tree.name].append(  # type: ignore
                    C_only_tree.tree.C_S_star * Q_(tree.N_per_m2, "1/m^2")
                    if tree.is_alive
                    else np.nan
                )

            additional_vars["stand_basal_area"].append(stand.basal_area)  # type: ignore
            additional_vars["dominant_tree_height"].append(stand.dominant_tree_height)  # type: ignore
            additional_vars["mean_tree_height"].append(stand.mean_tree_height)  # type: ignore

            return False

        start_date = forcing.index[0]
        end_date = forcing.index[-1]
        log = f"Running simulation from {start_date} until {end_date}"
        write_to_logfile(logfile_path, log)
        print(log, flush=True)

        # log species parameters
        species_names = {tree_in_stand.species for tree_in_stand in stand.trees}
        for species in species_names:
            write_to_logfile(
                logfile_path,
                species,
                species_params_to_latex_table(species, custom_species_params) + "\n",
            )

        # log simulation setting
        write_to_logfile(logfile_path, f"Simulation profile:\n{sim_profile}")
        write_to_logfile(logfile_path, f"Stand:\n{stand}")
        write_to_logfile(logfile_path, "\nSimulation:")

        if recorded_spinup_simulation is None:
            simulation = Simulation(
                sim_name,
                stand,
                emergency_action_str,
                emergency_direction,
                emergency_q,
                emergency_stand_action_str,
            )
        else:
            spinup_simulation = recorded_spinup_simulation.simulation
            simulation = Simulation(
                sim_name,
                stand,
                emergency_action_str,
                emergency_direction,
                emergency_q,
                emergency_stand_action_str,
                times=spinup_simulation.times[:-1],
                GPP_totals=spinup_simulation.GPP_totals,
                GPP_years=spinup_simulation.GPP_years,
            )

        simulation.run(
            forcing,
            light_model,
            #        final_felling=final_felling,
            callback=callback,
            logfile_path=logfile_path,
            show_pbar=True,
        )

        #        if recorded_spinup_simulation is None:
        #            total_forcing = forcing
        #        else:
        #            total_forcing = pd.concat([recorded_spinup_simulation.forcing, forcing])

        recorded_simulation = cls(
            #            total_forcing,
            simulation,
            additional_vars,
            sim_utils.get_simulation_record_ds(simulation, additional_vars),
        )
        return recorded_simulation

    @classmethod
    def from_file(cls, filepath: Path) -> "RecordedSimulation":
        """Load instance from file."""
        pint.set_application_registry(ureg)
        with open(filepath, "rb") as f:
            loaded_recorded_simulation = dill.load(f)

        return loaded_recorded_simulation

    def save_to_file(self, filepath: Path):
        """Save instance to file."""
        with open(filepath, "wb") as f:
            dill.dump(self, f)
