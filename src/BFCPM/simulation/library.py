"""Library for simulations and useful functions associated."""
from __future__ import annotations

from typing import Dict, Union

import pandas as pd

from .. import utils
from ..simulation_parameters import simulation_params
from ..type_aliases import SimulationProfile


def prepare_forcing(nr_copies: int, year_offset: int = 0) -> pd.Dataframe:
    """Prepare a detrended forcing for a simulation.

    Args:
        nr_copies: number of 20 year batches of forcing
        year_offset: add this number to the year 2000, it can be helpful if part of
            the simulation is a spinup or if the simulation would end later than
            what ´pd.Datetime´ supports

    Returns:
        A detrended forcing dataset.
    """
    min_year_offset = -250
    if year_offset < min_year_offset:
        raise ValueError(f"Do not start earlier than {2000+min_year_offset}.")

    # simulation data
    forcing_base = utils.load_forcing(
        simulation_params["fpath"],
        #    start_date="1997-01-01",
        start_date="2000-01-01",
        end_date="2019-12-31",
    )

    forcing_base.index = forcing_base.index + pd.DateOffset(years=year_offset)

    # variables to detrend or extend
    variable_names = ["dirPar", "diffPar", "Tair", "H2O", "CO2", "U", "Prec", "P"]

    forcing = utils.detrend_forcing(forcing_base, nr_copies, variable_names)

    if forcing_base.index.min().year < 2000 + min_year_offset:
        raise OverflowError("Date exceeded 2262, not supoorted by DatetimeIndex.")

    if nr_copies == 8:
        # otherwise we will have one day in 2160
        forcing = forcing[forcing.index.year < 2160 + year_offset]
    if nr_copies == 12:
        forcing = forcing[forcing.index.year < 2240 + year_offset]
    if nr_copies == 16:
        forcing = forcing[forcing.index.year < 2320 + year_offset]

    return forcing


def create_mixed_aged_sim_profile(
    species: str, N: float, clear_cut_year: Union[int, None]
) -> SimulationProfile:
    management_strategies = [
        [
            ("StandAge3", "Plant"),
            ("StandAge259", "CutWait3AndReplant"),
            ("StandAge179", "CutWait3AndReplant"),
            ("StandAge99", "CutWait3AndReplant"),
            ("StandAge19", "CutWait3AndReplant"),
        ],
        [
            ("StandAge3", "Plant"),
            ("StandAge279", "CutWait3AndReplant"),
            ("StandAge199", "CutWait3AndReplant"),
            ("StandAge119", "CutWait3AndReplant"),
            ("StandAge39", "CutWait3AndReplant"),
        ],
        [
            ("StandAge3", "Plant"),
            ("StandAge299", "CutWait3AndReplant"),
            ("StandAge219", "CutWait3AndReplant"),
            ("StandAge139", "CutWait3AndReplant"),
            ("StandAge59", "CutWait3AndReplant"),
        ],
        [
            ("StandAge3", "Plant"),
            ("StandAge239", "CutWait3AndReplant"),
            ("StandAge159", "CutWait3AndReplant"),
            ("StandAge79", "CutWait3AndReplant"),
        ],
    ]

    if clear_cut_year is not None:
        clean_management_strategies = list()
        for management_strategy in management_strategies:

            def is_before_clear_cut(trigger_name: str) -> bool:
                idx = trigger_name.find("StandAge")
                if idx != -1:
                    trigger_year = int(trigger_name[(idx + len("StandAge")) :])
                    if trigger_year >= clear_cut_year - 1:  # type: ignore
                        return False

                return True

            # remove all actions after the clear cut
            clean_management_strategy = [
                tup for tup in management_strategy if is_before_clear_cut(tup[0])
            ]
            # add clear clear cut action
            clean_management_strategy.append((f"StandAge{clear_cut_year-1}", "Cut"))
            clean_management_strategies.append(clean_management_strategy)

        management_strategies = clean_management_strategies

    sim_profile = [
        (species, 1.0, N / 10_000.0 / 4, management_strategy, "waiting")
        for management_strategy in management_strategies
    ]

    return sim_profile  # type: ignore


def load_clear_cut_sim_profiles(
    N: float, spinup_length: int, sim_length: int
) -> Dict[str, SimulationProfile]:
    """Load simulation profiles that start with a clear cut after the spinup.

    The number in the name stands for the length of the spinup.
    """
    sim_end = spinup_length + sim_length

    management_strategy = [
        (f"OnDelayOneTime{spinup_length+1}", "Wait3AndPlant"),
        ("PCT", "T0.75"),
        (f"DBH35-{sim_end}", "CutWait3AndReplant"),
        (f"SBA25-{sim_end}", "ThinStandToSBA18"),
    ]

    clear_cut_sim_profiles = {
        "even-aged_pine_long": [
            ("pine", 1.0, N / 10_000 / 4, management_strategy, "waiting"),
            ("pine", 1.2, N / 10_000 / 4, management_strategy, "waiting"),
            ("pine", 1.4, N / 10_000 / 4, management_strategy, "waiting"),
            ("pine", 1.6, N / 10_000 / 4, management_strategy, "waiting"),
        ],
        "even-aged_spruce_long": [
            ("spruce", 1.0, 0.2 / 4, management_strategy, "waiting"),
            ("spruce", 1.2, 0.2 / 4, management_strategy, "waiting"),
            ("spruce", 1.4, 0.2 / 4, management_strategy, "waiting"),
            ("spruce", 1.6, 0.2 / 4, management_strategy, "waiting"),
        ],
        "even-aged_mixed_long": [
            ("pine", 1.2, 0.2 / 4, management_strategy, "waiting"),
            ("pine", 1.4, 0.2 / 4, management_strategy, "waiting"),
            ("spruce", 1.2, 0.2 / 4, management_strategy, "waiting"),
            ("spruce", 1.4, 0.2 / 4, management_strategy, "waiting"),
        ],
    }

    return clear_cut_sim_profiles  # type: ignore
