"""Library readily of available triggers and management actions."""
from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np

from .. import Q_
from ..type_aliases import (MSData, MSDataList, SimulationProfile,
                            SpeciesSettings)
from . import management_strategy as ms_module
from .management_strategy import (ManagementActionABC, ManagementStrategy,
                                  TriggerABC)

triggers: Dict[str, Dict[str, Union[str, Any]]] = {
    # pre-commercial thinning
    "PCT": {"cls_name": "PCT", "kwargs": {"mth_lim": Q_(3.0, "m")}},
    "StandAge3": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("3 yr")]},
    },
    "StandAge9": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("9 yr")]},
    },
    "StandAge19": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("19 yr")]},
    },
    "StandAge39": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("39 yr")]},
    },
    "StandAge59": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("59 yr")]},
    },
    "StandAge79": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("79 yr")]},
    },
    "StandAge99": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("99 yr")]},
    },
    "StandAge119": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("119 yr")]},
    },
    "StandAge139": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("139 yr")]},
    },
    "StandAge159": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("159 yr")]},
    },
    "StandAge179": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("179 yr")]},
    },
    "StandAge199": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("199 yr")]},
    },
    "StandAge219": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("219 yr")]},
    },
    "StandAge239": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("239 yr")]},
    },
    "StandAge259": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("259 yr")]},
    },
    "StandAge279": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("279 yr")]},
    },
    "StandAge299": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("299 yr")]},
    },
    "StandAge319": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("319 yr")]},
    },
    "StandAge339": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("339 yr")]},
    },
    "StandAge359": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("359 yr")]},
    },
    "StandAge379": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("379 yr")]},
    },
    "StandAge399": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("399 yr")]},
    },
    "StandAge419": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("419 yr")]},
    },
    "StandAge439": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("439 yr")]},
    },
    "StandAge459": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("459 yr")]},
    },
    "StandAge479": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("479 yr")]},
    },
    "StandAge160": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("160 yr")]},
    },
    "StandAge180": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("180 yr")]},
    },
    "StandAge200": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("200 yr")]},
    },
    "StandAge220": {
        "cls_name": "OnStandAges",
        "kwargs": {"stand_ages": [Q_("220 yr")]},
    },
    #####
    "OnDelayOneTime1": {
        "cls_name": "OnDelayOneTime",
        "kwargs": {"delay_length": Q_(1.0, "yr")},
    },
    "OnDelayOneTime11": {
        "cls_name": "OnDelayOneTime",
        "kwargs": {"delay_length": Q_(11.0, "yr")},
    },
    "OnDelayOneTime81": {
        "cls_name": "OnDelayOneTime",
        "kwargs": {"delay_length": Q_(81.0, "yr")},
    },
    "OnDelayOneTime161": {
        "cls_name": "OnDelayOneTime",
        "kwargs": {"delay_length": Q_(161.0, "yr")},
    },
    "OnDelayOneTime241": {
        "cls_name": "OnDelayOneTime",
        "kwargs": {"delay_length": Q_(241.0, "yr")},
    },
    "SBAvsDTHBrownLower80": {
        "cls_name": "OnSBAvsDTH",
        "kwargs": {"blocked_stand_ages": [(Q_(70, "yr"), Q_(np.inf, "yr"))]},
    },
    "SBAvsDTHBrownLower160": {
        "cls_name": "OnSBAvsDTH",
        "kwargs": {"blocked_stand_ages": [(Q_(150, "yr"), Q_(np.inf, "yr"))]},
    },
    "SBAvsDTHBrownLower80-160": {
        "cls_name": "OnSBAvsDTH",
        "kwargs": {
            "blocked_stand_ages": [
                (Q_(70, "yr"), Q_(80, "yr")),
                (Q_(150, "yr"), Q_(np.inf, "yr")),
            ]
        },
    },
    "DBH25": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("25 cm"),
            "blocked_stand_ages": list(),
            "remain_active": False,
        },
    },
    "DBH35": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": list(),
            "remain_active": True,
        },
    },
    "DBH35-240": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(230, "yr"), Q_(240, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-320": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(310, "yr"), Q_(320, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-20-100": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(10, "yr"), Q_(20, "yr")),
                (Q_(90, "yr"), Q_(100, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-20-80": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(10, "yr"), Q_(20, "yr")),
                (Q_(70, "yr"), Q_(80, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-40-80": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(30, "yr"), Q_(40, "yr")),
                (Q_(70, "yr"), Q_(80, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-60-80": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(50, "yr"), Q_(60, "yr")),
                (Q_(70, "yr"), Q_(80, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-80": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [(Q_(70, "yr"), Q_(80, "yr"))],
            "remain_active": True,
        },
    },
    "DBH35-20-100-160": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(10, "yr"), Q_(20, "yr")),
                (Q_(90, "yr"), Q_(100, "yr")),
                (Q_(150, "yr"), Q_(160, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-40-120-160": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(30, "yr"), Q_(40, "yr")),
                (Q_(110, "yr"), Q_(120, "yr")),
                (Q_(150, "yr"), Q_(160, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-60-140-160": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(50, "yr"), Q_(60, "yr")),
                (Q_(130, "yr"), Q_(140, "yr")),
                (Q_(150, "yr"), Q_(160, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-20-100-180-240": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(10, "yr"), Q_(20, "yr")),
                (Q_(90, "yr"), Q_(100, "yr")),
                (Q_(170, "yr"), Q_(180, "yr")),
                (Q_(230, "yr"), Q_(240, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-20-100-180-260-320": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(10, "yr"), Q_(20, "yr")),
                (Q_(90, "yr"), Q_(100, "yr")),
                (Q_(170, "yr"), Q_(180, "yr")),
                (Q_(250, "yr"), Q_(260, "yr")),
                (Q_(310, "yr"), Q_(320, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-40-120": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(30, "yr"), Q_(40, "yr")),
                (Q_(110, "yr"), Q_(120, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-40-120-200-240": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(30, "yr"), Q_(40, "yr")),
                (Q_(110, "yr"), Q_(120, "yr")),
                (Q_(190, "yr"), Q_(200, "yr")),
                (Q_(230, "yr"), Q_(240, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-40-120-200-280-320": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(30, "yr"), Q_(40, "yr")),
                (Q_(110, "yr"), Q_(120, "yr")),
                (Q_(190, "yr"), Q_(200, "yr")),
                (Q_(270, "yr"), Q_(280, "yr")),
                (Q_(310, "yr"), Q_(320, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-60-140-220-240": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(50, "yr"), Q_(60, "yr")),
                (Q_(130, "yr"), Q_(140, "yr")),
                (Q_(210, "yr"), Q_(220, "yr")),
                (Q_(230, "yr"), Q_(240, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-60-140-220-300-320": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(50, "yr"), Q_(60, "yr")),
                (Q_(130, "yr"), Q_(140, "yr")),
                (Q_(210, "yr"), Q_(220, "yr")),
                (Q_(290, "yr"), Q_(300, "yr")),
                (Q_(310, "yr"), Q_(320, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-80-160": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(70, "yr"), Q_(80, "yr")),
                (Q_(150, "yr"), Q_(160, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-80-160-240": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(70, "yr"), Q_(80, "yr")),
                (Q_(150, "yr"), Q_(160, "yr")),
                (Q_(230, "yr"), Q_(240, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-80-160-240-320": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(70, "yr"), Q_(80, "yr")),
                (Q_(150, "yr"), Q_(160, "yr")),
                (Q_(230, "yr"), Q_(240, "yr")),
                (Q_(310, "yr"), Q_(320, "yr")),
            ],
            "remain_active": True,
        },
    },
    "DBH35-20-100-160-180-240": {
        "cls_name": "OnDBHLimit",
        "kwargs": {
            "dbh_lim": Q_("35 cm"),
            "blocked_stand_ages": [
                (Q_(70, "yr"), Q_(80, "yr")),
                (Q_(150, "yr"), Q_(160, "yr")),
                (Q_(230, "yr"), Q_(240, "yr")),
            ],
            "remain_active": True,
        },
    },
    "SBA25": {
        "cls_name": "OnSBALimit",
        "kwargs": {"sba_lim": Q_("25 m^2/ha"), "blocked_stand_ages": []},
    },
    "SBA25-80": {
        "cls_name": "OnSBALimit",
        "kwargs": {
            "sba_lim": Q_("25 m^2/ha"),
            "blocked_stand_ages": [(Q_(70, "yr"), Q_(80, "yr"))],
        },
    },
    "SBA25-160": {
        "cls_name": "OnSBALimit",
        "kwargs": {
            "sba_lim": Q_("25 m^2/ha"),
            "blocked_stand_ages": [(Q_(150, "yr"), Q_(160, "yr"))],
        },
    },
    "SBA25-240": {
        "cls_name": "OnSBALimit",
        "kwargs": {
            "sba_lim": Q_("25 m^2/ha"),
            "blocked_stand_ages": [(Q_(230, "yr"), Q_(240, "yr"))],
        },
    },
    "SBA25-320": {
        "cls_name": "OnSBALimit",
        "kwargs": {
            "sba_lim": Q_("25 m^2/ha"),
            "blocked_stand_ages": [(Q_(310, "yr"), Q_(320, "yr"))],
        },
    },
    "SBA25-80-160": {
        "cls_name": "OnSBALimit",
        "kwargs": {
            "sba_lim": Q_("25 m^2/ha"),
            "blocked_stand_ages": [
                (Q_(70, "yr"), Q_(80, "yr")),
                (Q_(150, "yr"), Q_(np.inf, "yr")),
            ],
        },
    },
    "SBA25-40-80": {
        "cls_name": "OnSBALimit",
        "kwargs": {
            "sba_lim": Q_("25 m^2/ha"),
            "blocked_stand_ages": [
                (Q_(30, "yr"), Q_(40, "yr")),
                (Q_(70, "yr"), Q_(np.inf, "yr")),
            ],
        },
    },
    "SBA25-80-160-240": {
        "cls_name": "OnSBALimit",
        "kwargs": {
            "sba_lim": Q_("25 m^2/ha"),
            "blocked_stand_ages": [
                (Q_(70, "yr"), Q_(80, "yr")),
                (Q_(150, "yr"), Q_(160, "yr")),
                (Q_(230, "yr"), Q_(np.inf, "yr")),
            ],
        },
    },
    "NLimit0025": {"cls_name": "OnNLimit", "kwargs": {"N_lim": Q_(0.0025, "1/m^2")}},
}
"""List of easy to access triggers with pre-defined behavior."""


actions: Dict[str, Dict[str, Union[str, Any]]] = {
    "T0.5": {"cls_name": "Thin", "kwargs": {"q": 0.5}},
    "T0.6": {"cls_name": "Thin", "kwargs": {"q": 1 - 1500.0 / 2500.0}},
    "T0.75": {"cls_name": "Thin", "kwargs": {"q": 1.0 - 1500.0 / 2000.0}},
    "T0.8": {"cls_name": "Thin", "kwargs": {"q": 1 - 2000.0 / 2500.0}},
    "T0.9": {"cls_name": "Thin", "kwargs": {"q": 1 - 1800.0 / 2000.0}},
    "ThinStandGreenLower": {"cls_name": "ThinStand", "kwargs": {}},
    "ThinStandToSBA18": {"cls_name": "ThinStand", "kwargs": {"f": lambda dth: 18.0}},
    "CutAndReplant": {"cls_name": "CutAndReplant", "kwargs": {}},
    "CutWaitAndReplant": {"cls_name": "CutWaitAndReplant", "kwargs": {"nr_waiting": 0}},
    "CutWait3AndReplant": {
        "cls_name": "CutWaitAndReplant",
        "kwargs": {"nr_waiting": 3},
    },
    "Cut": {"cls_name": "Cut", "kwargs": {}},
    "Plant": {"cls_name": "Plant", "kwargs": {}},
    "Wait3AndPlant": {"cls_name": "WaitAndPlant", "kwargs": {"nr_waiting": 3}},
}
"""List of easy to access management actions with pre-defined behavior."""


def ma_data_to_ma(ma_data: str) -> ManagementActionABC:
    """Return a management action from the `actions` dictionary."""
    action_dict = actions[ma_data]
    action_cls = getattr(ms_module, action_dict["cls_name"])
    management_action = action_cls(**action_dict["kwargs"])

    return management_action


def ms_data_to_ms_item(ms_data: MSData) -> Tuple[TriggerABC, ManagementActionABC]:
    """Return a (trigger, action) tuple."""
    trigger_dict = triggers[ms_data[0]]
    trigger = getattr(ms_module, trigger_dict["cls_name"])(**trigger_dict["kwargs"])

    #    action_dict = actions[ms_data[1]]
    #    action_cls = getattr(ms_module, action_dict["cls_name"])
    #    action = action_cls(**action_dict["kwargs"])
    action = ma_data_to_ma(ms_data[1])

    return (trigger, action)


def ms_list_to_management_strategy(ms_list: MSDataList) -> ManagementStrategy:
    """Return a proper ManagementStrategy from a list of (trigger, action) tuples."""
    # we need to avoid that two trees use the ms instance
    # otherwise they will add their times together and act faster
    #    trigger_and_action_list = []
    #    for ms_data in ms_list:
    #        trigger_and_action_list.append(ms_item)
    #
    trigger_and_action_list = [ms_data_to_ms_item(ms_data) for ms_data in ms_list]
    ms = ManagementStrategy(trigger_and_action_list)
    return ms


def species_setting_from_sim_profile(sim_profile: SimulationProfile) -> SpeciesSettings:
    """Return species settings to initialize a :class:`~..trees.stand.Stand`."""
    species_settings: SpeciesSettings = {"pine": [], "spruce": [], "birch": []}
    wait_str = ""

    for tree_nr, tree_tuple in enumerate(sim_profile):
        # the last str "waiting" is optional, so the length of the list might differ
        if len(tree_tuple) == 4:
            species, dbh, N_per_m2, ms_data_list = tree_tuple  # type: ignore
        elif len(tree_tuple) == 5:
            species, dbh, N_per_m2, ms_data_list, wait_str = tree_tuple  # type: ignore
        else:
            raise ValueError("Tree tuple has unknown structure.")

        dbh = Q_(float(dbh), "cm")
        N_per_m2 = Q_(float(N_per_m2), "1/m^2")

        ms = ms_list_to_management_strategy(ms_data_list)

        #        waiting = True if wait_str == "waiting" else False
        #        waiting = bool(wait_str == "waiting")
        #        waiting = wait_str == "waiting"
        #        waiting: bool = wait_str == "waiting"

        #        species_settings[species].append((tree_nr, dbh, N_per_m2, ms, waiting))
        species_settings[species].append((tree_nr, dbh, N_per_m2, ms, wait_str))

    return species_settings


# seems outdated, code that calls these functions is likely to be outdated as well
# def emergency_action_from_str(ea: str) -> ManagementActionABC:
#    action_dict = actions[ea]
#    action_cls = getattr(ms_module, action_dict["cls_name"])
#    emergency_action = action_cls(**action_dict["kwargs"])
#
#    return emergency_action
#
#
# def tree_tuple_str(tree_tuple):
#    ms_list = tree_tuple[-1]
#    ms_str = "-".join(["-".join([s for s in tup]) for tup in ms_list])
#    return "-".join([str(x) for x in tree_tuple[:-1]]) + "-" + ms_str
#
#
# def sim_name_from_sim_profile(
#    sim_profile: tuple,
#    emergency_strategy,
#    emergency_action:str
# ) -> str:
#    sim_name = "_".join([tree_tuple_str(tree_tuple) for tree_tuple in sim_profile])
#    sim_name += "-" + emergency_strategy + "-" + emergency_action
#    return sim_name
