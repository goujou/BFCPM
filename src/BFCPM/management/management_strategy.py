"""This module several classes of strategies to manage a forest stand."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod  # , abstractproperty
from typing import TYPE_CHECKING, Callable, List, Tuple

import numpy as np

from .. import Q_

if TYPE_CHECKING:
    from ..stand import Stand
    from ..trees.mean_tree import MeanTree


# data for creating thinning functions, not used right now
_brown_upper_data = {"x_T": 20, "y_T": 32, "x_L": 12, "y_L": 24, "x_C": 17, "y_C": 31}
_brown_lower_data = {"x_T": 22, "y_T": 28, "x_L": 14, "y_L": 24, "x_C": 17, "y_C": 27}
_green_upper_data = {"x_T": 22, "y_T": 22, "x_L": 12, "y_L": 17, "x_C": 16, "y_C": 21}
_green_lower_data = {"x_T": 22, "y_T": 18, "x_L": 12, "y_L": 13.5, "x_C": 16, "y_C": 17}

_brown_upper_pure_birch_stand_data = {
    "x_T": 20,
    "y_T": 24,
    "x_L": 12,
    "y_L": 16,
    "x_C": 16,
    "y_C": 22,
}
_brown_lower_pure_birch_stand_data = {
    "x_T": 20,
    "y_T": 21,
    "x_L": 14,
    "y_L": 17,
    "x_C": 18,
    "y_C": 20,
}
_green_upper_pure_birch_stand_data = {
    "x_T": 20,
    "y_T": 16,
    "x_L": 12,
    "y_L": 9.5,
    "x_C": 16,
    "y_C": 14,
}
_green_lower_pure_birch_stand_data = {
    "x_T": 20,
    "y_T": 14,
    "x_L": 12,
    "y_L": 8,
    "x_C": 16,
    "y_C": 12,
}


def _compute_coeffs(d: dict) -> Tuple[float, float]:
    """Helper function to create coefficents for the function I invented
    to figure out how to interpolate Samuli's graphs for thining best.
    """
    num = np.log(d["y_T"] - d["y_L"]) - np.log(d["y_T"] - d["y_C"])
    denom = np.log(d["x_T"] - d["x_L"]) - np.log(d["x_T"] - d["x_C"])

    beta = num / denom
    alpha = (d["y_T"] - d["y_L"]) / (d["x_T"] - d["x_L"]) ** beta

    return alpha, beta


def _f(x: float, d: dict) -> float:
    """Threshold value whether to thin or not.

    Args:
        x: dbh [cm]
        d: coming from the _dictionaries up in this file
    """
    alpha, beta = _compute_coeffs(d)

    if x < d["x_L"]:
        return np.nan
    elif x < d["x_T"]:
        return d["y_T"] - alpha * (d["x_T"] - x) ** beta
    else:
        return d["y_T"]


class TriggerABC(metaclass=ABCMeta):
    """A meta class for triggers for management strategies."""

    def __init__(self):
        pass

    @abstractmethod
    def triggered(
        self, stand: "Stand", tree_in_stand: "MeanTree", Delta_t: Q_[float]
    ) -> bool:
        """Return bool telling whether trigger becomes activated."""

    def __repr__(self):
        return self.__str__()


class OnStandAges(TriggerABC):
    """Triggered if certain stand age is reached.

    Args:
        stand_ages: sorted (ascending) list of ages

    Note:
        There is no check whether the stand ages are sorted, be careful!
    """

    def __init__(self, stand_ages: List[Q_[float]]):
        super().__init__()
        self.stand_ages = list(stand_ages)

    def triggered(
        self,
        stand: "Stand",
        tree_in_stand: "MeanTree",
        Delta_t: Q_[float],
    ) -> bool:
        if len(self.stand_ages) == 0:
            return False

        if stand.age >= self.stand_ages[0]:
            self.stand_ages = self.stand_ages[1:]
            return True

        return False

    def __str__(self):
        return f"{str(self.__class__.__name__)} (stand_ages={self.stand_ages})"


class OnDBHLimit(TriggerABC):
    """Triggered if certain dbh limit of the tree is reached.

    Args:
        dbh_lim: if exceeded, triggered [cm]
        blocked_stand_ages: at these stand ages no trigger,
            list of pairs (lower_age, upper_age)
        remain_active: if false, triggered only once
    """

    def __init__(
        self,
        dbh_lim: Q_[float],
        blocked_stand_ages: List[Tuple[Q_[float], Q_[float]]],
        remain_active: bool,
    ):
        super().__init__()

        self.dbh_lim = dbh_lim
        # make it a new object
        self.blocked_stand_ages = list(blocked_stand_ages)
        self.remain_active = remain_active
        self.active = True

    def triggered(
        self,
        stand: "Stand",
        tree_in_stand: "MeanTree",
        Delta_t: Q_[float],
    ) -> bool:
        if not self.active:
            return False

        for lower, upper in self.blocked_stand_ages:
            if (stand.age >= lower) and (stand.age <= upper):
                return False

        if tree_in_stand.dbh >= self.dbh_lim:
            self.active = self.remain_active
            return True

        return False

    def __str__(self):
        return (
            f"{str(self.__class__.__name__)} (dbh_lim={self.dbh_lim:~P}, "
            f"(blocked={self.blocked_stand_ages}), remain_active={self.remain_active})"
        )


class PCT(TriggerABC):
    """Pre-commercial thinning.

    Thinning if mean tree height exceeds `mth_lim`, then deactivated.
    Reactivates itself as soon as `stand.mean_tree_height` falls under
    `mth_lim`, for example by an intermediate clear cut.

    Args:
        mth_lim: if exceeded, triggered [m]
    """

    def __init__(self, mth_lim: Q_[float]):
        super().__init__()

        self.mth_lim = mth_lim
        self.active = True

    def triggered(
        self,
        stand: "Stand",
        tree_in_stand: "MeanTree",
        Delta_t: Q_[float],
    ) -> bool:
        mth = stand.mean_tree_height
        if not self.active:
            if np.isnan(mth) or (mth < self.mth_lim):
                self.active = True
            return False

        if mth >= self.mth_lim:
            self.active = False
            return True

        return False

    def __str__(self):
        return f"{str(self.__class__.__name__)} (mth_lim={self.mth_lim:~P})"


class OnDTHLimit(TriggerABC):
    """Triggered if dominant tree height limit is reached.

    Args:
        dth_lim: if exceeded, triggered [m]
        remain_active: if false, triggered only once
    """

    def __init__(self, dth_lim: Q_[float], remain_active: bool):
        super().__init__()

        self.dth_lim = dth_lim
        self.remain_active = remain_active
        self.active = True

    def triggered(
        self,
        stand: "Stand",
        tree_in_stand: "MeanTree",
        Delta_t: Q_[float],
    ) -> bool:
        if not self.active:
            return False

        if stand.dominant_tree_height >= self.dth_lim:
            self.active = self.remain_active
            return True

        return False

    def __str__(self):
        return (
            f"{str(self.__class__.__name__)} (dth_lim={self.dth_lim:~P}, "
            f"remain_active={self.remain_active})"
        )


class OnSBALimit(TriggerABC):
    """Triggered if stand basal area limit is reached.

    Args:
        sba_lim: if exceeded, triggered [m]
        blocked_stand_ages: at these stand ages no trigger,
            list of pairs (lower_age, upper_age)
    """

    def __init__(
        self, sba_lim: Q_[float], blocked_stand_ages: List[Tuple[Q_[float], Q_[float]]]
    ):
        super().__init__()

        self.sba_lim = sba_lim

        # make it a new object
        self.blocked_stand_ages = list(blocked_stand_ages)

    def triggered(
        self,
        stand: "Stand",
        tree_in_stand: "MeanTree",
        Delta_t: Q_[float],
    ) -> bool:
        for lower, upper in self.blocked_stand_ages:
            if (stand.age >= lower) and (stand.age <= upper):
                return False

        if stand.basal_area >= self.sba_lim:
            return True

        return False

    def __str__(self):
        return (
            f"{str(self.__class__.__name__)} (sba_lim={self.sba_lim:~P}, "
            f"(blocked={self.blocked_stand_ages})"
        )


class OnNLimit(TriggerABC):
    """Triggered if tree's N_per_m2 goes by thinning below a limit.

    Args:
        N_lim: if undermatched, triggered [1/m^2]
    """

    def __init__(self, N_lim: Q_[float]):
        super().__init__()
        self.N_lim = N_lim

    def triggered(
        self,
        stand: "Stand",
        tree_in_stand: "MeanTree",
        Delta_t: Q_[float],
    ) -> bool:
        if Q_(tree_in_stand.N_per_m2, "1/m^2") < self.N_lim:
            return True

        return False

    def __str__(self):
        return f"{str(self.__class__.__name__)} N_lim={self.N_lim:~P}"


class OnPeriodically(TriggerABC):
    """Regularly activated after fixed time period.

    Args:
        period_length: length of time interval between actions
    """

    def __init__(self, period_length: Q_[float]):
        super().__init__()
        self.period_length = period_length
        self.time_elapsed = Q_(0, "yr")

    def triggered(
        self,
        stand: "Stand",
        tree_in_stand: "MeanTree",
        Delta_t: Q_[float],
    ) -> bool:
        self.time_elapsed += Delta_t

        if self.time_elapsed >= self.period_length:
            self.time_elapsed = Q_(0, "yr")
            return True

        return False

    def __str__(self):
        return f"{str(self.__class__.__name__)} (period={self.period_length})"


class OnDelayOneTime(TriggerABC):
    """One time activated after fixed time period.

    Args:
        delay_length: length of time interval before the action [yr]
    """

    def __init__(self, delay_length: Q_[float]):
        super().__init__()
        self.delay_length = delay_length
        self.time_elapsed = Q_(0, "yr")
        self.active = True

    def triggered(
        self,
        stand: "Stand",
        tree_in_stand: "MeanTree",
        Delta_t: Q_[float],
    ) -> bool:
        if not self.active:
            return False

        self.time_elapsed += Delta_t

        if self.time_elapsed >= self.delay_length:
            self.time_elapsed = Q_(0, "yr")
            self.active = False
            return True

        return False

    def __str__(self):
        return f"{str(self.__class__.__name__)} (period={self.delay_length})"


class OnTimeIntervals(TriggerABC):
    """Triggered based on given time intervals.

    Args:
        interval_lengths: list of lengths of time intervals between actions
    """

    def __init__(self, interval_lengths: List[Q_[float]]):
        super().__init__()
        self.interval_lengths = interval_lengths
        self.time_elapsed = Q_(0, "yr")

    @property
    def current_interval_length(self):
        """Current length of time interval between last and next cutting."""
        if len(self.interval_lengths) == 0:
            return Q_(np.inf, "yr")

        return self.interval_lengths[0]

    def triggered(
        self,
        stand: "Stand",
        tree_in_stand: "MeanTree",
        Delta_t: Q_[float],
    ) -> bool:
        self.time_elapsed += Delta_t

        if self.time_elapsed >= self.current_interval_length:
            self.interval_lengths = self.interval_lengths[1:]
            self.time_elapsed = Q_(0, "yr")
            return True

        return False

    def __str__(self):
        return f"{str(self.__class__.__name__)} (intervals={self.interval_lengths})"


class OnSBAvsDTH(TriggerABC):
    """Triggered based on stand basal area and dominant tree height.

    Args:
        blocked_stand_ages: at these stand ages no trigger,
            list of pairs (lower_age, upper_age)
        f: function of dominant tree height, returns limit value
            for stand basal area, triggered if current stand
            basal area is above;
            if `None`, lower brown function is used
    """

    def __init__(
        self,
        blocked_stand_ages: List[Tuple[Q_[float], Q_[float]]],
        f: Callable[[float], float] = None,
    ):
        super().__init__()

        # make it a new object
        self.blocked_stand_ages = list(blocked_stand_ages)
        self.f = f

    def triggered(
        self,
        stand: "Stand",
        tree_in_stand: "MeanTree",
        Delta_t: Q_[float],
    ) -> bool:
        for lower, upper in self.blocked_stand_ages:
            if (stand.age >= lower) and (stand.age <= upper):
                return False

        pure_birch_stand = True
        for tree in stand.living_trees:
            if tree.species != "birch":
                pure_birch_stand = False

        pure_spruce_stand = True
        for tree in stand.living_trees:
            if tree.species != "spruce":
                pure_spruce_stand = False

        if self.f is None:
            if pure_birch_stand:
                f = lambda dth: _f(dth, _brown_lower_pure_birch_stand_data)
            elif pure_spruce_stand:
                f = lambda dth: _f(dth, _brown_lower_data)
            else:
                f = lambda dth: _f(dth, _brown_lower_data)
        else:
            f = self.f

        dth = stand.dominant_tree_height.to("m").magnitude
        sba = stand.basal_area.to("m^2 / ha").magnitude
        if sba > f(dth):
            return True

        return False

    def __str__(self):
        return f"{str(self.__class__.__name__)} (blocked={self.blocked_stand_ages})"


class ManagementActionABC(metaclass=ABCMeta):
    """A forest stand management action metaclass."""

    def __init__(self):
        pass

    def __str__(self):
        s = [f"{str(self.__class__.__name__)}"]
        return "\n".join(s)

    def __repr__(self):
        return str(self)

    @abstractmethod
    def do(self, stand: "Stand", tree_in_stand: "MeanTree") -> List[str]:
        """Return list of management actions as str."""


class Thin(ManagementActionABC):
    """Management action to thin a mean tree.

    Args:
        q [-]: the fraction of ``N_per_m2`` to cut.
    """

    def __init__(
        self,
        q: float,
    ):
        super().__init__()
        if (q <= 0) or (q >= 1):
            raise ValueError("Thinning fraction not between 0 and 1.")
        self.q = q

    def __str__(self):
        s = [f"{str(self.__class__.__name__)} (q={self.q:2.2f})"]

        return "\n".join(s)

    def do(self, stand: "Stand", tree_in_stand: "MeanTree") -> List[str]:
        actions = list()
        if tree_in_stand.is_alive:
            actions = ["thin"]
            tree_in_stand._new_N_per_m2 = (1 - self.q) * tree_in_stand.N_per_m2

        return actions


class ThinStand(ManagementActionABC):
    """Management action to thin the entire stand.

    Args:
        f: goal value (stand basal area [m2 ha-1]) depending on
            stand dominant tree height [m]
            if `None`, lower green function will be used
    """

    def __init__(self, f: Callable[[float], float] = None):
        super().__init__()
        self.f = f

    def __str__(self):
        s = [f"{str(self.__class__.__name__)}"]
        return "\n".join(s)

    def do(self, stand: "Stand", tree_in_stand: "MeanTree") -> List[str]:
        if not tree_in_stand.is_alive:
            return list()

        actions = ["thin"]

        pure_birch_stand = True
        for tree in stand.living_trees:
            if tree.species != "birch":
                pure_birch_stand = False

        pure_spruce_stand = True
        for tree in stand.living_trees:
            if tree.species != "spruce":
                pure_spruce_stand = False

        if self.f is None:
            if pure_birch_stand:
                f = lambda dth: _f(dth, _green_lower_pure_birch_stand_data)
            elif pure_spruce_stand:
                f = lambda dth: _f(dth, _green_lower_data)
            else:
                f = lambda dth: _f(dth, _green_lower_data)
        else:
            f = self.f

        sba_current = stand.basal_area.to("m^2 / ha").magnitude
        dth = stand.dominant_tree_height.to("m").magnitude
        sba_goal = f(dth)

        if sba_goal >= sba_current:
            return list()

        q = 1 - sba_goal / sba_current

        tree_in_stand._new_N_per_m2 = (1 - q) * tree_in_stand.N_per_m2

        return actions


class Cut(ManagementActionABC):
    """Management action to cut MeanTree down."""

    def __init__(self):
        super().__init__()

    def do(self, stand: "Stand", tree_in_stand: "MeanTree") -> List[str]:
        if not tree_in_stand.is_alive:
            return list()

        actions = ["cut"]
        return actions


class CutAndReplant(ManagementActionABC):
    """Management action to cut MeanTree down and replant it.

    Replanting is done with ``dbh`` and ``N_per_m2`` as in the original tree.
    """

    def do(self, stand: "Stand", tree_in_stand: "MeanTree") -> List[str]:
        if (not tree_in_stand.is_alive) and (tree_in_stand.current_status != "removed"):
            return list()

        tree_in_stand._new_species = tree_in_stand.species
        tree_in_stand._new_dbh = tree_in_stand.C_only_tree.tree.initial_dbh
        tree_in_stand._new_tree_age = Q_("0 yr")
        tree_in_stand._new_N_per_m2 = tree_in_stand.base_N_per_m2

        if tree_in_stand.is_alive:
            actions = ["cut", "replant"]
        elif tree_in_stand.current_status == "removed":
            actions = ["wait", "replant"]

        return actions


class CutWaitAndReplant(ManagementActionABC):
    """Management action to cut MeanTree down and replant it after some break break.

    Replanting is done with ``dbh`` and ``N_per_m2`` as in the original tree.

    Args:
        - nr_waiting: number of years to wait before replanting
    """

    def __init__(self, nr_waiting: int):
        super().__init__()
        self.nr_waiting = nr_waiting

    def do(self, stand: "Stand", tree_in_stand: "MeanTree") -> List[str]:
        if (not tree_in_stand.is_alive) and (tree_in_stand.current_status != "removed"):
            return list()

        tree_in_stand._new_species = tree_in_stand.species
        tree_in_stand._new_dbh = tree_in_stand.C_only_tree.tree.initial_dbh
        tree_in_stand._new_tree_age = Q_(self.nr_waiting + 1, "yr")
        tree_in_stand._new_N_per_m2 = tree_in_stand.base_N_per_m2

        if tree_in_stand.is_alive:
            actions = ["cut"] + ["wait"] * (self.nr_waiting + 1) + ["replant"]
        elif tree_in_stand.current_status == "removed":
            # actually, cut and the first wait happen at the same time step
            actions = ["wait"] * (self.nr_waiting + 1) + ["replant"]

        return actions

    def __str__(self):
        return f"{str(self.__class__.__name__)} (nr_waiting={self.nr_waiting})"


class Plant(ManagementActionABC):
    """Management action to plant a MeanTree that is `waiting`."""

    def __init__(
        self,
    ):
        super().__init__()

    def do(self, stand: "Stand", tree_in_stand: "MeanTree") -> List[str]:
        if tree_in_stand.is_alive:
            return list()

        actions = ["plant"]
        return actions


class WaitAndPlant(ManagementActionABC):
    """Management action to plant a MeanTree that is `waiting` after some delay.

    Args:
        - nr_waiting: number of years to wait before planting
    """

    def __init__(self, nr_waiting: int):
        super().__init__()
        self.nr_waiting = nr_waiting

    def do(self, stand: "Stand", tree_in_stand: "MeanTree") -> List[str]:
        if tree_in_stand.is_alive:
            return list()

        actions = ["wait"] * self.nr_waiting + ["plant"]
        return actions

    def __str__(self):
        return f"{str(self.__class__.__name__)} (nr_waiting={self.nr_waiting})"


class ManagementStrategy:
    """Class for a management strategy of aclass:`~..trees.mean_tree.MeanTree`.

    All triggers will be checked at the end of every year, and and if they get active,
    the according management action is being executed.

    Args:
        trigger_action_list: list of (trigger, action) pairs.
    """

    def __init__(
        self, trigger_action_list: List[Tuple[TriggerABC, ManagementActionABC]]
    ):
        self.trigger_action_list = trigger_action_list

    def __str__(self):
        s = [f"{t}: {a}" for t, a in self.trigger_action_list]
        return "\n".join(s)

    def __repr__(self):
        return str(self)

    def add(self, ms_item: Tuple[TriggerABC, ManagementActionABC]):
        """Add a (trigger, action) pair to the trigger action list."""
        self.trigger_action_list.append(ms_item)

    def post_processing_update(
        self, stand: "Stand", tree_in_stand: "MeanTree", Delta_t: Q_[float]
    ) -> Tuple[List[str], str]:
        """Check all trigger and in case execute action.

        Args:
            stand: the current stand
            tree_in_stand: the MeanTree in consideration
            Delta_t: the time step size [yr]

        Returns:
            (list of management actions, log str)
        """
        actions = []

        log: List[str] = []
        for trigger, management_action in self.trigger_action_list:
            if trigger.triggered(stand, tree_in_stand, Delta_t):
                log += [f"Triggered {trigger}"]
                #                if tree_in_stand.current_status in ["alive", "waiting"]:
                if True:  # always try, some actions might not happen
                    actions = management_action.do(stand, tree_in_stand)
                    log += [f"{tree_in_stand.name}, actions: {actions}"]
                    print("\n".join(log))
                    break

        return actions, "\n".join(log)
