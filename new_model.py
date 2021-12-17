from data_loader import load_full_data, preprocess, load_config, load_unusable_space, load_shuffle_slots, load_shuffle_to_range
from action_group import group_actions, get_action_groups_str
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import ray
from ray.rllib import agents
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.tune.logger import pretty_print
import tensorflow as tf
import numpy as np
import copy
import os
from pathlib import Path
import json
import time
from view import generate_summary, generate_slots_summary
from collections import defaultdict, namedtuple

import gym
from gym.spaces import Discrete, Dict, Box
from functools import total_ordering

from datetime import datetime
from functools import cmp_to_key
import pandas as pd
from enum import Enum

"""
Some of the code such as the state, environment and container definition needs to be included/duplicated here due to parallelism from the Ray library.
"""

"""
UTILITY FUNCTIONS
"""
wt_map = {
    "L": 0,
    "M": 1,
    "H": 2,
    "X": 3,
    "U": 4
}

def get_hours_difference(time_1, time_2):
    """
    Get time difference in hours between two timestamps.
    """
    return int((time_1 - time_2).total_seconds() // 3600)


def map_group_to_num(groups):
    """
    Map each container group to a number.
    """
    encoding_dict = {}
    encode_num = 1
    for group in groups:
        encoding_dict[group] = encode_num
        encode_num += 1

    return encoding_dict


def convert_action_to_move(action, max_rows, index=0):
    """
    Convert action to according move which consists of (row_origin, row_dest) subject to 0 or 1-based index.
    """
    row_origin = action // (max_rows - 1)  # pos of container to move
    mod = action % (max_rows - 1)  # modulo of current action
    if mod < row_origin:
        row_dest = mod
    else:
        row_dest = mod + 1
    return (row_origin, row_dest) if index == 0 else (row_origin + 1, row_dest + 1)


def print_legal_moves(action_mask, max_rows):
    """
    @param1: action mask of size (action_space,) where entry 1 at index i denote i as a legal action.
    @param2: max_rows caters to different scenarios when number of rows is 6 or 10 or 6/10 * number of slots in the case of multiple slot shuffling.
    @return: list of legal moves in the form (row_origin, row_dest)
    """
    return [convert_action_to_move(action, max_rows) for action in np.where(action_mask == 1)[0]]


def map_group_to_color(groups):
    """
    Map each container group to a different color for color printing.
    """
    group_map_dict = map_group_to_num(groups)
    colors = ["black", "red", "green", "orange", "blue", "purple", "dark green", "white", "cyan", "brown", "light coral", "dark salmon", "gold", "dark khaki", "plum", "chocolate", "royal blue", "ivory", "azure", "lavender", "old lace", "misty rose",
              "moccasin", "bisque", "light pink", "sky blue", "turquoise", "yellow green", "golden rod", "tan", "olive drab", "lime green", "dark sea green", "sea green", "deep sky blue", "medium purple", "medium orchid", "thistle", "pale violet red", "peru"]
    return {key: colors[val-1] for key, val in group_map_dict.items()}


def convert_move_to_action(row_origin, row_dest, max_rows):
    """
    Convert from move to action for testing.
    """
    move_to_action_map = {convert_action_to_move(
        a, max_rows): a for a in range(max_rows * (max_rows - 1))}
    return move_to_action_map[(row_origin, row_dest)]

"""
Background colors for printing each slot state.
"""
BACKGROUND_COLORS = {"black": ("0;{};40", "#A9A9A9"),  # hex color different
                     "red": ("0;{};41", "#FF0000"),
                     "green": ("0;{};42", "#008000"),
                     "orange": ("0;{};43", "#FFA500"),
                     "blue": ("0;{};44", "#0000FF"),
                     "purple": ("0;{};45", "#800080"),
                     "dark green": ("0;{};46", "#006400"),
                     "white": ("0;{};47", "#F0FFF0"),  # hex color different
                     "cyan": ("0;{};48;2;0;255;255", "#00FFFF"),
                     "brown": ("0;{};48;2;165;42;42", "#A52A2A"),
                     "light coral": ("0;{};48;2;240;128;128", "#F08080"),
                     "dark salmon": ("0;{};48;2;233;150;122", "#E9967A"),
                     "gold": ("0;{};48;2;255;215;0", "#FFD700"),
                     "dark khaki": ("0;{};48;2;189;183;107", "#BDB76B"),
                     "plum": ("0;{};48;2;221;160;221", "#DDA0DD"),
                     "chocolate": ("0;{};48;2;210;105;30", "#D2691E"),
                     "royal blue": ("0;{};48;2;65;105;225", "#4169E1"),
                     "ivory": ("0;{};48;2;255;255;240", "#FFFFF0"),
                     "azure": ("0;{};48;2;240;255;255", "#F0FFFF"),
                     "lavender": ("0;{};48;2;230;230;250", "#E6E6FA"),
                     "old lace": ("0;{};48;2;253;245;230", "#FDF5E6"),
                     "misty rose": ("0;{};48;2;255;228;225", "#FFE4E1"),
                     "moccasin": ("0;{};48;2;255;228;181", "#FFE4B5"),
                     "bisque": ("0;{};48;2;255;228;196", "#FFE4C4"),
                     "light pink": ("0;{};48;2;255;182;193", "#FFB6C1"),
                     "sky blue": ("0;{};48;2;135;206;235", "#87CEEB"),
                     "turquoise": ("0;{};48;2;64;224;208", "#40E0D0"),
                     "yellow green": ("0;{};48;2;154;205;50", "#9ACD32"),
                     "golden rod": ("0;{};48;2;218;165;32", "#DAA520"),
                     "tan": ("0;{};48;2;210;180;140", "#D2B48C"),
                     "olive drab": ("0;{};48;2;107;142;35", "#6B8E23"),
                     "lime green": ("0;{};48;2;50;205;50", "#32CD32"),
                     "dark sea green": ("0;{};48;2;143;188;143", "#8FBC8F"),
                     "sea green": ("0;{};48;2;46;139;87", "#2E8B57"),
                     "deep sky blue": ("0;{};48;2;0;191;255", "#00BFFF"),
                     "medium purple": ("0;{};48;2;147;112;219", "#9370DB"),
                     "medium orchid": ("0;{};48;2;186;85;211", "#BA55D3"),
                     "thistle": ("0;{};48;2;216;191;216", "#BA55D3"),
                     "pale violet red": ("0;{};48;2;219;112;147", "#DB7093"),
                     "peru": ("0;{};48;2;205;133;63", "#CD853F"),
                     }

COLORS = {"black": "30",
          "red": "31",
          "green": "32",
          "orange": "33",
          "blue": "34",
          "purple": "35",
          "olive green": "36",
          "white": "37"}


def color_print(text, color, background_color, end=""):
    """
    Prints text with color.
    """
    color_string = BACKGROUND_COLORS[background_color][0].format(COLORS[color])
    text = f"\x1b[{color_string}m{text}\x1b[0m{end}"
    return text

"""
CONTAINER CLASS
"""
@total_ordering
class Container:
    def __init__(self, df_row, early_vsl_threshold):
        """
        @param1: the corresponding container row in the dataframe. So it is a Pandas Series.
        @param2: the threshold timing to prioritize containers with etb below this threshold.
        @return: process this Series to populate each container's attributes
        """
        self.name = df_row['CNTR_N']
        self.row = df_row['row']
        self.level = df_row['level']
        self.container_name = df_row['CNTR_N']
        self.vessel = df_row['vsl2']
        self.snap_dt = self.convert_datetime(df_row['SNAP_DT'])
        self.etb = self.convert_datetime(df_row['etb2'])
        self.etu = self.convert_datetime(df_row['etu2'])
        self.port_mark = ''
        self.size = 0
        self.category = ''
        self.weight = 0
        self._process_pscw(df_row['pscw'])
        self.is_freeze = False
        self.container_short_name = self.name.split(" ")[0]
        self.is_high_priority = self.get_etb_hours_from_snap_dt() <= early_vsl_threshold

    def _process_pscw(self, pscw):
        """
        Populate portmark, size, cat, weight attributes.
        """
        ptmk, sz, cat, wt = pscw.split("_")
        self.port_mark, self.size, self.category, self.weight = ptmk, int(
            sz), cat, wt_map[wt]

    def get_days_from_snap_dt(self):
        """
        Get days from etb to snap_dt.
        """
        return int((self.etb - self.snap_dt).total_seconds() // 3600)

    def get_etb_hours_from_snap_dt(self):
        """
        Get number of hours from etb to snap_dt.
        """
        time_delta = self.etb - self.snap_dt
        total_seconds = time_delta.total_seconds()
        return int(total_seconds // 3600)

    def get_etu_hours_from_snap_dt(self):
        """
        Get number of hours from etu to snap_dt.
        """
        time_delta = self.etu - self.snap_dt
        total_seconds = time_delta.total_seconds()
        return int(total_seconds // 3600)

    def convert_datetime(self, dt):
        """
        Convert to python datetime.
        """
        if isinstance(dt, pd.Timestamp):
            return dt.to_pydatetime()
        elif isinstance(dt, str):
            return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

    def equal_without_weight(self, other):
        """
        @param: other container
        @return: whether two containers are equal for attributes except weight.
        """
        return self == other

    def equal_mixPK(self, other):
        """
        Check if two containers are equal vessel and port mark.
        """
        return self.vessel == other.vessel and self.port_mark == other.port_mark
    
    def equal_mixCat(self, other):
        """
        Check if two containers are equal vessel and category.
        """
        return self.vessel == other.vessel and self.category == other.category

    def equal_with_weight(self, other):
        """
        @param: other container
        @return: whether two containers are equal for attributes including weight.
        """
        return self == other and self.weight == other.weight

    def __eq__(self, other):
        """
        Overload == comparison.
        """
        return self.vessel == other.vessel and self.port_mark == other.port_mark and self.size == other.size and self.category == other.category

    def __ne__(self, other):
        """
        Overload != comparison.
        """
        return not (self == other)

    def __lt__(self, other):
        return self.etb > other.etb

    def __str__(self):
        return f"{self.vessel}_{self.get_etb_hours_from_snap_dt()}:{self.get_etu_hours_from_snap_dt()}_{self.port_mark}_{self.size}_{self.category}_{self.weight}"

    def __repr__(self):
        """
        Container representation.
        """
        return f"{self.vessel}_{self.get_etb_hours_from_snap_dt()}:{self.get_etu_hours_from_snap_dt()}_{self.port_mark}_{self.size}_{self.category}_{self.weight}"

    def short_name(self):
        return f"{self.vessel}_{self.get_etb_hours_from_snap_dt()}:{self.get_etu_hours_from_snap_dt()}_{self.port_mark}_{self.size}_{self.category}"

    def ui_repr(self):
        """
        UI representation to include freeze information.
        """
        is_freeze_text = "_(F)" if self.is_freeze else ""
        return f"{self.get_etb_hours_from_snap_dt()}:{self.get_etu_hours_from_snap_dt()}_{self.port_mark}_{self.size}_{self.category}_{list(wt_map.keys())[list(wt_map.values()).index(self.weight)]}{is_freeze_text}"

"""
STATE CLASS
"""
class State:
    def __init__(self, max_rows, max_levels, shuffle_config, is_combined=False, num_slots_combined=1, each_slot_num_rows=6, unusable_rows=None):
        """
        This class encapsulates the current state of a slot.
        @param1: number of possible rows
        @param2: number of possible levels
        @param3: shuffling configuration which is the sheet "action" in "config_smartShuffling.xlsx" file
        @param4: whether it is single-slot or multiple-slot shuffling
        @param5: number of combined slots
        @param6: number of rows from each slot in case of multiple-slot shuffling
        @param7: which rows are unusable
        """
        self.max_rows = max_rows
        self.max_levels = max_levels
        # Keep track of each row of containers
        self.rows = [[] for i in range(self.max_rows)]
        # Keep track of the past move from row_origin -> row_dest so container is not moved back row_dest -> row_origin
        self.past_move = None
        # Vessel groups are groups of containers with the same vessel
        self.vessel_groups = defaultdict(int)
        # Container groups are groups of containers with the same vessel, portmark, size, category except for weight
        self.container_groups = defaultdict(int)
        # Keep track of all the weights for each container group
        self.container_groups_weights = defaultdict(list)
        self.vessel_groups_name = []
        # Map each group to their color and number
        self.group_to_num_map = {}
        self.group_to_color_map = {}
        self.longest_container_name = 0
        self.shuffle_config = shuffle_config
        self.unusable_rows = [
            row - 1 for row in unusable_rows] if unusable_rows else None
        # Whether to enable box freeze
        self.box_freeze = False
        # Keep tracks of highest freeze level for each row so all containers below are freezed
        self.highest_freeze_level_per_row = None
        # Sort container and vessel groups and calculate rows required for each group in case we want to compute a desired done cost as terminal point of the episode
        self.container_groups_sorted = None
        self.vessel_groups_sorted_count = None
        self.vessel_groups_sorted_overstow = None
        self.container_groups_rows_required = None
        self.vessel_groups_rows_required = None
        # Actual available rows taking into account unusable rows
        self.actual_available_rows = self.max_rows - \
            len(self.unusable_rows) if self.unusable_rows else self.max_rows
        self.is_combined = is_combined
        self.num_slots_combined = num_slots_combined
        # For multiple-slot shuffling, each slot's row 0 still have max levels == 4 
        self.safety_rows = [
            i * each_slot_num_rows for i in range(num_slots_combined)]

    @property
    def c_group_len(self):
        return len(self.container_groups)

    @property
    def actual_c_group_len(self):
        return sum([1 if group_count != 0 else 0 for container_group, group_count in self.container_groups.items()])

    @property
    def v_group_len(self):
        return len(self.vessel_groups)

    @property
    def actual_v_group_len(self):
        return sum([1 if group_count != 0 else 0 for vessel_group, group_count in self.vessel_groups.items()])

    @property
    def total_cost(self):
        """
        Get total violation cost.
        """
        return sum(self.get_row_costs())

    @property
    def fit_by_container_group(self):
        """
        Whether we can fit by container group.
        """
        return self.container_groups_rows_required <= self.actual_available_rows

    @property
    def fit_by_vessel_group(self):
        """
        Whether we can fit by vessel group.
        """
        return not self.fit_by_container_group and self.vessel_groups_rows_required <= self.actual_available_rows

    @property
    def shuffling_type(self):
        """
        Get shuffle type. 0 is for overstow only, 1 is for overstow + mixPK/mixCat and 2 is for overstow + mixPSCW 
        """
        if self.shuffle_config['overstow'] > 0 and self.shuffle_config['mixPK'] == 0 and self.shuffle_config['mixCat'] == 0 and self.shuffle_config['mixWeight'] == 0:
            return 0
        elif self.shuffle_config['overstow'] > 0 and self.shuffle_config['mixPK'] > 0 and self.shuffle_config['mixCat'] > 0 and self.shuffle_config['mixWeight'] > 0:
            return 2
        else:
            return 1

    def summary(self):
        """
        Generate a summary of the slot.
        """
        final_string = ""
        final_string += f"Number of container groups: {self.c_group_len}\n  "
        final_string += f"Number of vessel groups: {self.v_group_len}\n  "
        final_string += f"Initial cost: {self.total_cost}\n  "
        if self.fit_by_container_group:
            fit_by = 'container groups'
        elif self.fit_by_vessel_group:
            fit_by = 'vessel groups'
        else:
            fit_by = 'mixed groups'
        final_string += f"Fit by: {fit_by}\n  "
        final_string += f"State done cost: {self.compute_done_cost()}\n  "
        final_string += f"Ground rows: {self.get_ground_rows()}\n  "
        final_string += f"Overstow rows: {self.get_overstow_rows()}\n  "
        return final_string

    def fill_rows(self, df):
        """
        Fill up self.rows and other attributes from the dataframe.
        """

        # Container and vessel group.
        ContainerGroup = namedtuple(
            'ContainerGroup', ['vessel', 'etb', 'etu', 'port_mark', 'size', 'category'])

        VesselGroup = namedtuple(
            'VesselGroup', ['vessel', 'etb', 'etu']
        )

        # Whether box freeze is taken into account
        if 'boxFreeze' in df.columns:
            self.box_freeze = True

        # Fill rows with each container
        for _, row in df.iterrows():
            row_idx = row['row'] - 1

            # Create Container object
            container = Container(row, self.shuffle_config['earlyVsl_ETBHr'])

            # Whether this container is to be freezed
            if self.box_freeze:
                container.is_freeze = True if row['boxFreeze'] == 1 else False

            # Get container and vessel namedtuple
            container_group = ContainerGroup(container.vessel, container.get_etb_hours_from_snap_dt(
            ), container.get_etu_hours_from_snap_dt(), container.port_mark, container.size, container.category)
            vessel_group = VesselGroup(container.vessel, container.get_etb_hours_from_snap_dt(
            ), container.get_etu_hours_from_snap_dt())

            # Fill dictionaries
            self.container_groups[container_group] += 1
            self.vessel_groups[vessel_group] += 1
            self.container_groups_weights[container_group].append(
                container.weight)

            # Append container to rows
            self.rows[row_idx].append(container)

        # If box freeze, re-adjust calculations
        if self.box_freeze:
            self.highest_freeze_level_per_row = self.cal_highest_freeze_level_per_row()

            # Calculate remaining container and vessel group count after taking freeze into account
            for row, level in enumerate(self.highest_freeze_level_per_row):
                if level != -1:
                    for level in range(level + 1):
                        container = self.rows[row][level]
                        container_group = ContainerGroup(container.vessel, container.get_etb_hours_from_snap_dt(
                        ), container.get_etu_hours_from_snap_dt(), container.port_mark, container.size, container.category)
                        vessel_group = VesselGroup(container.vessel, container.get_etb_hours_from_snap_dt(
                        ), container.get_etu_hours_from_snap_dt())
                        self.container_groups[container_group] -= 1
                        self.vessel_groups[vessel_group] -= 1

            # Calculate actual available rows
            self.actual_available_rows -= sum([1 if row_freeze_level != -
                                               1 else 0 for row_freeze_level in self.highest_freeze_level_per_row])

        # Container groups sorted by count and num rows required
        self.container_groups_sorted = sorted(
            self.container_groups.items(), key=lambda item: item[1], reverse=True)
        self.container_groups_rows_required = sum(
            [count // self.max_levels if count % self.max_levels == 0 else count //
                self.max_levels + 1 for group, count in self.container_groups_sorted])

        # Vessel groups sorted by count and overstow and num rows required
        self.vessel_groups_sorted_count = sorted(
            self.vessel_groups.items(), key=lambda item: item[1], reverse=True)
        vessel_group_compare = cmp_to_key(self.cmp_vessel_groups)
        self.vessel_groups_sorted_overstow = list(
            (vessel_group, count) for vessel_group, count in self.vessel_groups.items())
        self.vessel_groups_sorted_overstow.sort(
            key=vessel_group_compare, reverse=True)
        self.vessel_groups_rows_required = sum(
            [count // self.max_levels if count % self.max_levels == 0 else count //
                self.max_levels + 1 for group, count in self.vessel_groups_sorted_count])

        # Container group weights and name for mixWeight and color
        self.container_groups_weights = {container: sorted(
            weights) for container, weights in self.container_groups_weights.items()}
        self.container_group_names = [
            f"{container_group.vessel}_{container_group.etb}:{container_group.etu}_{container_group.port_mark}_{container_group.size}_{container_group.category}" for container_group in self.container_groups.keys()]
        self.group_to_num_map = map_group_to_num(
            self.container_group_names)
        # self.group_to_color_map = map_group_to_color(
        #     self.container_group_names)
        self.longest_container_name = max(
            [len(container.__str__()) for row in self.rows for container in row])

    def cmp_vessel_groups(self, v_group_1, v_group_2):
        """
        Function to compare vessel groups by etb-etu for sorting.
        """
        if v_group_1[0].etb > v_group_2[0].etu - 2:  # v_group_1 should be below
            return 1
        elif v_group_1[0].etb == v_group_2[0].etb and v_group_1[0].etu == v_group_2[0].etu:
            return 0
        else:
            return -1

    def cal_highest_freeze_level_per_row(self):
        """
        Find the highest freeze level for each row. All containers from the level to below will be freezed.
        """
        highest_freeze_level_per_row = [0] * self.max_rows
        for i, row in enumerate(self.rows):
            highest_freeze_level = -1
            for j, container in enumerate(row):
                if container.is_freeze:
                    highest_freeze_level = max(highest_freeze_level, j)
            highest_freeze_level_per_row[i] = highest_freeze_level

        return highest_freeze_level_per_row

    def move_container(self, row_origin, row_dest):
        """ 
        Move container to reflect the current state in `self.rows`. Past move is updated. 
        """
        from_idx = row_origin - 1
        to_idx = row_dest - 1

        if not self.is_valid_move(from_idx, to_idx):
            return

        from_row = self.rows[from_idx]
        to_row = self.rows[to_idx]
        container_to_move = from_row.pop()
        to_row.append(container_to_move)
        container_to_move.level = self._get_row_height(to_idx)

        self.past_move = (row_origin, row_dest)

    def is_valid_move(self, row_origin, row_dest):
        """
        Check if a move is valid based on row_origin and row_dest.
        """
        # If state is not combined from many slots
        if not self.is_combined:
            if self.max_rows == 6:
                # additional constraint for row 1 with max_levels == 4.
                if self._is_row_full(row_dest) or self._is_row_empty(row_origin) or self.past_move == (row_dest + 1, row_origin + 1) or (row_dest == 0 and len(self.rows[row_dest]) == 4):
                    return False
                # If the container is freezed, it cannot be moved
                if self.box_freeze and self.highest_freeze_level_per_row and len(self.rows[row_origin]) == self.highest_freeze_level_per_row[row_origin] + 1:
                    return False
            elif self.max_rows == 10:
                if self._is_row_full(row_dest) or self._is_row_empty(row_origin) or self.past_move == (row_dest + 1, row_origin + 1):
                    return False
                if self.box_freeze and self.highest_freeze_level_per_row and len(self.rows[row_origin]) == self.highest_freeze_level_per_row[row_origin] + 1:
                    return False

            # Container cannot be moved to an unusable row
            if self.unusable_rows:
                if row_dest in self.unusable_rows:
                    return False

            return True
        # If state is a combination of slots
        else:
            # If any violation and also row dest is among the safety rows (row 1 of each slot) which is already full
            if self._is_row_full(row_dest) or self._is_row_empty(row_origin) or self.past_move == (row_dest + 1, row_origin + 1) or (row_dest in self.safety_rows and len(self.rows[row_dest]) == 4):
                return False

            return True

    def _is_row_empty(self, row_num):
        """
        Check if a row is empty.
        """
        return self._get_row_height(row_num) == 0

    def _is_row_full(self, row_num):
        """
        Check if a row is full.
        """
        if not self.is_combined:
            if self.max_rows == 6:
                return self._get_row_height(row_num) == self.max_levels if row_num != 0 else self._get_row_height(row_num) == 4
            elif self.max_rows == 10:
                return self._get_row_height(row_num) == self.max_levels
        else:
            return self._get_row_height(row_num) == self.max_levels if row_num not in self.safety_rows else self._get_row_height(row_num) == 4

    def _get_row_height(self, row_num):
        """
        Return the number of containers in this row.
        """
        return len(self.rows[row_num])

    def _compute_row_cost(self, row):
        """
        Compute each row's cost depending on shuffling configuration. 
        """
        if self.shuffling_type == 0:
            return row.sum(axis=0, where=[False, True]).sum()
        elif self.shuffling_type == 2:
            return row.sum(axis=0, where=[False, True, True, True, True]).sum()
        else:
            return row.sum(axis=0, where=[False, True, True]).sum()

    def get_row_costs(self):
        """
        Get all rows' cost.
        """
        return [self._compute_row_cost(row) for row in self.to_numpy_state()]

    def to_numpy_state(self):
        """
        The main representation for training in the neural network. Depending on the shuffle config, dimension of the state is (rows, levels, 2) for overstow only, (rows, levels, 3) for overstow and mixPK/mixCat and (rows, levels, 5) for overstow, mixPK, mixCat, mixWeight
        """
        if self.shuffling_type == 0:  # overstow only
            np_state = np.zeros(
                (self.max_rows, self.max_levels, 2), dtype=np.int32)
            # Which row
            for i, row in enumerate(self.rows):  
                 # Which container
                for j, container in enumerate(row): 
                    # Represent that container with its number
                    np_state[i][j][0] = self.group_to_num_map[container.short_name()]

                    # Skip this container if it's freezed
                    if self.box_freeze and self.highest_freeze_level_per_row and j < self.highest_freeze_level_per_row[i]:
                        continue

            for i, row in enumerate(self.rows):
                for j in range(len(row) - 1):
                    if self.box_freeze and self.highest_freeze_level_per_row and j < self.highest_freeze_level_per_row[i]:
                        continue
                    container = row[j]
                    top_container = row[j + 1]
                    # Check for overstow if different vessels
                    if container.vessel != top_container.vessel: 
                        if get_hours_difference(top_container.etu, container.etb) >= 2:
                            # If high priority, use 'earlyVsl_weightage'
                            if container.is_high_priority:
                                for h in range(j + 1, len(row)):
                                    np_state[i][h][1] = self.shuffle_config['overstow'] * self.shuffle_config['earlyVsl_weightage']
                                break
                            else:
                                for h in range(j + 1, len(row)):
                                    np_state[i][h][1] = self.shuffle_config['overstow'] * self.shuffle_config['lateVsl_weightage']
                                break

            return np_state

        elif self.shuffling_type == 1:
            np_state = np.zeros(
                (self.max_rows, self.max_levels, 3), dtype=np.int32)

            for i, row in enumerate(self.rows): 
                for j, container in enumerate(row): 
                    np_state[i][j][0] = self.group_to_num_map[container.short_name()]

                    if self.box_freeze and self.highest_freeze_level_per_row and j < self.highest_freeze_level_per_row[i]:
                        continue

                    for k in range(j + 1, len(row)): 
                        top_container = row[k]

                        # Whether mixPK ro mixCat
                        if self.shuffle_config['mixPK'] > 0:
                            if container.vessel == top_container.vessel and container.port_mark != top_container.port_mark:
                                for h in range(k, len(row)):
                                    np_state[i][h][2] = self.shuffle_config['mixPK']
                        else:
                            if container.vessel == top_container.vessel and container.category != top_container.category:
                                for h in range(k, len(row)):
                                    np_state[i][h][2] = self.shuffle_config['mixCat']

            for i, row in enumerate(self.rows):
                for j in range(len(row) - 1):
                    if self.box_freeze and self.highest_freeze_level_per_row and j < self.highest_freeze_level_per_row[i]:
                        continue
                    container = row[j]
                    top_container = row[j + 1]
                    if container.vessel != top_container.vessel:
                        if get_hours_difference(top_container.etu, container.etb) >= 2:
                            if container.is_high_priority:
                                for h in range(j + 1, len(row)):
                                    np_state[i][h][1] = self.shuffle_config['overstow'] * self.shuffle_config['earlyVsl_weightage']
                                break
                            else:
                                for h in range(j + 1, len(row)):
                                    np_state[i][h][1] = self.shuffle_config['overstow'] * self.shuffle_config['lateVsl_weightage']
                                break

            return np_state

        else:
            np_state = np.zeros(
                (self.max_rows, self.max_levels, 5), dtype=np.int32)
            # Only consider mixWeight costs when the row is all of same container group. This is to simplify the cost 
            same_type_rows = [
                row for row in range(len(self.rows)) if self.check_row_same_type(row)]

            for i, row in enumerate(self.rows): 
                for j, container in enumerate(row):  
                    np_state[i][j][0] = self.group_to_num_map[container.short_name()]

                    if self.box_freeze and self.highest_freeze_level_per_row and j < self.highest_freeze_level_per_row[i]:
                        continue

                    for k in range(j + 1, len(row)):  
                        top_container = row[k]

                        if self.shuffle_config['mixPK'] > 0:
                            if container.vessel == top_container.vessel and container.port_mark != top_container.port_mark:
                                for h in range(k, len(row)):
                                    np_state[i][h][2] = self.shuffle_config['mixPK']

                        if self.shuffle_config['mixCat'] > 0:
                            if container.vessel == top_container.vessel and container.category != top_container.category:
                                for h in range(k, len(row)):
                                    np_state[i][h][3] = self.shuffle_config['mixCat']

                        if 'mixWeight' in self.shuffle_config:
                            if container == top_container and container.weight > top_container.weight and i in same_type_rows:
                                for h in range(k, len(row)):
                                    np_state[i][h][4] = self.shuffle_config['mixWeight']

            for i, row in enumerate(self.rows):
                for j in range(len(row) - 1):
                    if self.box_freeze and self.highest_freeze_level_per_row and j < self.highest_freeze_level_per_row[i]:
                        continue
                    container = row[j]
                    top_container = row[j + 1]
                    if container.vessel != top_container.vessel:
                        if get_hours_difference(top_container.etu, container.etb) >= 2:
                            if container.is_high_priority:
                                for h in range(j + 1, len(row)):
                                    np_state[i][h][1] = self.shuffle_config['overstow'] * self.shuffle_config['earlyVsl_weightage']
                                break
                            else:
                                for h in range(j + 1, len(row)):
                                    np_state[i][h][1] = self.shuffle_config['overstow'] * self.shuffle_config['lateVsl_weightage']
                                break

        return np_state
    
    def get_rows_overstow_priority_container(self):
        """
        Get which rows are currently having priority containers which are overstowed.
        """
        rows = np.zeros(self.max_rows)
        for i, row in enumerate(self.rows):
            for j in range(len(row) - 1):
                if self.box_freeze and self.highest_freeze_level_per_row and j < self.highest_freeze_level_per_row[i]:
                    continue
                container = row[j]
                top_container = row[j + 1]
                if container.vessel != top_container.vessel:  # overstow and mix
                    if get_hours_difference(top_container.etu, container.etb) >= 2:
                        if container.is_high_priority:
                            rows[i] = 1
                            break
        return rows

    def to_env_state(self):
        """
        Vertically stack numpy state to flatten the input for the neural network. This is the dimension of the observation space as well.
        """
        return np.vstack(self.to_numpy_state()).flatten('F')

    def compute_done_cost(self):
        """
        *** CURRENTLY NOT USED. Try to estimate done cost for early termination if possible. With container priority in place, this estimation might need to be updated.
        """
        # overstow only
        if self.shuffling_type == 0:
            # be optimistic, assume can reduce cost to 0
            return 0

        else:
            # can fit by container groups, assume can reduce cost to 0
            if self.fit_by_container_group:  
                return 0

            # fit by vessel group, require minimal mixing
            elif self.fit_by_vessel_group:  
                done_cost = 0

                for vessel_group, _ in self.vessel_groups_sorted_count:
                    # get all matching container groups exclude first group with most containers (sorted). Assume that first group will take up one row.
                    matching_container_groups = [(container_group, count) for container_group,
                                                 count in self.container_groups_sorted if container_group.vessel == vessel_group.vessel][1:]  
                    done_cost += sum([count for container_group,
                                      count in matching_container_groups])

                # Slightly raise done_cost so as not to be too optimistic
                return int(done_cost * 1.5)
            
            # fit by mix group, require more mixing
            else:  
                total_rows = 0
                done_cost = 0
                stop_index = 0
                # Iterate among vessel groups with latest time
                for index, (container_group, group_count) in enumerate(self.container_groups_sorted):
                    if group_count == 0:
                        continue

                    # group_count can fit in 1 row
                    if group_count <= self.max_levels:  
                        total_rows += 1
                        stop_index = index + 1
                        if total_rows == self.actual_available_rows:  # max rows reached, remaining groups needs to be mixed
                            remaining_groups = self.container_groups_sorted[stop_index:]
                            done_cost = sum([group_count for container_group,
                                            group_count in remaining_groups])
                            return int(done_cost * 1.5)
                    # group_count requires multiple rows   
                    else:
                        num_full_rows = group_count // self.max_levels if group_count % self.max_levels == 0 else group_count // self.max_levels + 1
                        remaining_rows = self.actual_available_rows - total_rows
                        # if enough space
                        if remaining_rows >= num_full_rows:  
                            total_rows += num_full_rows
                            stop_index = index + 1
                            # max rows reached, remaining groups needs to be mixed
                            if total_rows == self.actual_available_rows:  
                                remaining_groups = self.container_groups_sorted[stop_index:]
                                done_cost = sum([group_count for container_group,
                                                group_count in remaining_groups])
                                return int(done_cost * 1.5)
                        # max rows reached or exceeded, both remainder of current group and remaining groups need to be mixed
                        else:  
                            rows_left = num_full_rows - remaining_rows
                            group_count_left = group_count - rows_left * self.max_levels
                            stop_index = index + 1
                            remaining_groups = self.vessel_groups_sorted_overstow[stop_index:]
                            done_cost = ((group_count_left %
                                          self.actual_available_rows) + sum([group_count for container_group,
                                                                             group_count in remaining_groups]))
                            return int(done_cost * 1.5)

    def check_ground_row(self, row_idx):
        """
        Check whether this is a ground row. This is to help with mixWeight.
        """
        row = self.rows[row_idx]
        container = row[0]

        ContainerGroup = namedtuple(
            'ContainerGroup', ['vessel', 'etb', 'etu', 'port_mark', 'size', 'category'])

        container_group = ContainerGroup(container.vessel, container.get_etb_hours_from_snap_dt(
        ), container.get_etu_hours_from_snap_dt(), container.port_mark, container.size, container.category)

        # Get all the weights of this container group
        container_group_weights = self.container_groups_weights[container_group]
        # Get the min weight of this group
        group_min_weight = container_group_weights[0]
        # Get number of containers with the min weight
        group_min_weight_count = len(
            [weight for weight in container_group_weights if weight == group_min_weight])

        # Get the number of containers in this row with the same min weight as the group
        min_row_weight_count = len(
            [container.weight for container in row if container.weight == group_min_weight])

        # If weights in the row does not follow weight order of the group from bottom up, it's not considered a ground row
        for i, container in enumerate(self.rows[row_idx]):
            if container.weight != container_group_weights[i]:
                return False

        if min_row_weight_count < group_min_weight_count:
            if min_row_weight_count == self.max_levels:
                return True
            else:
                return False
        else:
            return True

    def check_row_same_type(self, row_idx):
        """
        Check if a row is of the same type for ground rows consideration.
        """
        if self.shuffling_type == 2:
            # If mixPSCW, row same type only if all containers are of the same container group
            return all(container.equal_without_weight(self.rows[row_idx][0]) for container in self.rows[row_idx])
        elif self.shuffling_type == 1:
            if self.shuffle_config['mixPK'] > 0:
                # If overstow + mixPK
                return all(container.equal_mixPK(self.rows[row_idx][0]) for container in self.rows[row_idx])
            else:
                # If overstow + mixCat
                return all(container.equal_mixCat(self.rows[row_idx][0]) for container in self.rows[row_idx])
        else:
            # If only overstow
            return all(container.vessel == self.rows[row_idx][0].vessel for container in self.rows[row_idx])

    def get_ground_rows(self): 
        """
        Get all the ground rows depending on shuffling type.
        """
        if self.shuffling_type == 2:
            same_type_rows = [row_idx for row_idx, row in enumerate(self.rows) if len(
                self.rows[row_idx]) >= 2 and self.check_row_same_type(row_idx)]
            return np.array([row_idx for row_idx in same_type_rows if self.check_ground_row(row_idx)])
        else:
            return np.array([row_idx for row_idx, row in enumerate(self.rows) if len(self.rows[row_idx]) >= 3 and self.check_row_same_type(row_idx)])

    def get_empty_levels_per_row(self, numpy_state):
        """
        Returns the number of empty levels for each row.
        """
        empty_levels = [np.where(numpy_state[row].T[0] == 0)[0]
                        for row in range(self.max_rows)]
        empty_levels_per_row = [len(row) if i != 0 else len(
            row) - 1 for i, row in enumerate(empty_levels)]
        return empty_levels_per_row

    def get_rows_overstow_values(self):
        """ 
        Returns the overstow values at each row
        """
        if self.shuffling_type == 0:
            return np.array([row.sum(axis=0, where=[False, True]).sum() for row in self.to_numpy_state()])
        elif self.shuffling_type == 2:
            return np.array([row.sum(axis=0, where=[False, True, False, False, False]).sum() for row in self.to_numpy_state()])
        else:
            return np.array([row.sum(axis=0, where=[False, True, False]).sum() for row in self.to_numpy_state()])

    def get_overstow_rows(self):
        """ 
        Returns rows which have overstow.
        """
        return np.where(self.get_rows_overstow_values() > 0)[0]

    def get_non_overstow_rows(self):
        """ 
        Returns rows which do not have overstow.
        """
        return np.setdiff1d(np.arange(self.max_rows), self.get_overstow_rows())

    def __repr__(self):
        """
        Represent this state.
        """
        output = ""
        for i, row in enumerate(self.rows):
            output += f"Row {i + 1}:\n"
            for j, container in enumerate(row):
                output += f"  Level {container.level}: {container}\n"
            output += "\n"

        return output

    def cprint(self, mode='plain'):
        """
        Print this state either in plain text or color mode.
        """
        final_string = ''
        for level in range(self.max_levels + 1):
            if level == self.max_levels:
                for i in range(self.max_rows):
                    if self.unusable_rows and i in self.unusable_rows:
                        string = f'Row {i+1} (Unusable)'.ljust(
                            self.longest_container_name, ' ')
                    else:
                        string = f'Row {i+1}'.ljust(
                            self.longest_container_name, ' ')
                    final_string += string + '  '
                break
            for row in range(self.max_rows):
                container_row = self.rows[row]
                row_len = len(container_row)
                corresponding_level = self.max_levels - level
                if corresponding_level > row_len:
                    final_string += ' '*self.longest_container_name + '  '
                else:
                    container = container_row[corresponding_level - 1]
                    container_name = container.__str__()
                    container_str = container_name if len(
                        container_name) == self.longest_container_name else container_name.ljust(self.longest_container_name, ' ')
                    if mode == 'color':
                        final_string += color_print(container_str, color,
                                                    self.group_to_color_map[f'{container.short_name()}'], end='  ')
                        color = 'white' if self.group_to_color_map[
                        f'{container.short_name()}'] == 'black' else 'black'
                    elif mode == 'plain':
                        final_string += container_str + '  '
            final_string += '\n'
        final_string += '\n'
        return final_string

"""
ACTION MASKING
"""
class ParametricActionsModel(TFModelV2):
    """
    Mask away invalid actions through the neural network.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):

        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        self.action_embed_model = FullyConnectedNetwork(
            Box(-1, 1, shape=(obs_space.shape[0] - action_space.n,)),
            action_space, num_outputs,
            model_config, name + "_action_embed"
        )

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict['obs']['action_mask']

        action_embed, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]
        })

        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

"""
ShuffleType Enumeration
"""
class ShuffleType(Enum):
    OVERSTOW_ONLY = 1
    MIX_NO_WEIGHT = 2
    MIX_W_WEIGHT = 3

"""
ENVIRONMENT
"""
class RayContainerShuffleEnv(gym.Env):

    def __init__(self, environment_config):
        """
        This class represents the Environment through which the agent will interact with and receive feedback in the form of observation, reward, termination.
        @param1: configuration for this environment such as the intial state.
        """
        # Default state for reset
        self.default_state = environment_config["oop_state"]
        # Change state for interaction
        self.change_state = copy.deepcopy(self.default_state)
        # Attributes from the config and state
        self.invalid_move_penalty = environment_config['invalid_move_penalty']
        self.episode_length = environment_config['shuffle_moves_limit']
        self.max_rows = self.default_state.max_rows
        self.max_levels = self.default_state.max_levels
        self.num_actions = self.max_rows * (self.max_rows - 1)
        # Action space and observation space to specify their type, data type and dimension
        self.action_space = Discrete(self.num_actions)
        self.shuffle_config = self.default_state.shuffle_config
        if self.shuffle_config['overstow'] > 0 and self.shuffle_config['mixPK'] == 0 and self.shuffle_config['mixCat'] == 0 and self.shuffle_config['mixWeight'] == 0:
            self.shuffle_type = ShuffleType.OVERSTOW_ONLY
            self.observation_space = Dict({
                "action_mask": Box(0, 1, shape=(self.num_actions,), dtype=np.float32),
                "state": Box(0, 100, shape=(self.max_rows*self.max_levels*2,), dtype=np.int32)
            })
        elif self.shuffle_config['overstow'] > 0 and self.shuffle_config['mixPK'] > 0 and self.shuffle_config['mixCat'] > 0 and self.shuffle_config['mixWeight'] > 0:
            self.shuffle_type = ShuffleType.MIX_W_WEIGHT
            self.observation_space = Dict({
                "action_mask": Box(0, 1, shape=(self.num_actions,), dtype=np.float32),
                "state": Box(0, 100, shape=(self.max_rows*self.max_levels*5,), dtype=np.int32)
            })
        else:
            self.shuffle_type = ShuffleType.MIX_NO_WEIGHT
            self.observation_space = Dict({
                "action_mask": Box(0, 1, shape=(self.num_actions,), dtype=np.float32),
                "state": Box(0, 100, shape=(self.max_rows*self.max_levels*3,), dtype=np.int32)
            })
        self.longest_vessel_name = max([len(container.short_name(
        )) for row in self.default_state.rows for container in row])
        self.move_dict = {a: self._convert_action_to_move(
            a) for a in range(self.num_actions)}
        self.fit_by_container_group = self.default_state.fit_by_container_group
        self.fit_by_vessel_group = self.default_state.fit_by_vessel_group
        # Current cost of the slot for reward assignment
        self.current_cost = self.default_state.total_cost
        # Done cost which can be specified or estimated
        self.done_cost = self.default_state.compute_done_cost(
        ) if environment_config['done_cost'] == 'auto' else environment_config['done_cost']
        # Current ground rows to support mix weight
        self.ground_rows = set(self.default_state.get_ground_rows())
        # Global variables to keep track of the shortest action sequence so far that reached the goal
        # self.actions = []
        # self.shortest_actions = []
        # self.shortest_len = environment_config['shuffle_moves_limit']

    def step(self, action, logging=False):
        """
        The agent takes an action and receive feedback from the environment.
        """
        # Keep track of each action sequence to determine the shortest one
        # self.actions.append(action)

        # Convert numerical action to move
        row_origin, row_dest = self._convert_action_to_move(action)

        # Get the total cost before this movement
        before_row_costs_sum = sum(self.change_state.get_row_costs())

        # Get the overstow situation before this movement
        before_row_overstow = self.change_state.get_rows_overstow_values()

        before_row_overstow_sum = sum(before_row_overstow)

        # The state takes this action and gets updated
        self.change_state.move_container(row_origin + 1, row_dest + 1)

        # Get the total cost after this movement
        after_row_costs_sum = sum(self.change_state.get_row_costs())

        # Get the overstow situation after this movement
        after_row_overstow = self.change_state.get_rows_overstow_values()

        after_row_overstow_sum = sum(after_row_overstow)

        # Get which rows with priority containers being overstowed
        priority_overstow_rows = self.change_state.get_rows_overstow_priority_container()

        # Take number of ground rows into account if shuffling type includes mixWeight
        if self.shuffle_type == ShuffleType.MIX_W_WEIGHT:
            after_ground_rows = set(self.change_state.get_ground_rows())

        # Episode termination
        done = bool(after_row_costs_sum <= self.done_cost)

        if done:
            # If episode terminates within move limits, update global action sequence if it's shorter than previously
            # if len(self.actions) <= self.shortest_len:
            #     self.shortest_actions = copy.deepcopy(self.actions)
            #     self.shortest_len = len(self.shortest_actions)
            #     # Keep track of global shortest actions
            #     global_actions = ray.get_actor('global_actions')
            #     current_shotest_actions = ray.get(global_actions.get.remote())
            #     if not current_shotest_actions or len(self.shortest_actions) < len(current_shotest_actions):
            #         global_actions.update.remote(self.shortest_actions)
            reward = self.current_cost - after_row_costs_sum
        else:
            # If a movement leads to overstow of a row with priority container, this is highly undesirable hence negatively rewarded
            if priority_overstow_rows[row_dest] == 1:
                reward = -1.0

            # Positive reward if the cost after movement reduces to below current cost
            elif after_row_costs_sum < self.current_cost:
                reward = self.current_cost - after_row_costs_sum
                self.current_cost = after_row_costs_sum

            # Negative reward if the movement worsens existing overstow situation
            elif after_row_overstow_sum > before_row_overstow_sum:
                reward = -0.2

            # Negative reward for every extra move taken
            else:
                reward = -0.1

            # Positive reward for every new ground rows created when shuffling type includes mix weight
            if self.shuffle_type == ShuffleType.MIX_W_WEIGHT and after_ground_rows.difference(self.ground_rows):
                reward += 1.0
                self.ground_rows = self.ground_rows.union(after_ground_rows)

        # The state is a combination of numpy state and action mask
        self.state = {
            "action_mask": self.get_legal_moves(),
            "state": self.change_state.to_env_state()
        }

        # Feedback emitted by the environment
        return self.state, reward, done, {}

    def render(self, mode='plain'):
        """
        Render the current state of the environment
        """
        print(self.change_state.cprint(mode=mode))

    def reset(self):
        """
        Reset this environment for a new episode.
        """
        self.current_cost = self.default_state.total_cost
        # self.actions.clear()
        self.change_state = copy.deepcopy(self.default_state)
        self.ground_rows = set(self.default_state.get_ground_rows())
        self.state = {
            "action_mask": self.get_legal_moves(),
            "state": self.change_state.to_env_state()
        }
        return self.state

    def get_legal_moves(self):
        """
        Generate the action mask with all valid actions being 1 and invalid being 0.
        """
        all_valid_actions = self._get_valid_actions()
        all_actions = np.zeros(self.num_actions, dtype=np.float32)
        all_actions[all_valid_actions] = 1
        return all_actions

    def _get_valid_actions(self):
        """
        Get all valid actions after eliminating all invalid actions.
        """
        ground_rows = self.change_state.get_ground_rows()
        all_ground_row_invalid_actions = []
        for ground_row in ground_rows:
            invalid_actions = [action for action, (row_origin, row_dest) in self.move_dict.items(
            ) if row_origin == ground_row]
            all_ground_row_invalid_actions += invalid_actions

        additional_invalid_actions = [action for action in range(
            self.num_actions) if not self.change_state.is_valid_move(*self._convert_action_to_move(action))]

        all_invalid_actions = np.union1d(
            np.array(all_ground_row_invalid_actions), np.array(additional_invalid_actions))
        all_valid_actions = np.setdiff1d(
            np.arange(self.num_actions), all_invalid_actions, assume_unique=True)

        return all_valid_actions

    def _convert_action_to_move(self, action):
        """
        Convert a numerical action to a movement.
        """
        row_origin = action // (self.max_rows - 1)  # pos of container to move
        mod = action % (self.max_rows - 1)  # modulo of current action
        if mod < row_origin:
            row_dest = mod
        else:
            row_dest = mod + 1
        return row_origin, row_dest

"""
Global Ray actor keeping track of shortest action sequence for each slot. Used when slot can be solved.
"""
@ray.remote
class ShortestActions:
    def __init__(self):
        self.shortest_actions = []

    def update(self, actions):
        self.shortest_actions = copy.deepcopy(actions)

    def get(self):
        return self.shortest_actions

if __name__ == '__main__':
    # Load information from main_config.json
    with open('./config/main_config.json', 'r') as f:
        content = json.load(f)

    # Environment config
    environment_config = content['environment']

    # Terminal shuffle config, for different shuffling mode (overstow only/ overstow + mix)
    all_terminals_shuffle_config = pd.read_excel('./config/config_smartShuffling.xlsx', sheet_name='action')
    all_terminals_shuffle_config.set_index('ct', inplace=True)
    
    # Load all data in
    full_df = load_full_data(content['data_path'])

    # Load unusable space in
    unusable_space = load_unusable_space()

    # Load slots to shuffle in
    shuffle_slots = load_shuffle_slots()

    # Load shuffle_to range
    shuffle_to_range_df = load_shuffle_to_range()

    # All unique blocks
    all_blks = full_df['blk'].unique()

    # Which blocks to shuffle based on main config excel file or main_config.json for testing individual slots
    blks_to_shuffle = all_blks if content['blocks'] == 'all' else content['blocks']
    
    # Create shuffle_solutions folder
    solution_path = Path('shuffle_solutions')
    solution_path.mkdir(exist_ok=True)
    root_path = Path(os.getcwd()).joinpath('shuffle_solutions')

    # Timer for complete shuffling of all blocks
    start = time.time()
    print("START ALL BLOCK SHUFFLING")

    for blk in blks_to_shuffle:
        # Timer for each block, create block solution folder
        start_block_time = time.time()
        print(f"START BLOCK {blk} SHUFFLING")

        # Keep track of all the range of slots to shuffle for this block
        blk_shuffle_ranges = []

        if blk in shuffle_slots.index:
            print("SOLVING BLOCK: ", blk)

            # Config file may specify non-contiguous slots for each block
            blk_shuffle_slots = shuffle_slots.loc[blk]

            # Check if it's non-contiguous
            if isinstance(blk_shuffle_slots, pd.DataFrame):
                for _, row in blk_shuffle_slots.iterrows():
                    from_slot = row['slotFm']
                    to_slot = row['slotTo']
                    blk_shuffle_ranges.append((from_slot, to_slot))
            else:  # Only a single range
                from_slot = blk_shuffle_slots['slotFm']
                to_slot = blk_shuffle_slots['slotTo']
                blk_shuffle_ranges.append((from_slot, to_slot))
        else:
            continue

        # Start block timing
        start_blk_time = time.time()

        # Move to this block folder
        path = root_path.joinpath(blk)
        path.mkdir(exist_ok=True)
        os.chdir(path)

        # Get block dataframe
        block_df = full_df[full_df['blk'] == blk]

        # Get block terminal to map to shuffle config
        block_terminal = block_df['ct'].unique()[0]

        # Get unusable space based on block terminal and block, then retrieve all affected slots
        block_unusable_space = unusable_space[(
        unusable_space['ct'] == block_terminal) & (unusable_space['blk'] == blk)]
        
        affected_slots_in_block = block_unusable_space['slotTo'].unique()

        # Get the shuffle config for this block
        block_shuffle_config = all_terminals_shuffle_config.loc[block_terminal]

        # Determine the agent config depending on block shuffle config
        if block_shuffle_config['overstow'] > 0 and block_shuffle_config['mixPK'] == 0 and block_shuffle_config['mixCat'] == 0 and block_shuffle_config['mixWeight'] == 0:
            agent_config = content['ray_agent_overstow_only']
        elif block_shuffle_config['overstow'] > 0 and block_shuffle_config['mixPK'] > 0 and block_shuffle_config['mixCat'] > 0 and block_shuffle_config['mixWeight'] > 0:
            agent_config = content['ray_agent_mix_w_weight']
        else:
            agent_config = content['ray_agent_mix_no_weight']

        # Get block yard crane type
        block_yc_type = block_df['ycType'].unique()[0]

        # Get shuffle_to config for this yard crane type
        shuffle_to_range = shuffle_to_range_df.loc[block_yc_type]

        # Get range of slots to left and right of this slot in case of multiple slot shuffling
        to_left, to_right = shuffle_to_range['toLeft'], shuffle_to_range['toRight']

        # Generate block summary in the current block directory
        generate_summary(blk, block_df, block_shuffle_config, block_unusable_space,
                            box_freeze=content['box_freeze'])

        # Get all distinct slots in this block
        distinct_slots = block_df['slotTo'].unique()

        # Which slots to shuffle based on main config excel file or main_config.json for testing individual slots
        slots_to_shuffle = distinct_slots if content['slots'] == 'all' else content['slots']

        for slot_no in slots_to_shuffle:
            # Flag to check if this slot is within any shuffle range from excel config
            is_within_any_shuffle_range = False

            for from_slot, to_slot in blk_shuffle_ranges:
                if slot_no >= from_slot and slot_no <= to_slot:
                    is_within_any_shuffle_range = True

            # Skip if this slot is not within any shuffle range, else shuffle
            if not is_within_any_shuffle_range:
                continue

            # If this yard crane requires multiple-slot shuffling
            if to_left or to_right:

                # Get number of slot rows
                max_rows = 6
                max_levels = 5

                to_left_slots = [
                    slot for slot in distinct_slots if slot >= slot_no - to_left and slot < slot_no]
                print("TO LEFT: ", to_left_slots)

                to_right_slots = [
                    slot for slot in distinct_slots if slot <= slot_no + to_right and slot > slot_no]
                print("TO RIGHT: ", to_right_slots)

                # Initialize to combine all slots
                all_slots = to_left_slots + [slot_no] + to_right_slots
                combined_slots_df = pd.DataFrame()
                total_max_rows = 0

                # Get combined slot dataframe
                for slot in all_slots:
                    slot_df = preprocess(
                        block_df[block_df['slotTo'] == slot], box_freeze=False)

                    # Increment slot rows
                    slot_df['row'] = slot_df['row'].apply(
                        lambda x: x + total_max_rows)

                    # Get a new combined slots dataframe
                    combined_slots_df = pd.concat([combined_slots_df, slot_df])
                    total_max_rows += max_rows

                    # Mark slot as already shuffled

                print("Total max rows: ", total_max_rows)

                # Get the combined state here
                state = State(total_max_rows, max_levels,
                              block_shuffle_config, is_combined=True, num_slots_combined=len(all_slots), each_slot_num_rows=max_rows, unusable_rows=None)
                state.fill_rows(combined_slots_df)

                # Check whether there's any violating row in this slot
                row_len_exceeding = [
                    len(row) > max_levels for row in state.rows]

                if any(row_len_exceeding):
                    print(f"SLOT {slot} cannot be solved!")
                    continue

                generate_slots_summary(
                    slot_no, state, total_max_rows, max_levels)

                # Register custom model with action masking
                ModelCatalog.register_custom_model('cs_mask_multiple', ParametricActionsModel)

                # Environment configuration
                env_config = {
                    'oop_state': state,
                    'invalid_move_penalty': environment_config['invalid_move_penalty'],
                    'shuffle_moves_limit': environment_config['shuffle_moves_limit'],
                    'done_cost': environment_config['done_cost']
                }

                # Overall config
                agent_train_config = {
                    'env': RayContainerShuffleEnv,
                    'env_config': env_config,
                    'num_gpus': agent_config['num_gpus'],
                    'model': {
                        'custom_model': 'cs_mask_multiple'
                    },
                    'num_workers': agent_config['num_workers'],
                    "rollout_fragment_length": agent_config['rollout_fragment_length'],
                    "horizon": agent_config['horizon'],
                    "train_batch_size": agent_config['train_batch_size'],
                }

                # Get PPO's default config, update with train config
                ppo_config = agents.ppo.DEFAULT_CONFIG.copy()
                ppo_config.update(agent_train_config)
                ppo_config['lr'] = 1e-3
            
                if agent_config.get('num_envs_per_worker'):
                    agent_train_config["num_envs_per_worker"] = agent_config["num_envs_per_worker"]

                # Start slot shuffling
                start_slot_time = time.time()
                print(f"START SLOT {slot_no} SHUFFLING")
                
                # Initialize Ray
                ray.shutdown()
                ray.init(dashboard_port=8266)

                # Get trainer
                trainer = agents.ppo.PPOTrainer(config=ppo_config)

                # Get stopping criteria
                stop = agent_config['stop']

                # Training until stopping criteria
                for _ in range(stop['training_iteration']):
                    result = trainer.train()
                    print(pretty_print(result))
                    if result['timesteps_total'] >= stop['timesteps_total'] or result['episode_reward_mean'] >= result['episode_reward_max'] - 1:
                        break
                
                # Finish with this slot shuffling
                end_slot_time = time.time()

                # Placeholder for actions
                actions = []

                # Write block solution in corresponding directory
                with open(f'{blk}_visual_solution.txt', 'a') as f:
                    # Initialize environment
                    env = RayContainerShuffleEnv(env_config)
                    obs = env.reset()
                    done = False
                    print(env.render(mode='color'))
                    f.write(f'Slot {slot_no} solution: ')

                    # Compute action at each step and take that action
                    for _ in range(environment_config['shuffle_moves_limit']):
                        if done:
                            print(env.render(mode='color'))
                            print(env.change_state.get_row_costs())
                            break
                        action = trainer.compute_action(obs)
                        actions.append(action)
                        f.write(f'{action} ')
                        obs, reward, done, info = env.step(action)
                    f.write('\n')

                # Output this slot shuffling steps in standard format
                with open(f'{slot_no}_output_solution.txt', 'w') as f:
                    f.write(f'Slot {slot_no} solved in {end_slot_time - start_slot_time} seconds: \n')
                    f.write('CNTR,vsl2,voy2,fm_CT,fm_blk,fm_sloTFm,fm_slotTo,fm_row,fm_level,to_CT,to_blk,to_slotFm,to_slotTo,to_row,to_level\n')
                    slot_df = combined_slots_df.set_index('CNTR_N')
                    for action in actions:
                        from_row, to_row = convert_action_to_move(action, index=1, max_rows=state.max_rows)
                        moved_container = state.rows[from_row-1][-1]
                        moved_container_series = slot_df.loc[moved_container.name]
                        from_level = len(state.rows[from_row-1])
                        state.move_container(from_row, to_row)
                        to_level = len(state.rows[to_row-1])
                        output = f'{moved_container.name},{moved_container_series.vsl2},{moved_container_series.voy2},{moved_container_series.ct},{moved_container_series.blk},{moved_container_series.slotFm},{moved_container_series.slotTo},{from_row},{from_level},{moved_container_series.ct},{moved_container_series.blk},{moved_container_series.slotFm},{moved_container_series.slotTo},{to_row},{to_level}\n'
                        f.write(output)

            else:
                
                # Preprocess slot dataframe, remove unneccary columns
                slot = preprocess(block_df[block_df['slotTo'] == slot_no], box_freeze=content['box_freeze'])

                # Check slot type (6/10 rows)
                max_rows = 10 if np.any(slot['row'].unique() > 6) else 6
                max_levels = 5

                # Get unusable rows in this slot if required
                unusable_rows = None
                if slot_no in affected_slots_in_block:
                    unusable_rows = block_unusable_space[block_unusable_space['slotTo'] == slot_no]['row'].unique()
                
                # Get current slot state
                state = State(max_rows, max_levels, block_shuffle_config, unusable_rows=unusable_rows)
                state.fill_rows(slot)

                # Check whether there's any violating row in this slot
                row_len_exceeding = [len(row) > max_levels for row in state.rows]

                if any(row_len_exceeding):
                    print(f"SLOT {slot_no} cannot be solved!")
                    continue

                # Get numpy state
                np_state = state.to_numpy_state()

                # Check for violations in each 6/10 slot, also check if enough empty levels
                if max_rows == 6 and (len(state.rows[0]) == max_levels or sum(state.get_empty_levels_per_row(np_state)) < content['empty_levels_threshold']):
                    continue
                elif max_rows == 10 and sum(state.get_empty_levels_per_row(np_state)) < content['empty_levels_threshold']:
                    continue
                    
                # Register action masking
                ModelCatalog.register_custom_model('cs_mask_single', ParametricActionsModel)

                # Define env_config, following RLLib
                env_config = {
                    'oop_state': state,
                    'invalid_move_penalty': environment_config['invalid_move_penalty'],
                    'shuffle_moves_limit': environment_config['shuffle_moves_limit'],
                    'done_cost': environment_config['done_cost']
                }

                # Get train config using main_config.json
                agent_train_config = {
                    'env': RayContainerShuffleEnv,
                    'env_config': env_config,
                    'num_gpus': agent_config['num_gpus'],
                    'model': {
                        'custom_model': 'cs_mask_single'
                    },
                    'num_workers': agent_config['num_workers'],
                    "rollout_fragment_length": agent_config['rollout_fragment_length'],
                    "horizon": agent_config['horizon'],
                    "train_batch_size": agent_config['train_batch_size'],
                }

                if agent_config.get('num_envs_per_worker'):
                    agent_train_config["num_envs_per_worker"] = agent_config["num_envs_per_worker"]

                # Start slot shuffling
                start_slot_time = time.time()
                print(f"START SLOT {slot_no} SHUFFLING")
                
                # Initialize Ray
                ray.shutdown()
                ray.init(dashboard_port=8266)

                # Get PPO's default config, update with train config
                ppo_config = agents.ppo.DEFAULT_CONFIG.copy()
                ppo_config.update(agent_train_config)
                ppo_config['lr'] = 1e-3

                # Get trainer
                trainer = agents.ppo.PPOTrainer(config=ppo_config)

                # Get stopping criteria
                stop = agent_config['stop']

                # Training until stopping criteria
                for _ in range(stop['training_iteration']):
                    result = trainer.train()
                    print(pretty_print(result))
                    if result['timesteps_total'] >= stop['timesteps_total'] or result['episode_reward_mean'] >= result['episode_reward_max'] - 0.1:
                        break
                
                # Finish with this slot shuffling
                end_slot_time = time.time()

                # Placeholder for actions
                actions = []

                # Write block solution in corresponding directory
                with open(f'{blk}_visual_solution.txt', 'a') as f:
                    # Initialize environment
                    env = RayContainerShuffleEnv(env_config)
                    obs = env.reset()
                    done = False
                    print(env.render(mode='plain'))
                    f.write(f'Slot {slot_no} solution: ')

                    # Compute action at each step and take that action
                    for _ in range(environment_config['shuffle_moves_limit']):
                        if done:
                            print(env.render(mode='plain'))
                            print(env.change_state.get_row_costs())
                            break
                        action = trainer.compute_action(obs)
                        actions.append(action)
                        f.write(f'{action} ')
                        obs, reward, done, info = env.step(action)
                    f.write('\n')

                # Output this slot shuffling steps in standard format
                with open(f'{slot_no}_output_solution.txt', 'w') as f:
                    f.write(f'Slot {slot_no} solved in {end_slot_time - start_slot_time} seconds: \n')
                    f.write('CNTR,vsl2,voy2,fm_CT,fm_blk,fm_sloTFm,fm_slotTo,fm_row,fm_level,to_CT,to_blk,to_slotFm,to_slotTo,to_row,to_level\n')
                    slot_df = slot.set_index('CNTR_N')
                    for action in actions:
                        from_row, to_row = convert_action_to_move(action, index=1, max_rows=state.max_rows)
                        moved_container = state.rows[from_row-1][-1]
                        moved_container_series = slot_df.loc[moved_container.name]
                        from_level = len(state.rows[from_row-1])
                        state.move_container(from_row, to_row)
                        to_level = len(state.rows[to_row-1])
                        output = f'{moved_container.name},{moved_container_series.vsl2},{moved_container_series.voy2},{moved_container_series.ct},{moved_container_series.blk},{moved_container_series.slotFm},{moved_container_series.slotTo},{from_row},{from_level},{moved_container_series.ct},{moved_container_series.blk},{moved_container_series.slotFm},{moved_container_series.slotTo},{to_row},{to_level}\n'
                        f.write(output)

            # Shutdown ray after each slot
            ray.shutdown()
        
        # Finish shuffling this block
        end_blk_time = time.time()

        with open(f'{blk}_visual_solution.txt', 'a') as f:
            f.write(f'Finished block {blk} in {end_blk_time - start_blk_time} seconds.')