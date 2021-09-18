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


wt_map = {
    "L": 0,
    "M": 1,
    "H": 2,
    "X": 3,
    "U": 4
}


def get_hours_difference(time_1, time_2):
    return int((time_1 - time_2).total_seconds() // 3600)


def map_group_to_num(groups):
    encoding_dict = {}
    encode_num = 1
    for group in groups:
        encoding_dict[group] = encode_num
        encode_num += 1

    return encoding_dict


def convert_action_to_log(action, max_rows):
    row_origin = action // (max_rows - 1)  # pos of container to move
    mod = action % (max_rows - 1)  # modulo of current action
    if mod < row_origin:
        row_dest = mod
    else:
        row_dest = mod + 1
    return f"Moving container from row {row_origin + 1} to row {row_dest + 1}"


def convert_action_to_move(action, max_rows, index=0):
    row_origin = action // (max_rows - 1)  # pos of container to move
    mod = action % (max_rows - 1)  # modulo of current action
    if mod < row_origin:
        row_dest = mod
    else:
        row_dest = mod + 1
    return (row_origin, row_dest) if index == 0 else (row_origin + 1, row_dest + 1)


def print_legal_moves(action_mask, max_rows):
    return [convert_action_to_move(action, max_rows) for action in np.where(action_mask == 1)[0]]


def map_group_to_color(groups):
    group_map_dict = map_group_to_num(groups)
    colors = ["black", "red", "green", "orange", "blue", "purple", "dark green", "white", "cyan", "brown", "light coral", "dark salmon", "gold", "dark khaki", "plum", "chocolate", "royal blue", "ivory", "azure", "lavender", "old lace", "misty rose",
              "moccasin", "bisque", "light pink", "sky blue", "turquoise", "yellow green", "golden rod", "tan", "olive drab", "lime green", "dark sea green", "sea green", "deep sky blue", "medium purple", "medium orchid", "thistle", "pale violet red", "peru"]
    return {key: colors[val-1] for key, val in group_map_dict.items()}


def convert_move_to_action(row_origin, row_dest, max_rows):
    move_to_action_map = {convert_action_to_move(
        a, max_rows): a for a in range(max_rows * (max_rows - 1))}
    return move_to_action_map[(row_origin, row_dest)]


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


@total_ordering
class Container:
    def __init__(self, df_row, n_days_from_snap_dt=3):
        self.n_days_from_snap_dt = n_days_from_snap_dt
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

    def _process_pscw(self, pscw):
        ptmk, sz, cat, wt = pscw.split("_")
        self.port_mark, self.size, self.category, self.weight = ptmk, int(
            sz), cat, wt_map[wt]

    def get_days_from_snap_dt(self):
        return int((self.etb - self.snap_dt).total_seconds() // 3600)

    def is_near_snap_dt(self):
        return self.get_days_from_snap_dt() < self.n_days_from_snap_dt

    def get_etb_hours_from_snap_dt(self):
        time_delta = self.etb - self.snap_dt
        total_seconds = time_delta.total_seconds()
        return int(total_seconds // 3600)

    def get_etu_hours_from_snap_dt(self):
        time_delta = self.etu - self.snap_dt
        total_seconds = time_delta.total_seconds()
        return int(total_seconds // 3600)

    def convert_datetime(self, dt):
        if isinstance(dt, pd.Timestamp):
            return dt.to_pydatetime()
        elif isinstance(dt, str):
            return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

    def equal_without_weight(self, other):
        return self == other

    def equal_with_weight(self, other):
        return self == other and self.weight == other.weight

    def __eq__(self, other):
        return self.vessel == other.vessel and self.port_mark == other.port_mark and self.size == other.size and self.category == other.category

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.etb > other.etb

    def __str__(self):
        return f"{self.vessel}_{self.get_etb_hours_from_snap_dt()}:{self.get_etu_hours_from_snap_dt()}_{self.port_mark}_{self.size}_{self.category}_{self.weight}"

    def __repr__(self):
        return f"{self.vessel}_{self.get_etb_hours_from_snap_dt()}:{self.get_etu_hours_from_snap_dt()}_{self.port_mark}_{self.size}_{self.category}_{self.weight}"

    def short_name(self):
        return f"{self.vessel}_{self.get_etb_hours_from_snap_dt()}:{self.get_etu_hours_from_snap_dt()}_{self.port_mark}_{self.size}_{self.category}"

    def ui_repr(self):
        is_freeze_text = "_(F)" if self.is_freeze else ""
        return f"{self.container_short_name}_{self.get_etb_hours_from_snap_dt()}:{self.get_etu_hours_from_snap_dt()}_{self.port_mark}_{self.size}_{self.category}_{list(wt_map.keys())[list(wt_map.values()).index(self.weight)]}{is_freeze_text}"


class State:
    def __init__(self, max_rows, max_levels, shuffle_config, is_combined=False, num_slots_combined=1, each_slot_num_rows=6, unusable_rows=None):
        self.max_rows = max_rows
        self.max_levels = max_levels
        self.rows = [[] for i in range(self.max_rows)]
        self.past_move = None
        self.vessel_groups = defaultdict(int)
        self.container_groups = defaultdict(int)
        self.container_groups_weights = defaultdict(list)
        self.vessel_groups_name = []
        self.group_to_num_map = {}
        self.group_to_color_map = {}
        self.longest_container_name = 0
        self.shuffle_config = shuffle_config
        self.unusable_rows = [
            row - 1 for row in unusable_rows] if unusable_rows else None
        self.box_freeze = False
        self.highest_freeze_level_per_row = None
        self.container_groups_sorted = None
        self.vessel_groups_sorted_count = None
        self.vessel_groups_sorted_overstow = None
        self.container_groups_rows_required = None
        self.vessel_groups_rows_required = None
        self.actual_available_rows = self.max_rows - \
            len(self.unusable_rows) if self.unusable_rows else self.max_rows
        self.is_combined = is_combined
        self.num_slots_combined = num_slots_combined
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
        return sum(self.get_row_costs())

    @property
    def fit_by_container_group(self):
        return self.container_groups_rows_required <= self.actual_available_rows

    @property
    def fit_by_vessel_group(self):
        return not self.fit_by_container_group and self.vessel_groups_rows_required <= self.actual_available_rows

    def summary(self):
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
        final_string += f"Is solvable: {self.is_solvable()}\n  "
        return final_string

    def fill_rows(self, df):

        ContainerGroup = namedtuple(
            'ContainerGroup', ['vessel', 'etb', 'etu', 'port_mark', 'size', 'category'])

        VesselGroup = namedtuple(
            'VesselGroup', ['vessel', 'etb', 'etu']
        )

        if 'boxFreeze' in df.columns:
            self.box_freeze = True

        # Fill rows with each container
        for _, row in df.iterrows():
            row_idx = row['row'] - 1
            container = Container(row)

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

            # Append container to row
            self.rows[row_idx].append(container)

        # If box freeze, re-adjust calculations
        if self.box_freeze:
            self.highest_freeze_level_per_row = self.cal_highest_freeze_level_per_row()

            # Calculate remaining container and vessel group count
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
        self.group_to_color_map = map_group_to_color(
            self.container_group_names)
        self.longest_container_name = max(
            [len(container.__str__()) for row in self.rows for container in row])

    def cmp_vessel_groups(self, v_group_1, v_group_2):
        if v_group_1[0].etb > v_group_2[0].etu - 2:  # v_group_1 should be below
            return 1
        elif v_group_1[0].etb == v_group_2[0].etb and v_group_1[0].etu == v_group_2[0].etu:
            return 0
        else:
            return -1

    def cal_highest_freeze_level_per_row(self):
        highest_freeze_level_per_row = [0] * self.max_rows
        for i, row in enumerate(self.rows):
            highest_freeze_level = -1
            for j, container in enumerate(row):
                if container.is_freeze:
                    highest_freeze_level = max(highest_freeze_level, j)
            highest_freeze_level_per_row[i] = highest_freeze_level

        return highest_freeze_level_per_row

    def move_container(self, row_origin, row_dest):
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

    def remove_row(self, row):
        row_index = row - 1
        self.rows[row_index].pop()

    def is_valid_move(self, row_origin, row_dest):
        # If state is not combined from many slots
        if not self.is_combined:
            if self.max_rows == 6:
                # additional constraint row 1
                if self._is_row_full(row_dest) or self._is_row_empty(row_origin) or self.past_move == (row_dest + 1, row_origin + 1) or (row_dest == 0 and len(self.rows[row_dest]) == 4):
                    return False
                if self.box_freeze and self.highest_freeze_level_per_row and len(self.rows[row_origin]) == self.highest_freeze_level_per_row[row_origin] + 1:
                    return False
            elif self.max_rows == 10:
                if self._is_row_full(row_dest) or self._is_row_empty(row_origin) or self.past_move == (row_dest + 1, row_origin + 1):
                    return False
                if self.box_freeze and self.highest_freeze_level_per_row and len(self.rows[row_origin]) == self.highest_freeze_level_per_row[row_origin] + 1:
                    return False

            if self.unusable_rows:
                if row_dest in self.unusable_rows:
                    return False

            return True
        # If state is a combination of slots
        else:
            # If any violation and also row dest is among the safety rows which is already full
            if self._is_row_full(row_dest) or self._is_row_empty(row_origin) or self.past_move == (row_dest + 1, row_origin + 1) or (row_dest in self.safety_rows and len(self.rows[row_dest]) == 4):
                return False

            return True

    def _is_row_empty(self, row_num):
        return self._get_row_height(row_num) == 0

    def _is_row_full(self, row_num):
        if not self.is_combined:
            if self.max_rows == 6:
                return self._get_row_height(row_num) == self.max_levels if row_num != 0 else self._get_row_height(row_num) == 4
            elif self.max_rows == 10:
                return self._get_row_height(row_num) == self.max_levels
        else:
            return self._get_row_height(row_num) == self.max_levels if row_num not in self.safety_rows else self._get_row_height(row_num) == 4

    def _get_row_height(self, row_num):
        return len(self.rows[row_num])

    def _compute_row_cost(self, row):
        if len(self.shuffle_config) > 1 and 'mixWeight' not in self.shuffle_config:
            return row.sum(axis=0, where=[False, True, True]).sum()
        elif len(self.shuffle_config) > 1 and 'mixWeight' in self.shuffle_config:
            return row.sum(axis=0, where=[False, True, True, True]).sum()
        else:
            return row.sum(axis=0, where=[False, True]).sum()

    def get_row_costs(self):
        return [self._compute_row_cost(row) for row in self.to_numpy_state()]

    def to_numpy_state(self):
        if len(self.shuffle_config) == 1 and "overstow" in self.shuffle_config:  # overstow only
            np_state = np.zeros(
                (self.max_rows, self.max_levels, 2), dtype=np.int32)
            for i, row in enumerate(self.rows):  # which row
                # print(f"Row {i}\n")
                for j, container in enumerate(row):  # which container
                    # np_state[i][j][0] = self.group_to_num_map[container.short_name()]
                    np_state[i][j][0] = 1

                    if self.box_freeze and self.highest_freeze_level_per_row and j < self.highest_freeze_level_per_row[i]:
                        continue

                    if self.box_freeze and self.highest_freeze_level_per_row and j == self.highest_freeze_level_per_row[i]:
                        for k in range(j + 1, len(row)):  # which top container
                            top_container = row[k]
                            np_state[i][k][0] = self.group_to_num_map[top_container.short_name(
                            )]
                            np_state[i][k][1] = 2
                        break  # skip this row
                    else:
                        # print("Below: ", container)
                        for k in range(j + 1, len(row)):  # which top container
                            top_container = row[k]
                            # print("Top: ", top_container)
                            if container.vessel != top_container.vessel:  # only if different vessel
                                # overstow
                                if get_hours_difference(top_container.etu, container.etb) >= 2:
                                    for h in range(k, len(row)):
                                        np_state[i][h][1] = 1
                                    break
            return np_state

        elif len(self.shuffle_config) > 1 and "mixWeight" not in self.shuffle_config:
            np_state = np.zeros(
                (self.max_rows, self.max_levels, 3), dtype=np.int32)

            for i, row in enumerate(self.rows):  # which row
                # print(f"Row {i}\n")
                for j, container in enumerate(row):  # which container
                    np_state[i][j][0] = self.group_to_num_map[container.short_name()]

                    if self.box_freeze and self.highest_freeze_level_per_row and j < self.highest_freeze_level_per_row[i]:
                        continue

                    if j == self.highest_freeze_level_per_row[i]:
                        for k in range(j + 1, len(row)):  # which top container
                            top_container = row[k]
                            np_state[i][k][0] = self.group_to_num_map[top_container.short_name(
                            )]
                            np_state[i][k][1] = 3
                        break  # skip this row
                    else:
                        # print("Below: ", container)
                        for k in range(j + 1, len(row)):  # which top container
                            top_container = row[k]
                            # print("Top: ", top_container)

                            if container.vessel != top_container.vessel:  # overstow and mix
                                if get_hours_difference(top_container.etu, container.etb) >= 2:
                                    for h in range(k, len(row)):
                                        np_state[i][h][1] = 1

                                for h in range(k, len(row)):
                                    np_state[i][h][2] = 1

                            if 'mixPK' in self.shuffle_config:
                                if container.vessel == top_container.vessel and container.port_mark != top_container.port_mark:
                                    for h in range(k, len(row)):
                                        np_state[i][h][2] = 1

                            if 'mixCat' in self.shuffle_config:
                                if container.vessel == top_container.vessel and container.category != top_container.category:
                                    for h in range(k, len(row)):
                                        np_state[i][h][2] = 1

                            if 'mixSz' in self.shuffle_config:
                                if container.vessel == top_container.vessel and container.size != top_container.size:
                                    for h in range(k, len(row)):
                                        np_state[i][h][2] = 1
            return np_state

        else:  # Everything included
            np_state = np.zeros(
                (self.max_rows, self.max_levels, 4), dtype=np.int32)
            same_type_rows = [
                row for row in range(len(self.rows)) if self.check_row_same_type(row)]

            for i, row in enumerate(self.rows):  # which row
                # print(f"Row {i}\n")
                for j, container in enumerate(row):  # which container
                    np_state[i][j][0] = self.group_to_num_map[container.short_name()]

                    if self.box_freeze and self.highest_freeze_level_per_row and j < self.highest_freeze_level_per_row[i]:
                        continue

                    # print("Below: ", container)
                    for k in range(j + 1, len(row)):  # which top container
                        top_container = row[k]
                        # print("Top: ", top_container)

                        if container.vessel != top_container.vessel:  # overstow and mix
                            if get_hours_difference(top_container.etu, container.etb) >= 2:
                                for h in range(k, len(row)):
                                    np_state[i][h][1] = 1

                            for h in range(k, len(row)):
                                np_state[i][h][2] = 1

                        if 'mixPK' in self.shuffle_config:
                            if container.vessel == top_container.vessel and container.port_mark != top_container.port_mark:
                                for h in range(k, len(row)):
                                    np_state[i][h][2] = 1

                        if 'mixCat' in self.shuffle_config:
                            if container.vessel == top_container.vessel and container.category != top_container.category:
                                for h in range(k, len(row)):
                                    np_state[i][h][2] = 1

                        if 'mixSz' in self.shuffle_config:
                            if container.vessel == top_container.vessel and container.size != top_container.size:
                                for h in range(k, len(row)):
                                    np_state[i][h][2] = 1

                        # if 'mixWeight' in self.shuffle_config and i in same_type_rows:
                        if 'mixWeight' in self.shuffle_config:
                            if container == top_container and container.weight > top_container.weight:
                                for h in range(k, len(row)):
                                    np_state[i][h][3] = 1

            return np_state

    def to_env_state(self):
        return np.vstack(self.to_numpy_state()).flatten('F')

    def compute_done_cost(self):
        # overstow only
        if len(self.shuffle_config) == 1 and "overstow" in self.shuffle_config:

            return 0

        # port mark, size, category

        else:
            if self.fit_by_container_group:  # can fit by container groups
                return 0

            elif self.fit_by_vessel_group:  # fit by vessel group, require minimal mixing
                done_cost = 0

                for vessel_group, _ in self.vessel_groups_sorted_count:
                    matching_container_groups = [(container_group, count) for container_group,
                                                 count in self.container_groups_sorted if container_group.vessel == vessel_group.vessel][1:]  # get all matching container groups exclude first group with most containers (sorted)
                    done_cost += sum([count for container_group,
                                      count in matching_container_groups])

                return int(done_cost * 1.5)

            else:  # fit by mix group, require more mixing
                total_rows = 0
                done_cost = 0
                stop_index = 0
                # Iterate among vessel groups with latest time
                for index, (container_group, group_count) in enumerate(self.container_groups_sorted):
                    if group_count == 0:
                        continue

                    if group_count <= self.max_levels:  # group_count can fit in 1 row
                        total_rows += 1
                        stop_index = index + 1
                        if total_rows == self.actual_available_rows:  # max rows reached, remaining groups needs to be mixed
                            remaining_groups = self.container_groups_sorted[stop_index:]
                            done_cost = sum([group_count for container_group,
                                            group_count in remaining_groups])
                            return int(done_cost * 1.5)
                    else:
                        num_full_rows = group_count // self.max_levels if group_count % self.max_levels == 0 else group_count // self.max_levels + 1
                        remaining_rows = self.actual_available_rows - total_rows
                        if remaining_rows >= num_full_rows:  # if enough space
                            total_rows += num_full_rows
                            stop_index = index + 1
                            if total_rows == self.actual_available_rows:  # max rows reached, remaining groups needs to be mixed
                                remaining_groups = self.container_groups_sorted[stop_index:]
                                done_cost = sum([group_count for container_group,
                                                group_count in remaining_groups])
                                return int(done_cost * 1.5)
                        else:  # max rows reached or exceeded, both remainder of current group and remaining groups need to be mixed
                            rows_left = num_full_rows - remaining_rows
                            group_count_left = group_count - rows_left * self.max_levels
                            stop_index = index + 1
                            remaining_groups = self.vessel_groups_sorted_overstow[stop_index:]
                            done_cost = ((group_count_left %
                                          self.actual_available_rows) + sum([group_count for container_group,
                                                                             group_count in remaining_groups]))
                            return int(done_cost * 1.5)

    def is_solvable(self):
        np_state = self.to_numpy_state()
        overstow_costs = sum(self.get_rows_overstow_values())
        avail_levels = sum(self.get_empty_levels_per_row(np_state))
        return overstow_costs < avail_levels

    def check_ground_row(self, row_idx):
        row = self.rows[row_idx]
        container = row[0]

        ContainerGroup = namedtuple(
            'ContainerGroup', ['vessel', 'etb', 'etu', 'port_mark', 'size', 'category'])

        container_group = ContainerGroup(container.vessel, container.get_etb_hours_from_snap_dt(
        ), container.get_etu_hours_from_snap_dt(), container.port_mark, container.size, container.category)

        container_group_weights = self.container_groups_weights[container_group]
        group_min_weight = container_group_weights[0]
        group_min_weight_count = len(
            [weight for weight in container_group_weights if weight == group_min_weight])

        min_row_weight_count = len(
            [container.weight for container in row if container.weight == group_min_weight])

        # print(container_group_weights)
        # print(self.rows[row_idx])
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
        if 'mixWeight' in self.shuffle_config:
            # include strict same weight
            return all(container.equal_without_weight(self.rows[row_idx][0]) for container in self.rows[row_idx])
        return all(container.vessel == self.rows[row_idx][0].vessel for container in self.rows[row_idx])

    def get_non_empty_rows(self):
        """ Returns rows which are not empty """
        return np.array([i for i, row in enumerate(self.rows) if len(row) > 0])

    def get_ground_rows(self):  # TODO: Double check ground rows
        """ Returns rows with 2 or more containers of the same type """
        if 'mixWeight' in self.shuffle_config:
            same_type_rows = [row_idx for row_idx, row in enumerate(self.rows) if len(
                self.rows[row_idx]) >= 2 and self.check_row_same_type(row_idx)]
            return np.array([row_idx for row_idx in same_type_rows if self.check_ground_row(row_idx)])
        else:
            return np.array([row_idx for row_idx, row in enumerate(self.rows) if len(self.rows[row_idx]) >= 3 and self.check_row_same_type(row_idx)])

    def get_empty_levels_per_row(self, numpy_state):
        """
        Returns both empty configuration and number of empty levels for each row
        """
        empty_levels = [np.where(numpy_state[row].T[0] == 0)[0]
                        for row in range(self.max_rows)]
        empty_levels_per_row = [len(row) if i != 0 else len(
            row) - 1 for i, row in enumerate(empty_levels)]
        return empty_levels_per_row

    def get_good_and_avail_rows(self):
        """ Returns rows which are neither overstow nor mix and not full """
        non_overstow_non_mix_rows = np.where(
            np.array(self.get_row_costs()) == 0)[0]
        return np.array([row for row in non_overstow_non_mix_rows if not self._is_row_full(row)])

    def get_overstow_or_mix_rows(self):
        """ Returns rows which are either overstow or mix or both """
        return np.where(np.array(self.get_row_costs()) > 0)[0]

    def get_rows_overstow_values(self):
        """ Returns the overstow values at each row """
        if len(self.shuffle_config) > 1 and 'mixWeight' not in self.shuffle_config:
            return np.array([row.sum(axis=0, where=[False, True, False]).sum() for row in self.to_numpy_state()])
        elif len(self.shuffle_config) > 1 and 'mixWeight' in self.shuffle_config:
            return np.array([row.sum(axis=0, where=[False, True, False, False]).sum() for row in self.to_numpy_state()])
        else:
            return np.array([row.sum(axis=0, where=[False, True]).sum() for row in self.to_numpy_state()])

    def get_overstow_rows(self):
        """ Returns rows which have overstow """
        return np.where(self.get_rows_overstow_values() > 0)[0]

    def get_non_overstow_rows(self):
        """ Returns rows which do not have overstow """
        return np.setdiff1d(np.arange(self.max_rows), self.get_overstow_rows())

    def __repr__(self):
        output = ""
        for i, row in enumerate(self.rows):
            output += f"Row {i + 1}:\n"
            for j, container in enumerate(row):
                output += f"  Level {container.level}: {container}\n"
            output += "\n"

        return output

    def cprint(self, mode='plain'):
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
                    color = 'white' if self.group_to_color_map[
                        f'{container.short_name()}'] == 'black' else 'black'

                    if mode == 'color':
                        final_string += color_print(container_str, color,
                                                    self.group_to_color_map[f'{container.short_name()}'], end='  ')
                    elif mode == 'plain':
                        final_string += container_str + '  '
            final_string += '\n'
        final_string += '\n'
        return final_string


class ShuffleType(Enum):
    OVERSTOW_ONLY = 1
    MIX_NO_WEIGHT = 2
    MIX_W_WEIGHT = 3


class RayContainerShuffleEnv(gym.Env):

    def __init__(self, environment_config):
        self.default_state = environment_config["oop_state"]
        self.change_state = copy.deepcopy(self.default_state)
        self.invalid_move_penalty = environment_config['invalid_move_penalty']
        self.episode_length = environment_config['shuffle_moves_limit']
        self.max_rows = self.default_state.max_rows
        self.max_levels = self.default_state.max_levels
        self.num_actions = self.max_rows * (self.max_rows - 1)
        self.action_space = Discrete(self.num_actions)
        self.shuffle_config = self.default_state.shuffle_config
        if len(self.shuffle_config) == 1 and "overstow" in self.shuffle_config:
            self.shuffle_type = ShuffleType.OVERSTOW_ONLY
            self.observation_space = Dict({
                "action_mask": Box(0, 1, shape=(self.num_actions,), dtype=np.float32),
                "state": Box(0, 100, shape=(self.max_rows*self.max_levels*2,), dtype=np.int32)
            })
        elif len(self.shuffle_config) > 1 and "mixWeight" not in self.shuffle_config:
            self.shuffle_type = ShuffleType.MIX_NO_WEIGHT
            self.observation_space = Dict({
                "action_mask": Box(0, 1, shape=(self.num_actions,), dtype=np.float32),
                "state": Box(0, 100, shape=(self.max_rows*self.max_levels*3,), dtype=np.int32)
            })
        else:
            self.shuffle_type = ShuffleType.MIX_W_WEIGHT
            self.observation_space = Dict({
                "action_mask": Box(0, 1, shape=(self.num_actions,), dtype=np.float32),
                "state": Box(0, 100, shape=(self.max_rows*self.max_levels*4,), dtype=np.int32)
            })
        self.longest_vessel_name = max([len(container.short_name(
        )) for row in self.default_state.rows for container in row])
        self.move_dict = {a: self._convert_action_to_move(
            a) for a in range(self.num_actions)}
        self.fit_by_container_group = self.default_state.fit_by_container_group
        self.fit_by_vessel_group = self.default_state.fit_by_vessel_group
        self.current_cost = self.default_state.total_cost
        self.done_cost = self.default_state.compute_done_cost(
        ) if environment_config['done_cost'] == 'auto' else environment_config['done_cost']
        # self.ground_rows = set(self.default_state.get_ground_rows())
        self.actions = []
        self.shortest_actions = []
        self.shortest_len = environment_config['shuffle_moves_limit']

    def step(self, action, logging=False):
        self.actions.append(action)
        row_origin, row_dest = self._convert_action_to_move(action)

        # before_row_costs = self.change_state.get_row_costs()

        # before_row_costs_sum = sum(before_row_costs)

        self.change_state.move_container(row_origin + 1, row_dest + 1)

        after_row_costs = self.change_state.get_row_costs()

        after_row_costs_sum = sum(after_row_costs)

        if self.shuffle_type == ShuffleType.MIX_W_WEIGHT:
            after_ground_rows = set(self.change_state.get_ground_rows())

        done = bool(after_row_costs_sum <= self.done_cost)

        if done:
            if len(self.actions) <= self.shortest_len:
                self.shortest_actions = copy.deepcopy(self.actions)
                self.shortest_len = len(self.shortest_actions)
                # Keep track of global shortest actions
                global_actions = ray.get_actor('global_actions')
                current_shotest_actions = ray.get(global_actions.get.remote())
                if not current_shotest_actions or len(self.shortest_actions) < len(current_shotest_actions):
                    global_actions.update.remote(self.shortest_actions)
            reward = 100.0
        else:
            if after_row_costs_sum < self.current_cost:
                reward = 1.0
                self.current_cost = after_row_costs_sum
            else:
                reward = -1.0

            if self.shuffle_type == ShuffleType.MIX_W_WEIGHT and after_ground_rows.difference(self.ground_rows):
                reward += 10.0
                self.ground_rows = self.ground_rows.union(after_ground_rows)

        self.state = {
            "action_mask": self.get_legal_moves(),
            "state": self.change_state.to_env_state()
        }

        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(self.change_state.cprint(mode='color'))

    def reset(self):
        self.current_cost = self.default_state.total_cost
        self.actions.clear()
        self.change_state = copy.deepcopy(self.default_state)
        self.ground_rows = set(self.default_state.get_ground_rows())
        self.state = {
            "action_mask": self.get_legal_moves(),
            "state": self.change_state.to_env_state()
        }
        return self.state

    def get_legal_moves(self):
        all_valid_actions = self._get_valid_actions()
        all_actions = np.zeros(self.num_actions, dtype=np.float32)
        all_actions[all_valid_actions] = 1
        return all_actions

    def _get_valid_actions(self):
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
        row_origin = action // (self.max_rows - 1)  # pos of container to move
        mod = action % (self.max_rows - 1)  # modulo of current action
        if mod < row_origin:
            row_dest = mod
        else:
            row_dest = mod + 1
        return row_origin, row_dest


class ParametricActionsModel(TFModelV2):
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


@ray.remote
class ShortestActions:
    def __init__(self):
        self.shortest_actions = []

    def update(self, actions):
        self.shortest_actions = copy.deepcopy(actions)

    def get(self):
        return self.shortest_actions


if __name__ == "__main__":
    with open('./config/main_config.json', 'r') as f:
        content = json.load(f)

    # Environment config
    environment_config = content['environment']

    # Terminal shuffle config, for different shuffling mode (overstow only/ overstow + mix)
    all_terminals_shuffle_config = load_config()

    # Load all data in
    full_df = load_full_data(data_path=content['data_path'])

    # Load unusable space in
    unusable_space = load_unusable_space()

    # Load slots to shuffle in
    shuffle_slots = load_shuffle_slots()

    # Load shuffle_to range
    shuffle_to_range_df = load_shuffle_to_range(
        data_path=[content['data_path']])

    # All unique blocks
    all_blks = full_df['blk'].unique()

    # Which blocks to shuffle based on config
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

        blk_shuffle_ranges = []

        if blk in shuffle_slots.index:
            print("SOLVING BLOCK: ", blk)
            # There can be non-contiguous rows for each block
            blk_shuffle_rows = shuffle_slots.loc[blk]

            # Check if it's non-contiguous
            if isinstance(blk_shuffle_rows, pd.DataFrame):
                for _, row in blk_shuffle_rows.iterrows():
                    from_slot = row['slotFm']
                    to_slot = row['slotTo']
                    blk_shuffle_ranges.append((from_slot, to_slot))
            else:  # Only a single range
                from_slot = blk_shuffle_rows['slotFm']
                to_slot = blk_shuffle_rows['slotTo']
                blk_shuffle_ranges.append((from_slot, to_slot))
        else:
            continue

        start_blk_time = time.time()
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
        block_shuffle_config = all_terminals_shuffle_config[block_terminal]

        # Determine the agent config depending on block shuffle config
        if len(block_shuffle_config) == 1 and "overstow" in block_shuffle_config:
            agent_config = content['ray_agent_overstow_only']
        elif len(block_shuffle_config) > 1 and "mixWeight" not in block_shuffle_config:
            agent_config = content['ray_agent_mix_no_weight']
        else:
            agent_config = content['ray_agent_mix_w_weight']

        # Get block yard crane type
        block_yc_type = block_df['ycType'].unique()[0]

        # Get shuffle_to config for this yard crane type
        shuffle_to_range = shuffle_to_range_df.loc[block_yc_type]

        to_left, to_right = shuffle_to_range['toLeft'], shuffle_to_range['toRight']

        # Generate block summary for viewing
        generate_summary(blk, block_df, block_shuffle_config, block_unusable_space,
                         box_freeze=content['box_freeze'])

        distinct_slots = block_df['slotTo'].unique()

        slots_to_shuffle = distinct_slots if content['slots'] == 'all' else content['slots']

        # Set denoting which slots have already been shuffled together, only for inter-slot shuffling
        already_shuffled_slots = set()

        for slot_no in slots_to_shuffle:
            # Flag to check if this slot is to be shuffled
            is_within_any_shuffle_range = False

            for from_slot, to_slot in blk_shuffle_ranges:
                if slot_no >= from_slot and slot_no <= to_slot:
                    is_within_any_shuffle_range = True

            # Skip if this slot is not to be shuffled
            if not is_within_any_shuffle_range:
                continue

            if slot_no in already_shuffled_slots:
                continue

            # If this yard crane requires inter-slot shuffling
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
                    already_shuffled_slots.add(slot)

                print("Total max rows: ", total_max_rows)

                # Get the combined state here
                state = State(total_max_rows, max_levels,
                              block_shuffle_config, is_combined=True, num_slots_combined=len(all_slots), each_slot_num_rows=max_rows, unusable_rows=None)
                state.fill_rows(combined_slots_df)

                print("Container group num: ", state.actual_c_group_len)

                print("DONE COST: ", state.compute_done_cost())

                # Check whether there's any violating row in this slot
                row_len_exceeding = [
                    len(row) > max_levels for row in state.rows]

                if any(row_len_exceeding):
                    print(f"SLOT {slot} cannot be solved!")
                    continue

                generate_slots_summary(
                    slot_no, state, total_max_rows, max_levels)

                # Register custom model with action masking
                ModelCatalog.register_custom_model(
                    'cs_mask_multiple', ParametricActionsModel)

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

                appo_config = agents.impala.DEFAULT_CONFIG.copy()
                appo_config.update(agent_train_config)
                appo_config['lr'] = 1e-3
                stop = agent_config['stop']

                ray.shutdown()
                ray.init(dashboard_port=8266)

                # Tune Analysis
                trainer = agents.impala.ImpalaTrainer(config=appo_config)

                # initialize ray actor
                shortest_actions = ShortestActions.options(
                    name='global_actions').remote()

                for _ in range(stop['training_iteration']):
                    result = trainer.train()
                    print(pretty_print(result))
                    if result['timesteps_total'] >= stop['timesteps_total'] or result['episode_reward_mean'] >= stop['episode_reward_mean']:
                        break

                # End training time
                end_block_time = time.time()
                print(f"END BLOCK {blk} SHUFFLING")

                final_shortest_actions = ray.get(shortest_actions.get.remote())
                print("Shortest actions: ", final_shortest_actions)

                if final_shortest_actions:
                    with open('ray_solution.txt', 'a') as f:
                        f.write(f'Slot {slot_no} solution: ')
                        f.write(
                            f'{" ".join([str(action) for action in final_shortest_actions])}')
                        f.write('\n')
                    with open('group_solution.txt', 'a') as f:
                        f.write(f'Slot {slot_no} solution: \n')
                        all_action_groups = group_actions(final_shortest_actions)
                        f.write(get_action_groups_str(all_action_groups))
                        f.write('\n')
                else:
                    with open('ray_solution.txt', 'a') as f:
                        env = RayContainerShuffleEnv(env_config)
                        obs = env.reset()
                        done = False
                        print(env.render(mode='color'))
                        f.write(f'Slot {slot_no} solution: ')
                        for _ in range(environment_config['shuffle_moves_limit']):
                            if done:
                                print(env.render(mode='color'))
                                print(env.change_state.get_row_costs())
                                break
                            action = trainer.compute_action(obs)
                            f.write(f'{action} ')
                            obs, reward, done, info = env.step(action)
                        f.write('\n')
                    with open('group_solution.txt', 'a') as f:
                        f.write(f'Slot {slot_no} solution: \n')
                        all_action_groups = group_actions(final_shortest_actions)
                        f.write(get_action_groups_str(all_action_groups))
                        f.write('\n')

            else:

                print("SOLVING SLOT: ", slot_no)

                # Get slot dataframe
                slot = preprocess(
                    block_df[block_df['slotTo'] == slot_no], box_freeze=content['box_freeze'])

                # Get number of slot rows
                max_rows = 10 if np.any(slot['row'].unique() > 6) else 6
                max_levels = 5

                # Get unusable rows in this slot if required
                unusable_rows = None
                if slot_no in affected_slots_in_block:
                    unusable_rows = block_unusable_space[block_unusable_space['slotTo'] == slot_no]['row'].unique(
                    )

                print(
                    f"These rows are not usable: {unusable_rows}" if unusable_rows else "ALL ROWS USABLE")

                # Get current slot state
                state = State(max_rows, max_levels,
                              block_shuffle_config, unusable_rows=unusable_rows)
                state.fill_rows(slot)

                print("DONE COST: ", state.compute_done_cost())

                # Check whether there's any violating row in this slot
                row_len_exceeding = [
                    len(row) > max_levels for row in state.rows]

                if any(row_len_exceeding):
                    print(f"SLOT {slot_no} cannot be solved!")
                    continue

                np_state = state.to_numpy_state()

                # Check for violations in each 6/10 slot, also check if enough empty levels
                if max_rows == 6 and (len(state.rows[0]) == max_levels or sum(state.get_empty_levels_per_row(np_state)) < content['empty_levels_threshold']):
                    continue
                elif max_rows == 10 and sum(state.get_empty_levels_per_row(np_state)) < content['empty_levels_threshold']:
                    continue

                # Register custom model with action masking
                ModelCatalog.register_custom_model(
                    'cs_mask_single', ParametricActionsModel)

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
                        'custom_model': 'cs_mask_single'
                    },
                    'num_workers': agent_config['num_workers'],
                    "rollout_fragment_length": agent_config['rollout_fragment_length'],
                    "horizon": agent_config['horizon'],
                    "train_batch_size": agent_config['train_batch_size'],
                }

                if agent_config.get('num_envs_per_worker'):
                    agent_train_config["num_envs_per_worker"] = agent_config["num_envs_per_worker"]

                ray.shutdown()
                ray.init(dashboard_port=8266)

                if max_rows == 6:
                    ppo_config = agents.ppo.DEFAULT_CONFIG.copy()
                    ppo_config.update(agent_train_config)
                    ppo_config['lr'] = 1e-3
                    trainer = agents.ppo.PPOTrainer(config=ppo_config)
                else:
                    appo_config = agents.ppo.appo.DEFAULT_CONFIG.copy()
                    appo_config.update(agent_train_config)
                    appo_config['lr'] = 1e-3
                    trainer = agents.ppo.APPOTrainer(config=appo_config)

                stop = agent_config['stop']

                # initialize ray actor
                shortest_actions = ShortestActions.options(
                    name='global_actions').remote()

                for _ in range(stop['training_iteration']):
                    result = trainer.train()
                    print(pretty_print(result))
                    if result['timesteps_total'] >= stop['timesteps_total'] or result['episode_reward_mean'] >= stop['episode_reward_mean']:
                        break

                final_shortest_actions = ray.get(shortest_actions.get.remote())
                print("Shortest actions: ", final_shortest_actions)

                if final_shortest_actions:
                    with open('ray_solution.txt', 'a') as f:
                        f.write(f'Slot {slot_no} solution: ')
                        f.write(
                            f'{" ".join([str(action) for action in final_shortest_actions])}')
                        f.write('\n')
                    with open('group_solution.txt', 'a') as f:
                        f.write(f'Slot {slot_no} solution: \n')
                        all_action_groups = group_actions(final_shortest_actions)
                        f.write(get_action_groups_str(all_action_groups))
                        f.write('\n')
                    with open('output_solution.txt', 'a') as f:
                        f.write(f'Slot {slot_no} solutions: \n')
                        f.write('CNTR,vsl2,voy2,fm_CT,fm_blk,fm_sloTFm,fm_slotTo,fm_row,fm_level,to_CT,to_blk,to_slotFm,to_slotTo,to_row,to_level\n')
                        slot_df = slot.set_index('CNTR_N')
                        for action in final_shortest_actions:
                            from_row, to_row = convert_action_to_move(action, index=1, max_rows=state.max_rows)
                            moved_container = state.rows[from_row-1][-1]
                            moved_container_series = slot_df.loc[moved_container.name]
                            from_level = len(state.rows[from_row-1])
                            state.move_container(from_row, to_row)
                            to_level = len(state.rows[to_row-1])
                            output = f'{moved_container.name},{moved_container_series.vsl2},{moved_container_series.voy2},{moved_container_series.ct},{moved_container_series.blk},{moved_container_series.slotFm},{moved_container_series.slotTo},{from_row},{from_level},{moved_container_series.ct},{moved_container_series.blk},{moved_container_series.slotFm},{moved_container_series.slotTo},{to_row},{to_level}\n'
                            f.write(output)
                else:
                    with open('ray_solution.txt', 'a') as f:
                        env = RayContainerShuffleEnv(env_config)
                        obs = env.reset()
                        done = False
                        print(env.render(mode='color'))
                        f.write(f'Slot {slot_no} solution: ')
                        for _ in range(environment_config['shuffle_moves_limit']):
                            if done:
                                print(env.render(mode='color'))
                                print(env.change_state.get_row_costs())
                                break
                            action = trainer.compute_action(obs)
                            f.write(f'{action} ')
                            obs, reward, done, info = env.step(action)
                        f.write('\n')
                    with open('group_solution.txt', 'a') as f:
                        f.write(f'Slot {slot_no} solution: \n')
                        all_action_groups = group_actions(
                            final_shortest_actions)
                        f.write(get_action_groups_str(all_action_groups))
                        f.write('\n')
            ray.shutdown()

    end = time.time()
    print(f"END ALL BLOCK SHUFFLING IN {end - start} SECONDS")
