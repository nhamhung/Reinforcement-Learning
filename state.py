from container import Container
from collections import Counter, defaultdict, namedtuple
from utils import map_group_to_num, map_group_to_color, get_hours_difference, color_print
import numpy as np
from functools import cmp_to_key


class State:
    def __init__(self, max_rows, max_levels, shuffle_config, unusable_rows=None):
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

    def cmp_vessel_groups(self, v_group_1, v_group_2):
        if v_group_1[0].etb > v_group_2[0].etu - 2:  # v_group_1 should be below
            return 1
        elif v_group_1[0].etb == v_group_2[0].etb and v_group_1[0].etu == v_group_2[0].etu:
            return 0
        else:
            return -1

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

    def cal_highest_freeze_level_per_row(self):
        highest_freeze_level_per_row = [0] * 6
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

    def _is_row_empty(self, row_num):
        return self._get_row_height(row_num) == 0

    def _is_row_full(self, row_num):
        if self.max_rows == 6:
            return self._get_row_height(row_num) == self.max_levels if row_num != 0 else self._get_row_height(row_num) == 4
        elif self.max_rows == 10:
            return self._get_row_height(row_num) == self.max_levels

    def _get_row_height(self, row_num):
        return len(self.rows[row_num])

    def _compute_row_cost(self, row):
        if len(self.shuffle_config) > 1:
            return row.sum(axis=0, where=[False, True, True]).sum()
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
                    np_state[i][j][0] = self.group_to_num_map[container.short_name()]

                    if self.box_freeze and self.highest_freeze_level_per_row and j < self.highest_freeze_level_per_row[i]:
                        continue

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

        else:
            np_state = np.zeros(
                (self.max_rows, self.max_levels, 3), dtype=np.int32)

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
                                    np_state[i][h][1] = 2

                            for h in range(k, len(row)):
                                np_state[i][h][2] = 2

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

                        if 'mixWeight' in self.shuffle_config:
                            if container.vessel == top_container.vessel and container.weight > top_container.weight:
                                for h in range(k, len(row)):
                                    np_state[i][h][2] = 1

            return np_state

    def to_env_state(self):
        return np.vstack(self.to_numpy_state()).flatten('F')

    def compute_done_cost(self):
        # overstow only
        if len(self.shuffle_config) == 1 and "overstow" in self.shuffle_config:

            return 0

        # port mark, size, category

        elif len(self.shuffle_config) > 1 and "mixWeight" not in self.shuffle_config:

            if self.fit_by_container_group:  # can fit by container groups
                return 0

            elif self.fit_by_vessel_group:  # fit by vessel group, require minimal mixing
                done_cost = 0

                for vessel_group, _ in self.vessel_groups_sorted_count:
                    matching_container_groups = [(container_group, count) for container_group,
                                                 count in self.container_groups_sorted if container_group.vessel == vessel_group.vessel][1:]  # get all matching container groups exclude first group with most containers (sorted)
                    done_cost += sum([count for container_group,
                                      count in matching_container_groups])

                return done_cost

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
                                            group_count in remaining_groups]) * 2
                            return done_cost
                    else:
                        num_full_rows = group_count // self.max_levels if group_count % self.max_levels == 0 else group_count // self.max_levels + 1
                        remaining_rows = self.actual_available_rows - total_rows
                        if remaining_rows >= num_full_rows:  # if enough space
                            total_rows += num_full_rows
                            stop_index = index + 1
                            if total_rows == self.actual_available_rows:  # max rows reached, remaining groups needs to be mixed
                                remaining_groups = self.container_groups_sorted[stop_index:]
                                done_cost = sum([group_count for container_group,
                                                group_count in remaining_groups]) * 2
                                return done_cost
                        else:  # max rows reached or exceeded, both remainder of current group and remaining groups need to be mixed
                            rows_left = num_full_rows - remaining_rows
                            group_count_left = group_count - rows_left * self.max_levels
                            stop_index = index + 1
                            remaining_groups = self.vessel_groups_sorted_overstow[stop_index:]
                            done_cost = ((group_count_left %
                                          self.actual_available_rows) + sum([group_count for container_group,
                                                                             group_count in remaining_groups])) * 2
                            return done_cost

        # weight

    def is_solvable(self):
        np_state = self.to_numpy_state()
        overstow_costs = sum(self.get_rows_overstow_values())
        avail_levels = sum(self.get_empty_levels_per_row(np_state))
        return overstow_costs < avail_levels

    def check_ground_row(self, row_idx):
        row = self.rows[row_idx]
        container = row[0]

        container_group_weights = self.container_groups_weights[container.short_name(
        )]
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

        if group_min_weight_count <= self.max_levels:
            return False if min_row_weight_count < group_min_weight_count else True
        else:
            return True

    def check_row_same_type(self, row_idx):
        if self.fit_by_container_group:
            if 'mixWeight' in self.shuffle_config:
                # include strict same weight
                return all(container.equal_with_weight(self.rows[row_idx][0]) for container in self.rows[row_idx])
            # no weight consideration
            return all(container.equal_without_weight(self.rows[row_idx][0]) for container in self.rows[row_idx])
        else:
            return all(container.vessel == self.rows[row_idx][0].vessel for container in self.rows[row_idx])

    def get_non_empty_rows(self):
        """ Returns rows which are not empty """
        return np.array([i for i, row in enumerate(self.rows) if len(row) > 0])

    def get_ground_rows(self):  # TODO: Double check ground rows
        """ Returns rows with 2 or more containers of the same type """
        if self.fit_by_container_group and 'mixWeight' in self.shuffle_config:
            same_type_rows = [row_idx for row_idx, row in enumerate(self.rows) if len(
                self.rows[row_idx]) >= 1 and self.check_row_same_type(row_idx)]
            return np.array([row_idx for row_idx in same_type_rows if self.check_ground_row(row_idx)])
        else:
            return np.array([row_idx for row_idx, row in enumerate(self.rows) if len(self.rows[row_idx]) >= 2 and self.check_row_same_type(row_idx)])

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
        if len(self.shuffle_config) > 1:
            return np.array([row.sum(axis=0, where=[False, True, False]).sum() for row in self.to_numpy_state()])
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
