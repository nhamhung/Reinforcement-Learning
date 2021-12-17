from container import Container
from collections import Counter, defaultdict, namedtuple
from utils import map_group_to_num, map_group_to_color, get_hours_difference, color_print
import numpy as np
from functools import cmp_to_key


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