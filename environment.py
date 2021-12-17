import gym
from gym.spaces import Discrete, Dict, Box
from utils import convert_action_to_move
import math
import numpy as np
import copy
from enum import Enum


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

    def render(self, mode='human'):
        """
        Render the current state of the environment
        """
        print(self.change_state.cprint(mode='color'))

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
