from tf_agents.environments import py_environment
from tf_agents.specs import tensor_spec, array_spec
from tf_agents.trajectories import time_step as ts
from utils import convert_action_to_move
import math
import numpy as np
import copy


class ContainerShuffleEnv(py_environment.PyEnvironment):

    def __init__(self, oop_state, environment_config):
        super(ContainerShuffleEnv, self).__init__()
        self.default_state = oop_state
        self.change_state = copy.deepcopy(self.default_state)
        self.invalid_move_penalty = environment_config['invalid_move_penalty']
        self.num_steps = 0
        self.episode_length = environment_config['shuffle_moves_limit']
        self.max_rows = self.default_state.max_rows
        self.max_levels = self.default_state.max_levels
        self.num_actions = self.max_rows * (self.max_rows - 1)
        self.observation_shape = self.max_rows * self.max_levels * \
            3 if len(
                self.default_state.shuffle_config) > 1 else self.max_rows * self.max_levels * 2
        self._action_spec = array_spec.BoundedArraySpec(
            (), dtype=np.int32, minimum=0, maximum=self.num_actions-1, name='action')
        self._observation_spec = {
            'observation': array_spec.BoundedArraySpec((self.observation_shape,), dtype=np.int32, minimum=0, maximum=10, name='observation'),
            'valid_actions': array_spec.ArraySpec(name="valid_actions", shape=(self.num_actions, ), dtype=np.int32)
        }
        self._state = self.default_state.to_env_state()
        self._episode_ended = False
        self.longest_vessel_name = max([len(container.short_name(
        )) for row in self.default_state.rows for container in row])
        self.move_dict = {a: self._convert_action_to_move(
            a) for a in range(self.num_actions)}
        self.fit_by_container_group = self.default_state.fit_by_container_group
        self.fit_by_vessel_group = self.default_state.fit_by_vessel_group
        self.initial_cost = self.default_state.total_cost
        self.done_cost = self.default_state.compute_done_cost(
        ) if environment_config['done_cost'] == 'auto' else environment_config['done_cost']

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self.default_state.to_env_state()
        self.num_steps = 0
        self.change_state = copy.deepcopy(self.default_state)
        self._episode_ended = False
        return ts.restart({'observation': np.array(self._state, dtype=np.int32), 'valid_actions': self.get_legal_moves()})

    def _step(self, action, logging=False):
        if self._episode_ended:
            return self.reset()

        self.num_steps += 1

        # Before movement state
        row_origin, row_dest = self._convert_action_to_move(action)

        before_row_costs = self.change_state.get_row_costs()

        self.change_state.move_container(row_origin + 1, row_dest + 1)

        # After movement state
        after_row_costs = self.change_state.get_row_costs()

        self._state = self.change_state.to_env_state()

        is_final = bool(sum(after_row_costs) <=
                        self.done_cost or self.num_steps == self.episode_length)

        if is_final:
            self._episode_ended = True
            if self.num_steps == self.episode_length:
                reward = self.invalid_move_penalty
            else:
                reward = 100.0
            return ts.termination({'observation': np.array(self._state, dtype=np.int32), 'valid_actions': self.get_legal_moves()}, reward)
        else:
            """
            This case allows for intermediate step where overstowed container may be put on top of ground rows and does not worsen existing overstow
            """
            reward = sum(before_row_costs) - sum(after_row_costs)
            reward = reward if reward != 0.0 else -0.5

            return ts.transition({'observation': np.array(self._state, dtype=np.int32), 'valid_actions': self.get_legal_moves()}, reward=reward, discount=0.99)

    def render(self):
        print(self.change_state.cprint(mode='color'))

    def get_legal_moves(self):
        all_valid_actions = self._get_valid_actions()
        all_actions = np.zeros(self.num_actions, dtype=np.int32)
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
