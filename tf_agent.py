from utils import convert_action_to_log, convert_action_to_move
import tqdm
from environment import ContainerShuffleEnv
from state import State
import tensorflow as tf
import numpy as np
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import categorical_q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
tf.compat.v1.enable_v2_behavior()


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def tf_agent_initialize_and_train(agent_config, environment_config, block_shuffle_config, unusable_rows, slot_data, slot_no, max_rows, max_levels, all_slot_actions):

    # Load current state
    print(max_rows, max_levels)
    oop_state = State(max_rows, max_levels,
                      block_shuffle_config, unusable_rows=unusable_rows)
    oop_state.fill_rows(slot_data)
    print(oop_state.cprint(mode='color'))

    # Load environment
    train_py_env = ContainerShuffleEnv(oop_state, environment_config)
    eval_py_env = ContainerShuffleEnv(oop_state, environment_config)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Define neural network
    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec()['observation'],
        train_env.action_spec(),
        num_atoms=agent_config['num_atoms'],
        fc_layer_params=agent_config['fc_layer_params'])

    # Define otimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=agent_config['learning_rate'])

    # Define train step counter
    train_step_counter = tf.compat.v2.Variable(0)

    # Observation and action constrant splitter
    def observation_and_action_constraint_splitter(observation):
        return observation['observation'], observation['valid_actions']

    # Define rainbow DQN agent and initialize
    agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=categorical_q_net,
        observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
        optimizer=optimizer,
        min_q_value=agent_config['min_q_value'],
        max_q_value=agent_config['max_q_value'],
        n_step_update=agent_config['n_step_update'],
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=agent_config['gamma'],
        train_step_counter=train_step_counter)

    agent.initialize()

    # Initialize some random policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec(),
                                                    observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)

    # Prepare replay_buffer, fill it up with some initial collections and return its iterator
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=agent_config['replay_buffer_capacity'])

    def collect_step(environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(
            time_step, action_step, next_time_step)

        replay_buffer.add_batch(traj)

    for _ in range(agent_config['initial_collect_steps']):
        collect_step(train_env, random_policy)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=agent_config['batch_size'],
        num_steps=agent_config['n_step_update'] + 1).prefetch(3)

    iterator = iter(dataset)

    # Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training
    avg_return = compute_avg_return(
        eval_env, agent.policy, agent_config['num_eval_episodes'])

    prev_avg_return = float('-inf')

    returns = [avg_return]

    with tqdm.trange(agent_config['num_iterations']) as t:
        for _ in t:

            # Collect a few steps using collect_policy and save to replay buffer
            for _ in range(agent_config['collect_steps_per_iteration']):
                # Exploit and explore by using agent's epsilon-greedy policy
                collect_step(train_env, agent.collect_policy)

            # Sample a batch of data from the buffer and update the agent's network
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience)

            step = agent.train_step_counter.numpy()

            if step % agent_config['log_interval'] == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss.loss))

            if step % agent_config['eval_interval'] == 0:
                # evaluate using agent's greedy policy
                avg_return = compute_avg_return(
                    eval_env, agent.policy, agent_config['num_eval_episodes'])
                print('step = {0}: Average Return = {1:.2f}'.format(
                    step, avg_return))

                # Overwrite previous best with current best solution
                if avg_return > prev_avg_return:

                    all_slot_actions[f'{slot_no}'].clear()

                    with open(f'slot_{slot_no}_tf_solution.txt', 'w') as f:
                        # Denote which shuffle config this slot has
                        f.write(f'Slot shuffle config: {block_shuffle_config}')

                        # Get agent's optimal policy
                        policy = agent.policy

                        # Start from initial state
                        time_step = eval_env.reset()
                        actions = []

                        for _ in range(environment_config['shuffle_moves_limit']):
                            if time_step.is_last():
                                break

                            action_step = policy.action(time_step)

                            # Which action to take, append to list of actions for this slot
                            action = action_step.action.numpy()[0]
                            all_slot_actions[f'{slot_no}'].append(action)

                            f.write(
                                f"\n{convert_action_to_log(action, max_rows)}\n")

                            # Get from row and to row
                            from_row, to_row = convert_action_to_move(
                                action, max_rows, index=0)

                            f.write(
                                f"Row origin cost before: {eval_env.pyenv.envs[0].change_state.get_row_costs()[from_row]}\n")
                            f.write(
                                f"Row dest cost before: {eval_env.pyenv.envs[0].change_state.get_row_costs()[to_row]}\n")
                            # Take this action
                            time_step = eval_env.step(action_step.action)

                            # Append each move to actions dictionary
                            actions.append(action_step.action.numpy()[0])

                            f.write(
                                f"Row origin cost after: {eval_env.pyenv.envs[0].change_state.get_row_costs()[from_row]}\n")
                            f.write(
                                f"Row dest cost after: {eval_env.pyenv.envs[0].change_state.get_row_costs()[to_row]}\n")
                            f.write(f"Reward: {time_step.reward.numpy()[0]}\n")

                        prev_avg_return = avg_return

                    print(
                        f"Completed overwriting of solution at avg return {avg_return}")

                # Terminate immediately if a solution has been found
                if agent_config['terminate_early'] and avg_return > 80:
                    break

                returns.append(avg_return)
