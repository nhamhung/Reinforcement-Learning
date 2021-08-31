import time
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from container import Container
from environment import ContainerShuffleEnv
from data_loader import load_full_data, load_blk_data, load_unusable_space, preprocess, load_config, load_shuffle_slots, load_shuffle_to_range
from state import State
from tf_agent import tf_agent_initialize_and_train
from utils import convert_action_to_log, convert_action_to_move, shorten_actions
from view import generate_summary
from action_group import group_actions, get_action_groups_str
import argparse
import os
import sys
from pathlib import Path
import datetime
import json
import gym
from gym.spaces import Discrete, Dict, Box
import copy


if __name__ == '__main__':
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

        # TODOS: whether to resolve this or not
        if blk in shuffle_slots.index:
            print("SOLVING BLOCK: ", blk)
            blk_shuffle_row = shuffle_slots.loc[blk]
            from_slot = blk_shuffle_row['slotFm']
            to_slot = blk_shuffle_row['slotTo']
        else:
            continue

        # Timer for each block, create block solution folder
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

        # Get block yard crane type
        block_yc_type = block_df['ycType'].unique()[0]

        # Get shuffle_to config for this yard crane type
        shuffle_to_range = shuffle_to_range_df.loc[block_yc_type]

        to_left, to_right = shuffle_to_range['toLeft'], shuffle_to_range['toRight']
        print(to_left, to_right)

        # Generate block summary for viewing
        generate_summary(blk, block_df, block_shuffle_config, block_unusable_space,
                         box_freeze=content['box_freeze'])

        # Dictionary to hold all actions for each slot in block
        all_slot_actions = {}
        distinct_slots = block_df['slotTo'].unique()

        slots_to_shuffle = distinct_slots if content['slots'] == 'all' else content['slots']

        # Set denoting which slots have already been shuffled together, only for inter-slot shuffling
        already_shuffled_slots = set()

        for slot_no in slots_to_shuffle:

            if slot_no < from_slot or slot_no > to_slot:
                continue

            if slot_no in already_shuffled_slots:
                continue

            # If this yard crane requires inter-slot shuffling
            if to_left or to_right:

                to_left_slots = [
                    slot for slot in distinct_slots if slot >= slot_no - to_left and slot < slot_no]
                print("TO LEFT: ", to_left_slots)

                to_right_slots = [
                    slot for slot in distinct_slots if slot <= slot_no + to_right and slot > slot_no]
                print("TO RIGHT: ", to_right_slots)

                all_slots = to_left_slots + [slot_no] + to_right_slots

                slot_states = []

                for slot in all_slots:
                    # Get each slot dataframe
                    slot = preprocess(
                        block_df[block_df['slotTo'] == slot_no], box_freeze=content['box_freeze'])

            # Just shuffle within the slot
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

                # Try to solve if state is solvable or done_cost is lower
                if state.is_solvable() or state.total_cost > state.compute_done_cost():

                    all_slot_actions[f'{slot_no}'] = []

                    # Train agent and use it to output solution for each slot
                    agent_config = content['tf_agent']
                    # Twitch number of iterations for different scenarios
                    # if not state.is_solvable():
                    #     agent_config['num_iterations'] = 5000

                    tf_agent_initialize_and_train(
                        agent_config, environment_config, block_shuffle_config, unusable_rows, slot, slot_no, max_rows, max_levels, all_slot_actions)

        # Timer for end of block training
        end_blk_time = time.time()
        time_now = datetime.datetime.now()
        file_name = time_now.strftime('%Y%m%d_%H%M.txt')

        # Output timing for each block
        with open(f"{file_name}", "w") as f:
            f.write(f"\n{block_terminal}, {blk}, {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_blk_time))}, {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_blk_time))}, {end_blk_time - start_blk_time}\n\n")

        # Output shuffling steps for each slot in the block
        with open('block_solution_visual.txt', 'w') as f:
            f.write(
                f"Block {blk} is solved in: {(end_blk_time - start_blk_time) / 60} minutes\n")
            for slot, actions in all_slot_actions.items():
                f.write(f"Slot {slot} solution: ")
                f.write(" ".join(map(str, actions)))
                f.write("\n")

        with open('block_solution_group.txt', 'w') as f:
            f.write(
                f"Block {blk} is solved in: {(end_blk_time - start_blk_time) / 60} minutes\n")
            for slot, actions in all_slot_actions.items():
                all_action_groups = group_actions(actions)
                f.write(f"Slot {slot} solution in groups: \n")
                f.write(get_action_groups_str(all_action_groups))
                f.write("\n")

    # End overall shuffling time
    end = time.time()
    print("END ALL BLOCK SHUFFLING")
    print("TOTAL TIME: ", end - start)
