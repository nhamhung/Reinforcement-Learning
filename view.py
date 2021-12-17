import argparse
import os
import sys
from pathlib import Path
import xlsxwriter
import re
import json
import numpy as np
import pandas as pd
import copy

from utils import get_hex_bg_color, convert_action_to_move, shorten_actions, convert_move_to_action
from state import State
from data_loader import load_full_data, preprocess, load_config, load_unusable_space, load_shuffle_to_range
from environment import RayContainerShuffleEnv


def get_slot_records(slot_state, max_rows, max_levels):
    """
    Get all the shortened text for each container in this slot.
    """
    records = [['' for _ in range(max_rows)] for _ in range(max_levels)]

    container_records = [
        ['' for _ in range(max_rows)] for _ in range(max_levels)]

    for level in range(max_levels):
        for row in range(max_rows):
            container_row = slot_state.rows[row]
            row_len = len(container_row)
            corresponding_level = max_levels - level
            if corresponding_level > row_len:
                records[level][row] = ' ' * \
                    slot_state.longest_container_name + ' '
                container_records[level][row] = ' ' * \
                    slot_state.longest_container_name + ' '
            else:
                container = container_row[corresponding_level - 1]
                container_name = container.__str__()
                container_str = container_name if len(
                    container_name) == slot_state.longest_container_name else container_name.ljust(slot_state.longest_container_name, ' ')
                records[level][row] = container_str
                container_records[level][row] = container.ui_repr().ljust(
                    slot_state.longest_container_name, ' ')

    return records, container_records


def write_state(workbook, worksheet, slot_state, records, container_records, row, col, max_rows, max_levels, highest_freeze_level_per_row, from_row=None, to_row=None, color=False):
    """
    Write the current slot state to excel.
    """

    if from_row is not None and to_row is not None:
        from_level = max_levels - len(slot_state.rows[from_row]) - 1
        to_level = max_levels - len(slot_state.rows[to_row])

    cell_from_format = workbook.add_format()
    cell_freeze_format = workbook.add_format()
    cell_to_format = workbook.add_format()

    cell_from_format.set_bg_color('gray')
    cell_to_format.set_border(5)
    cell_to_format.set_border_color('red')
    cell_freeze_format.set_border(6)
    cell_freeze_format.set_border_color('blue')

    for i, level in enumerate(records):
        row += 1
        col = 0
        for j, row_text in enumerate(level):
            cell_color_format = workbook.add_format()

            if row_text.strip(' ') != '':
                if color:
                    cell_color_format.set_bg_color(get_hex_bg_color(
                        slot_state.group_to_color_map[row_text.strip(' ')[:-2]]))
                    if highest_freeze_level_per_row[j] != -1 and i == max_levels - highest_freeze_level_per_row[j] - 1:
                        cell_color_format.set_border(5)
                        cell_color_format.set_border_color('black')
                        worksheet.write_string(
                            row, col, container_records[i][j], cell_color_format)
                    else:
                        worksheet.write_string(
                            row, col, container_records[i][j], cell_color_format)
                else:
                    if from_row is not None and to_row is not None and j == to_row and i == to_level:
                        worksheet.write_string(
                            row, col, container_records[i][j], cell_to_format)
                    elif highest_freeze_level_per_row[j] != -1 and i == max_levels - highest_freeze_level_per_row[j] - 1:
                        worksheet.write_string(
                            row, col, container_records[i][j], cell_freeze_format)
                    else:
                        worksheet.write_string(
                            row, col, container_records[i][j])
                col += 1
            else:
                if from_row is not None and to_row is not None and j == from_row and i == from_level:
                    worksheet.write_string(
                        row, col, container_records[i][j], cell_from_format)
                elif highest_freeze_level_per_row[j] != -1 and i == max_levels - highest_freeze_level_per_row[j] - 1:
                    worksheet.write_string(
                        row, col, container_records[i][j], cell_freeze_format)
                else:
                    worksheet.write_string(row, col, container_records[i][j])
                col += 1
    row += 1
    for col in range(max_rows):
        if slot_state.unusable_rows and col in slot_state.unusable_rows:
            worksheet.write_string(row, col, f"Row {col + 1}: Unusable")
        else:
            worksheet.write_string(row, col, f"Row {col + 1}")

    return row, col


def create_excel_output(blk_name, blk_df, block_solutions, block_shuffle_config, block_unusable_space, shuffle_moves_limit, to_left, to_right, view_tf_agent=True, box_freeze=True, color=False):
    """
    Write the state after each action for each slot in this block. Each state gets a sheet.
    """
    if view_tf_agent:
        workbook = xlsxwriter.Workbook(
            f'{blk_name}_tf_agent_solution.xlsx')
    else:
        workbook = xlsxwriter.Workbook(f'{blk_name}_ray_solution.xlsx')

    affected_slots_in_block = block_unusable_space['slotTo'].unique()

    distinct_slots = blk_df['slotTo'].unique()

    # print("Block solutions: ", block_solutions)

    for slot_no, actions in block_solutions.items():

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
                    blk_df[blk_df['slotTo'] == slot], box_freeze=False)

                # Increment slot rows
                slot_df['row'] = slot_df['row'].apply(
                    lambda x: x + total_max_rows)

                # Get a new combined slots dataframe
                combined_slots_df = pd.concat([combined_slots_df, slot_df])
                total_max_rows += max_rows

            slot_state = State(total_max_rows, max_levels,
                               block_shuffle_config, unusable_rows=None)
            slot_state.fill_rows(combined_slots_df)
            records, container_records = get_slot_records(
                slot_state, total_max_rows, max_levels)

            is_solved = True if len(actions) < shuffle_moves_limit else False

            worksheet = workbook.add_worksheet(
                name=f'{slot_no}_solved') if is_solved else workbook.add_worksheet(name=f'{slot_no}_unsolved')

            worksheet.set_column(0, total_max_rows, 23)

            highest_freeze_level_per_row = [-1] * total_max_rows

            row, col = 0, 0
            row, col = write_state(workbook, worksheet,
                                   slot_state, records, container_records, row, col, total_max_rows, max_levels, highest_freeze_level_per_row, color=color)
            row += 2

            actions = [convert_action_to_move(
                action, total_max_rows, index=1) for action in actions]

            actions = shorten_actions(actions)

            if not is_solved:
                actions = shuffle_moves_termination(slot_state, actions)

            for i, action in enumerate(actions):
                merge_format = workbook.add_format({
                    'bold': 1,
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter',
                    'fg_color': 'yellow'
                })

                # Move container
                from_row, to_row = action
                before_cost = slot_state.total_cost
                slot_state.move_container(from_row, to_row)
                after_cost = slot_state.total_cost

                worksheet.merge_range(
                    f'B{row+1}:E{row+1}', f"{i+1}. Row {action[0]} to row {action[1]}. Before cost: {before_cost}. After cost: {after_cost}", merge_format)

                row += 1

                worksheet.merge_range(f'A{row+1}:D{row+1}',
                                      f"{slot_state.to_numpy_state()}")
                row += 1

                # Get the new records after movement
                records, container_records = get_slot_records(
                    slot_state, total_max_rows, max_levels)

                row, col = write_state(workbook, worksheet,
                                       slot_state, records, container_records, row, col, total_max_rows, max_levels, highest_freeze_level_per_row, from_row=from_row-1, to_row=to_row-1, color=color)
                row += 2

        else:
            slot_df = blk_df[blk_df['slotTo'] == slot_no]
            slot_df = preprocess(slot_df, box_freeze=box_freeze)

            max_rows = 10 if np.any(slot_df['row'].unique() > 6) else 6
            max_levels = 5

            unusable_rows = None
            if slot_no in affected_slots_in_block:
                unusable_rows = block_unusable_space[block_unusable_space['slotTo'] == slot_no]['row'].unique(
                )

            slot_state = State(max_rows, max_levels,
                               block_shuffle_config, unusable_rows=unusable_rows)
            slot_state.fill_rows(slot_df)

            env_config = {
                'oop_state': slot_state,
                'invalid_move_penalty': 0,
                'shuffle_moves_limit': 50,
                'done_cost': 0
            }

            env = RayContainerShuffleEnv(env_config)
            env.reset()

            records, container_records = get_slot_records(
                slot_state, max_rows, max_levels)

            # Get highest freeze level per row
            highest_freeze_level_per_row = slot_state.highest_freeze_level_per_row

            is_solved = True if len(actions) < shuffle_moves_limit else False

            worksheet = workbook.add_worksheet(
                name=f'{slot_no}_solved') if is_solved else workbook.add_worksheet(name=f'{slot_no}_unsolved')

            worksheet.set_column(0, 10, 23)

            row, col = 0, 0
            row, col = write_state(workbook, worksheet,
                                   slot_state, records, container_records, row, col, max_rows, max_levels, highest_freeze_level_per_row, color=color)
            row += 2

            actions = [convert_action_to_move(
                action, max_rows, index=1) for action in actions]

            actions = shorten_actions(actions)

            print(actions)

            if not is_solved:
                actions = shuffle_moves_termination(slot_state, actions)

            for i, action in enumerate(actions):
                merge_format = workbook.add_format({
                    'bold': 1,
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter',
                    'fg_color': 'yellow'
                })

                # Move container
                from_row, to_row = action
                if from_row == to_row:
                    continue
                before_cost = slot_state.total_cost
                slot_state.move_container(from_row, to_row)
                after_cost = slot_state.total_cost

                numeric_action = convert_move_to_action(from_row - 1, to_row - 1, max_rows)
                obs, reward, done, info = env.step(numeric_action)

                if max_rows == 6:
                    worksheet.merge_range(
                        f'B{row+1}:E{row+1}', f"{i+1}. Row {action[0]} to row {action[1]}. Before cost: {before_cost}. After cost: {after_cost}. Reward: {reward}", merge_format)
                elif max_rows == 10:
                    worksheet.merge_range(
                        f'B{row+1}:I{row+1}', f"{i+1}. Row {action[0]} to row {action[1]}. Before cost: {before_cost}. After cost: {after_cost}. Reward: {reward}", merge_format)

                row += 1

                worksheet.merge_range(f'A{row+1}:D{row+1}',
                                      f"{slot_state.to_numpy_state()}")
                row += 1

                # Get the new records after movement
                records, container_records = get_slot_records(
                    slot_state, max_rows, max_levels)

                row, col = write_state(workbook, worksheet,
                                       slot_state, records, container_records, row, col, max_rows, max_levels, highest_freeze_level_per_row, from_row=from_row-1, to_row=to_row-1, color=color)
                row += 2

    workbook.close()


def shuffle_moves_termination(state, actions):
    """
    Only write output if the cost is reduced. Else terminates.
    """
    slot_state = copy.deepcopy(state)
    cost = slot_state.total_cost
    action_stop = 0

    for i, action in enumerate(actions):
        from_row, to_row = action
        slot_state.move_container(from_row, to_row)
        current_cost = slot_state.total_cost
        if current_cost < cost:
            cost = current_cost
            action_stop = i + 1

    return actions[:action_stop]


def generate_slots_summary(slot_strings, state, max_rows, max_levels):
    """
    Generate a summary for this slot.
    """

    workbook = xlsxwriter.Workbook(f'{slot_strings}_summary.xlsx')

    worksheet = workbook.add_worksheet()

    worksheet.set_column(0, 24, 23)

    row, col = 0, 0

    records, container_records = get_slot_records(
        state, max_rows, max_levels)

    highest_freeze_level_per_row = [-1] * max_rows

    cell_format = workbook.add_format({'bold': True, 'font_color': 'blue'})

    row, col = write_state(workbook, worksheet, state,
                           records, container_records, row, col, max_rows, max_levels, highest_freeze_level_per_row)

    row += 2

    worksheet.merge_range(
        f'A{row+1}:D{row+1}', f"Number of container groups: {state.c_group_len}. Actual groups after box free: {state.actual_c_group_len}. Require {state.container_groups_rows_required} rows.", cell_format)

    row += 1
    worksheet.merge_range(
        f'A{row+1}:D{row+1}', f"Number of vessel groups: {state.v_group_len}. Actual groups after box free: {state.actual_v_group_len}. Require {state.vessel_groups_rows_required} rows", cell_format)
    row += 1

    worksheet.merge_range(f'A{row+1}:D{row+1}',
                          f"Initial cost: {state.total_cost}", cell_format)
    row += 1
    worksheet.merge_range(f'A{row+1}:D{row+1}',
                          f"Numpy state: {state.to_numpy_state()}", cell_format)
    row += 1
    worksheet.merge_range(
        f'A{row+1}:D{row+1}', f"Overstow rows: {state.get_overstow_rows()}", cell_format)
    row += 1

    fit_by = 'container_group' if state.fit_by_container_group else 'vessel groups' if state.fit_by_vessel_group else "mixed group"
    worksheet.merge_range(f'A{row+1}:D{row+1}',
                          f"Fit by: {fit_by}", cell_format)
    row += 1
    worksheet.merge_range(f'A{row+1}:D{row+1}',
                          f"State done cost: {state.compute_done_cost()}", cell_format)
    row += 1

    worksheet.merge_range(
        f'A{row+1}:D{row+1}', f"Ground rows: {state.get_ground_rows()}", cell_format)
    row += 1

    worksheet.merge_range(
        f'A{row+1}:D{row+1}', f"Highest freeze level per row: {highest_freeze_level_per_row}", cell_format)
    row += 1

    worksheet.merge_range(
        f'A{row+1}:D{row+1}', f"Actual available rows: {state.actual_available_rows}", cell_format)
    row += 1

    workbook.close()


def generate_summary(blk_no, block_df, block_shuffle_config, block_unusable_space, box_freeze=False):
    """
    Generate a summary for all the slots in this block.
    """

    distinct_slots = block_df['slotTo'].unique()

    affected_slots_in_block = block_unusable_space['slotTo'].unique()

    workbook = xlsxwriter.Workbook(f'{blk_no}_summary.xlsx')

    worksheet = workbook.add_worksheet(name=f'block_{blk_no}')

    worksheet.set_column(0, 10, 23)

    row, col = 0, 0

    merge_format_slot = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'fg_color': 'yellow'})

    cell_format = workbook.add_format({'bold': True, 'font_color': 'blue'})

    for slot_no in distinct_slots:

        slot = preprocess(
            block_df[block_df['slotTo'] == slot_no], box_freeze=box_freeze)
        max_rows = 10 if np.any(slot['row'].unique() > 6) else 6
        max_levels = 5

        unusable_rows = None
        if slot_no in affected_slots_in_block:
            unusable_rows = block_unusable_space[block_unusable_space['slotTo'] == slot_no]['row'].unique(
            )

        state = State(max_rows, max_levels, block_shuffle_config,
                      unusable_rows=unusable_rows)
        state.fill_rows(slot)

        highest_freeze_level_per_row = state.highest_freeze_level_per_row

        row_len_exceeding = [
            len(row) > max_levels for row in state.rows]

        records, container_records = get_slot_records(
            state, max_rows, max_levels)

        worksheet.merge_range(f'A{row+1}:D{row+1}',
                              f'Slot {slot_no}', merge_format_slot)

        row += 2

        row, col = write_state(workbook, worksheet, state,
                               records, container_records, row, col, max_rows, max_levels, highest_freeze_level_per_row)

        row += 2

        worksheet.merge_range(
            f'A{row+1}:D{row+1}', f"Number of container groups: {state.c_group_len}. Actual groups after box free: {state.actual_c_group_len}. Require {state.container_groups_rows_required} rows.", cell_format)
        row += 1
        worksheet.merge_range(
            f'A{row+1}:D{row+1}', f"Number of vessel groups: {state.v_group_len}. Actual groups after box free: {state.actual_v_group_len}. Require {state.vessel_groups_rows_required} rows", cell_format)
        row += 1

        if any(row_len_exceeding):
            worksheet.merge_range(f'A{row+1}:D{row+1}',
                                  f"Initial cost: Cannot be computed due to levels exceeding {max_levels}", cell_format)
            row += 1
            worksheet.merge_range(
                f'A{row+1}:D{row+1}', f"Is solvable: False", cell_format)
            row += 2
            continue

        worksheet.merge_range(f'A{row+1}:D{row+1}',
                              f"Initial cost: {state.total_cost}", cell_format)
        row += 1
        worksheet.merge_range(f'A{row+1}:D{row+1}',
                              f"Numpy state: {state.to_numpy_state()}", cell_format)
        row += 1
        worksheet.merge_range(
            f'A{row+1}:D{row+1}', f"Overstow rows: {state.get_overstow_rows()}", cell_format)
        row += 1

        fit_by = 'container_group' if state.fit_by_container_group else 'vessel groups' if state.fit_by_vessel_group else "mixed group"
        worksheet.merge_range(f'A{row+1}:D{row+1}',
                              f"Fit by: {fit_by}", cell_format)
        row += 1
        worksheet.merge_range(f'A{row+1}:D{row+1}',
                              f"State done cost: {state.compute_done_cost()}", cell_format)
        row += 1

        worksheet.merge_range(
            f'A{row+1}:D{row+1}', f"Ground rows: {state.get_ground_rows()}", cell_format)
        row += 1

        worksheet.merge_range(
            f'A{row+1}:D{row+1}', f"Highest freeze level per row: {highest_freeze_level_per_row}", cell_format)
        row += 1

        worksheet.merge_range(
            f'A{row+1}:D{row+1}', f"Actual available rows: {state.actual_available_rows}", cell_format)
        row += 1

        # worksheet.merge_range(
        #     f'A{row+1}:D{row+1}', f"Is solvable: {state.is_solvable()}", cell_format)

        row += 2

    workbook.close()


if __name__ == '__main__':
    # Get content from main config file
    with open('./config/main_config.json', 'r') as f:
        content = json.load(f)
        view_config = content['view']

    # Get shuffling move limit
    shuffle_moves_limit = content['environment']['shuffle_moves_limit']

    root_dir_path = Path(os.getcwd())

    # Shuffle config for each terminal
    all_terminals_shuffle_config = load_config()

    # Load data
    full_df = load_full_data(data_path=content['data_path'])
    all_blks = full_df['blk'].unique()

    # Load shuffle_to range
    shuffle_to_range_df = load_shuffle_to_range(
        data_path=[content['data_path']])

    # Load unusable space in
    unusable_space = load_unusable_space()

    # Get blocks to view
    blks_to_view = all_blks if content['blocks'] == 'all' else content['blocks']

    # Generate view for all block
    for blk in blks_to_view:
        path_to_blk_solution = Path(root_dir_path, 'shuffle_solutions', blk)
        os.chdir(path_to_blk_solution)
        view_file = f'{blk}_visual_solution.txt'
        with open(view_file, 'r') as f:
            text = f.read()
            block_solutions = text.split('\n')[:-1]
            block_solutions = {int(slot[0]): list(map(int, slot[1:])) for slot in [re.findall(
                '\d+', slot) for slot in block_solutions if len(slot) > 0]}

        
        os.chdir(root_dir_path)
        solution_view_path = Path(root_dir_path, 'solution_visuals')
        solution_view_path.mkdir(exist_ok=True)
        os.chdir(solution_view_path)

        blk_df = full_df[full_df['blk'] == blk]

        # Get block terminal to map to shuffle config
        block_terminal = blk_df['ct'].unique()[0]

        # Get unusable space based on block terminal and block, then retrieve all affected slots
        block_unusable_space = unusable_space[(
            unusable_space['ct'] == block_terminal) & (unusable_space['blk'] == blk)]

        # Get the shuffle config for this block
        block_shuffle_config = all_terminals_shuffle_config.loc[block_terminal]

        # Get block yard crane type
        block_yc_type = blk_df['ycType'].unique()[0]

        # Get shuffle_to config for this yard crane type
        shuffle_to_range = shuffle_to_range_df.loc[block_yc_type]

        # Get which slots on left and right to combine
        to_left, to_right = shuffle_to_range['toLeft'], shuffle_to_range['toRight']

        # Create excel output for this block
        create_excel_output(blk, blk_df, block_solutions,
                            block_shuffle_config, block_unusable_space, shuffle_moves_limit, to_left, to_right, view_tf_agent=view_config['view_tf_agent'], box_freeze=content['box_freeze'], color=view_config['color'])

    print(f"GENERATED VIEWS FOR BLOCKS: {blks_to_view}")
