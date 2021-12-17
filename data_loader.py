import pandas as pd
import numpy as np
from datetime import timedelta


def load_full_data(data_path='./data/shuffleSlotsNew.xlsx'):
    """
    Load entire dataframe.
    """
    full_df = pd.read_excel(data_path, sheet_name='shuffleData')
    full_df = full_df[~full_df['CNTR_N'].isna()]
    return full_df


def load_blk_data(blk_no, data_path='./data/shuffleSlotsNew.xlsx'):
    """
    Only load data from a certain block.
    """
    full_df = load_full_data(data_path=data_path)
    return full_df[full_df['blk'] == blk_no]


def load_unusable_space(data_path='./data/unusableSpace.xlsx'):
    """
    Load unusable space.
    """
    unusable_space = pd.read_excel(data_path)
    return unusable_space


def load_shuffle_slots(data_path='./config/config_smartShuffling.xlsx'):
    """
    Load the slots to be shuffled.
    """
    origin_shuffle_blks = pd.read_excel(
        './config/config_smartShuffling.xlsx', sheet_name='shuffleBlk')

    origin_shuffle_slots = pd.read_excel(
        './config/config_smartShuffling.xlsx', sheet_name='shuffleSlot')

    shuffle_blks = origin_shuffle_blks[origin_shuffle_blks['isShuffleBlk']
                                       == 1]['blk'].unique()

    shuffle_slots = origin_shuffle_slots[origin_shuffle_slots['blk'].isin(
        shuffle_blks) & (origin_shuffle_slots['isShuffleSlot'] == 1)]

    shuffle_slots = shuffle_slots.set_index('blk')

    return shuffle_slots


def load_config(data_path='./config/config_smartShuffling.xlsx'):
    """
    Load shuffling config.
    """
    excel_config = pd.read_excel(
        './config/config_smartShuffling.xlsx', sheet_name='action')
    excel_config.set_index('ct', inplace=True)
    return excel_config


def load_shuffle_to_range(data_path='./config/config_smartShuffling.xlsx'):
    """
    Load the shuffling range.
    """
    shuffle_to_range = pd.read_excel(
        './config/config_smartShuffling.xlsx', sheet_name='shuffleToRange')
    shuffle_to_range = shuffle_to_range.set_index('ycType')
    return shuffle_to_range


def preprocess(slot_df, box_freeze=False):
    """
    Preprocess a slot dataframe.
    """
    if box_freeze:
        columns = ['row', 'level', 'CNTR_N', 'vsl2', 'pscw',
                   'SNAP_DT', 'etb2', 'etu2', 'pk', 'sz',	'cat',	'wtClass', 'boxFreeze', 'voy2', 'ct', 'blk', 'slotFm', 'slotTo']
    else:
        columns = ['row', 'level', 'CNTR_N', 'vsl2', 'pscw',
                   'SNAP_DT', 'etb2', 'etu2', 'pk', 'sz',	'cat',	'wtClass', 'voy2', 'ct', 'blk', 'slotFm', 'slotTo']
    slot_df = slot_df[columns]
    slot_df = slot_df[slot_df['level'] != 0]
    slot_df.sort_values(['row', 'level'], inplace=True)
    slot_df['SNAP_DT'] = pd.to_datetime(slot_df['SNAP_DT'])
    slot_df['etb2'] = pd.to_datetime(slot_df['etb2'])
    slot_df['etu2'] = pd.to_datetime(slot_df['etu2'])
    # slot_df.fillna(slot_df['SNAP_DT'].iloc[0] + timedelta(30), inplace=True)
    return slot_df