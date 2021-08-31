import functools
import time
import numpy as np


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer


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


def get_hex_bg_color(color):
    return BACKGROUND_COLORS[color][1]


def shorten_actions(actions):
    enhanced_actions = []
    current_index = -1
    for i, action in enumerate(actions):
        if i <= current_index:
            continue
        if i == len(actions) - 1:
            enhanced_actions.append(action)
            break
        next_action = actions[i + 1]
        # print(action, next_action)
        if action[1] != next_action[0]:
            enhanced_actions.append(action)
        else:
            current_index = i
            next_action = actions[current_index + 1]
            while current_index < len(actions) - 1 and action[1] == next_action[0]:
                action = (action[0], next_action[1])
                current_index += 1
                if current_index == len(actions) - 1:
                    break
                next_action = actions[current_index + 1]
            enhanced_actions.append(action)

    return enhanced_actions


# print(shorten_actions([(5, 2), (5, 2), (5, 2), (2, 1), (2, 5), (2, 5), (4, 5), (1, 4), (4, 2), (4, 5), (2, 4), (4, 1), (1, 2), (2, 4),
#                        (4, 1), (1, 2), (2, 4), (4, 1), (1, 2), (2, 4), (4, 1), (1, 2), (2, 4), (4, 1), (1, 2), (2, 4), (4, 1), (1, 2), (2, 4), (4, 1)]))
