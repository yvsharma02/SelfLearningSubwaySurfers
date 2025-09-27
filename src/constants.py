# from enum import Enum

# Its just easier to not deal with enums :)

# class Action(Enum):
ACTION_NOTHING = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4

GAME_STATE_ONGOING = 0
GAME_STATE_NON_FATAL_MISTAKE = 1
GAME_STATE_OVER = 2
# GAME_STATE_WILL_BE_OVER = 2

CAPTURE_WIDTH = 480
CAPTURE_HEIGHT = 800

CAPTURE_SCALE_FACTOR = 4

SCALED_HEIGHT = int(CAPTURE_HEIGHT / CAPTURE_SCALE_FACTOR)
SCALED_WIDTH = int(CAPTURE_WIDTH / CAPTURE_SCALE_FACTOR)

def scale_dimensions(x1, y1, x2, y2):
    return int(x1 / CAPTURE_SCALE_FACTOR), int(y1 / CAPTURE_SCALE_FACTOR), int(x2 / CAPTURE_SCALE_FACTOR), int(y2 / CAPTURE_SCALE_FACTOR)

def action_to_name(action):
    if (action == ACTION_NOTHING): return "NOTHING"
    if (action == ACTION_UP): return "UP"
    if (action == ACTION_DOWN): return "DOWN"
    if (action == ACTION_LEFT): return "LEFT"
    if (action == ACTION_RIGHT): return "RIGHT"
    
    return "N/A"