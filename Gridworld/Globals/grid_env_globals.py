#Constants, per run
IS_SPARSE = None
IS_STOCHASTIC = None

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

ACTION_SET = [NORTH, EAST, SOUTH, WEST]

MAX_ROW = 5
MAX_COLUMN = 8
MIN_ROW = 0
MIN_COLUMN = 0

NUM_ROWS = 6
NUM_COLUMNS = 9

START_STATE = [3, 0]
GOAL_STATE = [5, 8]
OBSTACLE_STATES = [[2, 2], [3, 2], [4, 2], [1, 5], [3, 7], [4, 7], [5, 7]]
NUM_STATE_COORDINATES  = 2 #(x, y)

#Variables
current_state = None
