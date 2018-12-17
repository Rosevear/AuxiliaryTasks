from collections import namedtuple

#Directory locations for the agents and environments, relative to the Gridworld directory
AGENT_DIR = 'Agents'
ENV_DIR = 'Envs'
MODELS_DIR = 'Models/'
RESULTS_DIR = 'Results/'

#Environments
GRID = "grid"
CONTINUOUS = 'continuous_grid'
WINDY = 'windy_grid'

#MISC
GRAPH_COLOURS = ('r', 'g', 'b', 'c', 'm', 'y', 'k', 'w')
GRAPH_COLOUR_CYCLE_PREFIX = 'C'

VALID_MOVE_SETS = [4, 8, 9]
NUM_AUX_AGENT_PARAMS = 2

LOG_FILE_NAME = 'sweep_log'

Results_tuple = namedtuple("Results_tuple", ["plot_title", "agent_type", "data", "x_values", "x_value_frequency", "x_label", "x_max_val", "y_label", "y_max_val"])
Transition = namedtuple("Transition", ["states", "actions", "reward", "next_state"])

NORMAL_POINT = 'b'
EMPHASIS_POINT = 'r'
