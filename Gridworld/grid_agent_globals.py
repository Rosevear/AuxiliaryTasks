#Constants, per run
#TODO: Move these to environment file
#ToDO: maybe just merge agent and enviro globals in one file them in one
NUM_ROWS = 6
NUM_COLUMNS = 9
GOAL_STATE = (5, 8)
OBSTACLE_STATES = [[2, 2], [3, 2], [4, 2], [1, 5], [3, 7], [4, 7], [5, 7]]

#Parameters
EPSILON = 1.0
LAMBDA = None
ALPHA = None
GAMMA = None
EPSILON_MIN = None
N = None
IS_STOCHASTIC = None
IS_1_HOT = None
NUM_ACTIONS = 4

FEATURE_VECTOR_SIZE = None
AUX_FEATURE_VECTOR_SIZE = None

#Used for sampling in the auxiliary tasks
BUFFER_SIZE = 2000

#Number of output nodes used in the noisy and redundant auxiliary tasks, respectively
NUM_NOISE_NODES = 10
NUM_REDUNDANT_TASKS = 5

#The number of times to run the auxiliary task during a single time step
SAMPLES_PER_STEP = 1

#How long to wait before updating the target networks
NUM_STEPS_TO_UPDATE = 1

HOT_SUFFIX = 'hot'
COORD_SUFFIX = 'coord'

BUFFER_SAMPLE_BIAS_PROBABILITY = 0.50

#Agents: non auxiliary task based
RANDOM = 'random'
NEURAL = 'neural'
TABULAR = 'tabularQ'

#Agents: auxiliary task based
REWARD = 'reward'
STATE = 'state'
REDUNDANT = 'redundant'
NOISE = 'noise'
AGENT = None

#Variables
state_action_values = None
observed_state_action_pairs = None
observed_states = None
model = None
target_network = None
cur_epsilon = None
zero_reward_buffer = None
zero_buffer_count = None
non_zero_reward_buffer = None
non_zero_buffer_count = None
deterministic_state_buffer = None
deterministic_state_buffer_count = None
stochastic_state_buffer = None
stochastic_state_buffer_count = None
cur_state = None
cur_action = None
cur_context = None
cur_context_actions = None
