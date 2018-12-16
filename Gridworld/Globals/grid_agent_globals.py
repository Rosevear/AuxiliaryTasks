#Parameters
EPSILON = 1.0
LAMBDA = None
ALPHA = None
GAMMA = None
EPSILON_MIN = None
EPSILON_DECAY_RATE = 0.05
N = None

NUM_NERONS_LAYER_1 = 50
NUM_NERONS_LAYER_2 = 50

#For Sarsa agent specifically
TRACE = 0.90
NUM_TILINGS = 8
IHT_SIZE = 4096
TILE_WIDTH = 1/4

IS_STOCHASTIC = None
IS_1_HOT = None
NUM_ACTIONS = 4
IS_DISCRETE = None

FEATURE_VECTOR_SIZE = None
AUX_FEATURE_VECTOR_SIZE = None

OPTIMIZER = 'RMSprop'
INIT = 'glorot'

#Number of output nodes used in the noisy and redundant auxiliary tasks, respectively
NUM_NOISE_NODES = 10
NUM_REDUNDANT_TASKS = 5

#Used for sampling in the auxiliary tasks
BUFFER_SIZE = 1
BUFFER_SAMPLE_BIAS_PROBABILITY = 0.50

#The number of samples to select from the replay buffer on each time step
BATCH_SIZE = 1

#How long to wait before updating the target networks
NUM_STEPS_TO_UPDATE = 1

HOT_SUFFIX = 'hot'
COORD_SUFFIX = 'coord'

#Agents: non auxiliary task based
RANDOM = 'random'
NEURAL = 'neural'
TABULAR = 'tabular'
SARSA_LAMBDA = 'sarsa_lambda'

#Agents: auxiliary task based
REWARD = 'reward'
STATE = 'state'
REDUNDANT = 'redundant'
NOISE = 'noise'
AGENT = None

ENV = None

#General Variables
cur_epsilon = None
cur_state = None
cur_action = None
cur_context = None
is_trial_episode = False

#The buffer that holds all of the sub buffers (which are used to do biased sampling)
buffer_container = None

#Buffer for single-task neural network
generic_buffer = None

#Sarsa lambda agent
weights = None
iht = None
e_trace = None

#Variables: Auxiliary Neural Networks
state_action_values = None
observed_state_action_pairs = None
observed_states = None
model = None
target_network = None

#Buffers for various auxiliary tasks
zero_reward_buffer = None
non_zero_reward_buffer = None
deterministic_state_buffer = None
stochastic_state_buffer = None
cur_context_actions = None
