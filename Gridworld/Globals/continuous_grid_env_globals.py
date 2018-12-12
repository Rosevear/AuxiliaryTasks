#Constants, per run
IS_SPARSE = None
IS_STOCHASTIC = None

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

ACTION_SET = [NORTH, EAST, SOUTH, WEST]
ACTION_EFFFECT_SIZE = 0.01

#Gaussian distributed noise
#On average, the agent acieves what it set out to do, so the man is 0
ACTION_NOISE_MEAN = 0
#W want ~99% of the noise to have an effect less than or equal to the action effect size
ACTION_NOISE_VARIANCE = 0.00333
#Unifromly distributed noise
ACTION_NOISE_UNIFORM_RANGE = 0.005

#Used to scale the range of the noise when using the uniform distribution for generating noise samples
#ACTION_NOISE_SCALING_FACTOR = 0.01

NUM_STATE_COORDINATES  = 2 #(x, y)

MIN_ROW = 0
MIN_COLUMN = 0
MAX_ROW = 1
MAX_COLUMN = 1

START_STATE = [0.50, 0.50]
GOAL_STATE = [1.0, 1.0]
GOAL_STATE_RELATIVE_TOLERANCE = GOAL_STATE_ABSOLUTE_TOLERANCE = 0.01

#Obstacle state info
#NOTE: Each obstacle is represented by the four corners of its rectangle,
#specified as a list of tuples, in the following order: bottom_left, top_left, top_right, bottom_right
#NOTE: Tuple Co-ordinates are in format (y, x), to map to (row, column) state representation in the environment file
# VERTICAL_MOVEMENT_OBSTACLES = []
# HORIZONTAL_MOVEMENT_OBSTACLES = []
VERTICAL_MOVEMENT_OBSTACLES = [[(0.625, 0.25), (0.75, 0.25), (0.75, 0.75), (0.625, 0.75)]]
HORIZONTAL_MOVEMENT_OBSTACLES = [[(0.25, 0.25), (0.75, 0.25), (0.75, 0.375), (0.25, 0.375)], [(0.25, 0.625), (0.75, 0.625), (0.75, 0.75), (0.25, 0.75)]]

#Variables
current_state = None
