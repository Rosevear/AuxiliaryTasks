#!/usr/bin/env python

from __future__ import division

import Globals.continuous_grid_env_globals as e_globs
from Utils.utils import rand_norm, rand_in_range, rand_un

import numpy as np
import json

def env_init():
    return

def env_start():
    e_globs.current_state
    e_globs.current_state = e_globs.START_STATE
    return e_globs.current_state

def env_step(action):

    if not action in e_globs.ACTION_SET:
        print "Invalid action taken!!"
        print "action : ", action
        print "e_globs.current_state : ", e_globs.current_state
        exit(1)


    old_state = e_globs.current_state
    cur_row = e_globs.current_state[0]
    cur_column = e_globs.current_state[1]

    #Change the state based on the agent action and some noise
    noise = rand_norm(e_globs.ACTION_NOISE_MEAN, e_globs.ACTION_NOISE_VARIANCE)
    state_modifier = e_globs.ACTION_EFFFECT_SIZE + noise
    if action == e_globs.NORTH:
        e_globs.current_state = [cur_row + state_modifier, cur_column]
    elif action == e_globs.EAST:
        e_globs.current_state = [cur_row, cur_column + state_modifier]
    elif action == e_globs.SOUTH:
        e_globs.current_state = [cur_row - state_modifier, cur_column]
    elif action == e_globs.WEST:
        e_globs.current_state = [cur_row, cur_column - state_modifier]

    #Enforce the constraint that actions do not leave the grid world, except for when they lead to exceeding the goal state
    if not is_goal_state(e_globs.current_state):
        if e_globs.current_state[0] > e_globs.MAX_ROW:
            e_globs.current_state[0] = e_globs.MAX_ROW
        elif e_globs.current_state[0] < e_globs.MIN_ROW:
            e_globs.current_state[0] = e_globs.MIN_ROW

        if e_globs.current_state[1] > e_globs.MAX_COLUMN:
            e_globs.current_state[1] = e_globs.MAX_COLUMN
        elif e_globs.current_state[1] < e_globs.MIN_COLUMN:
            e_globs.current_state[1] = e_globs.MIN_COLUMN

    #Enforce the constraint that some squares are out of bounds, so we go nowhere if we try to step into or through them
    if agent_is_blocked(old_state, e_globs.current_state):
       e_globs.current_state = old_state

    # print('old state')
    # print(old_state)
    # print('action')
    # print(action)
    # print('new state')
    # print(e_globs.current_state)

    #Set the reward structure for the environment
    if e_globs.IS_SPARSE:
        if is_goal_state(e_globs.current_state):
            is_terminal = True
            reward = 1
        else:
            is_terminal = False
            reward = 0
    else:
        if is_goal_state(e_globs.current_state):
            is_terminal = True
            reward = 0
        else:
            is_terminal = False
            reward = -1

    result = {"reward": reward, "state": e_globs.current_state, "isTerminal": is_terminal}

    return result

def env_cleanup():
    return

def env_message(in_message):
    """
    """
    params = json.loads(in_message)
    e_globs.IS_SPARSE = params['IS_SPARSE']

    return

def is_goal_state(state):
    """
    Return whether the agent is appropriately close enough to the goal state to
    terminate the current episode.
    """
    return np.isclose(state[0], e_globs.GOAL_STATE[0], rtol=e_globs.GOAL_STATE_RELATIVE_TOLERANCE, atol=e_globs.GOAL_STATE_ABSOLUTE_TOLERANCE, equal_nan=False) and np.isclose(state[1], e_globs.GOAL_STATE[1], rtol=0.001, atol=0.001, equal_nan=False)

#NOTE: What about edge cases where you walk into the top/bottom of a horizontal obstacle?
#To handle this, we check all obstacle types, rather than differentiating based on the action taken
def agent_is_blocked(start_state, destination_state):
    """
    Return True iff the current attempt to reach destination_state is being blocked by an obstacle between it and start_state
    """

    return is_blocked_vertically(start_state, destination_state) or is_blocked_horizontally(start_state, destination_state)

#TODO: The near duplication here is a code smell. Look into refactoring this
def is_blocked_vertically(start_state, destination_state):
    "Return True iff there is a vertical obstacle between start_state and destination_state"

    #NOTE: Each obstacle is represented by the four corners of its rectangle,
    #pecified as a list of tuples, in the following order: bottom_left, top_left, top_right, bottom_right
    #NOTE: Tuple Co-ordinates are in format (y, x), to map to (row, column) state representation in the environment file

    # print('vert')
    # print('start')
    # print(start_state)
    # print('destination')
    # print(destination_state)

    for obstacle in e_globs.VERTICAL_MOVEMENT_OBSTACLES:
        obstacle_bottom_left_pt = obstacle[0]
        obstacle_top_left_pt = obstacle[1]
        obstacle_bottom_right_pt = obstacle[3]
        # print('bottom left reference')
        # print(obstacle_bottom_left_pt)
        # print('top left reference')
        # print(obstacle_top_left_pt)
        #Check we do not go through the bottom of the obstacle
        if (start_state[0] < obstacle_bottom_left_pt[0] and destination_state[0] >= obstacle_bottom_left_pt[0]) and (start_state[1] >= obstacle_bottom_left_pt[1] and start_state[1] <= obstacle_bottom_right_pt[1]):
            # print('Blocked vertically from above!')
            return True
        #Check that we do not go through the top of the obstacle
        elif (start_state[0] > obstacle_top_left_pt[0] and destination_state[0] <= obstacle_top_left_pt[0]) and (start_state[1] >= obstacle_bottom_left_pt[1] and start_state[1] <= obstacle_bottom_right_pt[1]):
            # print('Blocked vertically from below!')
            return True

    return False


def is_blocked_horizontally(start_state, destination_state):
    "Return True iff there is a horizontal obstacle between start_state and destination_state"

    #NOTE: Each obstacle is represented by the four corners of its rectangle,
    #specified as a list of tuples, in the following order: bottom_left, top_left, top_right, bottom_right
    #NOTE: Tuple Co-ordinates are in format (y, x), to map to (row, column) state representation in the environment file

    # print('horizontal')
    # print('start')
    # print(start_state)
    # print('destination')
    # print(destination_state)

    for obstacle in e_globs.HORIZONTAL_MOVEMENT_OBSTACLES:
        obstacle_bottom_left_pt = obstacle[0]
        obstacle_top_left_pt = obstacle[1]
        obstacle_bottom_right_pt = obstacle[3]
        # print('bottom left reference')
        # print(obstacle_bottom_left_pt)
        # print('top left reference')
        # print(obstacle_top_left_pt)
        # print('bottom right reference')
        # print(obstacle_bottom_right_pt)
        #Check we do not go through the obstacle left to right
        if (start_state[1] < obstacle_bottom_left_pt[1] and destination_state[1] >= obstacle_bottom_left_pt[1]) and (start_state[0] >= obstacle_bottom_left_pt[0] and start_state[0] <= obstacle_top_left_pt[0]):
            # print('Blocked horizontally to the right!')
            return True
        #Check that we do not go through the obstacle right to left
        elif (start_state[1] > obstacle_bottom_right_pt[1] and destination_state[1] <= obstacle_bottom_right_pt[1]) and (start_state[0] >= obstacle_bottom_left_pt[0] and start_state[0] <= obstacle_top_left_pt[0]):
            # print('Blocked horizontally to the left!')
            return True

    return False
