#!/usr/bin/env python

from __future__ import division
from utils import rand_norm, rand_in_range, rand_un
import numpy as np
import json
import grid_env_globals as e_globs

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

    """
    If we are in an obstacle state with a stochastic environment, the effect of
    an action is random, so we resample an action uniformly at random to introduce
    stochasticity to the current state from the point of view of the agent
    """
    if e_globs.IS_STOCHASTIC and e_globs.current_state in e_globs.OBSTACLE_STATES:
        action = np.random.choice(e_globs.ACTION_SET, 1, p=[0.10, 0.10, 0.40, 0.40])

    #Change the state based on the agent action
    if action == e_globs.NORTH:
        e_globs.current_state = [cur_row + 1, cur_column]
    elif action == e_globs.EAST:
        e_globs.current_state = [cur_row, cur_column + 1]
    elif action == e_globs.SOUTH:
        e_globs.current_state = [cur_row - 1, cur_column]
    elif action == e_globs.WEST:
        e_globs.current_state = [cur_row, cur_column - 1]

    #Enforce the constraint that actions do not leave the grid world
    if e_globs.current_state[0] > e_globs.MAX_ROW:
        e_globs.current_state[0] = e_globs.MAX_ROW
    elif e_globs.current_state[0] < e_globs.MIN_ROW:
        e_globs.current_state[0] = e_globs.MIN_ROW

    if e_globs.current_state[1] > e_globs.MAX_COLUMN:
        e_globs.current_state[1] = e_globs.MAX_COLUMN
    elif e_globs.current_state[1] < e_globs.MIN_COLUMN:
        e_globs.current_state[1] = e_globs.MIN_COLUMN

    #Enforce the constraint that some squares are out of bounds, so we go nowhere if we try to step into them
    if not e_globs.IS_STOCHASTIC and e_globs.current_state in e_globs.OBSTACLE_STATES:
        e_globs.current_state = old_state

    if e_globs.IS_SPARSE:
        if e_globs.current_state == e_globs.GOAL_STATE:
            is_terminal = True
            reward = 1
        else:
            is_terminal = False
            reward = 0
    else:
        if e_globs.current_state == e_globs.GOAL_STATE:
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
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    params = json.loads(in_message)
    e_globs.IS_SPARSE = params['IS_SPARSE']
    e_globs.IS_STOCHASTIC = params['IS_STOCHASTIC']
    return
