#!/usr/bin/env python
from __future__ import division
from utils import rand_in_range, rand_un
from random import randint
import numpy as np
import random
import json
from tiles3 import IHT, tiles
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

#Parameters
EPSILON = None
GAMMA = 1
LAMBDA = None
NUM_TILINGS = None
ALPHA = None
IHT_SIZE = None

#Actions
NUM_ACTIONS = 3
DECELERATE = 0
COAST = 1
ACCELERATE = 2

#STATE RANGES
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

def agent_init():
    global weights, iht, e_trace

    iht = IHT(IHT_SIZE)
    weights = np.array([random.uniform(-0.001, 0) for weight in range(IHT_SIZE)])
    weights = weights[np.newaxis, :]
    e_trace = np.zeros(IHT_SIZE)
    e_trace = e_trace[np.newaxis, :]

def agent_start(state):
    global cur_state, cur_action, weights
    cur_state = state

    #Choose the next action, epislon-greedy style
    if rand_un() < 1 - EPSILON:
        actions = [approx_value(cur_state, action, weights)[0] for action in range(NUM_ACTIONS)]
        cur_action = actions.index(max(actions))
    else:
        cur_action = rand_in_range(NUM_ACTIONS)

    return cur_action

def agent_step(reward, state):
    global cur_state, cur_action, weights, e_trace

    next_state = state

    #Update the weights
    delta = reward
    cur_state_feature_indices = approx_value(cur_state, cur_action, weights)[1]
    for index in cur_state_feature_indices:
        delta = delta - weights[0][index]
        e_trace[0][index] = 1

    #Choose the next action, epislon-greedy style
    if rand_un() < 1 - EPSILON:
        actions = [approx_value(cur_state, action, weights)[0] for action in range(NUM_ACTIONS)]
        next_action = actions.index(max(actions))
    else:
        next_action = rand_in_range(NUM_ACTIONS)

    next_state_feature_indices = approx_value(next_state, next_action, weights)[1]
    for index in next_state_feature_indices:
        delta = delta + GAMMA * weights[0][index]
    weights += ALPHA * delta * e_trace
    e_trace = GAMMA * LAMBDA * e_trace

    cur_state = next_state
    cur_action = next_action
    return cur_action

def agent_end(reward):
    global weights, e_trace

    delta = reward
    feature_indices = approx_value(cur_state, cur_action, weights)[1]
    for index in feature_indices:
        delta = delta - weights[0][index]
        e_trace[0][index] = 1
    weights += ALPHA * delta * e_trace
    return

def agent_cleanup():
    """
    Does nothing
    """
    return

def agent_message(in_message): # returns string, in_message: string
    global EPSILON, ALPHA, LAMBDA, NUM_TILINGS, IHT_SIZE
    if in_message[0] == 'PARAMS':
        params = json.loads(in_message[1])
        EPSILON = params["EPSILON"]
        LAMBDA = params['LAMBDA']
        NUM_TILINGS = params['NUM_TILINGS']
        ALPHA = params['ALPHA'] / NUM_TILINGS
        IHT_SIZE = params['IHT_SIZE']
    elif in_message[0] == 'PLOT':
        #Compute the values for use in the 3D plot
        plot_range = in_message[1]
        plot_values = []
        position_values = []
        velocity_values = []
        for position in range(plot_range):
            scaled_position = POSITION_MIN + (position * (POSITION_MAX - POSITION_MIN) / plot_range)
            for velocity in range(plot_range):
                #TODO: Check if we have to call the tile coder directly: the scaling
                #code below might be like what is already done internally in approx value
                #instead of just being a way to cycle through valid state values
                scaled_velocity = VELOCITY_MIN + (velocity * (VELOCITY_MAX - VELOCITY_MIN) / plot_range)
                cur_state = [scaled_position, scaled_velocity]
                best_action_val = -max([approx_value(cur_state, action, weights)[0] for action in range(NUM_ACTIONS)])
                position_values.append(scaled_position)
                velocity_values.append(scaled_velocity)
                plot_values.append(best_action_val)
        return np.array(position_values)[:, np.newaxis], np.array(velocity_values)[:, np.newaxis], np.array(plot_values)[:, np.newaxis]

def approx_value(state, action, weights):
    global iht
    """
    Return the current approximated value for state and action given weights,
    and the indices for the active features for the  for the state action pair.
    """
    scaled_position = NUM_TILINGS * state[0] / (POSITION_MAX + abs(POSITION_MIN))
    scaled_velocity = NUM_TILINGS * state[1] / (VELOCITY_MAX + abs(VELOCITY_MIN))
    cur_tiles = tiles(iht, NUM_TILINGS, [scaled_position, scaled_velocity], [action])
    estimate = 0
    for tile in cur_tiles:
        estimate += weights[0][tile]
    return (estimate, cur_tiles)
