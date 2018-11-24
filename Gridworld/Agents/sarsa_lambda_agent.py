#!/usr/bin/env python
from __future__ import division
from Utils.utils import rand_in_range, rand_un
from Utils.tiles3 import IHT, tiles
import Globals.grid_agent_globals as a_globs
from Globals.generic_globals import *
from Utils.agent_helpers import *
from rl_glue import RL_num_episodes, RL_num_steps

from random import randint
import numpy as np
import random
import json
import platform

import matplotlib as mpl
if platform.system() == 'Darwin':
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
import matplotlib.pyplot as plt


def agent_init():

    a_globs.cur_epsilon = a_globs.EPSILON
    print("Epsilon at run start: {}".format(a_globs.cur_epsilon))

    a_globs.iht = IHT(a_globs.IHT_SIZE)
    a_globs.weights = np.array([random.uniform(-0.001, 0) for weight in range(a_globs.IHT_SIZE)])
    a_globs.weights = a_globs.weights[np.newaxis, :]
    a_globs.e_trace = np.zeros(a_globs.IHT_SIZE)
    a_globs.e_trace = a_globs.e_trace[np.newaxis, :]

def agent_start(state):
    a_globs.cur_state = state

    #Choose the next action, epislon-greedy style
    if rand_un() < 1 - a_globs.cur_epsilon:
        actions = [approx_value(a_globs.cur_state, action, a_globs.weights)[0] for action in range(a_globs.NUM_ACTIONS)]
        a_globs.cur_action = actions.index(max(actions))
    else:
        a_globs.cur_action = rand_in_range(a_globs.NUM_ACTIONS)

    return a_globs.cur_action

def agent_step(reward, state):

    next_state = state

    #Update delta and the eligibility trace
    delta = reward
    _, a_globs.cur_state_feature_indices = approx_value(a_globs.cur_state, a_globs.cur_action, a_globs.weights)
    for index in a_globs.cur_state_feature_indices:
        delta = delta - a_globs.weights[0][index]
        a_globs.e_trace[0][index] = 1

    #Choose the next action, epislon-greedy style
    if rand_un() < 1 - a_globs.cur_epsilon:
        actions = [approx_value(a_globs.cur_state, action, a_globs.weights)[0] for action in range(a_globs.NUM_ACTIONS)]
        next_action = actions.index(max(actions))
    else:
        next_action = rand_in_range(a_globs.NUM_ACTIONS)

    #Update the a_globs.weights
    _, next_state_feature_indices = approx_value(next_state, next_action, a_globs.weights)
    for index in next_state_feature_indices:
        delta = delta + a_globs.GAMMA * a_globs.weights[0][index]
    a_globs.weights += (a_globs.ALPHA / a_globs.NUM_TILINGS) * delta * a_globs.e_trace
    a_globs.e_trace = a_globs.GAMMA * a_globs.TRACE * a_globs.e_trace

    a_globs.cur_state = next_state
    a_globs.cur_action = next_action
    # print(state)
    # print(reward)
    # print(next_action)
    return a_globs.cur_action

def agent_end(reward):

    delta = reward
    feature_indices = approx_value(a_globs.cur_state, a_globs.cur_action, a_globs.weights)[1]
    for index in feature_indices:
        delta = delta - a_globs.weights[0][index]
        a_globs.e_trace[0][index] = 1
    a_globs.weights += (a_globs.ALPHA / a_globs.NUM_TILINGS) * delta * a_globs.e_trace
    return

def agent_cleanup():
    "Perform miscellaneous state management at the end of the current run"

    #Decay epsilon at the end of the episode
    a_globs.cur_epsilon = max(a_globs.EPSILON_MIN, a_globs.cur_epsilon - a_globs.EPSILON_DECAY_RATE)
    print('Cur epsilon at episode end: {}'.format(a_globs.cur_epsilon))
    return

def agent_message(in_message):
    "Retrieves the parameters from grid_exp.py, sent via the RL glue interface"

    if in_message[0] == 'PLOT':
        #Compute the values for use in the 3D plot
        if a_globs.ENV == CONTINUOUS:
            plot_range = in_message[1]
            return compute_state_action_values_continuous(plot_range)
        else:
            return compute_state_action_values_discrete()

    elif in_message[0] == 'GET_SNAPSHOT':
        pass

    else:
        params = json.loads(in_message)
        a_globs.EPSILON_MIN = params["EPSILON"]
        a_globs.ALPHA = params['ALPHA']
        a_globs.GAMMA = params['GAMMA']
        a_globs.TRACE = params['TRACE']
        a_globs.AGENT = params['AGENT']
        a_globs.IS_STOCHASTIC = params['IS_STOCHASTIC']
        a_globs.IS_1_HOT = params['IS_1_HOT']
        a_globs.ENV = params['ENV']

        if a_globs.ENV == GRID:
            a_globs.IS_DISCRETE = True
        else:
            a_globs.IS_DISCRETE = False
