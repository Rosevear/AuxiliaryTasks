#!/usr/bin/env python

from __future__ import division

import Globals.grid_agent_globals as a_globs
import Globals.grid_env_globals as e_globs
import Globals.continuous_grid_env_globals as cont_e_globs
from Globals.generic_globals import *
from Utils.agent_helpers import *
from Utils.utils import rand_in_range, rand_un

import json
import platform

from rl_glue import RL_num_episodes, RL_num_steps

import matplotlib as mpl
if platform.system() == 'Darwin':
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
import matplotlib.pyplot as plt

def agent_init():

    a_globs.cur_epsilon = a_globs.EPSILON
    a_globs.state_action_values = [[[0 for action in range(a_globs.NUM_ACTIONS)] for column in range(e_globs.NUM_COLUMNS)] for row in range(e_globs.NUM_ROWS)]


def agent_start(state):

    a_globs.cur_state = state

    if rand_un() < 1 - a_globs.cur_epsilon or a_globs.is_trial_episode:
        a_globs.cur_action = get_max_action_tabular(a_globs.cur_state)
    else:
        a_globs.cur_action = rand_in_range(a_globs.NUM_ACTIONS)
    return a_globs.cur_action

def agent_step(reward, state):

    next_state = state

    #Choose the next action, epsilon greedy style
    if rand_un() < 1 - a_globs.cur_epsilon or a_globs.is_trial_episode:
        next_action = get_max_action_tabular(next_state)
    else:
        next_action = rand_in_range(a_globs.NUM_ACTIONS)

    #Update the state action values
    if not a_globs.is_trial_episode:
        next_state_max_action = a_globs.state_action_values[next_state[0]][next_state[1]].index(max(a_globs.state_action_values[next_state[0]][next_state[1]]))
        a_globs.state_action_values[a_globs.cur_state[0]][a_globs.cur_state[1]][a_globs.cur_action] += a_globs.ALPHA * (reward + a_globs.GAMMA * a_globs.state_action_values[next_state[0]][next_state[1]][next_state_max_action] - a_globs.state_action_values[a_globs.cur_state[0]][a_globs.cur_state[1]][a_globs.cur_action])

    a_globs.cur_state = next_state
    a_globs.cur_action = next_action
    return next_action

def agent_end(reward):

    if not a_globs.is_trial_episode:
        a_globs.state_action_values[a_globs.cur_state[0]][a_globs.cur_state[1]][a_globs.cur_action] += a_globs.ALPHA * (reward - a_globs.state_action_values[a_globs.cur_state[0]][a_globs.cur_state[1]][a_globs.cur_action])

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

    else:
        params = json.loads(in_message)
        a_globs.EPSILON_MIN = params["EPSILON"]
        a_globs.ALPHA = params['ALPHA']
        a_globs.GAMMA = params['GAMMA']
        a_globs.AGENT = params['AGENT']
        a_globs.IS_STOCHASTIC = params['IS_STOCHASTIC']
        a_globs.IS_1_HOT = params['IS_1_HOT']
        a_globs.ENV = params['ENV']

        #These parameters are for auxiliary tasks only, and always occur together
        if 'N' in params and 'LAMBDA' in params:
            a_globs.N = params['N']
            a_globs.LAMBDA = params['LAMBDA']
    return
