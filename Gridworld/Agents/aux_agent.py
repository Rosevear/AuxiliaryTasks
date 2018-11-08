#!/usr/bin/env python

from __future__ import division

import Globals.grid_agent_globals as a_globs
import Globals.grid_env_globals as e_globs
import Globals.continuous_grid_env_globals as cont_e_globs
from Globals.generic_globals import *
from Utils.agent_helpers import *
from Utils.utils import rand_in_range, rand_un

from rl_glue import RL_num_episodes, RL_num_steps
from collections import namedtuple
from random import randint

import numpy as np
import pickle
import random
import json
import platform
import copy

from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Activation, Input, concatenate
from keras.initializers import he_normal, glorot_uniform
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model

import matplotlib as mpl
if platform.system() == 'Darwin':
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
import matplotlib.pyplot as plt

def agent_init():

    a_globs.cur_epsilon = a_globs.EPSILON
    print("Epsilon at run start: {}".format(a_globs.cur_epsilon))

    if a_globs.AGENT == a_globs.REWARD:
        num_outputs = 1
        cur_activation = 'sigmoid'
        loss={'main_output': 'mean_squared_error', 'aux_output': 'mean_squared_error'}
        a_globs.non_zero_reward_buffer = []
        a_globs.zero_reward_buffer = []
        a_globs.buffer_container = [a_globs.non_zero_reward_buffer, a_globs.zero_reward_buffer]

    elif a_globs.AGENT == a_globs.NOISE:
        num_outputs = a_globs.NUM_NOISE_NODES
        cur_activation = 'linear'
        loss={'main_output': 'mean_squared_error', 'aux_output': 'mean_squared_error'}

        a_globs.generic_buffer = []
        a_globs.buffer_container = [a_globs.generic_buffer]

    elif a_globs.AGENT == a_globs.STATE :
        num_outputs = a_globs.FEATURE_VECTOR_SIZE
        cur_activation = 'softmax'
        if a_globs.IS_1_HOT:
            loss={'main_output': 'mean_squared_error', 'aux_output': 'categorical_crossentropy'}
        else:
            loss={'main_output': 'mean_squared_error', 'aux_output': 'mean_squared_error'}

        a_globs.deterministic_state_buffer = []
        a_globs.stochastic_state_buffer = []
        if a_globs.IS_STOCHASTIC:
            a_globs.buffer_container = [a_globs.deterministic_state_buffer, a_globs.stochastic_state_buffer]
        else:
            a_globs.buffer_container = [a_globs.deterministic_state_buffer]

    elif a_globs.AGENT == a_globs.REDUNDANT:
        num_outputs = a_globs.NUM_ACTIONS * a_globs.NUM_REDUNDANT_TASKS
        cur_activation = 'linear'
        loss={'main_output': 'mean_squared_error', 'aux_output': 'mean_squared_error'}

        a_globs.generic_buffer = []
        a_globs.buffer_container = [a_globs.generic_buffer]

    #Specify the model
    init_weights = he_normal()
    main_input = Input(shape=(a_globs.FEATURE_VECTOR_SIZE,))
    shared_1 = Dense(164, activation='relu', kernel_initializer=init_weights, name='shared_1')(main_input)
    main_task_full_layer = Dense(150, activation='relu', kernel_initializer=init_weights, name='main_task_full_layer')(shared_1)
    aux_task_full_layer = Dense(150, activation='relu', kernel_initializer=init_weights)(shared_1)

    main_output = Dense(a_globs.NUM_ACTIONS, activation='linear', kernel_initializer=init_weights, name='main_output')(main_task_full_layer)
    aux_output = Dense(num_outputs, activation=cur_activation, kernel_initializer=init_weights, name='aux_output')(aux_task_full_layer)

    #Initialize the model
    loss_weights = {'main_output': 1.0, 'aux_output': a_globs.LAMBDA}
    a_globs.model = Model(inputs=main_input, outputs=[main_output, aux_output])
    a_globs.model.compile(optimizer=Adam(lr=a_globs.ALPHA), loss=loss, loss_weights=loss_weights)
    summarize_model(a_globs.model, a_globs.AGENT)

    #Create the target network to use in the update rule
    a_globs.target_network = clone_model(a_globs.model)
    a_globs.target_network.set_weights(a_globs.model.get_weights())


def agent_start(state):

    #Context is a sliding window of the previous n states that gets added to the replay buffer used by auxiliary tasks
    a_globs.cur_context = []
    a_globs.cur_context_actions = []
    a_globs.cur_state = state

    if rand_un() < 1 - a_globs.cur_epsilon:
        q_vals = get_q_vals_aux(a_globs.cur_state, False)
        a_globs.cur_action = np.argmax(q_vals[0])
    else:
        a_globs.cur_action = rand_in_range(a_globs.NUM_ACTIONS)
    return a_globs.cur_action


def agent_step(reward, state):

    next_state = state

    update_replay_buffer(a_globs.cur_state, a_globs.cur_action, reward, next_state)
    next_state_formatted = format_states([next_state])

    #Choose the next action, epsilon greedy style
    if rand_un() < 1 - a_globs.cur_epsilon:
        #Get the best action over all actions possible in the next state, ie max_a(Q(s + 1), a))
        q_vals = get_q_vals_aux(next_state, False)
        next_action = np.argmax(q_vals)
    else:
        next_action = rand_in_range(a_globs.NUM_ACTIONS)

    do_auxiliary_learning(a_globs.cur_state, next_state, reward)

    if RL_num_steps() % a_globs.NUM_STEPS_TO_UPDATE == 0:
        update_target_network()

    a_globs.cur_state = next_state
    a_globs.cur_action = next_action
    return next_action

def agent_end(reward):

    do_auxiliary_learning(a_globs.cur_state, None, reward)

    return

def agent_cleanup():
    "Perform miscellaneous state management at the end of the current run"

    #Decay epsilon at the end of the episode
    a_globs.cur_epsilon = max(a_globs.EPSILON_MIN, a_globs.cur_epsilon - (1 / (RL_num_episodes() + 1)))
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

        if a_globs.IS_1_HOT:
            a_globs.FEATURE_VECTOR_SIZE = e_globs.NUM_ROWS * e_globs.NUM_COLUMNS
        else:
            a_globs.FEATURE_VECTOR_SIZE = e_globs.NUM_STATE_COORDINATES

        #These parameters are for auxiliary tasks only, and always occur together
        if 'N' in params and 'LAMBDA' in params:
            a_globs.N = params['N']
            a_globs.LAMBDA = params['LAMBDA']
    return
