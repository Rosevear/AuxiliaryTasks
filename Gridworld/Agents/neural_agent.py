#!/usr/bin/env python

from __future__ import division

import Globals.grid_agent_globals as a_globs
from Globals.generic_globals import *
from Utils.agent_helpers import *
from Utils.utils import rand_in_range, rand_un

import numpy as np
import pickle
import random
import json
import platform
import copy

from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Activation, Input, concatenate
from keras.initializers import he_normal
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model

from rl_glue import RL_num_episodes, RL_num_steps

import matplotlib as mpl
if platform.system() == 'Darwin':
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
import matplotlib.pyplot as plt

def agent_init():

    a_globs.cur_epsilon = a_globs.EPSILON

    #Initialize a generic replay buffers
    a_globs.generic_buffer = []

    #Initialize the neural network
    a_globs.model = Sequential()
    init_weights = he_normal()

    a_globs.model.add(Dense(164, activation='relu', kernel_initializer=init_weights, input_shape=(a_globs.FEATURE_VECTOR_SIZE,)))
    a_globs.model.add(Dense(150, activation='relu', kernel_initializer=init_weights))
    a_globs.model.add(Dense(a_globs.NUM_ACTIONS, activation='linear', kernel_initializer=init_weights))

    a_globs.model.compile(loss='mse', optimizer=Adam(lr=a_globs.ALPHA))
    summarize_model(a_globs.model, a_globs.AGENT)

    #Create the target network
    a_globs.target_network = clone_model(a_globs.model)
    a_globs.target_network.set_weights(a_globs.model.get_weights())

def agent_start(state):

    #Context is a sliding window of the previous n states that gets added to the replay buffer used by auxiliary tasks
    a_globs.cur_context = []
    a_globs.cur_context_actions = []
    a_globs.cur_state = state

    if rand_un() < 1 - a_globs.cur_epsilon:
        a_globs.cur_action = get_max_action(a_globs.cur_state)
    else:
        a_globs.cur_action = rand_in_range(a_globs.NUM_ACTIONS)
    return a_globs.cur_action

def agent_step(reward, state):

    next_state = state
    next_state_formatted = format_states([next_state])
    update_replay_buffer(a_globs.cur_state, a_globs.cur_action, reward, next_state)

    #Choose the next action, epsilon greedy style
    if rand_un() < 1 - a_globs.cur_epsilon:
        #Get the best action over all actions possible in the next state, max_a(Q(s + 1), a))
        q_vals = a_globs.model.predict(next_state_formatted, batch_size=1)
        next_action = np.argmax(q_vals)
    else:
        next_action = rand_in_range(a_globs.NUM_ACTIONS)

    #Get the target value for the update from the target network
    q_vals = a_globs.target_network.predict(next_state_formatted, batch_size=1)
    cur_action_target = reward + a_globs.GAMMA * np.max(q_vals)

    #Get the value for the current state of the action which was just taken, ie Q(S, A),
    #and set the target for the specifc action taken (we need to pass in the
    #whole vector of q_values, since our network takes state only as input)
    cur_state_formatted = format_states([a_globs.cur_state])
    q_vals = a_globs.model.predict(cur_state_formatted, batch_size=1)
    q_vals[0][a_globs.cur_action] = cur_action_target


    #Update the weights
    a_globs.model.fit(cur_state_formatted, q_vals, batch_size=1, epochs=1, verbose=0)

    #Check and see if the relevant buffer is non-empty
    #TODO: Put an actual length check here instead of just sampling from the buffer
    observation_present = do_buffer_sampling()
    if observation_present and a_globs.SAMPLES_PER_STEP > 0:

        #print('I am replay buffer!')
        #Create the target training batch
        # batch_inputs = np.zeros(shape=(a_globs.SAMPLES_PER_STEP, a_globs.FEATURE_VECTOR_SIZE,))
        # batch_targets = np.zeros(shape=(a_globs.SAMPLES_PER_STEP, a_globs.NUM_ACTIONS))

        #Use the replay buffer to learn from previously visited states
        for i in range(a_globs.SAMPLES_PER_STEP):
            cur_observation = do_buffer_sampling()
            # print('states')
            # print(cur_observation.states)
            # print('actions')
            # print(cur_observation.actions)
            # print('reward')
            # print(cur_observation.reward)
            # print('next state')
            # print(cur_observation.next_state)
            #NOTE: For now If N > 1 we only want the most recent state associated with the reward and next state (effectively setting N > 1 changes nothing right now since we want to use the same input type as in the regular singel task case)
            #print('cur obs in learning')
            #print(cur_observation.states)
            most_recent_obs_state = cur_observation.states[-1]
            sampled_state_formatted = format_states([most_recent_obs_state])
            sampled_next_state_formatted = format_states([cur_observation.next_state])
            # print('sampled_state_formatted')
            # print(sampled_state_formatted)
            # print('sample state next formatted')
            # print(sampled_next_state_formatted)

            #Get the best action over all actions possible in the next state, ie max_a(Q(s + 1), a))
            q_vals = a_globs.target_network.predict(sampled_next_state_formatted, batch_size=1)
            cur_action_target = reward + (a_globs.GAMMA * np.max(q_vals))

            #Get the q_vals to adjust the learning target for the current action taken
            q_vals = a_globs.model.predict(sampled_state_formatted, batch_size=1)
            q_vals[0][a_globs.cur_action] = cur_action_target

            # batch_inputs[i] = sampled_state_formatted
            # batch_targets[i] = q_vals

            # print(sampled_state_formatted)
            # print(q_vals)
            a_globs.model.fit(sampled_state_formatted, q_vals, batch_size=1, epochs=1, verbose=0)

    if RL_num_steps() % a_globs.NUM_STEPS_TO_UPDATE == 0:
        update_target_network()

    a_globs.cur_state = next_state
    a_globs.cur_action = next_action
    return next_action

def agent_end(reward):

    #Update the network weights
    cur_state_formatted = format_states([a_globs.cur_state])
    q_vals = a_globs.model.predict(cur_state_formatted, batch_size=1)
    q_vals[0][a_globs.cur_action] = reward
    a_globs.model.fit(cur_state_formatted, q_vals, batch_size=1, epochs=1, verbose=1)

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
            a_globs.FEATURE_VECTOR_SIZE = a_globs.NUM_ROWS * a_globs.NUM_COLUMNS
            a_globs.AUX_FEATURE_VECTOR_SIZE = a_globs.FEATURE_VECTOR_SIZE * a_globs.NUM_ACTIONS
        else:
            a_globs.FEATURE_VECTOR_SIZE = 2
            a_globs.AUX_FEATURE_VECTOR_SIZE = a_globs.FEATURE_VECTOR_SIZE + 1

        #These parameters are for auxiliary tasks only, and always occur together
        if 'N' in params and 'LAMBDA' in params:
            a_globs.N = params['N']
            a_globs.LAMBDA = params['LAMBDA']
    return
