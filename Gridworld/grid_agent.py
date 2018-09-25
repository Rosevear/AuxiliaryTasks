#!/usr/bin/env python

from __future__ import division
from collections import namedtuple
from utils import rand_in_range, rand_un
from random import randint

import numpy as np
import pickle
import random
import json
import platform

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, concatenate
from keras.initializers import he_normal
from keras.optimizers import RMSprop
from keras.utils import plot_model

from rl_glue import RL_num_episodes, RL_num_steps
import grid_agent_globals as a_globs

import matplotlib as mpl
if platform.system() == 'Darwin':
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
import matplotlib.pyplot as plt

#TODO: Refactor some of the neural network and auxiliary task code to reduce duplication
#TODO: Refactor the update exeprience replay code to reduce duplication
#TODO: Make the input type a user set parameter and then use functional programming to set which function is used to format the state

def agent_init():

    a_globs.cur_epsilon = a_globs.EPSILON
    print("Epsilon at run start: {}".format(a_globs.cur_epsilon))

    if a_globs.AGENT == a_globs.RANDOM:
        pass

    elif a_globs.AGENT == a_globs.TABULAR :
        a_globs.state_action_values = [[[0 for action in range(a_globs.NUM_ACTIONS)] for column in range(a_globs.NUM_COLUMNS)] for row in range(a_globs.NUM_ROWS)]
    elif a_globs.AGENT == a_globs.NEURAL:

        #Initialize the neural network
        a_globs.model = Sequential()
        init_weights = he_normal()

        a_globs.model.add(Dense(164, activation='relu', kernel_initializer=init_weights, input_shape=(a_globs.FEATURE_VECTOR_SIZE,)))
        a_globs.model.add(Dense(150, activation='relu', kernel_initializer=init_weights))
        a_globs.model.add(Dense(a_globs.NUM_ACTIONS, activation='linear', kernel_initializer=init_weights))

        rms = RMSprop(lr=a_globs.ALPHA)
        a_globs.model.compile(loss='mse', optimizer=rms)

    else:

        #Initialize the replay buffers for use by the auxiliary prediction tasks
        a_globs.non_zero_reward_buffer = []
        a_globs.zero_reward_buffer = []
        a_globs.non_zero_buffer_count = 0
        a_globs.zero_buffer_count = 0

        a_globs.deterministic_state_buffer = []
        a_globs.stochastic_state_buffer_count = []
        a_globs.deterministic_state_buffer_count = 0
        stochastic_state_buffer_count = 0

        if a_globs.AGENT == a_globs.REWARD:
            num_outputs = 1
            cur_activation = 'sigmoid'
            loss={'main_output': 'mean_squared_error', 'aux_output': 'binary_crossentropy'}

        elif a_globs.AGENT == a_globs.NOISE:
            num_outputs = a_globs.NUM_NOISE_NODES
            cur_activation = 'linear'
            loss={'main_output': 'mean_squared_error', 'aux_output': 'mean_squared_error'}

        elif a_globs.AGENT == a_globs.STATE :
            num_outputs = a_globs.FEATURE_VECTOR_SIZE
            cur_activation = 'softmax'
            loss={'main_output': 'mean_squared_error', 'aux_output': 'categorical_crossentropy'}

        elif a_globs.AGENT == a_globs.REDUNDANT:
            num_outputs = a_globs.NUM_ACTIONS * a_globs.NUM_REDUNDANT_TASKS
            cur_activation = 'linear'
            loss={'main_output': 'mean_squared_error', 'aux_output': 'mean_squared_error'}

        #Specify the a_globs.model
        init_weights = he_normal()
        if a_globs.AGENT == a_globs.REDUNDANT:
            main_input = Input(shape=(a_globs.FEATURE_VECTOR_SIZE * 2,))
        else:
            main_input = Input(shape=(a_globs.FEATURE_VECTOR_SIZE + (a_globs.AUX_FEATURE_VECTOR_SIZE * a_globs.N),))

        shared_1 = Dense(164, activation='relu', kernel_initializer=init_weights)(main_input)
        main_task_full_layer = Dense(150, activation='relu', kernel_initializer=init_weights)(shared_1)
        aux_task_full_layer = Dense(150, activation='relu', kernel_initializer=init_weights)(shared_1)

        main_output = Dense(a_globs.NUM_ACTIONS, activation='linear', kernel_initializer=init_weights, name='main_output')(main_task_full_layer)
        aux_output = Dense(num_outputs, activation=cur_activation, kernel_initializer=init_weights, name='aux_output')(aux_task_full_layer)

        #Initialize the model
        rms = RMSprop(lr=a_globs.ALPHA)
        loss_weights = {'main_output': 1.0, 'aux_output': 1.0}
        a_globs.model = Model(inputs=main_input, outputs=[main_output, aux_output])
        a_globs.model.compile(optimizer=rms, loss=loss, loss_weights=loss_weights)

        #Save a visual and textual summary of the a_globs.model
        plot_model(a_globs.model, to_file='{} agent a_globs.model.png'.format(a_globs.AGENT), show_shapes=True)
        with open('{} agent a_globs.model.txt'.format(a_globs.AGENT), 'w') as model_summary_file:
            # Pass the file handle in as a lambda function to make it callable
            a_globs.model.summary(print_fn=lambda x: model_summary_file.write(x + '\n'))


def agent_start(state):

    #Context is a sliding window of the previous n states that gets added to the replay buffer used by auxiliary tasks
    a_globs.cur_context = []
    a_globs.cur_context_actions = []
    a_globs.cur_state = state

    if rand_un() < 1 - a_globs.cur_epsilon:
        if a_globs.AGENT == a_globs.TABULAR :
            a_globs.cur_action = get_max_action_tabular(a_globs.cur_state)
        elif a_globs.AGENT == a_globs.NEURAL:
            a_globs.cur_action = get_max_action(a_globs.cur_state)
        elif a_globs.AGENT == a_globs.RANDOM:
            a_globs.cur_action = rand_in_range(a_globs.NUM_ACTIONS)
        else:
            a_globs.cur_action = get_max_action_aux(a_globs.cur_state)
    else:
        a_globs.cur_action = rand_in_range(a_globs.NUM_ACTIONS)
    return a_globs.cur_action


def agent_step(reward, state):

    next_state = state

    if a_globs.AGENT == a_globs.TABULAR :
        #Choose the next action, epsilon greedy style
        if rand_un() < 1 - a_globs.cur_epsilon:
            next_action = get_max_action_tabular(next_state)
        else:
            next_action = rand_in_range(a_globs.NUM_ACTIONS)

        #Update the state action values
        next_state_max_action = a_globs.state_action_values[next_state[0]][next_state[1]].index(max(a_globs.state_action_values[next_state[0]][next_state[1]]))
        a_globs.state_action_values[a_globs.cur_state[0]][a_globs.cur_state[1]][a_globs.cur_action] += a_globs.ALPHA * (reward + a_globs.GAMMA * a_globs.state_action_values[next_state[0]][next_state[1]][next_state_max_action] - a_globs.state_action_values[a_globs.cur_state[0]][a_globs.cur_state[1]][a_globs.cur_action])

    elif a_globs.AGENT == a_globs.NEURAL:
        #Get the best action over all actions possible in the next state, ie max_a(Q, a)
        next_state_coded = format_state(next_state)
        q_vals = a_globs.model.predict(next_state_coded, batch_size=1)
        q_max = np.max(q_vals)
        cur_action_target = reward + a_globs.GAMMA * q_max

        #Choose the next action, epsilon greedy style
        if rand_un() < 1 - a_globs.cur_epsilon:
            next_action = np.argmax(q_vals)
        else:
            next_action = rand_in_range(a_globs.NUM_ACTIONS)

        #Get the value for the current state of the action which was just taken ie Q(S, A)
        cur_state_coded = format_state(next_state)
        q_vals = a_globs.model.predict(cur_state_coded, batch_size=1)
        q_vals[0][a_globs.cur_action] = cur_action_target

        #Update the weights
        a_globs.model.fit(cur_state_coded, q_vals, batch_size=1, epochs=1, verbose=0)

    elif a_globs.AGENT == a_globs.RANDOM:
        next_action = rand_in_range(a_globs.NUM_ACTIONS)

    #All auxiliary tasks
    else:

        update_replay_buffer(a_globs.cur_state, a_globs.cur_action, reward, next_state)
        aux_dummy = set_up_empty_aux_input()
        next_state_coded = format_state(next_state)

        #Get the best action over all actions possible in the next state, ie max_a(Q(s + 1), a))
        q_vals, _ = a_globs.model.predict(np.concatenate([aux_dummy, next_state_coded], axis=1), batch_size=1)
        q_max = np.max(q_vals)
        cur_action_target = reward + (a_globs.GAMMA * q_max)

        #Choose the next action, epsilon greedy style
        if rand_un() < 1 - a_globs.cur_epsilon:
            next_action = np.argmax(q_vals)
        else:
            next_action = rand_in_range(a_globs.NUM_ACTIONS)

        #Get the learning target q-value for the current state
        cur_state_coded = format_state(a_globs.cur_state)
        q_vals, _ = a_globs.model.predict(np.concatenate([aux_dummy, cur_state_coded], axis=1), batch_size=1)
        q_vals[0][a_globs.cur_action] = cur_action_target

        #Sample a transition from the replay buffer to use for auxiliary task training
        cur_transition = None
        if a_globs.zero_reward_buffer and a_globs.non_zero_reward_buffer and a_globs.AGENT != a_globs.STATE :
            cur_transition = sample_from_buffers(a_globs.zero_reward_buffer, a_globs.non_zero_reward_buffer)
        elif a_globs.AGENT == a_globs.STATE  and a_globs.IS_STOCHASTIC:
            if a_globs.deterministic_state_buffer and a_globs.a_globs.stochastic_state_buffer_count:
                cur_transition = sample_from_buffers(a_globs.deterministic_state_buffer, a_globs.a_globs.stochastic_state_buffer_count)
        elif a_globs.AGENT == a_globs.STATE  and not a_globs.IS_STOCHASTIC:
            if a_globs.deterministic_state_buffer:
                cur_transition = sample_from_buffers(a_globs.deterministic_state_buffer)

        #Update the current q-value and auxiliary task output towards their respective targets
        if cur_transition:
            #Set the auxiliary input depending on the task
            if a_globs.AGENT == a_globs.REDUNDANT:
                aux_input = cur_state_coded
            else:
                aux_input = format_state_actions(cur_transition.states, cur_transition.actions)

            if a_globs.AGENT == a_globs.REWARD:
                #We make the rewards positive since we care only about the binary
                #distinction between zero and non zero rewards and theano binary
                #cross entropy loss requires targets to be 0 or 1
                aux_target = np.array([abs(cur_transition.reward)])
            elif a_globs.AGENT == a_globs.STATE :
                aux_target = format_state(cur_transition.next_state)
            elif a_globs.AGENT == a_globs.NOISE:
                aux_target = np.array([rand_un() for i in range(a_globs.NUM_NOISE_NODES)]).reshape(1, a_globs.NUM_NOISE_NODES)
            elif a_globs.AGENT == a_globs.REDUNDANT:
                nested_q_vals = [q_vals for i in range(a_globs.NUM_REDUNDANT_TASKS)]
                aux_target = np.array([item for sublist in nested_q_vals for item in sublist]).reshape(1, a_globs.NUM_ACTIONS * a_globs.NUM_REDUNDANT_TASKS)
            a_globs.model.fit(np.concatenate([aux_input, cur_state_coded], axis=1), {'main_output' : q_vals, 'aux_output' : aux_target}, batch_size=1, epochs=1, verbose=0)


    a_globs.cur_state = next_state
    a_globs.cur_action = next_action
    return next_action

def agent_end(reward):

    if a_globs.AGENT == a_globs.TABULAR :
        a_globs.state_action_values[a_globs.cur_state[0]][a_globs.cur_state[1]][a_globs.cur_action] += a_globs.ALPHA * (reward - a_globs.state_action_values[a_globs.cur_state[0]][a_globs.cur_state[1]][a_globs.cur_action])
    elif a_globs.AGENT == a_globs.NEURAL:
        #Update the network weights
        cur_state_coded = format_state(a_globs.cur_state)
        q_vals = a_globs.model.predict(cur_state_coded, batch_size=1)
        q_vals[0][a_globs.cur_action] = reward
        a_globs.model.fit(cur_state_coded, q_vals, batch_size=1, epochs=1, verbose=1)

    elif a_globs.AGENT == a_globs.RANDOM:
        pass

    #All auxiliary tasks
    else:
        update_replay_buffer(a_globs.cur_state, a_globs.cur_action, reward, a_globs.GOAL_STATE)

        #Get the Q-value for the current state
        aux_dummy = set_up_empty_aux_input()
        cur_state_coded = format_state(a_globs.cur_state)
        q_vals, _ = a_globs.model.predict(np.concatenate([aux_dummy, cur_state_coded], axis=1), batch_size=1)
        q_vals[0][a_globs.cur_action] = reward

        #Sample a transition from the replay buffer to use for auxiliary task training
        cur_transition = None
        if a_globs.zero_reward_buffer and a_globs.non_zero_reward_buffer and a_globs.AGENT != a_globs.STATE :
            cur_transition = sample_from_buffers(a_globs.zero_reward_buffer, a_globs.non_zero_reward_buffer)
        elif a_globs.AGENT == a_globs.STATE  and a_globs.IS_STOCHASTIC:
            if a_globs.deterministic_state_buffer and a_globs.a_globs.stochastic_state_buffer_count:
                cur_transition = sample_from_buffers(a_globs.deterministic_state_buffer, a_globs.a_globs.stochastic_state_buffer_count)
        elif a_globs.AGENT == a_globs.STATE  and not a_globs.IS_STOCHASTIC:
            if a_globs.deterministic_state_buffer:
                cur_transition = sample_from_buffers(a_globs.deterministic_state_buffer)

        #Update the current q-value and auxiliary task output towards their respective targets
        if cur_transition is not None:
            #Set the auxiliary input depending on the task
            if a_globs.AGENT == a_globs.REDUNDANT:
                aux_input = cur_state_coded
            else:
                aux_input = format_state_actions(cur_transition.states, cur_transition.actions)

            if a_globs.AGENT == a_globs.REWARD:
                #We make the rewards positive since we care only about the binary
                #distinction between zero and non zero rewards and theano binary
                #cross entropy loss requires targets to be 0 or 1
                aux_target = np.array([abs(cur_transition.reward)])
            elif a_globs.AGENT == a_globs.STATE :
                aux_target = format_state(cur_transition.next_state)
            elif a_globs.AGENT == a_globs.NOISE:
                aux_target = np.array([rand_un() for i in range(a_globs.NUM_NOISE_NODES)]).reshape(1, a_globs.NUM_NOISE_NODES)
            elif a_globs.AGENT == a_globs.REDUNDANT:
                nested_q_vals = [q_vals for i in range(a_globs.NUM_REDUNDANT_TASKS)]
                aux_target = np.array([item for sublist in nested_q_vals for item in sublist]).reshape(1, a_globs.NUM_ACTIONS * a_globs.NUM_REDUNDANT_TASKS)
            a_globs.model.fit(np.concatenate([aux_input, cur_state_coded], axis=1), {'main_output' : q_vals, 'aux_output' : aux_target}, batch_size=1, epochs=1, verbose=1)
    return

def agent_cleanup():

    #Decay epsilon at the end of the episode
    a_globs.cur_epsilon = max(a_globs.EPSILON_MIN, a_globs.cur_epsilon - (1 / (RL_num_episodes() + 1)))
    return

def agent_message(in_message):
    "Retrieves the parameters from grid_exp.py, sent via the RL glue interface"

    params = json.loads(in_message)
    a_globs.EPSILON_MIN = params["EPSILON"]
    a_globs.ALPHA = params['ALPHA']
    a_globs.GAMMA = params['GAMMA']
    a_globs.AGENT = params['AGENT']
    if 'N' in params:
        a_globs.N = params['N']
    a_globs.IS_STOCHASTIC = params['IS_STOCHASTIC']
    return

def get_max_action(state):
    "Return the maximum action to take given the current state"

    cur_state_coded = format_state(state)
    q_vals = a_globs.model.predict(cur_state_coded, batch_size=1)
    return np.argmax(q_vals[0])

def get_max_action_tabular(state):
    "Return the maximum action to take given the current state."

    #Need to ensure that an action is picked uniformly at random from among those that tie for maximum
    cur_max = a_globs.state_action_values[state[0]][state[1]][0]
    max_indices = [0]
    for i in range(1, len(a_globs.state_action_values[state[0]][state[1]])):
        if a_globs.state_action_values[state[0]][state[1]][i] > cur_max:
            cur_max = a_globs.state_action_values[state[0]][state[1]][i]
            max_indices = [i]
        elif a_globs.state_action_values[state[0]][state[1]][i] == cur_max:
            max_indices.append(i)
    next_action = max_indices[rand_in_range(len(max_indices))]
    return next_action

def get_max_action_aux(state):
    "Return the maximum acton to take given the current state"

    aux_dummy = set_up_empty_aux_input()
    cur_state_coded = format_state(state)
    q_vals, _ = a_globs.model.predict(np.concatenate([aux_dummy, cur_state_coded], axis=1), batch_size=1)

    return np.argmax(q_vals[0])

def update_replay_buffer(cur_state, cur_action, reward, next_state):
    """
    Update the replay buffer with the most recent transition, adding a_globs.cur_state to the current global historical context,
    and mapping that to reward and next_state if the current context == a_globs.a_globs.a_globs.a_globs.a_globs.N, the user set parameter for the context size
    """

    #Construct the historical context used in the prediciton tasks, and store them in the replay buffer according to their reward valence
    a_globs.cur_context.append(a_globs.cur_state)
    a_globs.cur_context_actions.append(a_globs.cur_action)
    cur_transition = None
    if len(a_globs.cur_context) == a_globs.N:
        #Construct the observation used by the auxiliary taks
        cur_transition = namedtuple("Transition", ["states", "actions", "reward", "next_state"])
        cur_transition.states = list(a_globs.cur_context)
        cur_transition.reward = reward
        cur_transition.next_state = next_state
        cur_transition.actions = list(a_globs.cur_context_actions)

        #Remove the oldest states from the context, to allow new ones to be added in a sliding window style
        a_globs.cur_context.pop(0)
        a_globs.cur_context_actions.pop(0)

    if cur_transition:
        if a_globs.AGENT == a_globs.STATE :
            if  cur_transition.states[-1] in a_globs.OBSTACLE_STATES:
                add_to_buffer(a_globs.a_globs.stochastic_state_buffer_count, cur_transition, stochastic_state_buffer_count)
                stochastic_state_buffer_count += 1
                if stochastic_state_buffer_count == a_globs.BUFFER_SIZE:
                    stochastic_state_buffer_count = 0
            else:
                add_to_buffer(a_globs.deterministic_state_buffer, cur_transition, a_globs.deterministic_state_buffer_count)
                a_globs.deterministic_state_buffer_count += 1
                if a_globs.deterministic_state_buffer_count == a_globs.BUFFER_SIZE:
                    a_globs.deterministic_state_buffer_count = 0
        else:
            if reward == 0:
                add_to_buffer(a_globs.zero_reward_buffer, cur_transition, a_globs.zero_buffer_count)
                a_globs.zero_buffer_count += 1
                if a_globs.zero_buffer_count == a_globs.BUFFER_SIZE:
                    a_globs.zero_buffer_count = 0
            else:
                add_to_buffer(a_globs.non_zero_reward_buffer, cur_transition, a_globs.non_zero_buffer_count)
                a_globs.non_zero_buffer_count += 1
                if a_globs.non_zero_buffer_count == a_globs.BUFFER_SIZE:
                    a_globs.non_zero_buffer_count = 0

def add_to_buffer(cur_buffer, to_add, buffer_count):
    """
    Add item to_add to cur_buffer at index buffer_count, otherwise append it to
    the end of the buffer in the case of buffer overflow
    """

    try:
        cur_buffer[buffer_count] = to_add
    except IndexError:
        cur_buffer.append(to_add)

def sample_from_buffers(buffer_one, buffer_two=None):
    """
    Sample a transiton uniformly at random from one of buffer_one and buffer_two.
    Which buffer is sampled is dependent on the current time step, and done in a
    way so as to sample equally from both buffers throughout an episode"
    """
    if RL_num_steps() % 2 == 0 or buffer_two is None:
        cur_transition = buffer_one[rand_in_range(len(buffer_one))]
    else:
        cur_transition = buffer_two[rand_in_range(len(buffer_two))]
    return cur_transition

def set_up_empty_aux_input():
    """
    Sets up empty auxiliary input of the correct dimensions and returns it.
    Used for when predicting the main output of networks with auxiliary tasks
    """

    if a_globs.AGENT == a_globs.REDUNDANT:
        aux_input = np.zeros(shape=(1, a_globs.FEATURE_VECTOR_SIZE,))
    else:
        aux_input = np.zeros(shape=(1, a_globs.AUX_FEATURE_VECTOR_SIZE * a_globs.N,))
    return aux_input

#NOTE: 1 hot encoding format
def format_state(state):
    "Return a one hot encoding of the current list of states"

    state = list(state)
    state_1_hot = np.zeros((a_globs.NUM_ROWS, a_globs.NUM_COLUMNS))
    state_1_hot[state[0]][state[1]] = 1
    state_1_hot_formatted = state_1_hot.reshape(1, a_globs.FEATURE_VECTOR_SIZE)

    return state_1_hot_formatted

def format_state_actions(states, actions):
    "Return a 1 hot encoding of the current list of states and the accompanying actions"

    all_states_1_hot = []
    for i in range(len(states)):
        state = states[i]
        action = actions[i]
        state_1_hot = np.zeros((a_globs.NUM_ROWS, a_globs.NUM_COLUMNS, a_globs.NUM_ACTIONS))
        state_1_hot[state[0]][state[1]][action] = 1
        state_1_hot = state_1_hot.reshape(1, a_globs.AUX_FEATURE_VECTOR_SIZE)
        all_states_1_hot.append(state_1_hot)

    return np.concatenate(all_states_1_hot, axis=1)

#NOTE: coordinate state formatting, with shifted origin
# def format_state(state):
#     """
#     Shift the x and y coordinates that define a state value so that it starts at (1, 1), from the point of new of the neural network.
#     We do this so that (0, 0) can be used as dummy auxiliary input when we want to predict just Q-vals from the multi-task neural network
#     We also turn the bare list representation into a numpy array to use with the network
#     """
#     #To guard against tuple inputs that are immutable
#     state = list(state)
#
#     state[0] += 1
#     state[1] += 1
#     formatted_state = np.array(state).reshape(1, a_globs.FEATURE_VECTOR_SIZE)
#
#     return formatted_state
#
# def format_state_actions(states, actions):
#
#     for i in range(len(states)):
#         states[i][0] += 1
#         states[i][1] += 1
#         actions[i] += 1
#
#     #flatten the states list
#     states = [coordinate for state in states for coordinate in state]
#     formatted_state_action = np.array(states + actions).reshape(1, a_globs.AUX_FEATURE_VECTOR_SIZE * a_globs.a_globs.a_globs.a_globs.a_globs.N)
#
#     return formatted_state_action
