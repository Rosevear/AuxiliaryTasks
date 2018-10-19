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
import copy

from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Activation, Input, concatenate
from keras.initializers import he_normal
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model

from rl_glue import RL_num_episodes, RL_num_steps
import grid_agent_globals as a_globs

import matplotlib as mpl
if platform.system() == 'Darwin':
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
import matplotlib.pyplot as plt

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

        a_globs.model.compile(loss='mse', optimizer=Adam(lr=a_globs.ALPHA))
        summarize_model(a_globs.model, a_globs.AGENT)

        #Create the target network
        a_globs.target_network = clone_model(a_globs.model)
        a_globs.target_network.set_weights(a_globs.model.get_weights())

    else:

        #Initialize the replay buffers for use by the auxiliary prediction tasks
        a_globs.non_zero_reward_buffer = []
        a_globs.zero_reward_buffer = []

        a_globs.deterministic_state_buffer = []
        a_globs.stochastic_state_buffer = []

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
            if a_globs.IS_1_HOT:
                loss={'main_output': 'mean_squared_error', 'aux_output': 'categorical_crossentropy'}
            else:
                loss={'main_output': 'mean_squared_error', 'aux_output': 'mean_squared_error'}

        elif a_globs.AGENT == a_globs.REDUNDANT:
            num_outputs = a_globs.NUM_ACTIONS * a_globs.NUM_REDUNDANT_TASKS
            cur_activation = 'linear'
            loss={'main_output': 'mean_squared_error', 'aux_output': 'mean_squared_error'}

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
        if a_globs.AGENT == a_globs.TABULAR :
            a_globs.cur_action = get_max_action_tabular(a_globs.cur_state)
        elif a_globs.AGENT == a_globs.NEURAL:
            a_globs.cur_action = get_max_action(a_globs.cur_state)
        elif a_globs.AGENT == a_globs.RANDOM:
            a_globs.cur_action = rand_in_range(a_globs.NUM_ACTIONS)
        else:
            q_vals = get_q_vals_aux(a_globs.cur_state, False)
            a_globs.cur_action = np.argmax(q_vals[0])
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

        next_state_formatted = format_states([next_state])

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
        #whole vector of q_values, since our network takes state and action as input)
        cur_state_formatted = format_states([a_globs.cur_state])
        q_vals = a_globs.model.predict(cur_state_formatted, batch_size=1)
        q_vals[0][a_globs.cur_action] = cur_action_target

        # print(a_globs.cur_state)
        # print('cur q vals')
        # print(q_vals)

        #Update the weights
        a_globs.model.fit(cur_state_formatted, q_vals, batch_size=1, epochs=1, verbose=0)

        if RL_num_steps() % a_globs.NUM_STEPS_TO_UPDATE == 0:
            update_target_network()

    elif a_globs.AGENT == a_globs.RANDOM:
        next_action = rand_in_range(a_globs.NUM_ACTIONS)

    #All auxiliary tasks
    else:

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

    if a_globs.AGENT == a_globs.TABULAR :
        a_globs.state_action_values[a_globs.cur_state[0]][a_globs.cur_state[1]][a_globs.cur_action] += a_globs.ALPHA * (reward - a_globs.state_action_values[a_globs.cur_state[0]][a_globs.cur_state[1]][a_globs.cur_action])
    elif a_globs.AGENT == a_globs.NEURAL:

        #Update the network weights
        cur_state_formatted = format_states([a_globs.cur_state])
        q_vals = a_globs.model.predict(cur_state_formatted, batch_size=1)
        q_vals[0][a_globs.cur_action] = reward
        a_globs.model.fit(cur_state_formatted, q_vals, batch_size=1, epochs=1, verbose=1)

    elif a_globs.AGENT == a_globs.RANDOM:
        pass

    #All auxiliary tasks
    else:
        update_replay_buffer(a_globs.cur_state, a_globs.cur_action, reward, a_globs.GOAL_STATE)
        do_auxiliary_learning(a_globs.cur_state, None, reward)

    return

def agent_cleanup():
    "Perform miscellaneous state management at the end of the current run"

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
    a_globs.IS_STOCHASTIC = params['IS_STOCHASTIC']
    a_globs.IS_1_HOT = params['IS_1_HOT']

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

def update_target_network():
    "Updates the network currently being used as the target so that it reflects the current network that is learning"

    a_globs.target_network.set_weights(a_globs.model.get_weights())

def get_max_action(state):
    "Return the maximum action to take given the current state"

    cur_state_formatted = format_states([state])
    q_vals = a_globs.model.predict(cur_state_formatted, batch_size=1)
    return np.argmax(q_vals[0])

def summarize_model(model, agent):
    "Save a visual and textual summary of the current neural network model"

    if a_globs.IS_1_HOT:
        suffix = a_globs.HOT_SUFFIX
    else:
        suffix = a_globs.COORD_SUFFIX

    plot_model(model, to_file='{} agent model {}.png'.format(agent, suffix), show_shapes=True)
    with open('{} agent model {}.txt'.format(agent, suffix), 'w') as model_summary_file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: model_summary_file.write(x + '\n'))


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

def get_q_vals_aux(state, use_target_network):
    "Return the maximum action to take given the current state. If use_target_netowkr is True the results will come from that network"

    cur_state_formatted = format_states([state])
    if use_target_network:
        q_vals, _ = a_globs.target_network.predict(np.concatenate([cur_state_formatted], axis=1), batch_size=1)
    else:
        q_vals, _ = a_globs.model.predict(np.concatenate([cur_state_formatted], axis=1), batch_size=1)

    return q_vals


def do_auxiliary_learning(cur_state, next_state, reward):
    "Update the weights for the auxiliary network based on both the current interaction with the environment and sampling from experience replay"

    # if RL_num_episodes() % 100 == 0:
    #     is_verbose = 1
    # else:
    #     is_verbose = 0

    is_verbose = 0

    #Perform direct learning on the current state and auxiliary information
    q_vals = get_q_vals_aux(cur_state, False)
    if next_state:
        #Get the best action over all actions possible in the next state, ie max_a(Q(s + 1), a))
        q_vals_next = get_q_vals_aux(next_state, True)
        cur_action_target = reward + (a_globs.GAMMA * np.max(q_vals_next))
        q_vals[0][a_globs.cur_action] = cur_action_target

    else:
        q_vals[0][a_globs.cur_action] = reward

    if a_globs.AGENT == a_globs.REWARD:
        #We make the rewards positive since we care only about the binary
        #distinction between zero and non zero rewards and theano binary
        #cross entropy loss requires targets to be 0 or 1
        aux_target = np.array([abs(reward)])
    elif a_globs.AGENT == a_globs.STATE :
        if next_state:
            aux_target = format_states([next_state])
        else:
            #If there is no next state represent the lack of such a state with the zero vector (since 1 hot encoding or shifted x y coordinates will not use this to refer to any actual state)
            aux_target = np.zeros(shape=(1, a_globs.FEATURE_VECTOR_SIZE,))
    elif a_globs.AGENT == a_globs.NOISE:
        aux_target = np.array([rand_un() for i in range(a_globs.NUM_NOISE_NODES)]).reshape(1, a_globs.NUM_NOISE_NODES)
    elif a_globs.AGENT == a_globs.REDUNDANT:
        nested_target = [q_vals for i in range(a_globs.NUM_REDUNDANT_TASKS)]
        aux_target = np.array([item for sublist in nested_target for item in sublist]).reshape(1, a_globs.NUM_ACTIONS * a_globs.NUM_REDUNDANT_TASKS)

    cur_state_formatted = format_states([cur_state])
    a_globs.model.fit(cur_state_formatted, {'main_output' : q_vals, 'aux_output' : aux_target}, batch_size=1, epochs=1, verbose=is_verbose)

    #Use the replay buffer to learn from previously visitied states
    for i in range(a_globs.SAMPLES_PER_STEP):
        cur_observation = do_buffer_sampling()
        if cur_observation:
            #NOTE: For now If N > 1 we only want the most recent state associated with the reward and next state (effectively setting N > 1 changes nothing right now since we want to use the same input type as in the regular singel task case)
            #print('cur obs in learning')
            #print(cur_observation.states)
            most_recent_obs_state = cur_observation.states[-1]
            sampled_state_formatted = format_states([most_recent_obs_state])

            #Get the best action over all actions possible in the next state, ie max_a(Q(s + 1), a))
            q_vals = get_q_vals_aux(cur_observation.next_state, True)
            cur_action_target = reward + (a_globs.GAMMA * np.max(q_vals))

            #Get the learning target q-value for the current state
            q_vals = get_q_vals_aux(most_recent_obs_state, False)
            q_vals[0][a_globs.cur_action] = cur_action_target

            if a_globs.AGENT == a_globs.REWARD:
                #We make the rewards positive since we care only about the binary
                #distinction between zero and non zero rewards and theano binary
                #cross entropy loss requires targets to be 0 or 1
                aux_target = np.array([abs(cur_observation.reward)])
            elif a_globs.AGENT == a_globs.STATE :
                aux_target = format_states([cur_observation.next_state])
            elif a_globs.AGENT == a_globs.NOISE:
                aux_target = np.array([rand_un() for i in range(a_globs.NUM_NOISE_NODES)]).reshape(1, a_globs.NUM_NOISE_NODES)
            elif a_globs.AGENT == a_globs.REDUNDANT:
                nested_target = [q_vals for i in range(a_globs.NUM_REDUNDANT_TASKS)]
                aux_target = np.array([item for sublist in nested_target for item in sublist]).reshape(1, a_globs.NUM_ACTIONS * a_globs.NUM_REDUNDANT_TASKS)


            a_globs.model.fit(sampled_state_formatted, {'main_output' : q_vals, 'aux_output' : aux_target}, batch_size=1, epochs=1, verbose=is_verbose)

def update_replay_buffer(cur_state, cur_action, reward, next_state):
    """
    Update the replay buffer with the most recent transition, adding cur_state to the current global historical context,
    and mapping that to its reward and next_state if the current context equals the observation size N, the user set parameter
    for the number of states per observation
    """

    #Construct the historical context used in the prediciton tasks, and store them in the replay buffer according to their reward valence
    a_globs.cur_context.append(cur_state)
    a_globs.cur_context_actions.append(cur_action)
    if len(a_globs.cur_context) == a_globs.N:

        #print('before pop')
        #print(a_globs.cur_context)
        cur_observation = construct_observation(a_globs.cur_context, a_globs.cur_context_actions, reward, next_state)

        #Remove the oldest states from the context, to allow new ones to be added in a sliding window style
        a_globs.cur_context.pop(0)
        a_globs.cur_context_actions.pop(0)
        #print('after pop')
        #print(a_globs.cur_context)

        if a_globs.AGENT == a_globs.STATE :
            if  cur_observation.states[-1] in a_globs.OBSTACLE_STATES:
                add_to_buffer(a_globs.stochastic_state_buffer, cur_observation)
            else:
                add_to_buffer(a_globs.deterministic_state_buffer, cur_observation)
        else:
            if reward == 0:
                add_to_buffer(a_globs.zero_reward_buffer, cur_observation)
            else:
                add_to_buffer(a_globs.non_zero_reward_buffer, cur_observation)

def construct_observation(cur_states, cur_actions, reward, next_state):
    "Construct the observation used by the auxiliary tasks"

    # if len(cur_states) == 0:
    #     print('empty states!')
    #     print(a_globs.cur_context)
    cur_observation = namedtuple("Transition", ["states", "actions", "reward", "next_state"])
    cur_observation.states = list(cur_states)
    cur_observation.actions = list(cur_actions)
    cur_observation.reward = reward
    cur_observation.next_state = next_state

    return cur_observation

def add_to_buffer(cur_buffer, item_to_add):
    """
    Append item_to_add to cur_buffer, maintaining the buffer to be within
    the size of the BUFFER_SIZE parameter by removing earlier items if necessary
    """

    if len(cur_buffer) > a_globs.BUFFER_SIZE:
        cur_buffer.pop(0)
    cur_buffer.append(item_to_add)

def do_buffer_sampling():
    "Determine from which buffer to sample, if any, based on the agent and environment type"

    cur_observation = None
    if a_globs.zero_reward_buffer and a_globs.non_zero_reward_buffer and a_globs.AGENT != a_globs.STATE:
        cur_observation = sample_from_buffers(a_globs.zero_reward_buffer, a_globs.non_zero_reward_buffer)
    elif a_globs.AGENT == a_globs.STATE  and a_globs.IS_STOCHASTIC:
        if a_globs.deterministic_state_buffer and a_globs.stochastic_state_buffer:
            cur_observation = sample_from_buffers(a_globs.deterministic_state_buffer, a_globs.stochastic_state_buffer)
    elif a_globs.AGENT == a_globs.STATE  and not a_globs.IS_STOCHASTIC:
        if a_globs.deterministic_state_buffer:
            cur_observation = sample_from_buffers(a_globs.deterministic_state_buffer)

    return cur_observation

def sample_from_buffers(buffer_one, buffer_two=None):
    """
    Sample a transition uniformly at random from one of buffer_one and buffer_two.
    but with even probability from each buffer
    """
    if buffer_two is None or rand_un() <= a_globs.BUFFER_SAMPLE_BIAS_PROBABILITY:
        cur_observation = buffer_one[rand_in_range(len(buffer_one))]
    else:
        cur_observation = buffer_two[rand_in_range(len(buffer_two))]
    # print('buffer 1')
    # print(len(buffer_one))
    # print('buffer 2')
    # print(len(buffer_two))
    # print('cur obs')
    # print(cur_observation.states)
    # print(cur_observation.actions)
    # print(cur_observation.reward)
    # print(cur_observation.next_state)
    return cur_observation

# def set_up_empty_aux_input():
#     """
#     Sets up empty auxiliary input of the correct dimensions and returns it.
#     Used for when predicting the main output of networks with auxiliary tasks
#     """
#
#     if a_globs.AGENT == a_globs.REDUNDANT:
#         aux_input = np.zeros(shape=(1, a_globs.FEATURE_VECTOR_SIZE,))
#     else:
#         aux_input = np.zeros(shape=(1, a_globs.AUX_FEATURE_VECTOR_SIZE * a_globs.N,))
#     return aux_input

def format_states(states):
    "Format the states according to the user defined input (1 hot encoding or (x, y) coordinates)"

    if a_globs.IS_1_HOT:
        formatted_states = states_encode_1_hot(states)
    else:
        formatted_states = coordinate_states_encoding(states)

    return formatted_states

def format_states_actions(states, actions):
    "Format the states and actions according to the user defined input (1 hot encoding or (x, y) coordinates)"

    if a_globs.IS_1_HOT:
        formatted_states_actions = states_actions_encode_1_hot(states, actions)
    else:
        formatted_states_actions = coordinate_states_actions_encoding(states, actions)

    return formatted_states_actions

def states_encode_1_hot(states):
    "Return a one hot encoding of the current list of states"

    #print(states)
    all_states_1_hot = []
    for state in states:
    #    print(state)
        state_1_hot = np.zeros((a_globs.NUM_ROWS, a_globs.NUM_COLUMNS))
        state_1_hot[state[0]][state[1]] = 1
        state_1_hot = state_1_hot.reshape(1, a_globs.FEATURE_VECTOR_SIZE)
        all_states_1_hot.append(state_1_hot)

    return np.concatenate(all_states_1_hot, axis=1)

def states_actions_encode_1_hot(states, actions):
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

def coordinate_states_encoding(states):
    """
    Format the x, y coordinates as a numpy array
    """

    # states_copy = list(copy.deepcopy(states))
    # print(states_copy)
    # for i in range(len(states)):
    #     states_copy[i][0] += 1
    #     states_copy[i][1] += 1

    #flatten the states list
    states = [coordinate for state in states for coordinate in state]
    formatted_states = np.array(states).reshape(1, a_globs.FEATURE_VECTOR_SIZE)

    return formatted_states

# def coordinate_states_actions_encoding(states, actions):
#     """
#     Shift the x and y coordinates that define a state value so that it starts at (1, 1), from the point of new of the neural network.
#     We do this so that (0, 0) can be used as dummy auxiliary input when we want to predict just Q-vals from the multi-task neural network
#     We also turn the bare list representation into a numpy array to use with the network
#     """
#
#     states_copy = list(copy.deepcopy(states))
#     actions_copy = list(copy.deepcopy(actions))
#     for i in range(len(states_copy)):
#         states_copy[i][0] += 1
#         states_copy[i][1] += 1
#         actions_copy[i] += 1
#
#     #flatten the states list
#     states_copy = [coordinate for state in states_copy for coordinate in state]
#     formatted_state_actions = np.array(states_copy + actions_copy).reshape(1, a_globs.AUX_FEATURE_VECTOR_SIZE * a_globs.N)
#
#     return formatted_state_actions
