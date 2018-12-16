
from __future__ import division

import Globals.grid_agent_globals as a_globs
import Globals.grid_env_globals as e_globs
import Globals.continuous_grid_env_globals as cont_e_globs
from Globals.generic_globals import *
from Utils.utils import rand_in_range, rand_un
from Utils.tiles3 import IHT, tiles

from collections import namedtuple
from utils import rand_in_range, rand_un
from random import randint

import numpy as np
import random
import json
import platform
import copy

from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Activation, Input, concatenate
from keras.initializers import he_normal
from keras.optimizers import RMSprop, Adam, Adagrad, SGD
from keras.utils import plot_model

from sklearn.manifold import TSNE
from Utils.cca_core import *

#This function has been taken from https://becominghuman.ai/visualizing-representations-bd9b62447e38 on November 17th, 2018
#It has been modified to suit the present purpose. Original author:  Siddhartha Banerjee
def create_truncated_model(trained_model):
    "Return a model equivalent to trained model in terms of architecture and weights, but without the last layer"

    optimizer_map = {'Adam' : Adam(lr=a_globs.ALPHA), 'RMSprop' : RMSprop(lr=a_globs.ALPHA), 'Adagrad' : Adagrad(lr=a_globs.ALPHA), 'SGD': SGD(lr=a_globs.ALPHA)}

    if a_globs.AGENT == a_globs.NEURAL:
        model = Sequential()
        model.add(Dense(a_globs.NUM_NERONS_LAYER_1, activation='relu', input_shape=(a_globs.FEATURE_VECTOR_SIZE,)))
        model.add(Dense(a_globs.NUM_NERONS_LAYER_2, activation='relu'))
    else:
        main_input = Input(shape=(a_globs.FEATURE_VECTOR_SIZE,))
        shared_1 = Dense(a_globs.NUM_NERONS_LAYER_1, activation='relu', name='shared_1')(main_input)
        main_task_full_layer = Dense(a_globs.NUM_NERONS_LAYER_2, activation='relu', name='main_task_full_layer')(shared_1)
        model = Model(inputs=main_input, outputs=main_task_full_layer)

    model.compile(loss='mse', optimizer=optimizer_map[a_globs.OPTIMIZER])
    for i, layer in enumerate(model.layers):
        layer.set_weights(trained_model.layers[i].get_weights())
    return model

def compute_CCA_discrete(model_snapshots):
    """
    Compute the CCA similarity for the network's final layer across all states
    between the last for the snapshot at the end of model_snapshots with all the
    other models in the snapshot list
    """

    max_x_val = e_globs.MAX_COLUMN + 1
    max_y_val = e_globs.MAX_ROW + 1
    truncated_trained_model = create_truncated_model(model_snapshots[-1])
    hidden_layer_feature_shape = truncated_trained_model.predict(format_states([[0, 0]])).shape

    mean_similarity_scores = []
    for model in model_snapshots:
        cur_truncated_model = create_truncated_model(model)
        cur_layer_representations = np.empty((hidden_layer_feature_shape[1], max_x_val * max_y_val))
        trained_layer_representations = np.empty((hidden_layer_feature_shape[1], max_x_val * max_y_val))

        #Compute the last hidden layer state representation for each state
        i = 0
        for x in range(max_x_val):
            for y in range(max_y_val):
                #State formatters expect a list of states in [row, column] format
                cur_state = [y, x]
                cur_state_formatted = format_states([cur_state])
                cur_layer_features = cur_truncated_model.predict(cur_state_formatted)
                trained_layer_features = truncated_trained_model.predict(cur_state_formatted)
                cur_layer_representations[:, i] = cur_layer_features
                trained_layer_representations[:, i] = trained_layer_features
                i += 1

        #print(cur_layer_representations.shape)
        cca_results = get_cca_similarity(cur_layer_representations, trained_layer_representations)
        #print(cca_results)
        # print('Neuron coefficients')
        # print('1')
        # print(x['neuron_coeffs1'])
        # print('2')
        #print(x['neuron_coeffs2'])
        # print('cca coefficients')
        # print('1')
        # print(cca_results['cca_coef1'])
        # print('2')
        # print(cca_results['cca_coef2'])
        # print('cca directions')
        # print(x['cca_dirns1'])
        if len(cca_results['cca_coef1']) > len(cca_results['cca_coef2']):
            mean_similarity_scores.append(np.mean(cca_results['cca_coef2']))
        else:
            mean_similarity_scores.append(np.mean(cca_results['cca_coef1']))
    # print('mean_similarity scores')
    # print(mean_similarity_scores)
    return mean_similarity_scores

def compute_CCA_continuous(plot_range, model_snapshots):
    """
    Compute the CCA similarity for the network's final layer across across a
    number of evenly sampled states equal to plot_range^2 between the last for
    the snapshot at the end of model_snapshots with all the other models in the
    snapshot list
    """

    truncated_trained_model = create_truncated_model(model_snapshots[-1])
    hidden_layer_feature_shape = truncated_trained_model.predict(format_states([[0, 0]])).shape
    mean_similarity_scores = []
    for model in model_snapshots:
        cur_truncated_model = create_truncated_model(model)
        cur_layer_representations = np.empty((hidden_layer_feature_shape[1], plot_range ** 2))
        trained_layer_representations = np.empty((hidden_layer_feature_shape[1], plot_range ** 2))

        #Compute the last hidden layer state representation for each state
        i = 0
        for x in range(plot_range):
            scaled_x = cont_e_globs.MIN_COLUMN + (x * (cont_e_globs.MAX_COLUMN - cont_e_globs.MIN_COLUMN) / plot_range)
            for y in range(plot_range):
                scaled_y = cont_e_globs.MIN_ROW + (y * (cont_e_globs.MAX_ROW - cont_e_globs.MIN_ROW) / plot_range)
                #State formatters expect a list of states in [row, column] format
                cur_state = [y, x]
                cur_state_formatted = format_states([cur_state])
                cur_layer_features = cur_truncated_model.predict(cur_state_formatted)
                trained_layer_features = truncated_trained_model.predict(cur_state_formatted)
                cur_layer_representations[:, i] = cur_layer_features
                trained_layer_representations[:, i] = trained_layer_features
                i += 1

        #print(cur_layer_representations.shape)
        cca_results = get_cca_similarity(cur_layer_representations, trained_layer_representations)
        # print(cca_results)
        # print('Neuron coefficients')
        # print('1')
        # print(x['neuron_coeffs1'])
        # print('2')
        #print(x['neuron_coeffs2'])
        # print('cca coefficients')
        # print('1')
        # print(cca_results['cca_coef1'])
        # print('2')
        # print(cca_results['cca_coef2'])
        # print('cca directions')
        # print(x['cca_dirns1'])
        if len(cca_results['cca_coef1']) > len(cca_results['cca_coef2']):
            mean_similarity_scores.append(np.mean(cca_results['cca_coef2']))
        else:
            mean_similarity_scores.append(np.mean(cca_results['cca_coef1']))
    # print('mean_similarity scores')
    # print(mean_similarity_scores)
    return mean_similarity_scores


def compute_t_SNE_discrete():
    "Computes the representations learned in the last hidden layer of the neural network for all states"

    #Get a truncated modle so that we can grab the predictions of the hidden layer
    truncated_model = create_truncated_model(a_globs.model)

    max_x_val = e_globs.MAX_COLUMN + 1
    max_y_val = e_globs.MAX_ROW + 1

    hidden_layer_feature_shape = truncated_model.predict(format_states([[0, 0]])).shape
    #print(hidden_layer_feature_shape)
    state_network_representations = np.empty((max_x_val * max_y_val, hidden_layer_feature_shape[1]))

    #Compute the last hidden layer state representation for each state
    i = 0
    for x in range(max_x_val):
        for y in range(max_y_val):
            #State formatters expect a list of states in [row, column] format
            cur_state = [y, x]
            cur_state_formatted = format_states([cur_state])
            hidden_layer_features = truncated_model.predict(cur_state_formatted)
            state_network_representations[i] = hidden_layer_features
            i += 1
            #x_values.append(x)
            #print('y value shape')
            #print(y_values.shape)
            #y_values.append(y)
            #y_values[0][y] = y
            # print(best_action_val)
            # print(plot_values)
            # print('x_values')
            # print(x_values)
            # print('y_values')
            # print(y_values)
            # print('plot_values')
            # print(plot_values)
        #x_values[0][x] = x

    #Perform dimensionality reduction
    # print(state_network_representations.shape)
    # print(state_network_representations)
    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(state_network_representations)
    #print(tsne_results)
    # print(tsne_results.shape)
    #print(tsne_results[:, 0])

    return tsne_results

def compute_t_SNE_continuous(plot_range):
    "Compute the values for the current value function across a number of evenly sampled states equal to plot_range^2"

    #Get a truncated modle so that we can grab the predictions of the hidden layer
    truncated_model = create_truncated_model(a_globs.model)
    hidden_layer_feature_shape = truncated_model.predict(format_states([[0, 0]])).shape
    state_network_representations = np.empty((plot_range ** 2, hidden_layer_feature_shape[1]))

    #Compute the last hidden layer state representation for each state
    i = 0
    for x in range(plot_range):
        scaled_x = cont_e_globs.MIN_COLUMN + (x * (cont_e_globs.MAX_COLUMN - cont_e_globs.MIN_COLUMN) / plot_range)
        for y in range(plot_range):
            #State formatters expect a list of states in [row, column] format
            scaled_y = cont_e_globs.MIN_ROW + (y * (cont_e_globs.MAX_ROW - cont_e_globs.MIN_ROW) / plot_range)
            cur_state = [scaled_y, scaled_x]
            cur_state_formatted = format_states([cur_state])
            hidden_layer_features = truncated_model.predict(cur_state_formatted)
            state_network_representations[i] = hidden_layer_features
            i += 1
            #x_values.append(x)
            #print('y value shape')
            #print(y_values.shape)
            #y_values.append(y)
            #y_values[0][y] = y
            # print(best_action_val)
            # print(plot_values)
            # print('x_values')
            # print(x_values)
            # print('y_values')
            # print(y_values)
            # print('plot_values')
            # print(plot_values)
        #x_values[0][x] = x

    #Perform dimensionality reduction
    # print(state_network_representations.shape)
    # print(state_network_representations)
    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(state_network_representations)
    #print(tsne_results)
    # print(tsne_results.shape)
    #print(tsne_results[:, 0])

    return tsne_results

def compute_state_action_values_discrete():
    "Compute the values for the current value function across all of the states"

    max_x_val = e_globs.MAX_COLUMN + 1
    max_y_val = e_globs.MAX_ROW + 1

    x_values = np.empty((1, max_x_val))
    y_values = np.empty((1, max_y_val))

    plot_values = np.empty((max_x_val, max_y_val))
    # print(a_globs.AGENT)
    # print('plot shape')
    # print(plot_values.shape)
    for x in range(max_x_val):
        for y in range(max_y_val):
            #State formatters expect a list of states in [row, column] format
            cur_state = [y, x]
            if a_globs.AGENT == a_globs.RANDOM:
                pass
            if a_globs.AGENT == a_globs.TABULAR:
                #State action table is int[row, column] format
                best_action_val = -max([a_globs.state_action_values[y][x][action] for action in range(a_globs.NUM_ACTIONS)])
            elif a_globs.AGENT == a_globs.SARSA_LAMBDA:
                best_action_val = -max([approx_value(cur_state, action, a_globs.weights)[0] for action in range(a_globs.NUM_ACTIONS)])
            else:
                cur_state_formatted = format_states([cur_state])
                if a_globs.AGENT == a_globs.NEURAL:
                    best_action_val = -max(a_globs.model.predict(cur_state_formatted, batch_size=1))[0]
                else:
                    best_action_val = -max(a_globs.model.predict(cur_state_formatted, batch_size=1)[0][0])

            #x_values.append(x)
            #print('y value shape')
            #print(y_values.shape)
            #y_values.append(y)
            y_values[0][y] = y
            # print(best_action_val)
            # print(plot_values)
            plot_values[x][y] = best_action_val
            # print('x_values')
            # print(x_values)
            # print('y_values')
            # print(y_values)
            # print('plot_values')
            # print(plot_values)
        x_values[0][x] = x
    return x_values, np.transpose(y_values), np.transpose(plot_values)

def compute_state_action_values_continuous(plot_range):
    "Compute the values for the current value function across a number of evenly sampled states equal to plot_range^2"

    x_values = np.empty((1, plot_range))
    y_values = np.empty((1, plot_range))
    plot_values = np.empty((plot_range, plot_range))

    for x in range(plot_range):
        # print('min column')
        # print(cont_e_globs.MIN_COLUMN)
        # print('max column')
        # print(cont_e_globs.MAX_COLUMN)
        # print('x')
        # print(x)
        scaled_x = cont_e_globs.MIN_COLUMN + (x * (cont_e_globs.MAX_COLUMN - cont_e_globs.MIN_COLUMN) / plot_range)
        #print('scaled x')
        #print(scaled_x)
        for y in range(plot_range):
            scaled_y = cont_e_globs.MIN_ROW + (y * (cont_e_globs.MAX_ROW - cont_e_globs.MIN_ROW) / plot_range)
            cur_state = [scaled_y, scaled_x]
            if a_globs.AGENT == a_globs.RANDOM:
                pass
            elif a_globs.AGENT == a_globs.SARSA_LAMBDA:
                best_action_val = -max([approx_value(cur_state, action, a_globs.weights)[0] for action in range(a_globs.NUM_ACTIONS)])
            else:
                cur_state_formatted = format_states([cur_state])
                if a_globs.AGENT == a_globs.NEURAL:
                    best_action_val = -max(a_globs.model.predict(cur_state_formatted, batch_size=1))[0]
                else:
                    best_action_val = -max(a_globs.model.predict(cur_state_formatted, batch_size=1)[0][0])

            #x_values.append(x)
            #print('y value shape')
            #print(y_values.shape)
            #y_values.append(y)
            y_values[0][y] = scaled_y
            # print(best_action_val)
            # print(plot_values)
            plot_values[x][y] = best_action_val
            # print('x_values')
            # print(x_values)
            # print('y_values')
            # print(y_values)
            # print('plot_values')
            # print(plot_values)
        x_values[0][x] = scaled_x
    return x_values, np.transpose(y_values), np.transpose(plot_values)

def approx_value(state, action, weights):
    """
    Return the current approximated value for state and action given weights,
    and the indices for the active features for the for the state action pair.
    """
    if a_globs.IS_DISCRETE:
        scaled_row = a_globs.NUM_TILINGS * state[0] / (e_globs.MAX_ROW + abs(e_globs.MIN_ROW))
        scaled_column = a_globs.NUM_TILINGS * state[1] / (e_globs.MAX_COLUMN + abs(e_globs.MIN_COLUMN))
    else:
        scaled_row = a_globs.NUM_TILINGS * state[0] / (cont_e_globs.MAX_ROW + abs(cont_e_globs.MIN_ROW))
        scaled_column = a_globs.NUM_TILINGS * state[1] / (cont_e_globs.MAX_COLUMN + abs(cont_e_globs.MIN_COLUMN))

    cur_tiles = tiles(a_globs.iht, a_globs.NUM_TILINGS, [scaled_row, scaled_column], [action])
    estimate = 0

    for tile in cur_tiles:
        estimate += weights[0][tile]
    return (estimate, cur_tiles)

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
        aux_target = np.array([[reward]])
    elif a_globs.AGENT == a_globs.STATE :
        if next_state:
            aux_target = format_states([next_state])
        else:
            aux_target = np.zeros(shape=(1, a_globs.FEATURE_VECTOR_SIZE,))
    elif a_globs.AGENT == a_globs.NOISE:
        aux_target = np.array([rand_un() for i in range(a_globs.NUM_NOISE_NODES)]).reshape(1, a_globs.NUM_NOISE_NODES)
    elif a_globs.AGENT == a_globs.REDUNDANT:
        nested_target = [q_vals for i in range(a_globs.NUM_REDUNDANT_TASKS)]
        aux_target = np.array([item for sublist in nested_target for item in sublist]).reshape(1, a_globs.NUM_ACTIONS * a_globs.NUM_REDUNDANT_TASKS)

    cur_state_formatted = format_states([cur_state])

    #Check and see if the relevant buffer is non-empty
    if buffers_are_ready(a_globs.buffer_container, a_globs.BUFFER_SIZE) and not a_globs.is_trial_episode:

        #print('I am replay buffer!')
        #Create the target training batch
        # print('aux target')
        # print(aux_target)
        # print('shape')
        # print(aux_target.shape)
        batch_inputs = np.empty(shape=(a_globs.BATCH_SIZE, a_globs.FEATURE_VECTOR_SIZE,))
        batch_targets = np.empty(shape=(a_globs.BATCH_SIZE, a_globs.NUM_ACTIONS))
        batch_aux_targets = np.empty(shape=(a_globs.BATCH_SIZE, aux_target.shape[1]))

        # print('batch aux targets')
        # print(batch_aux_targets)
        #
        # print('aux target 0')
        # print(aux_target[0])
        #
        # print('batch aux tagrets at 0')
        # print(batch_aux_targets[0])

        #Add the current observation to the mini-batch
        batch_inputs[0] = cur_state_formatted
        batch_targets[0] = q_vals
        batch_aux_targets[0] = aux_target[0]

        # print('post assignment of batch aux target')
        # print(batch_aux_targets[0])

        #Use the replay buffer to learn from previously visited states
        for i in range(1, a_globs.BATCH_SIZE):
            cur_observation = do_buffer_sampling()
            # print('states')
            # print(cur_observation.states)
            # print('actions')
            # print(cur_observation.actions)
            # print('reward')
            # print(cur_observation.reward)
            # print('next state')
            # print(cur_observation.next_state)

            #NOTE: For now If N > 1 we only want the most recent state associated
            #with the reward and next state (effectively setting N > 1 changes nothing right now since we want to use the same input type as in the regular singel task case)
            most_recent_obs_state = cur_observation.states[-1]
            sampled_state_formatted = format_states([most_recent_obs_state])
            sampled_next_state_formatted = format_states([cur_observation.next_state])

            # print('sampled_state_formatted')
            # print(sampled_state_formatted)
            # print('sample state next formatted')
            # print(sampled_next_state_formatted)

            #Get the best action over all actions possible in the next state, ie max_a(Q(s + 1), a))
            q_vals = get_q_vals_aux(cur_observation.next_state, True)
            cur_action_target = reward + (a_globs.GAMMA * np.max(q_vals))

            #Get the value for the current state of the action which was just taken, ie Q(S, A),
            #and set the target for the specifc action taken (we need to pass in the
            #whole vector of q_values, since our network takes state only as input)
            q_vals = get_q_vals_aux(most_recent_obs_state, False)
            q_vals[0][a_globs.cur_action] = cur_action_target

            if a_globs.AGENT == a_globs.REWARD:
                #We make the rewards positive since we care only about the binary
                #distinction between zero and non zero rewards and theano binary
                #cross entropy loss requires targets to be 0 or 1
                aux_target = np.array([[cur_observation.reward]])
            elif a_globs.AGENT == a_globs.STATE :
                aux_target = format_states([cur_observation.next_state])
            elif a_globs.AGENT == a_globs.NOISE:
                aux_target = np.array([rand_un() for _ in range(a_globs.NUM_NOISE_NODES)]).reshape(1, a_globs.NUM_NOISE_NODES)
            elif a_globs.AGENT == a_globs.REDUNDANT:
                nested_target = [q_vals for _ in range(a_globs.NUM_REDUNDANT_TASKS)]
                #print('nested_target')
                #print(nested_target)
                aux_target = np.array([item for sublist in nested_target for item in sublist]).reshape(1, a_globs.NUM_ACTIONS * a_globs.NUM_REDUNDANT_TASKS)

            #print('batch inpiuts')
            #print(batch_inputs)
            # print('sampled_state_formatted')
            # print(sampled_state_formatted)
            # print('q vals')
            # print(q_vals)

            #print(i)
            batch_inputs[i] = sampled_state_formatted
            batch_targets[i] = q_vals
            batch_aux_targets[i] = aux_target[0]

        # print('Fit ME')
        # print('batch inputs')
        # print(batch_inputs)
        # print('batch targets')
        # print(batch_targets)

        #Update the weights using the sampled batch
        # print('inputs')
        # print(batch_inputs)
        # print('targets')
        # print(batch_targets)
        # print('batch aux taergets')
        # print(batch_aux_targets)
        #a_globs.model.fit(batch_inputs, [batch_targets, batch_aux_targets], batch_size=a_globs.BATCH_SIZE , epochs=1, verbose=0)

def update_replay_buffer(cur_state, cur_action, reward, next_state):
    """
    Update the replay buffer with the most recent transition, adding cur_state to the current global historical context,
    and mapping that to its reward and next_state if the current context equals the observation size N, the user set parameter
    for the number of states per observation
    """

    #Construct the historical context used in the prediction tasks, and store them in the replay buffer according to their reward valence
    a_globs.cur_context.append(cur_state)
    a_globs.cur_context_actions.append(cur_action)
    if len(a_globs.cur_context) == a_globs.N or a_globs.AGENT == a_globs.NEURAL:
        #print('update buffer!')

        #print('before pop')
        #print(a_globs.cur_context)
        cur_observation = construct_observation(a_globs.cur_context, a_globs.cur_context_actions, reward, next_state)

        #Remove the oldest states from the context, to allow new ones to be added in a sliding window style
        a_globs.cur_context.pop(0)
        a_globs.cur_context_actions.pop(0)
        #print('after pop')
        #print(a_globs.cur_context)

        if a_globs.AGENT == a_globs.NEURAL:
                add_to_buffer(a_globs.generic_buffer, cur_observation)

        elif a_globs.AGENT == a_globs.STATE:
            #TODO: Set up a check for being in a stochastic obstacle case for the continuous case
            if  cur_observation.states[-1] in e_globs.OBSTACLE_STATES:
                add_to_buffer(a_globs.stochastic_state_buffer, cur_observation)
            else:
                add_to_buffer(a_globs.deterministic_state_buffer, cur_observation)

        elif a_globs.AGENT == a_globs.REWARD:
            if reward == 0:
                add_to_buffer(a_globs.zero_reward_buffer, cur_observation)
            else:
                add_to_buffer(a_globs.non_zero_reward_buffer, cur_observation)

        elif a_globs.AGENT == a_globs.REDUNDANT or a_globs.AGENT == a_globs.NOISE:
            add_to_buffer(a_globs.generic_buffer, cur_observation)
        else:
            exit("ERROR: Invalid agent specified! No corresponding buffer to use!")

#NOTE: THIS USES NAMED TUPLE WRONG. WE NEED TO TREAT THE RESULT OF NAMED TUPLE AS A FACTORY AND CREATE INSTANCES FROM THE FACTORY
def construct_observation(cur_states, cur_actions, reward, next_state):
    "Construct the observation used by the auxiliary tasks"

    cur_observation = Transition(list(cur_states), list(cur_actions), reward, next_state)
    # cur_observation.states = list(cur_states)
    # cur_observation.actions = list(cur_actions)
    # cur_observation.reward = reward
    # cur_observation.next_state = next_state

    return cur_observation

def add_to_buffer(cur_buffer, item_to_add):
    """
    Append item_to_add to cur_buffer, removing a prior entry if sum of all buffer sizes = BUFFER_SIZE parameter
    """

    cur_buffer_size = 0
    for sub_buffer in a_globs.buffer_container:
        cur_buffer_size += len(sub_buffer)
    if cur_buffer_size == a_globs.BUFFER_SIZE:
        cur_buffer.pop(0)
    cur_buffer.append(item_to_add)

def do_buffer_sampling():
    "Determine from which buffer(s) to sample based on the current agent and environment type"

    if a_globs.AGENT == a_globs.REWARD:
        cur_observation = sample_from_buffers(a_globs.zero_reward_buffer, a_globs.non_zero_reward_buffer)
    elif a_globs.AGENT == a_globs.STATE and a_globs.IS_STOCHASTIC:
        cur_observation = sample_from_buffers(a_globs.deterministic_state_buffer, a_globs.stochastic_state_buffer)
    elif a_globs.AGENT == a_globs.STATE and not a_globs.IS_STOCHASTIC:
        cur_observation = sample_from_buffers(a_globs.deterministic_state_buffer)
    elif a_globs.NEURAL:
        cur_observation = sample_from_buffers(a_globs.generic_buffer)
    else:
        exit("ERROR: Attempting to sample from a buffer with an agent ({}) for which no replay buffer is defined!".format(a_globs.AGENT))

    return cur_observation

def buffers_are_ready(buffer_container, buffer_size):
    "Return whether all of the buffers in buffer_container are non empty and that their sum adds up to buffer_size"
    cur_buffer_size = 0
    for sub_buffer in buffer_container:
        #print('I am ')
        #print(sub_buffer)
        if not sub_buffer:
            return False
        cur_buffer_size += len(sub_buffer)
    return cur_buffer_size == buffer_size


def sample_from_buffers(buffer_one, buffer_two=None):
    """
    Pick from one of buffer one and buffer two according to the buffer probability parameter.
    Then samples uniformly at random from the chosen buffer.
    """
    #NOTE: Will have to add a check here to force sampling from non empty buffer if I decided to not wait for nonempty buffers of both types in reward/state tasks
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
    all_states_1_hot = []
    for state in states:
        state_1_hot = np.zeros((e_globs.NUM_ROWS, e_globs.NUM_COLUMNS))
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
        state_1_hot = np.zeros((e_globs.NUM_ROWS, e_globs.NUM_COLUMNS, a_globs.NUM_ACTIONS))
        state_1_hot[state[0]][state[1]][action] = 1
        state_1_hot = state_1_hot.reshape(1, a_globs.AUX_FEATURE_VECTOR_SIZE)
        all_states_1_hot.append(state_1_hot)

    return np.concatenate(all_states_1_hot, axis=1)

def coordinate_states_encoding(states):
    """
    Format the x, y coordinates as a numpy array
    """

    #flatten the states list
    #[item for sublist in l for item in sublist]
    #print(a_globs.FEATURE_VECTOR_SIZE)
    states = [coordinate for state in states for coordinate in state]
    formatted_states = np.array(states).reshape(1, a_globs.FEATURE_VECTOR_SIZE)

    return formatted_states

def coordinate_states_actions_encoding(states, actions):
    """
    Formt the x,y coordinate and action as a numpy array
    """

    #flatten the states list
    states = [coordinate for state in states for coordinate in state]
    formatted_state_actions = np.array(states + actions).reshape(1, a_globs.AUX_FEATURE_VECTOR_SIZE * a_globs.N)

    return formatted_state_actions
