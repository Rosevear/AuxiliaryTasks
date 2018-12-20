#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

  Modified and extended by Cody Rosevear for use in CMPUT659, Winter 2018
  and CMPUT701, Fall 2018, University of Alberta
"""

from __future__ import division
from rl_glue import *  # Required for RL-Glue
from Globals.generic_globals import *
import Globals.grid_agent_globals as a_globs

import pickle
import argparse
import json
import random
import numpy as np
import platform
import os

from time import sleep
from itertools import product
from collections import namedtuple
from math import log
from mpl_toolkits.mplot3d import Axes3D
from keras.models import model_from_json, clone_model
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import TSNE

import matplotlib as mpl
if platform.system() == 'Darwin':
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

###### HELPER FUNCTIONS START ##############

#NOTE Taken from https://stackoverflow.com/questions/52303660/iterating-markers-in-plots/52303895#52303895, by user ImportanceOfBeingErnest, on December 18th 2018, and modified for the present purpose
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x, y, cmap=cm.jet, **kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

#NOTE:  Save and load model taken from https://machinelearningmastery.com/save-load-keras-deep-learning-models/, by Jason Brownlee, on December 13th 2018, and modified to suit the current purpose
def save_model(models, filename):
    "Save the current model architecture to disk in a in a json format and the weights in HDF5 format."

    # serialize model to JSON
    filename = filename.replace(' ', '_')
    try:
        i = 0
        for model in models:
            model_json = model.to_json()
            with open('{}{}_snapshot_{}.json'.format(MODELS_DIR, filename, str(i)), "w") as json_file:
                json_file.write(model_json)
            print("Succesfully saved the model architecture")

            # serialize weights to HDF5
            model.save_weights('{}{}_snapshot_{}.h5'.format(MODELS_DIR, filename, str(i)))
            print("Sucessfully saved the model weights")
            i += 1
    except IOError as cur_exception:
         print("ERROR: Could not save the model: {}".format(os.strerror(cur_exception.errno)))


def load_models(filenames):
    """
    Load the currently saved model architecture and weights from disk for each
    of the files specified in fileneames, a string of filenames delimited by spaces
    """

    # load weights into new model
    try:
        model_snapshots = []
        for model_filename in filenames:
            print(MODELS_DIR + model_filename + '.json')
            json_file = open('{}{}.json'.format(MODELS_DIR, model_filename), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            print("Succesfully loaded the model architecture")

            # load weights into new model
            loaded_model.load_weights("{}{}.h5".format(MODELS_DIR, model_filename))
            print("Successfully loaded the model weights")
            model_snapshots.append(loaded_model)
        return model_snapshots
    except IOError as cur_exception:
         print("ERROR: Could not load the model: {}".format(os.strerror(cur_exception.errno)))

def is_neural(cur_agent):
    return cur_agent == a_globs.NEURAL or cur_agent in AUX_AGENTS

def setup_plot(x_tick_size=None, plot_title=None):
    print("Plotting the results...")
    plt.ylabel('Steps per episode')
    plt.xlabel("Episode")
    plt.title(plot_title)

    # if x_tick_size:
    #     plt.xticks(np.arange(0, num_episodes + 1, x_tick_size))

    plt.axis([0, num_episodes, 0, max_steps + 1000])
    plt.legend(loc='center', bbox_to_anchor=(0.50, 0.90))

def do_plotting(suffix=0, filename=None):
    if filename:
        print("Saving the results...")
        if suffix:
            plt.savefig("{}.png".format(filename + str(suffix)), format="png")
        else:
            plt.savefig("{}.png".format(filename), format="png")
    else:
        print("Displaying the results...")
        plt.show()

def do_visualization(num_episodes, max_steps, plot_range, setting, suffix=0):
    "Create a 3D plot of plot_range samples of the agent's current value function across the state space for a single run of length num_episodes"

    cur_agent = setting[0]
    send_params(cur_agent, setting)
    RL_init()

    if args.load_models:
        #WE have we need to visualize, so skip training, load them, and jump to visualizing
        MODEL_SNAPSHOTS = load_models(args.load_models.split())
        a_globs.model = MODEL_SNAPSHOTS[-1]
    else:
        #Do the training to visualize
        MODEL_SNAPSHOTS = []
        cur_agent = setting[0]
        print("Performing a single {} episode long run to train the model for visualization for the {} agent".format(num_episodes, cur_agent))
        print('Parameters: agent = {}, alpha = {}, gamma = {}'.format(cur_agent, setting[1], setting[2]))
        np.random.seed(2)
        random.seed(2)
        for episode in range(num_episodes):
            print("episode number : {}".format(episode))

            if is_neural(cur_agent) and (episode % args.trial_frequency == 0 or episode == num_episodes - 1):
                 MODEL_SNAPSHOTS.append(RL_agent_message(('GET_SNAPSHOT',)))

            RL_episode(max_steps)
            RL_cleanup()

        if args.save_model:
            save_model(MODEL_SNAPSHOTS, RESULTS_FILE_NAME)

    #Perform all the visualization computation and plotting
    #Get the values for the value function
    #print(a_globs.model)
    (x_values, y_values, plot_values) = RL_agent_message(('PLOT', plot_range))

    print("Plotting the 3D value function plot")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('{} Agent Value Function'.format(cur_agent));
    ax.set_xlabel('Column position')
    ax.set_ylabel('Row position')
    ax.set_zlabel('Value')
    ax.plot_wireframe(x_values, y_values, plot_values)

    if RESULTS_FILE_NAME:
        print("Saving the results...")
        if suffix:
            plt.savefig("{} {} value function plot.png".format(RESULTS_FILE_NAME + str(suffix), cur_agent), format="png")
        else:
            plt.savefig("{} {} value function plot.png".format(RESULTS_FILE_NAME, cur_agent), format="png")
    else:
        print("Displaying the value function results results...")
        plt.show()

    #Get the last layer representation for each state and visualize using t-SNE
    if is_neural(cur_agent):
        tsne_results, marker_sizes, marker_colours, marker_styles, = RL_agent_message(('t-SNE', plot_range))
        #print(tsne_results)

        print("Plotting the t-SNE results")
        plt.figure(figsize=(10,10))
        scatter = mscatter(tsne_results[:, 0], tsne_results[:, 1], s=np.array(marker_sizes), c=np.array(marker_colours), m=np.array(marker_styles))
        #plt.legend(loc='center', bbox_to_anchor=(0.50, 0.90))
        plt.colorbar(scatter)
        plt.show()

        if RESULTS_FILE_NAME:
            print("Saving the results...")
            if suffix:
                plt.savefig("{} {} t-SNE plot.png".format(RESULTS_FILE_NAME + str(suffix), cur_agent), format="png")
            else:
                plt.savefig("{} {} t-SNE plot.png".format(RESULTS_FILE_NAME, cur_agent), format="png")
        else:
            print("Displaying the t-SNE results...")
            plt.show()

        #Get the SVCCA similarity score
        mean_similarity_scores, neurons = RL_agent_message(('CCA', plot_range, MODEL_SNAPSHOTS))
        save_results(mean_similarity_scores, cur_agent, RESULTS_FILE_NAME + 'SVCCA_similarity', visualize=True)
        episodes = [episode for episode in range(0, num_episodes + 1, args.trial_frequency)]

        plt.ylabel('SVCCA Similarity')
        plt.xlabel("Episode")
        plt.title("SVCCA Network Similarity Across Time")
        plt.legend(loc='center', bbox_to_anchor=(0.50, 0.90))

        if args.load_models:
            axes = plt.gca()
            axes.set_ylim([0.0, 1.0])
            plt.scatter(mean_similarity_scores, episodes, s=NORMAL_POINT, c=GRAPH_COLOURS[i])
        else:
            plt.axis([0, num_episodes, 0, 1.0])
            plt.scatter(mean_similarity_scores, episodes, s=NORMAL_POINT, c=GRAPH_COLOURS[i])
        plt.show()
        do_plotting(filename=RESULTS_FILE_NAME + "SVCCA_similarity")

        if RESULTS_FILE_NAME:
            print("Saving the results...")
            if suffix:
                plt.savefig("{} {} SVCCA plot.png".format(RESULTS_FILE_NAME + str(suffix), cur_agent), format="png")
            else:
                plt.savefig("{} {} SVCCA plot.png".format(RESULTS_FILE_NAME, cur_agent), format="png")
        else:
            print("Displaying the SVCCA results...")
            plt.show()

        #Do t-sne on the SVCCA representation of the layer
        # tsne = TSNE(n_components=2, verbose=1)
        # tsne_results = tsne.fit_transform(neurons)
        #
        # print("Plotting the t-SNE SVCCA preprocessed results")
        # plt.figure(figsize=(10,10))
        # plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        # plt.legend(loc='center', bbox_to_anchor=(0.50, 0.90))
        # plt.show()

        # if RESULTS_FILE_NAME:
        #     print("Saving the results...")
        #     if suffix:
        #         plt.savefig("{} {} t-SNE-SVCCA plot.png".format(RESULTS_FILE_NAME + str(suffix), cur_agent), format="png")
        #     else:
        #         plt.savefig("{} {} t-SNE-SVCCA plot.png".format(RESULTS_FILE_NAME, cur_agent), format="png")
        # else:
        #     print("Displaying the t-SNE-SVCCA results...")
        #     plt.show()

#NOTE: Taken from https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression on September 22nd 2018
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def sample_params_log_uniform(start, end, num_samples):
    "Randomly sample parameters from the base 10 log uniform distribution within the range start and stop, including start but not stop"

    log_uniform_start = log(start, 10)
    log_uniform_end = log(end, 10)
    param_logs = [log_uniform_start *  np.random.uniform(log_uniform_end / log_uniform_start, 1.0) for i in range(num_samples)]
    params = [round(10 ** cur_log, 5) for cur_log in param_logs]
    return params


def save_results(results, cur_agent, filename='Default File Name', q_plot=False, visualize=False):
    """
    Compute the mean and standard deviation of the current results,
    and save the results and statistics to the specified file.

    Both a human readable version of the file, and a pickled file that serializes
    the data, will be created.
    """

    if is_neural(cur_agent):
        agent_string = "Agent: " + cur_agent + ' Optimizer: ' + a_globs.OPTIMIZER + ' Initialization: ' + a_globs.INIT
        #print(agent_string)
    else:
        agent_string = cur_agent

    with open(RESULTS_DIR + '{} results'.format(filename), 'w') as results_file:
        results_file.write('{} average results\n'.format(agent_string))
        results_file.write(json.dumps(results))
        results_file.write('\n')
        results_file.write('{} agent summary statistics\n'.format(agent_string))
        results_file.write('mean: {} standard deviation: {}\n'.format(np.mean(results), np.std(results)))

        filename = filename.replace(' ', '_')
        if q_plot:
            with open(RESULTS_DIR + '{}_pickled'.format(filename), 'w') as results_file_pickled:
                episodes = [episode for episode in range(0, num_episodes + 1, args.trial_frequency)]
                results_tuple = Results_tuple("Steps Per Episode Across Time (Offline Evaluation)", agent_string, results, episodes, args.trial_frequency, 'Episode', num_episodes, 'Steps Per Episode', max_steps + 1000)
                print("Saving the Q plot results tuple!")
                print(results_tuple)
                pickle.dump(results_tuple, results_file_pickled)

        elif visualize:
            with open(RESULTS_DIR + '{}_pickled'.format(filename), 'w') as results_file_pickled:
                episodes = [episode for episode in range(0, num_episodes + 1, args.trial_frequency)]
                results_tuple = Results_tuple("Network Self-Similarity Across Time", agent_string, results, episodes, args.trial_frequency, 'Episode', num_episodes, 'SVCCA Similarity', 1.0)
                print("Saving the SVCCA results tuple!")
                print(results_tuple)
                pickle.dump(results_tuple, results_file_pickled)
        else:
            with open(RESULTS_DIR + '{}_pickled'.format(filename), 'w') as results_file_pickled:
                episodes = [episode for episode in range(num_episodes)]
                results_tuple = Results_tuple("Steps Per Episode Across Time (Online Evaluation)", agent_string, results, episodes, 1, 'Episode', num_episodes, 'Steps Per Episode', max_steps + 1000)
                print("Saving the online results tuple!")
                print(results_tuple)
                pickle.dump(results_tuple, results_file_pickled)

def load_data(filename):
    """
    Loads the data sepcified in filename from the Results directory.
    """

    with open(RESULTS_DIR + '{}'.format(filename), 'r') as results_file_pickled:
        results = pickle.load(results_file_pickled)
    return results

def compute_correlations(file_1, file_2):
    """
    Compute the corelation coefficient between the data stored in file1 and
    file 2, and report the associated p-value
    """

    file_1_data = load_data(file_1).data
    file_2_data = load_data(file_2).data

    #print(file_1_data)
    #print(file_2_data)
    print("Computing the pearson and spearman correlation coeffcients..")
    pearson_coeff, pearson_p_value = pearsonr(file_1_data, file_2_data)
    spearman_coeff, spearman_p_value = spearmanr(file_1_data, file_2_data)
    print("Pearson correlation coefficient: {}, with 2-sided p-value {}".format(pearson_coeff, pearson_p_value))
    print("Spearman correlation coefficient: {}, with 2-sided p-value {}".format(spearman_coeff, spearman_p_value))


def plot_files(result_files, q_plot):
    "Plot the results for the list of files in result_files, all on the same chart for easy comparison"

    i = 0
    classes = []
    recs = []
    for result_file in result_files:
        print("Loading results...")
        cur_results = load_data(result_file)
        print('Plotting results...')
        print(cur_results)
        if q_plot:
            plt.scatter(cur_results.x_values, cur_results.data, s=NORMAL_POINT, c=GRAPH_COLOURS[i])
            recs.append(mpatches.Rectangle((0,0), 1, 1, fc=GRAPH_COLOURS[i]))
            classes.append(cur_results.agent_type)
        else:
            plt.plot(cur_results.x_values, cur_results.data, GRAPH_COLOURS[i], label=cur_results.agent_type)
        i += 1
    if q_plot:
        plt.legend(recs, classes, loc='lower left')
    else:
        plt.legend(loc='upper right')

    plt.ylabel(cur_results.y_label)
    plt.xlabel(cur_results.x_label)
    plt.title(cur_results.plot_title)
    plt.axis([0, num_episodes, 0, max_steps + 1000])
    plt.show()

def write_to_log(contents, filename=LOG_FILE_NAME):
    "Write the contents to file filename."

    with open(filename, 'a+') as log_file:
        log_file.write(contents)
        log_file.write('\n')

def send_params(cur_agent, param_setting):
    "Send the specified agent and environment parameters to be used in the current run"

    #print(param_setting)
    if cur_agent in AUX_AGENTS:
        agent_params = {"EPSILON": EPSILON, "AGENT": param_setting[0], "ALPHA": param_setting[1], "GAMMA": param_setting[2], "N": param_setting[3], "LAMBDA": param_setting[4], "IS_STOCHASTIC": IS_STOCHASTIC, "IS_1_HOT": IS_1_HOT, "ENV": args.env}
    elif cur_agent == a_globs.NEURAL and args.sweep_neural:
        agent_params = {"EPSILON": EPSILON, "AGENT": param_setting[0], "ALPHA": param_setting[1], "GAMMA": param_setting[2], "BUFFER_SIZE": param_setting[3], "NUM_STEPS_TO_UPDATE": param_setting[4], "IS_STOCHASTIC": IS_STOCHASTIC, "IS_1_HOT": IS_1_HOT, "ENV": args.env}
    else:
        agent_params = {"EPSILON": EPSILON, "AGENT": param_setting[0], "ALPHA": param_setting[1], "GAMMA": param_setting[2], "TRACE": trace_params[0], "IS_STOCHASTIC": IS_STOCHASTIC, "IS_1_HOT": IS_1_HOT, "ENV": args.env}
    enviro_params = {"NUM_ACTIONS": NUM_ACTIONS, "IS_STOCHASTIC": IS_STOCHASTIC, "IS_SPARSE": IS_SPARSE}
    RL_agent_message(json.dumps(agent_params))
    RL_env_message(json.dumps(enviro_params))

###### HELPER FUNCTIONS END #################

#AUX_AGENTS = [', 'state', 'redundant', 'noise']
AUX_AGENTS = []
#AGENTS = []
AGENTS = ['neural']
#Use args.trial frequency to determine snapshot points instead
#CCA_SNAPSHOT_POINTS = [0, 25, 50, 75, 100, 125] #The episodes during a run for which a snapshot of the model should be taken for SVCCA analysis
MODEL_SNAPSHOTS = []

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solves the gridworld maze problem, as described in Sutton & Barto, 2018')
    #parser.add_argument('-update_frequency', nargs='?', type=int, default=1000, help='The number of time steps to wait before upddating the target network used by neural network agents. The default is update = 1000')
    #parser.add_argument('-batch_size', nargs='?', type=int, default=10, help='The batch size used when sampling from the experience replay buffer with neural network agents. The default is batch = 10')
    #parser.add_argument('-buffer_size', nargs='?', type=int, default=1000, help='The size of the buffer used in experience replay for neural network agents. The default is buffer_size = 1000')
    parser.add_argument('-plot_files', nargs='?', type=str, help='The file names of results files to load and plot. The files must be present in the Results directory.')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model snapshots trained for the current run of the program.')
    parser.add_argument('-load_models', nargs='?', type=str, help='The file names of a pre-trained model snapshots to load for visualization.')
    parser.add_argument('-trial_frequency', nargs='?', type=int, default=5, help='The number of episoodes after which to run a trial of the polic learned by the Q-Agent. The default is trial_frequency = 10')
    parser.add_argument('-max', nargs='?', type=int, default=1000, help='The maximum number of stepds the agent can take before the episode terminates. The default is max = 1000')
    parser.add_argument('-run', nargs='?', type=int, default=10, help='The number of independent runs per agent. Default value = 10.')
    parser.add_argument('-epi', nargs='?', type=int, default=50, help='The number of episodes per run for each agent. Default value = 50.')
    parser.add_argument('-l', nargs='?', type=float, default=1.0, help='Lambda parameter specifying the weighting for the auxiliary loss. Ranges from 0 to 1.0 inclusive. Default value = 1.0')
    parser.add_argument('-t', nargs='?', type=float, default=0.90, help='Parameter specifying the eligibility trace for sarsa lambda. Ranges from 0.0 to 1.0, inclusive. Default value = 0.90')
    parser.add_argument('-e', nargs='?', type=float, default=0.01, help='Epsilon paramter value to be used by the agent when selecting actions epsilon greedy style. Default = 0.01 This represents the minimum value epislon will decay to, since it initially starts at 1')
    parser.add_argument('-a', nargs='?', type=float, default=0.001, help='Alpha parameter which specifies the step size for the update rule. Default value = 0.001')
    parser.add_argument('-g', nargs='?', type=float, default=0.99, help='Discount factor, which determines how far ahead from the current state the agent takes into consideraton when updating its values. Default = 0.95')
    parser.add_argument('-n', nargs='?', type=int, default=1, help='The number of states to use in the auxiliary prediction tasks. This is currently disabled, and will default to n = 1')
    parser.add_argument('-env', choices={GRID, CONTINUOUS, WINDY}, type=str, default=GRID, help='Which environment to train the agent on. Options are: {}, {}, {}. Default is env = "grid"'. format(GRID, CONTINUOUS, WINDY))
    parser.add_argument('-name', nargs='?', default='default_name', type=str, help='The name of the file to save the experiment results to. File format is png.')
    parser.add_argument('-actions', nargs='?', type=int, default=4, help='The number of moves considered valid for the agent must be 4, 8, or 9. This only applies to the windy gridwordl experiment. Default value is actions = 4')
    parser.add_argument('--windy', action='store_true', help='Specify whether to use the windy gridworld environment')
    parser.add_argument('--continuous', action='store_true', help='Specify whether to use the continuous gridworld environment')
    parser.add_argument('--stochastic', action='store_true', help='Specify whether to train the agent with stochastic obstacle states, rather than simple wall states that the agent can\'t pass through.')
    parser.add_argument('--sparse', action='store_true', help='Specify whether the environment reward structure is rich or sparse. Rich rewards include -1 at every non-terminal state and 0 at the terminal state. Sparse rewards include 0 at every non-terminal state, and 1 at the terminal state')
    parser.add_argument('--sweep', action='store_true', help='Specify whether the program should ignore the input parameters provided and do a parameter sweep over all of its possible parameters and compare the best parameter setting per agent as well as all agent\'s at each parameter setting. See grid_exp.py to set the parameters that are swept over')
    parser.add_argument('--sweep_neural', action='store_true', help='Sweep the parameters of the single task neural network and display the results on a single plot. Parameters swept: alpha, buffer_size, target_network update size.')
    parser.add_argument('--hot', action='store_true', help='Whether to encode the neural network in 1 hot or (x, y) format.')
    parser.add_argument('--visualize', action='store_true', help='Whether to plot the value function, t-SNE, and CCA visualizations for each agent. Default = false')
    parser.add_argument('--q_plot', action='store_true', help='Whether to plot the performance of the q policy periodically. Default = false')
    parser.add_argument('-correlate', nargs='?', type=str, help='Compute the pearson and spearman correlation coefficients of the provided data files. The files must be present in the results directory.')

    args = parser.parse_args()

    #Validate user arguments
    float_params = [args.e, args.a, args.g, args.l, args.t]
    for param in float_params:
        if param < 0.0 or param > 1.0:
            exit("Epsilon, Alpha, Gamma, Lambda, and Trace, parameters must be a value between 0 and 1, inclusive.")

    if args.actions not in VALID_MOVE_SETS:
        exit("The valid move sets are 4, 8, and 9. Please choose one of those.")

    if args.env == WINDY and args.actions != 4:
        exit("The only valid action set for all non-windy gridworlds is 4 moves.")
    elif args.env == CONTINUOUS and (args.stochastic or args.hot):
        exit("The continuous gridworld environment does not support stochastic obstacle states, nor 1-hot vector encodings")
    else:
        print('Running the {} environment'.format(args.env))

    if args.sweep:
        #To ensure the same parameters are sampled from on multiple invocations of the program
        np.random.seed(0)
        random.seed(0)

        IS_SWEEP = True
        alpha_params = sample_params_log_uniform(0.001, 0.1, 6)
        gamma_params = GAMMA = [0.99]
        lambda_params = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05]
        trace_params = [0.90]
        replay_context_sizes = [1]

        #alpha_params = [1, 0.1]
        #lambda_params = sample_params_log_uniform(0.001, 1, 6)
        #lambda_params = [1]

        print('Sweeping alpha parameters: {}'.format(str(alpha_params)))
        print('Sweeping lambda parameters: {}'.format(str(lambda_params)))

    elif args.sweep_neural:
        #alpha_params = sample_params_log_uniform(0.001, 0.1, 6)
        gamma_params = [0.99]
        alpha_params = [0.001, 0.005, 0.0075, 0.01, 0.05, 0.1, 0.15, 0.25]
        buffer_size_params = [100, 1000, 10000]
        update_freq_params = [1000]

        #alpha_params = [0.001, 0.005]
        #buffer_size_params = [10000]

        sweeped_pairs = [(0.001, 100), (0.001, 1000)]

    else:
        alpha_params = [args.a]
        gamma_params = [args.g]
        lambda_params = [args.l]
        replay_context_sizes = [args.n]
        trace_params = [args.t]

    EPSILON = args.e
    IS_STOCHASTIC = args.stochastic
    IS_1_HOT = args.hot
    NUM_ACTIONS = args.actions
    IS_SPARSE = args.sparse
    RESULTS_FILE_NAME = args.name

    num_episodes = args.epi
    max_steps = args.max
    num_runs = args.run

    if args.correlate:
        data_files = args.correlate.split()
        compute_correlations(data_files[0], data_files[1])
        exit("Correlation computation completed!")

    #The main experiment loop
    if not args.sweep_neural:
        all_params = list(product(AGENTS, alpha_params, gamma_params)) + list(product(AUX_AGENTS, alpha_params, gamma_params, replay_context_sizes, lambda_params))
    else:
        all_params = list(product([a_globs.NEURAL], alpha_params, gamma_params, buffer_size_params, update_freq_params))

    all_results = []
    all_param_settings = []
    all_Q_results = []
    all_Q_param_settings = []

    print("Starting the experiment...")
    i = 0
    for param_setting in all_params:
        if not args.sweep_neural or not (param_setting[1], param_setting[3]) in sweeped_pairs:
            cur_agent = param_setting[0]

            #Load the appropriate agent and environment files
            if cur_agent == a_globs.SARSA_LAMBDA:
                print("Training agent: {} with alpha = {} gamma = {} trace = {}".format(param_setting[0], param_setting[1], param_setting[2], trace_params[0]))
                RLGlue("{}.{}_env".format(ENV_DIR, args.env), "{}.{}_agent".format(AGENT_DIR, cur_agent))
            elif cur_agent == a_globs.NEURAL and args.sweep_neural:
                log_contents = "Training agent: {} with alpha = {} gamma = {} buffer_size = {}, update_freq = {}".format(param_setting[0], param_setting[1], param_setting[2], param_setting[3], param_setting[4])
                print(log_contents)
                write_to_log(log_contents)
                RLGlue("{}.{}_env".format(ENV_DIR, args.env), "{}.{}_agent".format(AGENT_DIR, cur_agent))
            elif cur_agent in AGENTS:
                RLGlue("{}.{}_env".format(ENV_DIR, args.env), "{}.{}_agent".format(AGENT_DIR, cur_agent))
                print("Training agent: {} with alpha = {} gamma = {}".format(param_setting[0], param_setting[1], param_setting[2]))
            elif cur_agent in AUX_AGENTS:
                print("Training agent: {} with alpha = {} gamma = {}, context size = {}, and lambda = {}".format(param_setting[0], param_setting[1], param_setting[2], param_setting[3], param_setting[4]))
                RLGlue("{}.{}_env".format(ENV_DIR, args.env), "{}.aux_agent".format(AGENT_DIR))
            else:
                exit('ERROR: Invalid agent string {} provided!'.format(cur_agent))

            #These do not require the main experiment to finish, so we finish early
            if args.plot_files:
                plot_files(args.plot_files.split(), args.q_plot)
                exit("Performance Plotting Completed!")

            if args.visualize:
                do_visualization(num_episodes, max_steps, 10, all_params[0])
                exit("Visualization Completed!")

            cur_param_results = []
            cur_Q_param_results = []
            for run in range(num_runs):
                #Set random seeds to ensure replicability of results and the same trajectory of experience across agents for valid comparison
                np.random.seed(run)
                random.seed(run)

                #Send the agent and environment parameters to use for the current run
                send_params(cur_agent, param_setting)

                run_results = []
                Q_run_results = []
                print("Run number: {}".format(str(run)))
                RL_init()
                #online_model = clone_model(a_globs.model)
                for episode in range(num_episodes):
                    print("Episode number: {}".format(str(episode)))
                    # print('num steps prior to episode start')
                    # print(RL_num_steps())
                    RL_episode(max_steps)
                    # print('online results')
                    # print(run_results)
                    # print('num steps after episode')
                    # print(RL_num_steps())
                    run_results.append(RL_num_steps())
                    RL_cleanup()
                    #print(run_results)

                    #Run a test trial without learning or exploration to test the off-policy learned by the agent
                    if args.q_plot and is_neural(cur_agent) and (episode % args.trial_frequency == 0 or episode == num_episodes - 1):
                        print("Running a trial episode to test the Q-policy at episode: {} of run {} for agent: {}".format(episode, run, cur_agent))
                        old_weights = a_globs.model.get_weights()
                        a_globs.is_trial_episode = True
                        RL_episode(max_steps)
                        # print('offline results')
                        # print(Q_run_results)
                        Q_run_results.append(RL_num_steps())
                        RL_cleanup()
                        a_globs.is_trial_episode = False
                        a_globs.model.set_weights(old_weights)
                        # print(RL_num_steps())
                        # print(Q_run_results)

                cur_param_results.append(run_results)
                cur_Q_param_results.append(Q_run_results)

            all_results.append(cur_param_results)
            all_param_settings.append(param_setting)
            all_Q_results.append(cur_Q_param_results)
            all_Q_param_settings.append(param_setting)

            if args.sweep_neural:
                cur_agent = all_param_settings[i][0]
                if not is_neural(cur_agent):
                    exit('ERROR: The current agent is not a  neural network, but you are attempting to sweep it! Please ensure that the agent set up in grid_exp.py is a neural network!')
                #print(all_results)
                cur_data = [np.mean(run) for run in zip(*all_results[i])]
                episodes = [episode for episode in range(num_episodes)]

                #print(episodes)
                #print(cur_data)

                save_results(cur_data, cur_agent, RESULTS_FILE_NAME + str(i))

                plt.figure()
                plt.plot(episodes, cur_data, GRAPH_COLOURS[0], label="Agent = {}  Alpha = {} Buffer_size = {}, Update_freq = {}".format(cur_agent, str(all_param_settings[i][1]), str(all_param_settings[i][3]), str(all_param_settings[i][4])))
                setup_plot(plot_title='On Policy Results')
                do_plotting(i, RESULTS_FILE_NAME)
                plt.clf()

                #Plot the results for the optimal Q-value policy
                cur_agent = all_Q_param_settings[i][0]
                if not is_neural(cur_agent):
                    exit('ERROR: The current agent is not a  neural network, but you are attempting to sweep it! Please ensure that the agent set up in grid_exp.py is a neural network!')
                cur_data = [np.mean(run) for run in zip(*all_Q_results[i])]
                episodes = [episode for episode in range(0, num_episodes, args.trial_frequency)]

                save_results(cur_data, cur_agent, RESULTS_FILE_NAME + "Q_results" + str(i), q_plot=True)

                plt.figure()
                #print(episodes)
                #print(cur_data)
                #cur_data = [1100, 1500, 2000]
                #print(all_Q_param_settings)
                plt.plot(episodes, cur_data, GRAPH_COLOURS[0], label="Agent = {}  Alpha = {} Buffer_size = {}, Update_freq = {}".format(cur_agent, str(all_Q_param_settings[i][1]), str(all_Q_param_settings[i][3]), str(all_Q_param_settings[i][4])))
                setup_plot(args.trial_frequency, 'Off Policy Results')
                do_plotting(i, RESULTS_FILE_NAME + "Q_results")
                plt.clf()
                i += 1
        else:
            log_contents = "Skipping agent: {} with alpha = {} gamma = {} buffer_size = {}, update_freq = {}".format(param_setting[0], param_setting[1], param_setting[2], param_setting[3], param_setting[4])
            print(log_contents)
            write_to_log(log_contents)

    #Process and plot the results of a generic non-neural network specific sweep
    if args.sweep:
        #Average the results for each parameter setting over all of the runs and find the best parameter setting for each agent
        all_params_no_agent = list(product(alpha_params, gamma_params)) + list(product(alpha_params, gamma_params, replay_context_sizes, lambda_params))
        param_setting_results = {param_tuple : {} for param_tuple in all_params_no_agent}
        best_agent_results = {}
        avg_results = []
        for i in range(len(all_results)):
            cur_data = [np.mean(run) for run in zip(*all_results[i])]
            avg_results.append(cur_data)
            cur_overall_average = np.mean(cur_data) #NOTE: This is the average time spent getting to the goal across all runs for all episodes, for that agent, to give it a final overall score with which to compare it to other agents

            cur_agent = all_param_settings[i][0]
            agent_data = namedtuple("agent_data_tuple", ["data", "average", "params"])
            agent_data.average = cur_overall_average
            agent_data.data = cur_data
            agent_data.params = all_param_settings[i]

            #Group results by the best performing parameter setting for each agent
            if cur_agent not in best_agent_results or cur_overall_average < best_agent_results[cur_agent].average:
                best_agent_results[cur_agent] = agent_data

            #Group by each parameter setting to compare agent's across specific parameter settings
            cur_param_setting = tuple(all_param_settings[i][1:])
            param_setting_results[cur_param_setting][cur_agent] = agent_data

        #Create a table to show the best parameters for each agent
        i = 0
        for agent in best_agent_results.keys():
            #print(agent)
            #print(best_agent_results[agent].data)
            episodes = [episode for episode in range(num_episodes)]
            #print(best_agent_results[agent].params)

            save_results(best_agent_results[agent].data, agent, RESULTS_FILE_NAME)

            if agent in AUX_AGENTS:
                plt.plot(episodes, best_agent_results[agent].data, GRAPH_COLOURS[i], label="AGENT = {}  Alpha = {} Gamma = {} N = {}, Lambda = {}".format(best_agent_results[agent].params[0], str(best_agent_results[agent].params[1]), str(best_agent_results[agent].params[2]), str(best_agent_results[agent].params[3]), str(best_agent_results[agent].params[4])))
            elif agent == a_globs.SARSA_LAMBDA:
                plt.plot(episodes, best_agent_results[agent].data, GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {} Trace = {}".format(best_agent_results[agent].params[0], str(best_agent_results[agent].params[1]), str(best_agent_results[agent].params[2]), str(best_agent_results[agent].params[3])))
            else:
                plt.plot(episodes, best_agent_results[agent].data, GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {}".format(best_agent_results[agent].params[0], str(best_agent_results[agent].params[1]), str(best_agent_results[agent].params[2])))
            i += 1
        setup_plot()
        do_plotting(filename=RESULTS_FILE_NAME)

        #print('param setting results pre merge')
        #print(param_setting_results)

        #Merge the relevant regular and auxiliary agent parameter settings results so that we can compare them on the same tables
        reg_agent_param_settings = list(product(alpha_params, gamma_params))
        for reg_agent_params in reg_agent_param_settings:
            for key in param_setting_results.keys():
                if reg_agent_params != key and reg_agent_params == key[:len(key) - NUM_AUX_AGENT_PARAMS]:
                    param_setting_results[key] = merge_two_dicts(param_setting_results[key], param_setting_results[reg_agent_params])
            del param_setting_results[reg_agent_params]

        #Create a table for each parameter setting, showing all agents per setting
        file_name_suffix = 1
        #print('param setting results post merge')
        #print(param_setting_results)
        for param_setting in param_setting_results:
            plt.clf()
            cur_param_setting_result = param_setting_results[param_setting]
            i = 0
            #print('param setting')
            #print(param_setting)
            #print('param setting keys')
            #print(cur_param_setting_result.keys())
            for agent in cur_param_setting_result.keys():

                #print('param setting data')
                #print(cur_param_setting_result[agent].data)

                save_results(cur_param_setting_result[agent].data, agent, ''.join([RESULTS_FILE_NAME, str(file_name_suffix)]))
                episodes = [episode for episode in range(num_episodes)]
                if agent in AUX_AGENTS:
                    plt.plot(episodes, cur_param_setting_result[agent].data, GRAPH_COLOURS[i], label="AGENT = {}  Alpha = {} Gamma = {} N = {}, Lambda = {}".format(agent, str(cur_param_setting_result[agent].params[1]), str(cur_param_setting_result[agent].params[2]), str(cur_param_setting_result[agent].params[3]), str(cur_param_setting_result[agent].params[4])))
                elif cur_agent == a_globs.SARSA_LAMBDA:
                    plt.plot(episodes, cur_param_setting_result[agent].data, GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {} Trace = {}".format(agent, str(cur_param_setting_result[agent].params[1]), str(cur_param_setting_result[agent].params[2]), str(cur_param_setting_result[agent].params[2])))
                else:
                    plt.plot(episodes, cur_param_setting_result[agent].data, GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {}".format(agent, str(cur_param_setting_result[agent].params[1]), str(cur_param_setting_result[agent].params[2])))
                i += 1
            setup_plot()
            do_plotting(file_name_suffix, RESULTS_FILE_NAME)
            file_name_suffix += 1

    elif not args.sweep_neural:
        if args.save_model:
            save_model([a_globs.model], RESULTS_FILE_NAME)

        #Average the results over all of the runs for the single parameters setting provided
        avg_results = []
        for i in range(len(all_results)):
            avg_results.append([np.mean(run) for run in zip(*all_results[i])])
            cur_data = [episode for episode in range(num_episodes)]
            cur_agent = str(all_param_settings[i][0])

            save_results(avg_results[i], cur_agent, RESULTS_FILE_NAME)

            if cur_agent in AUX_AGENTS:
                plt.plot(cur_data, avg_results[i], GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {} N = {}, Lambda = {}".format(cur_agent, str(all_param_settings[i][1]), str(all_param_settings[i][2]), all_param_settings[i][3], str(all_param_settings[i][4])))
            elif cur_agent == a_globs.SARSA_LAMBDA:
                plt.plot(cur_data, avg_results[i], GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {} Trace = {}".format(cur_agent, str(all_param_settings[i][1]), str(all_param_settings[i][2]), args.t))
            else:
                plt.plot(cur_data, avg_results[i], GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {}".format(cur_agent, str(all_param_settings[i][1]), str(all_param_settings[i][2])))
        setup_plot()
        do_plotting(filename=RESULTS_FILE_NAME)
        plt.clf()

        if args.q_plot:
            avg_results = []
            for i in range(len(all_Q_results)):
                avg_results.append([np.mean(run) for run in zip(*all_Q_results[i])])
                cur_data = [episode for episode in range(0, num_episodes + 1, args.trial_frequency)]
                cur_agent = str(all_param_settings[i][0])

                save_results(avg_results[i], cur_agent, RESULTS_FILE_NAME + 'Q_results', q_plot=True)

                if cur_agent in AUX_AGENTS:
                    plt.plot(cur_data, avg_results[i], GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {} N = {}, Lambda = {}".format(cur_agent, str(all_Q_param_settings[i][1]), str(all_Q_param_settings[i][2]), all_Q_param_settings[i][3], str(all_Q_param_settings[i][4])))
                else:
                    plt.plot(cur_data, avg_results[i], GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {}".format(cur_agent, str(all_Q_param_settings[i][1]), str(all_Q_param_settings[i][2])))
            setup_plot(args.trial_frequency, 'Q_results')
            do_plotting(filename=RESULTS_FILE_NAME + "Q_results")

    print("Experiment completed!")
