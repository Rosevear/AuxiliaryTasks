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
from time import sleep

import argparse
import json
import random
import numpy as np
import platform
from itertools import product
from collections import namedtuple
from math import log

import matplotlib as mpl
if platform.system() == 'Darwin':
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
import matplotlib.pyplot as plt

###### HELPER FUNCTIONS START ##############
def setup_plot():
    print("Plotting the results...")
    plt.ylabel('Steps per episode')
    plt.xlabel("Episode")
    plt.axis([0, num_episodes, 0, max_steps + 1000])
    plt.legend(loc='center', bbox_to_anchor=(0.50, 0.90))

def do_plotting(suffix=0):
    if RESULTS_FILE_NAME:
        print("Saving the results...")
        if suffix:
            plt.savefig("{}.png".format(RESULTS_FILE_NAME + str(suffix)), format="png")
        else:
            plt.savefig("{}.png".format(RESULTS_FILE_NAME), format="png")
    else:
        print("Displaying the results...")
        plt.show()

#Taken from https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression on September 22nd 2018
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

def save_results(results, cur_agent, filename='Default File Name'):
    """
    Compute the mean and standard deviation of the current results,
    and save the results and statistics to the specified file
    """

    with open('{} results'.format(filename), 'a+') as results_file:
        results_file.write('{} average results\n'.format(cur_agent))
        results_file.write(json.dumps(results))
        results_file.write('\n')
        results_file.write('{} agent summary statistics\n'.format(cur_agent))
        results_file.write('mean: {} standard deviation: {}\n'.format(np.mean(results), np.std(results)))

###### HELPER FUNCTIONS END #################

#TODO: Consider creating a named tuple for each possible param combination, so that wen refer to params by name rather than having to keep the order in mind when accessing them
GRAPH_COLOURS = ('r', 'g', 'b', 'c', 'm', 'y', 'k')
#AUX_AGENTS = ['reward', 'state', 'redundant', 'noise']
AUX_AGENTS = ['reward', 'state']
AGENTS = ['random', 'neural']
#AGENTS = []
#AGENTS = ['neural']
VALID_MOVE_SETS = [4, 8, 9]
NUM_AUX_AGENT_PARAMS = 2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solves the gridworld maze problem, as described in Sutton & Barto, 2018')
    parser.add_argument('-run', nargs='?', type=int, default=10, help='The number of independent runs per agent. Default value = 10.')
    parser.add_argument('-epi', nargs='?', type=int, default=50, help='The number of episodes per run for each agent. Default value = 50.')
    parser.add_argument('-l', nargs='?', type=float, default=1.0, help=' Lambda parmaeter specifying the weighting for the auxiliary loss. Ranges from 0 to 1.0 inclusive. Default value = 1.0')
    parser.add_argument('-e', nargs='?', type=float, default=0.01, help='Epsilon paramter value for to be used by the agent when selecting actions epsilon greedy style. Default = 0.01 This represents the minimum value epislon will decay to, since it initially starts at 1')
    parser.add_argument('-a', nargs='?', type=float, default=0.001, help='Alpha parameter which specifies the step size for the update rule. Default value = 0.001')
    parser.add_argument('-g', nargs='?', type=float, default=0.99, help='Discount factor, which determines how far ahead from the current state the agent takes into consideraton when updating its values. Default = 0.95')
    parser.add_argument('-n', nargs='?', type=int, default=1, help='The number of states to use in the auxiliary prediction tasks. Default n = 1')
    parser.add_argument('-actions', nargs='?', type=int, default=4, help='The number of moves considered valid for the agent must be 4, 8, or 9. This only applies to the windy gridwordl experiment. Default value is actions = 4')
    parser.add_argument('--windy', action='store_true', help='Specify whether to use the windy gridworld environment')
    parser.add_argument('--continuous', action='store_true', help='Specify whether to use the continuous gridworld environment')
    parser.add_argument('--stochastic', action='store_true', help='Specify whether to train the agent with stochastic obstacle states, rather than simple wall states that the agent can\'t pass through.')
    parser.add_argument('--sparse', action='store_true', help='Specify whether the environment reward structure is rich or sparse. Rich rewards include -1 at every non-terminal state and 0 at the terminal state. Sparse rewards include 0 at every non-terminal state, and 1 at the terminal state.')
    parser.add_argument('--name', nargs='?', type=str, help='The name of the file to save the experiment results to. File format is png.')
    parser.add_argument('--sweep', action='store_true', help='Specify whether the agent should ignore the input parameters provided (alpha and experience buffer size) and do a parameter sweep')
    parser.add_argument('--hot', action='store_true', help='Whether to encode the neural network in 1 hot or (x, y) format.')

    args = parser.parse_args()

    if args.e < 0 or args.e > 1 or args.a < 0 or args.a > 1 or args.g < 0 or args.g > 1 or args.l < 0 or args.l > 1:
        exit("Epsilon, Alpha, Gamma, Lambda parameters must be a value between 0 and 1, inclusive.")

    if args.actions not in VALID_MOVE_SETS:
        exit("The valid move sets are 4, 8, and 9. Please choose one of those.")

    if args.windy:
        RLGlue("windy_grid_env", "grid_agent")
    else:
        if args.actions != 4:
            exit("The only valid action set for the all non-windy gridworlds is 4 moves.")
        if args.continuous:
            if args.stochastic or args.hot:
                exit("The continuous gridoworld environment does not support stochastic obstacle states nor 1-hot vector encodings")
            else:
                RLGlue("continuous_grid_env", "grid_agent")
        else:
            RLGlue("grid_env", "grid_agent")


    #Agent and environment parameters, and experiment settings
    if args.sweep:
        #To ensure the same parameters are sampled from on multiple invocations of the program
        np.random.seed(0)
        random.seed(0)

        IS_SWEEP = True
        alpha_params = sample_params_log_uniform(0.001, 0.1, 6)
        #alpha_params = [1, 0.1]
        gamma_params = GAMMA = [0.99]
        #lambda_params = sample_params_log_uniform(0.001, 1, 6)
        lambda_params = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05]
        #lambda_params = [1]
        replay_context_sizes =  [1]

        print('Sweeping alpha parameters: {}'.format(str(alpha_params)))
        print('Sweeping lambda parameters: {}'.format(str(lambda_params)))

    else:
        IS_SWEEP = False
        alpha_params = [args.a]
        gamma_params = [args.g]
        lambda_params = [args.l]
        replay_context_sizes = [args.n]

    EPSILON = args.e
    IS_STOCHASTIC = args.stochastic
    IS_1_HOT = args.hot
    NUM_ACTIONS = args.actions
    IS_SPARSE = args.sparse
    RESULTS_FILE_NAME = args.name

    num_episodes = args.epi
    max_steps = 1000
    num_runs = args.run

    #The main experiment loop
    all_params = list(product(AGENTS, alpha_params, gamma_params)) + list(product(AUX_AGENTS, alpha_params, gamma_params, replay_context_sizes, lambda_params))
    all_results = []
    all_param_settings = []
    print("Starting the experiment...")
    for param_setting in all_params:
        cur_agent = param_setting[0]
        if cur_agent in AUX_AGENTS:
            print("Training agent: {} with alpha = {} gamma = {}, context size = {}, and lambda = {}".format(param_setting[0], param_setting[1], param_setting[2], param_setting[3], param_setting[4]))
        else:
            print("Training agent: {} with alpha = {} gamma = {}".format(param_setting[0], param_setting[1], param_setting[2]))

        cur_param_results = []
        for run in range(1, num_runs + 1):
            #Set random seeds to ensure replicability of results and the same trajectory of experience across agents for valid comparison
            np.random.seed(run)
            random.seed(run)

            #Send the agent and environment parameters to use for the current run
            if cur_agent in AUX_AGENTS:
                agent_params = {"EPSILON": EPSILON, "AGENT": param_setting[0], "ALPHA": param_setting[1], "GAMMA": param_setting[2], "N": param_setting[3], "LAMBDA": param_setting[4], "IS_STOCHASTIC": IS_STOCHASTIC, "IS_1_HOT": IS_1_HOT}
            else:
                agent_params = {"EPSILON": EPSILON, "AGENT": param_setting[0], "ALPHA": param_setting[1], "GAMMA": param_setting[2], "IS_STOCHASTIC": IS_STOCHASTIC, "IS_1_HOT": IS_1_HOT}
            enviro_params = {"NUM_ACTIONS": NUM_ACTIONS, "IS_STOCHASTIC": IS_STOCHASTIC, "IS_SPARSE": IS_SPARSE}
            RL_agent_message(json.dumps(agent_params))
            RL_env_message(json.dumps(enviro_params))

            run_results = []
            print("Run number: {}".format(str(run)))
            RL_init()
            for episode in range(num_episodes):
                print("Episode number: {}".format(str(episode)))
                RL_episode(max_steps)
                run_results.append(RL_num_steps())
                RL_cleanup()
            cur_param_results.append(run_results)
        all_results.append(cur_param_results)
        all_param_settings.append(param_setting)

        #print(all_param_settings)
        #print("BREAK")
        #print(all_results)

    #Process and plot the results
    if IS_SWEEP:
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
            else:
                plt.plot(episodes, best_agent_results[agent].data, GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {}".format(best_agent_results[agent].params[0], str(best_agent_results[agent].params[1]), str(best_agent_results[agent].params[2])))
            i += 1
        setup_plot()
        do_plotting()

        # print('param setting results pre merge')
        # print(param_setting_results)

        #Merge the relevant regular and auxiliary agent parameter settings results so that we can compare them on the same tables
        reg_agent_param_settings = list(product(alpha_params, gamma_params))
        for reg_agent_params in reg_agent_param_settings:
            for key in param_setting_results.keys():
                if reg_agent_params != key and reg_agent_params == key[:len(key) - NUM_AUX_AGENT_PARAMS]:
                    param_setting_results[key] = merge_two_dicts(param_setting_results[key], param_setting_results[reg_agent_params])
            del param_setting_results[reg_agent_params]

        #Create a table for each parameter setting, showing all agents per setting
        file_name_suffix = 1
        # print('param setting results post merge')
        # print(param_setting_results)
        for param_setting in param_setting_results:
            plt.clf()
            cur_param_setting_result = param_setting_results[param_setting]
            i = 0
            # print('param setting keys')
            # print(cur_param_setting_result.keys())
            for agent in cur_param_setting_result.keys():

                save_results(cur_param_setting_result[agent].data, agent, ''.join([RESULTS_FILE_NAME, str(file_name_suffix)]))
                episodes = [episode for episode in range(num_episodes)]
                if agent in AUX_AGENTS:
                    plt.plot(episodes, cur_param_setting_result[agent].data, GRAPH_COLOURS[i], label="AGENT = {}  Alpha = {} Gamma = {} N = {}, Lambda = {}".format(agent, str(cur_param_setting_result[agent].params[1]), str(cur_param_setting_result[agent].params[2]), str(cur_param_setting_result[agent].params[3]), str(cur_param_setting_result[agent].params[4])))
                else:
                    plt.plot(episodes, cur_param_setting_result[agent].data, GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {}".format(agent, str(cur_param_setting_result[agent].params[1]), str(cur_param_setting_result[agent].params[2])))
                i += 1
            setup_plot()
            do_plotting(file_name_suffix)
            file_name_suffix += 1

    else:
        #Average the results over all of the runs
        avg_results = []
        for i in range(len(all_results)):
            avg_results.append([np.mean(run) for run in zip(*all_results[i])])

        for i in range(len(avg_results)):
            cur_data = [episode for episode in range(num_episodes)]
            cur_agent = str(all_param_settings[i][0])

            save_results(avg_results[i], cur_agent, RESULTS_FILE_NAME)

            if cur_agent in AUX_AGENTS:
                plt.plot(cur_data, avg_results[i], GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {} N = {}, Lambda = {}".format(cur_agent, str(all_param_settings[i][1]), str(all_param_settings[i][2]), all_param_settings[i][3], str(all_param_settings[i][4])))
            else:
                plt.plot(cur_data, avg_results[i], GRAPH_COLOURS[i], label="AGENT = {} Alpha = {} Gamma = {}".format(cur_agent, str(all_param_settings[i][1]), str(all_param_settings[i][2])))
        setup_plot()
        do_plotting()
    print("Experiment completed!")
