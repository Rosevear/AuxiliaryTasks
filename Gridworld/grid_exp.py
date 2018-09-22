#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

  Modified by Cody Rosevear for use in CMPUT659, Winter 2018
  and CMPUT701, Fall 2018, University of Alberta
"""

from __future__ import division
from rl_glue import *  # Required for RL-Glue
from time import sleep

import argparse
import json
import random
import pickle
import numpy as np
import platform
from itertools import product
from collections import namedtuple

import matplotlib as mpl
if platform.system() == 'Darwin':
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
import matplotlib.pyplot as plt


GRAPH_COLOURS = ('r', 'g', 'b', 'c', 'm', 'y', 'k')
#AGENTS = ['random', 'tabularQ', 'neural', 'reward', 'state', 'redundant', 'noise']
AGENTS = ['random', 'tabularQ', 'neural']
#AGENTS = ['random']
VALID_MOVE_SETS = [4, 8, 9]

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

def setup_plot():
    print("Plotting the results...")
    plt.ylabel('Steps per episode')
    plt.xlabel("Episode")
    plt.axis([0, num_episodes, 0, max_steps + 1000])
    plt.legend(loc='center', bbox_to_anchor=(0.50, 0.90))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solves the gridworld maze problem, as described in Sutton & Barto, 2018')
    parser.add_argument('-e', nargs='?', type=float, default=0.01, help='Epsilon paramter value for to be used by the agent when selecting actions epsilon greedy style. Default = 0.01 This represents the minimum value epislon will decay to, since it initially starts at 1')
    parser.add_argument('-a', nargs='?', type=float, default=0.001, help='Alpha parameter which specifies the step size for the update rule. Default value = 0.001')
    parser.add_argument('-g', nargs='?', type=float, default=1, help='Discount factor, which determines how far ahead from the current state the agent takes into consideraton when updating its values. Default = 1.0')
    parser.add_argument('-n', nargs='?', type=int, default=3, help='The number of states to use in the auxiliary prediction tasks. Default n = 3')
    parser.add_argument('-actions', nargs='?', type=int, default=4, help='The number of moves considered valid for the agent must be 4, 8, or 9. This only applies to the windy gridwordl experiment. Default value is actions = 4')
    parser.add_argument('--windy', action='store_true', help='Specify whether to use a single step or multistep agent.')
    parser.add_argument('--stochastic', action='store_true', help='Specify whether to train the agent with stochastic obstacle states, rather than simple wall states that the agent can\'t pass through.')
    parser.add_argument('--sparse', action='store_true', help='Specify whether the environment reward structure is rich or sparse. Rich rewards include -1 at every non-terminal state and 0 at the terminal state. Sparse rewards include 0 at every non-terminal state, and 1 at the terminal state.')
    parser.add_argument('--name', nargs='?', type=str, help='The name of the file to save the experiment results to. File format is png.')
    parser.add_argument('--sweep', action='store_true', help='Specify whether the agent should ignore the input parameters provided (alpha and experience buffer size) and do a parameter sweep')

    args = parser.parse_args()

    if args.e < 0 or args.e > 1 or args.a < 0 or args.a > 1 or args.g < 0 or args.g > 1:
        exit("Epsilon, Alpha, and Gamma parameters must be a value between 0 and 1, inclusive.")

    if args.actions not in VALID_MOVE_SETS:
        exit("The valid move sets are 4, 8, and 9. Please choose one of those.")

    if args.windy:
        RLGlue("windy_grid_env", "grid_agent")
    else:
        if args.actions != 4:
            exit("The only valid action set for the non-windy gridworld is 4 moves.")
        RLGlue("grid_env", "grid_agent")


    #Agent and environment parameters, and experiment settings
    if args.sweep:
        IS_SWEEP = True
        alpha_params = [0.1, 0.01, 0.001]
        replay_buffer_sizes =  [1, 10, 50]
        gamma_params = GAMMA = [0, 0.95, 1]

    else:
        IS_SWEEP = False
        alpha_params = [args.a]
        replay_buffer_sizes = [args.n]
        gamma_params = [args.g]

    EPSILON = args.e
    IS_STOCHASTIC = args.stochastic
    NUM_ACTIONS = args.actions
    IS_SPARSE = args.sparse
    RESULTS_FILE_NAME = args.name

    num_episodes = 10
    max_steps = 1000
    num_runs = 1

    #The main experiment loop
    all_params = product(AGENTS, alpha_params, replay_buffer_sizes, gamma_params)
    all_results = []
    all_param_settings = []
    print("Starting the experiment...")
    for param_setting in all_params:
        print("Training agent: {} with alpha = {} , gamma = {} and buffer size = {}".format(param_setting[0], param_setting[1], param_setting[3], param_setting[2]))
        cur_param_results = []
        for run in range(num_runs):
            #Set random seeds to ensure replicability of results and the same trajectory of experience across agents for valid comparison
            np.random.seed(run)
            random.seed(run)

            #Send the agent and environment parameters to use for the current run
            agent_params = {"EPSILON": EPSILON, "ALPHA": param_setting[1], "GAMMA": param_setting[3], "AGENT": param_setting[0], "N": param_setting[2], "IS_STOCHASTIC": IS_STOCHASTIC}
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

    #Process and plot the results
    if IS_SWEEP:
        #Average the results for each parameter setting over all of the runs and find the best parameter setting for each agent
        all_params_no_agent = product(alpha_params, replay_buffer_sizes, gamma_params)
        param_setting_results = {param_tuple : {} for param_tuple in all_params_no_agent}
        best_agent_results = {}
        avg_results = []
        for i in range(len(all_results)):
            cur_data = [np.mean(run) for run in zip(*all_results[i])]
            avg_results.append(cur_data)

            #Group results by the best performing parameter setting for each agent
            cur_agent = all_param_settings[i][0]
            cur_overall_average = np.mean(cur_data)#NOTE: This is the average time spent getting to the goal across all runs for all episodes, for that agent, to give it a final overall score with which to compare it to other agents
            if cur_agent not in best_agent_results or cur_overall_average < best_agent_results[cur_agent].average:
                agent_data = namedtuple("agent_data_tuple", ["data", "average", "params"])
                agent_data.average = cur_overall_average
                agent_data.data = cur_data
                agent_data.params = all_param_settings[i]
                best_agent_results[cur_agent] = agent_data

            #Group by each parameter setting to compare agent's across specific parameter settings
            cur_param_setting = tuple(all_param_settings[i][1:])
            param_setting_results[cur_param_setting][cur_agent] = cur_data

        #Create a table to show the best parameters for each agent
        i = 0
        for agent in best_agent_results.keys():
            episodes = [episode for episode in range(num_episodes)]
            plt.plot(episodes, best_agent_results[agent].data, GRAPH_COLOURS[i], label="Epsilon Min = {} Alpha = {} Gamma = {} N = {} AGENT = {}".format(EPSILON, str(best_agent_results[agent].params[1]), str(best_agent_results[agent].params[3]), str(best_agent_results[agent].params[2]), best_agent_results[agent].params[0]))
            i += 1
        setup_plot()
        do_plotting()

        #Create a table for each parameter setting, showing all agents per setting
        file_name_suffix = 1
        for param_setting in param_setting_results:
            plt.clf()
            cur_param_setting_result = param_setting_results[param_setting]
            i = 0
            for agent in cur_param_setting_result.keys():
                episodes = [episode for episode in range(num_episodes)]
                plt.plot(episodes, cur_param_setting_result[agent], GRAPH_COLOURS[i], label="Epsilon Min = {} Alpha = {} Gamma = {} N = {} AGENT = {}".format(EPSILON, str(param_setting[0]), str(param_setting[2]), str(param_setting[1]), agent))
                i += 1
            setup_plot()
            do_plotting(file_name_suffix)
            file_name_suffix += 1

    else:
        #Average the results for each parameter setting over all of the runs
        avg_results = []
        for i in range(len(all_results)):
            avg_results.append([np.mean(run) for run in zip(*all_results[i])])

        for i in range(len(avg_results)):
            cur_data = [episode for episode in range(num_episodes)]
            plt.plot(cur_data, avg_results[i], GRAPH_COLOURS[i], label="Epsilon Min = {} Alpha = {} Gamma = {} N = {} AGENT = {}".format(EPSILON, str(all_param_settings[i][1]), str(all_param_settings[i][3]), str(all_param_settings[i][2]), all_param_settings[i][0]))
        setup_plot()
        do_plotting()

    print("Experiment completed!")
