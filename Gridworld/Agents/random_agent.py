#!/usr/bin/env python

from __future__ import division

from Globals.generic_globals import *
from Utils.agent_helpers import *

from Utils.utils import rand_in_range, rand_un

import json
import platform

from rl_glue import RL_num_episodes, RL_num_steps
import Globals.grid_agent_globals as a_globs

import matplotlib as mpl
if platform.system() == 'Darwin':
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
import matplotlib.pyplot as plt

def agent_init():
    pass


def agent_start(state):

    return rand_in_range(a_globs.NUM_ACTIONS)


def agent_step(reward, state):

    return rand_in_range(a_globs.NUM_ACTIONS)

def agent_end(reward):

    pass

    return

def agent_cleanup():
    pass
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

    return
