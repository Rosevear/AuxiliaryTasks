Source code of experiments for fall 2018 capstone project at the University of Alberta, on visualizing and measuring neural network representations for reinforcement learning.

Written in Python 2.7.2

Installation

clone the repository and run pip install on requirements.txt to download the relevant dependencies.

To run

from the Gridworld directory run: python grid_exp.py args

Use python grid_exp.py --help to see a listing of the various arguments available.

The agent files currently used are random, tabular Q-learning, sarsa-lambda, and a neural network agent. In order to choose which to use, the agent file name must be specified within the AGENTS list in grid_exp.py, as there is no command line argument to specify the agent as of yet.

Similarly, the batch size, buffer size, and target network frequency size need to be specified by modifying the grid_ageny_globals.py file in the Globals directory, as, again, there is no command line argument to specify these parameters as of yet.


