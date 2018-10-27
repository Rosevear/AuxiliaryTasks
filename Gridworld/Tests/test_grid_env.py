import unittest
import json
import random
import numpy as np

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Agents.tabular_agent import *
from Envs.grid_env import *
import Globals.grid_env_globals as e_globs

class TestGridEnv(unittest.TestCase):

    def setUp(self):
        e_globs.IS_SPARSE = None
        e_globs.IS_STOCHASTIC = None

    def tearDown(self):
        e_globs.IS_SPARSE = None
        e_globs.IS_STOCHASTIC = None
        e_globs.current_state = e_globs.START_STATE

    def test_env_init(self):
        self.assertEqual(env_init(), None)

    def test_env_start(self):
        env_start()
        self.assertEqual(e_globs.current_state, e_globs.START_STATE)

    def test_env_step_deterministic(self):
        e_globs.IS_STOCHASTIC = False

        env_step(0)
        self.assertEqual(e_globs.current_state, [4, 0])
        env_step(1)
        self.assertEqual(e_globs.current_state, [4, 1])
        env_step(2)
        self.assertEqual(e_globs.current_state, [3, 1])
        env_step(3)
        self.assertEqual(e_globs.current_state, [3, 0])

    def test_env_step_deterministic_obstacle(self):
        e_globs.IS_STOCHASTIC = False

        env_step(1)
        env_step(1)
        self.assertEqual(e_globs.current_state, [3, 1])


    def test_env_step_stochastic_obstacle(self):
        e_globs.IS_STOCHASTIC = True
        np.random.seed(0)
        random.seed(0)
        env_step(1)
        env_step(1)
        env_step(1)
        #We know the state because we are using a random seed: I chekced what it should be in this case
        self.assertEqual(e_globs.current_state, [2, 2])


    def test_env_step_sparse(self):
        e_globs.IS_SPARSE = True

        new_context = env_step(0)
        self.assertEqual(new_context['reward'], 0)

        e_globs.current_state = [4, 8]
        new_context = env_step(0)

        self.assertEqual(new_context['reward'], 1)

    def test_env_step_rich(self):
        e_globs.IS_SPARSE = False

        new_context = env_step(0)
        self.assertEqual(new_context['reward'], -1)

        e_globs.current_state = [4, 8]
        new_context = env_step(0)
        self.assertEqual(new_context['reward'], 0)

    def test_env_step_borders(self):
        e_globs.current_state = [5, 0]
        env_step(0)
        self.assertEqual(e_globs.current_state, [5, 0])

        e_globs.current_state = [0, 8]
        env_step(1)
        self.assertEqual(e_globs.current_state, [0, 8])

        e_globs.current_state = [0, 0]
        env_step(2)
        self.assertEqual(e_globs.current_state, [0, 0])

        e_globs.current_state  = [0, 0]
        env_step(3)
        self.assertEqual(e_globs.current_state, [0, 0])

    def test_env_cleanup(self):
        self.assertEqual(env_cleanup(), None)

    def test_env_message_conditions_hold(self):
        env_message(json.dumps({"IS_STOCHASTIC": True, "IS_SPARSE": True}))
        self.assertTrue(e_globs.IS_STOCHASTIC)
        self.assertTrue(e_globs.IS_SPARSE)

    def test_env_message_conditions_fail(self):
        env_message(json.dumps({"IS_STOCHASTIC": False, "IS_SPARSE": False}))
        self.assertFalse(e_globs.IS_STOCHASTIC)
        self.assertFalse(e_globs.IS_SPARSE)


# def print_environment():
#     """
#     Print a graphical depiction of the environment in the agent is navigating.
#     Where X's signifiy open spaces and O's signify obstacles.
#     """

    # print("Printing a display of the discrete gridworld environment")
    # for row in range(e_globs.MAX_ROW, 0, -1):
    #     cur_row_display = ''
    #     for column in range(e_globs.MAX_COLUMN, 0, -1):
    #         cur_state = [row, column]
    #         if cur_state in e_globs.OBSTACLE_STATES:
    #             cur_row_display += 'O'
    #         else:
    #             cur_row_display += 'X'
    #     print(cur_row_display)

if __name__ == '__main__':
    #print_environment() TODO: Debug this
    print("Running tests for discrete gridworld environment...")
    np.random.seed(0)
    random.seed(0)
    unittest.main()
