import unittest
import json
import random
import numpy as np

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Agents.neural_agent import *
from Envs.continuous_grid_env import *
import Globals.continuous_grid_env_globals as e_globs

class TestContinuousGridEnv(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #Turn off the noise when testing that the obstaces work
        #we just want to make sure that walking into the wall keeps us where we are
        #and noise can make it difficult to pick out starting and expected values for assert statements
        #tmp = e_globs.ACTION_NOISE_UNIFORM_RANGE
        e_globs.ACTION_NOISE_UNIFORM_RANGE = 0.0

    def setUp(self):
        e_globs.IS_SPARSE = None

    def tearDown(self):
        e_globs.IS_SPARSE = None
        e_globs.IS_STOCHASTIC = None
        e_globs.current_state = e_globs.START_STATE

    def test_env_init(self):
        self.assertEqual(env_init(), None)

    def test_env_start(self):
        env_start()
        self.assertEqual(e_globs.current_state, e_globs.START_STATE)

    def test_env_step(self):
        old_state = e_globs.current_state

        env_step(e_globs.NORTH)
        self.assertTrue(e_globs.current_state[0] > old_state[0])

        old_state = e_globs.current_state
        env_step(e_globs.EAST)
        self.assertTrue(e_globs.current_state[1] > old_state[1])

        old_state = e_globs.current_state
        env_step(e_globs.SOUTH)
        self.assertTrue(e_globs.current_state[0] < old_state[0])

        old_state = e_globs.current_state
        env_step(e_globs.WEST)
        self.assertTrue(e_globs.current_state[1] < old_state[1])


    def test_env_step_sparse(self):
        e_globs.IS_SPARSE = True

        new_context = env_step(0)
        self.assertEqual(new_context['reward'], 0)

        e_globs.current_state = e_globs.GOAL_STATE
        new_context = env_step(0)

        self.assertEqual(new_context['reward'], 1)

    def test_env_step_rich(self):
        e_globs.IS_SPARSE = False

        new_context = env_step(0)
        self.assertEqual(new_context['reward'], -1)

        e_globs.current_state = e_globs.GOAL_STATE
        new_context = env_step(0)
        self.assertEqual(new_context['reward'], 0)

    def test_env_step_borders(self):
        e_globs.current_state = [1, 0]
        env_step(e_globs.NORTH)
        self.assertEqual(e_globs.current_state, [1, 0])

        e_globs.current_state = [0, 1]
        env_step(e_globs.EAST)
        self.assertEqual(e_globs.current_state, [0, 1])

        e_globs.current_state = [0, 0]
        env_step(e_globs.SOUTH)
        self.assertEqual(e_globs.current_state, [0, 0])

        e_globs.current_state  = [0, 0]
        env_step(e_globs.WEST)
        self.assertEqual(e_globs.current_state, [0, 0])

    def test_env_step_obstacles(self):

        #Test trying to get into the box
        e_globs.current_state = [0.26, 0.24]
        env_step(e_globs.EAST)
        self.assertEqual(e_globs.current_state, [0.26, 0.24])

        e_globs.current_state = [0.76, 0.27]
        env_step(e_globs.SOUTH)
        self.assertEqual(e_globs.current_state, [0.76, 0.27])

        e_globs.current_state = [0.26, 0.76]
        env_step(e_globs.WEST)
        self.assertEqual(e_globs.current_state, [0.26, 0.76])

        #Test trying to go out of the box
        e_globs.current_state = [0.27, 0.376]
        env_step(e_globs.WEST)
        self.assertEqual(e_globs.current_state, [0.27, 0.376])

        e_globs.current_state = [0.624, 0.30]
        env_step(e_globs.NORTH)
        self.assertEqual(e_globs.current_state, [0.624, 0.30])

        e_globs.current_state = [0.26, 0.624]
        env_step(e_globs.EAST)
        self.assertEqual(e_globs.current_state, [0.26, 0.624])

        #e_globs.ACTION_NOISE_UNIFORM_RANGE = tmp



    def test_env_is_in_goal_state(self):
        self.assertTrue(is_goal_state(e_globs.GOAL_STATE))
        self.assertTrue(is_goal_state([0.999, 0.999]))
        self.assertFalse(is_goal_state([0.99, 0.99]))

    def test_env_step_reach_goal(self):
        e_globs.current_state  = [0.99, 0.99]
        self.assertFalse(is_goal_state(e_globs.current_state))

        env_step(e_globs.NORTH)
        self.assertFalse(is_goal_state(e_globs.current_state))

        env_step(e_globs.EAST)
        self.assertTrue(is_goal_state(e_globs.current_state))

    def test_env_cleanup(self):
        self.assertEqual(env_cleanup(), None)

    def test_env_message_conditions_hold(self):
        env_message(json.dumps({"IS_SPARSE": True}))
        self.assertTrue(e_globs.IS_SPARSE)

    def test_env_message_conditions_fail(self):
        env_message(json.dumps({"IS_SPARSE": False}))
        self.assertFalse(e_globs.IS_SPARSE)

# def print_environment():
#     """
#     Print a graphical depiction of the environment in the agent is navigating.
#     Where X's signifiy open spaces and O's signify obstacles.
#     """
#
#     for row in range(0, e_globs.MAX_ROW, 0.001):
#         cur_row = ''
#         for column in range(0, e_globs.MAX_COLUMN, 0.001):
#             if obstacle:
#                 cur_row += 'O'
#             else:
#                 cur_row += 'X'
#         print(cur_row)


if __name__ == '__main__':
    #print_environment()
    print("Testing the continuous gridworld environment...")
    np.random.seed(0)
    random.seed(0)
    unittest.main()
