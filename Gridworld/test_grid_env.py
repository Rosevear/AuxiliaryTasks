import unittest
import grid_env
import grid_agent
import json
import grid_env_globals as e_globs
import random
import numpy as np

class TestGridEnv(unittest.TestCase):

    def setUp(self):
        e_globs.IS_SPARSE = None
        e_globs.IS_STOCHASTIC = None

    def tearDown(self):
        e_globs.IS_SPARSE = None
        e_globs.IS_STOCHASTIC = None
        e_globs.current_state = e_globs.START_STATE

    def test_env_init(self):
        self.assertEqual(grid_env.env_init(), None)

    def test_env_start(self):
        grid_env.env_start()
        self.assertEqual(e_globs.current_state, e_globs.START_STATE)

    def test_env_step_deterministic(self):
        e_globs.IS_STOCHASTIC = False

        grid_env.env_step(0)
        self.assertEqual(e_globs.current_state, [4, 0])
        grid_env.env_step(1)
        self.assertEqual(e_globs.current_state, [4, 1])
        grid_env.env_step(2)
        self.assertEqual(e_globs.current_state, [3, 1])
        grid_env.env_step(3)
        self.assertEqual(e_globs.current_state, [3, 0])

    def test_env_step_deterministic_obstacle(self):
        e_globs.IS_STOCHASTIC = False

        grid_env.env_step(1)
        grid_env.env_step(1)
        self.assertEqual(e_globs.current_state, [3, 1])


    def test_env_step_stochastic_obstacle(self):
        e_globs.IS_STOCHASTIC = True
        np.random.seed(0)
        random.seed(0)
        grid_env.env_step(1)
        grid_env.env_step(1)
        grid_env.env_step(1)
        #We know the state because we are using a random seed: I chekced what it should be in this case
        self.assertEqual(e_globs.current_state, [2, 2])




    def test_env_step_sparse(self):
        e_globs.IS_SPARSE = True

        new_context = grid_env.env_step(0)
        self.assertEqual(new_context['reward'], 0)

        e_globs.current_state = [4, 8]
        new_context = grid_env.env_step(0)

        self.assertEqual(new_context['reward'], 1)

    def test_env_step_rich(self):
        e_globs.IS_SPARSE = False

        new_context = grid_env.env_step(0)
        self.assertEqual(new_context['reward'], -1)

        e_globs.current_state = [4, 8]
        new_context = grid_env.env_step(0)
        self.assertEqual(new_context['reward'], 0)

    def test_env_step_borders(self):
        e_globs.current_state = [5, 0]
        grid_env.env_step(0)
        self.assertEqual(e_globs.current_state, [5, 0])

        e_globs.current_state = [0, 8]
        grid_env.env_step(1)
        self.assertEqual(e_globs.current_state, [0, 8])

        e_globs.current_state = [0, 0]
        grid_env.env_step(2)
        self.assertEqual(e_globs.current_state, [0, 0])

        current_state = [0, 0]
        grid_env.env_step(3)
        self.assertEqual(e_globs.current_state, [0, 0])

    def test_env_cleanup(self):
        self.assertEqual(grid_env.env_cleanup(), None)

    def test_env_message_conditions_hold(self):
        grid_env.env_message(json.dumps({"IS_STOCHASTIC": True, "IS_SPARSE": True}))
        self.assertTrue(e_globs.IS_STOCHASTIC)
        self.assertTrue(e_globs.IS_SPARSE)

    def test_env_message_conditions_fail(self):
        grid_env.env_message(json.dumps({"IS_STOCHASTIC": False, "IS_SPARSE": False}))
        self.assertFalse(e_globs.IS_STOCHASTIC)
        self.assertFalse(e_globs.IS_SPARSE)


if __name__ == '__main__':
    unittest.main()
