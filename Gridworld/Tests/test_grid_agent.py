import unittest
import grid_env
import grid_agent
import json
import grid_agent_globals as a_globs
import random
import numpy as np

class TestGridAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ALPHA = 0.001
        GAMMA = 0.95
        EPSILON_MIN = 0.1
        N = 3
        IS_STOCHASTIC = None
        AGENT = 'auxiliary'

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_agent_init(self):
        pass

    def test_agent_start(self):


    def test_add_to_buffer(self):
        mock_buffer = np.zeros()
        grid_agent.add_to_buffer(mock_buffer, 'test content', 1)

        self.assertEqual(len(mock_buffer))



if __name__ == '__main__':
    unittest.main()
