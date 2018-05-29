import unittest

from drl_lab.sim import Simulator
from tests.common import (
    env_hparams,
    nn_hparams,
)


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator(env_hparams, nn_hparams)

    def test_init(self):
        sim = self.sim

        expected = 1
        self.assertEqual(expected, sim.skip_frames)
        expected = sim.skip_frames
        self.assertEqual(expected, sim.skip_frame_timer)
        expected = 10000
        self.assertEqual(expected, sim.episode_max_length)

        expected = 100000
        self.assertEqual(expected, sim.max_iterations)
        expected = 20
        self.assertEqual(expected, sim.interval)
        expected = []
        self.assertEqual(expected, sim.rewards)
        expected = 0
        self.assertEqual(expected, sim.temp_rews)
        self.assertFalse(sim.done)
        self.assertIsNone(sim.last_action)

        expected = 1
        self.assertEqual(expected, sim.experience_data.a_size)
        expected = 50000
        self.assertEqual(expected, sim.experience_data.max_size)

        expected = 0.0
        self.assertEqual(expected, sim.times['total'])
        self.assertEqual(expected, sim.times['get_action'])
        self.assertEqual(expected, sim.times['train'])
        self.assertEqual(expected, sim.times['create_batch'])

    def test_run(self):
        sim = self.sim

        episode_reward = 0
        episode_mode = 'episode_reward' in locals()
        self.assertTrue(episode_mode)

        # Run as defalut
        t_get_action_before = sim.times['get_action']
        all_rewards = sim.run(iterations=10)
        t_get_action_after = sim.times['get_action']
        self.assertNotEqual(t_get_action_before, t_get_action_after)
        expected = 10
        self.assertEqual(expected, len(all_rewards))
        expected = 1
        self.assertEqual(expected, sim.skip_frame_timer)
        self.assertIsNotNone(sim.last_action)
        expected = 10
        self.assertEqual(expected, sim.experience_data.current_row)

        # Run as test_agent
        sim.run(test_agent=True, iterations=10)
        pass

        # Run with update=True
        learn_timer_before = sim.agent.learn_timer
        explore_chance_before = sim.agent.explore_chance
        sim.run(update=True, iterations=10)
        learn_timer_after = sim.agent.learn_timer
        explore_chance_after = sim.agent.explore_chance
        self.assertNotEqual(learn_timer_before, learn_timer_after)
        self.assertTrue(explore_chance_before > explore_chance_after)

        # Run as episode mode
        episode_reward = sim.run(iterations=-1)
        expected = float
        self.assertEqual(expected, type(episode_reward))

    def test_run_repeatedly(self):
        sim = self.sim
        results = sim.run_repeatedly(num_runs=10, iterations=-1, update=False)
        expected = 10
        self.assertEqual(expected, len(results))
