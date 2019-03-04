import unittest

import numpy as np

from drl_lab.agents import QnetworkAgent
from drl_lab.memory import Memory
from tests.common import (
    nn_hparams,
    weights_equal,
)

actions = [0, 1, 2, 3]
action_size = len(actions)
max_experience_size = 100
obs_shape = (5, 5, 3)
state_shape = obs_shape
state = np.random.randn(*state_shape)


class MockEnv:
    def __init__(self):
        self.actions = actions
        self.obs_shape = obs_shape


class MockSim:
    def __init__(self):
        self.last_action = None
        self.experience_data = Memory(state_shape, max_experience_size)
        _state, new_state = np.float32(state), np.float32(state)
        action, reward, done = 0, 0, False
        for i in range(max_experience_size):
            self.experience_data.add(_state, action, new_state, reward, done)
        self.times = {'train': 0.0}


class TestQnetworkAgent(unittest.TestCase):
    def setUp(self):
        self.env = MockEnv()
        self.parent_sim = MockSim()
        self.agent = QnetworkAgent(self.env, nn_hparams)

    def test_init(self):
        agent = self.agent
        self.assertEqual(agent.actions, actions)
        self.assertTrue(agent.next_action in actions)
        expected = 0.99
        self.assertEqual(expected, agent.next_reward_decay)
        expected = 0.0002
        self.assertEqual(expected, agent.update_rate)
        expected = 1
        self.assertEqual(expected, agent.fit_iterations)
        expected = 1.0
        self.assertEqual(expected, agent.explore_chance)
        expected = 'mini_batch'
        self.assertEqual(expected, agent.learn_style)
        expected = 3
        self.assertEqual(expected, agent.learn_interval)
        expected = -1000
        self.assertEqual(expected, agent.learn_timer)
        expected = 32
        self.assertEqual(expected, agent.batch_size)
        expected = 500
        self.assertEqual(expected, agent.target_network_update_freq)
        expected = 0.05
        self.assertEqual(expected, agent.exploration_final_eps)
        expected = 0
        self.assertEqual(expected, agent.step_counter)
        self.assertTrue(weights_equal(agent.nn.nn, agent.target_q_model))

    def test_update(self):
        agent = self.agent
        parent_sim = self.parent_sim

        parent_sim.last_action = None
        step_counter = agent.step_counter
        agent.update(parent_sim)
        expected = step_counter + 1
        self.assertEqual(expected, agent.step_counter)

        parent_sim.last_action = 0
        learn_timer = agent.learn_timer
        agent.update(parent_sim)
        expected = learn_timer + 1
        self.assertEqual(expected, agent.learn_timer)

        agent.learn_timer = agent.learn_interval
        agent.update(parent_sim)
        expected = 1
        self.assertEqual(expected, agent.learn_timer)

    def test_get_next_action(self):
        agent = self.agent

        self.assertTrue(agent.get_next_action(state) in actions)

        agent.explore_chance = 1.0
        got_actions1 = [agent.get_next_action(state) for i in range(10)]
        got_actions2 = [agent.get_next_action(state) for i in range(10)]
        self.assertNotEqual(got_actions1, got_actions2)

        agent.explore_chance = 0.0
        got_actions1 = [agent.get_next_action(state) for i in range(10)]
        got_actions2 = [agent.get_next_action(state) for i in range(10)]
        self.assertEqual(got_actions1, got_actions2)

    def test_get_random_action(self):
        agent = self.agent
        self.assertTrue(agent.get_random_action() in actions)
        got_actions1 = [agent.get_random_action() for i in range(10)]
        got_actions2 = [agent.get_random_action() for i in range(10)]
        self.assertNotEqual(got_actions1, got_actions2)

    def test_get_best_action(self):
        agent = self.agent
        self.assertTrue(agent.get_best_action(state) in actions)
        got_actions1 = [agent.get_best_action(state) for i in range(10)]
        got_actions2 = [agent.get_best_action(state) for i in range(10)]
        self.assertEqual(got_actions1, got_actions2)

    def test_learn(self):
        agent = self.agent
        parent_sim = self.parent_sim

        # all_experience
        batch = parent_sim.experience_data.get_batch(
                        parent_sim.experience_data.max_size-1)
        expected = max_experience_size-1
        self.assertEqual(expected, len(batch['states']))

        qout = agent.nn.forward_prop(batch['states'], agent.batch_size)
        expected = max_experience_size-1
        self.assertEqual(expected, len(qout))

        qout_next_state = agent.target_q_model.predict(batch['new_states'],
                                                       agent.batch_size)
        expected = max_experience_size-1
        self.assertEqual(expected, len(qout_next_state))

        agent.learn_style = 'all_experience'
        weights_before = agent.nn.nn.layers[0].get_weights()[0]
        agent.learn(parent_sim, epochs=1)
        weights_after = agent.nn.nn.layers[0].get_weights()[0]
        self.assertFalse(np.array_equal(weights_before, weights_after))
        self.assertTrue(weights_equal(agent.nn.nn, agent.target_q_model))

        # mini_batch
        batch = parent_sim.experience_data.get_batch(agent.batch_size)
        expected = agent.batch_size
        self.assertEqual(expected, len(batch['states']))

        qout = agent.nn.forward_prop(batch['states'], agent.batch_size)
        expected = agent.batch_size
        self.assertEqual(expected, len(qout))

        qout_next_state = agent.target_q_model.predict(batch['new_states'],
                                                       agent.batch_size)
        expected = agent.batch_size
        self.assertEqual(expected, len(qout_next_state))

        agent.learn_style = 'mini_batch'
        weights_before = agent.nn.nn.layers[0].get_weights()[0]
        agent.learn(parent_sim)
        weights_after = agent.nn.nn.layers[0].get_weights()[0]
        self.assertFalse(np.array_equal(weights_before, weights_after))
        agent.step_counter = agent.target_network_update_freq
        agent.learn(parent_sim)
        self.assertTrue(weights_equal(agent.nn.nn, agent.target_q_model))

    def test_target_train(self):
        agent = self.agent
        agent.target_train()
        self.assertTrue(weights_equal(agent.nn.nn, agent.target_q_model))
