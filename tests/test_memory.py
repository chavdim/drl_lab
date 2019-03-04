import unittest

import numpy as np

from drl_lab.memory import Memory

max_size = 100
state_shape = (5, 5, 3)
state = np.random.randn(*state_shape)


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.memory = Memory(state_shape, max_size)

    def test_init(self):
        memory = self.memory

        expected = (max_size, *state_shape)
        self.assertEqual(expected, memory.states.shape)
        expected = (max_size, 1)
        self.assertEqual(expected, memory.actions.shape)
        expected = (max_size, *state_shape)
        self.assertEqual(expected, memory.new_states.shape)
        expected = (max_size, 1)
        self.assertEqual(expected, memory.rewards.shape)
        expected = (max_size, 1)
        self.assertEqual(expected, memory.dones.shape)
        expected = np.float32
        self.assertEqual(expected, memory.states.dtype)
        self.assertEqual(expected, memory.actions.dtype)
        self.assertEqual(expected, memory.new_states.dtype)
        self.assertEqual(expected, memory.rewards.dtype)
        self.assertEqual(expected, memory.dones.dtype)
        expected = 0
        self.assertEqual(expected, memory.index)
        self.assertFalse(memory.filled)

    # called before test_get_batch
    def test_add(self):
        memory = self.memory

        expected = 0
        self.assertEqual(expected, memory.index)

        _state, new_state = np.float32(state), np.float32(state)
        action, reward, done = 0, 0, False
        memory.add(_state, action, new_state, reward, done)

        expected = 1
        self.assertEqual(expected, memory.index)

        memory_state = memory.states[memory.index-1]
        self.assertTrue(np.array_equal(_state, memory_state))
        expected = np.array([action], dtype=np.float32)
        memory_action = memory.actions[memory.index-1]
        self.assertTrue(np.array_equal(expected, memory_action))
        memory_new_state = memory.new_states[memory.index-1]
        self.assertTrue(np.array_equal(new_state, memory_new_state))
        expected = np.array([reward], dtype=np.float32)
        memory_reward = memory.rewards[memory.index-1]
        self.assertTrue(np.array_equal(expected, memory_reward))
        expected = np.array([done], dtype=np.float32)
        memory_done = memory.dones[memory.index-1]
        self.assertTrue(np.array_equal(expected, memory_done))

    def test_full(self):
        memory = self.memory
        memory.full()

        expected = 0
        self.assertEqual(expected, memory.index)
        self.assertTrue(memory.filled)
        memory.filled = False

        _state, new_state = np.float32(state), np.float32(state)
        action, reward, done = 0, 0, False
        for i in range(max_size):
            memory.add(_state, action, new_state, reward, done)

        expected = 0
        self.assertEqual(expected, memory.index)
        self.assertTrue(memory.filled)

    def test_get_batch(self):
        memory = self.memory
        _state, new_state = np.float32(state), np.float32(state)
        action, reward, done = 0, 0, False
        for i in range(max_size):
            memory.add(_state, action, new_state, reward, done)

        batch = memory.get_batch(1)
        expected = 1
        self.assertEqual(expected, len(batch['states']))
        self.assertEqual(expected, len(batch['actions']))
        self.assertEqual(expected, len(batch['new_states']))
        self.assertEqual(expected, len(batch['rewards']))
        self.assertEqual(expected, len(batch['dones']))
        self.assertTrue(np.array_equal(_state, batch['states'][0]))
        self.assertEqual(action, batch['actions'][0])
        self.assertTrue(np.array_equal(new_state, batch['new_states'][0]))
        self.assertEqual(reward, batch['rewards'][0])
        self.assertEqual(done, batch['dones'][0])

        batch = memory.get_batch(100)
        expected = 100
        self.assertEqual(expected, len(batch['states']))
        self.assertEqual(expected, len(batch['actions']))
        self.assertEqual(expected, len(batch['new_states']))
        self.assertEqual(expected, len(batch['rewards']))
        self.assertEqual(expected, len(batch['dones']))
        self.assertTrue(np.array_equal(_state, batch['states'][50]))
        self.assertEqual(action, batch['actions'][50])
        self.assertTrue(np.array_equal(new_state, batch['new_states'][50]))
        self.assertEqual(reward, batch['rewards'][50])
        self.assertEqual(done, batch['dones'][50])
