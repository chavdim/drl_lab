import unittest

from drl_lab.memory import Memory
from tests.common import (
    action_size,
    max_experience_size,
    np,
    reward_size,
    state,
    state_shape,
)


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.a_size = action_size
        self.r_size = reward_size
        self.max_size = max_experience_size
        self.memory = Memory(
            state_shape, self.a_size, self.r_size, self.max_size)

    def test_init(self):
        memory = self.memory

        expected = self.a_size
        self.assertEqual(expected, memory.a_size)
        expected = self.max_size
        self.assertEqual(expected, memory.max_size)
        expected = self.a_size + self.r_size + 1
        self.assertEqual(expected, memory.col_size)
        expected = (self.max_size, *state_shape)
        self.assertEqual(expected, memory.state_storage.shape)
        self.assertEqual(expected, memory.new_state_storage.shape)
        expected = (self.max_size, self.a_size + self.r_size + 1)
        self.assertEqual(expected, memory.storage.shape)
        expected = np.float32
        self.assertEqual(expected, memory.state_storage.dtype)
        self.assertEqual(expected, memory.new_state_storage.dtype)
        self.assertEqual(expected, memory.storage.dtype)
        expected = 0
        self.assertEqual(expected, memory.current_row)
        self.assertFalse(memory.filled_once)

    # called before test_get_batch
    def test_add_data(self):
        memory = self.memory

        expected = 0
        self.assertEqual(expected, memory.current_row)

        s, s_new = np.float32(state), np.float32(state)
        a, r, done = 0, 0, False
        memory.add_data(s, a, s_new, r, done)

        expected = 1
        self.assertEqual(expected, memory.current_row)
        expected = self.max_size
        self.assertEqual(expected, len(memory.state_storage))
        memory_state = memory.state_storage[memory.current_row-1]
        self.assertTrue(np.array_equal(s, memory_state))
        expected = self.max_size
        self.assertEqual(expected, len(memory.new_state_storage))
        memory_new_state = memory.new_state_storage[memory.current_row-1]
        self.assertTrue(np.array_equal(s_new, memory_new_state))
        expected = self.max_size
        self.assertEqual(expected, len(memory.storage))
        memory_all_data = memory.storage[memory.current_row-1]
        expected = np.array([a, r, done])
        self.assertTrue(np.array_equal(expected, memory_all_data))

    def test_full(self):
        memory = self.memory
        memory.full()

        expected = 0
        self.assertEqual(expected, memory.current_row)
        self.assertTrue(memory.filled_once)
        memory.filled_once = False

        s, s_new = np.float32(state), np.float32(state)
        a, r, done = 0, 0, False
        for i in range(self.max_size):
            memory.add_data(s, a, s_new, r, done)

        expected = 0
        self.assertEqual(expected, memory.current_row)
        self.assertTrue(memory.filled_once)

    def test_get_batch(self):
        memory = self.memory
        s, s_new = np.float32(state), np.float32(state)
        a, r, done = 0, 0, False
        for i in range(self.max_size):
            memory.add_data(s, a, s_new, r, done)

        batch = memory.get_batch(1)
        expected = 1
        self.assertEqual(expected, len(batch['state']))
        self.assertEqual(expected, len(batch['action']))
        self.assertEqual(expected, len(batch['new_state']))
        self.assertEqual(expected, len(batch['reward']))
        self.assertEqual(expected, len(batch['done']))
        self.assertTrue(np.array_equal(s, batch['state'][0]))
        self.assertEqual(a, batch['action'][0])
        self.assertTrue(np.array_equal(s_new, batch['new_state'][0]))
        self.assertEqual(r, batch['reward'][0])
        self.assertEqual(done, batch['done'][0])

        batch = memory.get_batch(100)
        expected = 100
        self.assertEqual(expected, len(batch['state']))
        self.assertEqual(expected, len(batch['action']))
        self.assertEqual(expected, len(batch['new_state']))
        self.assertEqual(expected, len(batch['reward']))
        self.assertEqual(expected, len(batch['done']))
        self.assertTrue(np.array_equal(s, batch['state'][50]))
        self.assertEqual(a, batch['action'][50])
        self.assertTrue(np.array_equal(s_new, batch['new_state'][50]))
        self.assertEqual(r, batch['reward'][50])
        self.assertEqual(done, batch['done'][50])
