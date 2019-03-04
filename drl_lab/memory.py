import numpy as np


class Memory:
    def __init__(self, state_shape, max_size=100000):
        self.max_size = max_size

        self.states = np.empty([max_size, *state_shape], dtype=np.float32)
        self.actions = np.empty([max_size, 1], dtype=np.float32)
        self.new_states = np.empty([max_size, *state_shape], dtype=np.float32)
        self.rewards = np.empty([max_size, 1], dtype=np.float32)
        self.dones = np.empty([max_size, 1], dtype=np.float32)

        self.index = 0
        self.filled = False

    def add(self, state, action, new_state, reward, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.new_states[self.index] = new_state
        self.rewards[self.index] = reward
        self.dones[self.index] = done

        self.index += 1

        # reset when full
        if self.index == self.max_size:
            self.full()

    def full(self):
        self.index = 0
        self.filled = True
        #print('memory full yo')

    def get_batch(self, batch_size=10):
        if self.filled:
            choices = np.random.randint(0, self.max_size, size=batch_size)
        else:
            choices = np.random.randint(0, self.index, size=batch_size)

        return {
            'states': self.states[choices],
            'actions': self.actions[choices],
            'new_states': self.new_states[choices],
            'rewards': self.rewards[choices],
            'dones': self.dones[choices],
        }
