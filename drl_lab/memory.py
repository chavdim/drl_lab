import numpy as np


class Memory:
    def __init__(self, s_shape, a_size, r_size, max_size=100000):
        s_shape = [max_size, *s_shape]
        self.a_size = a_size
        self.max_size = max_size
        self.col_size = (a_size+r_size+1)  # +1 for done (boolean)

        # state storages
        self.state_storage = np.empty(s_shape, dtype=np.float32)
        self.new_state_storage = np.empty(s_shape, dtype=np.float32)

        self.storage = np.empty([max_size, self.col_size], dtype=np.float32)

        self.current_row = 0
        self.filled_once = False

    def add_data(self, s, a, s_new, r, done):
        self.state_storage[self.current_row][0:, 0:, 0:] = np.copy(s)
        self.new_state_storage[self.current_row][0:, 0:, 0:] = np.copy(s_new)

        # all_data = np.append(a, r)
        # all_data = np.append(all_data, done)
        all_data = np.array([a, r, done])

        self.storage[self.current_row] = all_data
        self.current_row += 1

        if self.current_row == self.max_size:  # reset when full
            self.full()

    def full(self):
        self.current_row = 0
        self.filled_once = True
        print('memory full yo')

    def get_batch(self, batch_size=10):
        if self.filled_once:
            choices = np.random.randint(0, self.max_size, size=batch_size)
        else:
            choices = np.random.randint(0, self.current_row, size=batch_size)

        return {'state': self.state_storage[choices],
                'action': self.storage[choices][0:, 0:self.a_size],
                'new_state': self.new_state_storage[choices],
                'reward': self.storage[choices][0:, -2:-1],
                'done': self.storage[choices][0:, -1:]
                }
