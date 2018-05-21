import time
import random

import keras
import numpy as np

from drl_lab.models import QCNN, state2data


class QnetworkAgent:
    def __init__(self, env, nn_hparams, learn_style='mini_batch'):
        actions = env.actions
        self.actions_name_list = []
        self.action_indexes = {}
        for i in range(len(actions)):
            self.actions_name_list.append(actions[i].name)
            self.action_indexes[actions[i].name] = i
        self.next_action = self.get_random_action()

        # learning-related parameters
        self.next_reward_decay = 0.99
        self.update_rate = 0.0002  # learn rate
        self.fit_iterations = 1
        self.explore_chance = 1.0

        # learn style
        self.learn_style = learn_style
        self.learn_interval = 3
        self.learn_timer = -1000
        self.batch_size = 32
        self.target_network_update_freq = 500
        self.exploration_final_eps = 0.05
        self.step_counter = 0

        # neural network output q value for each action given state as input
        self.nn = QCNN(env.obs_shape, len(actions), nn_hparams)
        self.target_q_model = keras.models.clone_model(self.nn.nn)
        self.target_q_model.set_weights(self.nn.nn.get_weights())

    def update(self, parent_sim):
        # updating qnn
        self.step_counter += 1
        if parent_sim.last_action is not None:
            if self.learn_timer == self.learn_interval:
                self.learn_timer = 0
                self.learn(parent_sim)
            self.learn_timer += 1

    def get_next_action(self, state):
        # select next action (explore/exploit)
        if random.random() < self.explore_chance:
            self.next_action = self.get_random_action()
        else:
            self.next_action = self.get_best_action(state)
        return self.next_action

    def get_random_action(self):
        return random.choice(self.actions_name_list)

    def get_best_action(self, state):
        data = state2data(state)
        qout = self.nn.forward_prop(data, 1)
        action_index = np.argmax(qout[0])
        return int(action_index)

    def learn(self, parent_sim, epochs=10):
        # get a batch
        if self.learn_style == 'all_experience':
            batch = parent_sim.experience_data.get_batch(
                        parent_sim.experience_data.max_size-1)
        elif self.learn_style == 'mini_batch':
            batch = parent_sim.experience_data.get_batch(self.batch_size)

        # get a q-value
        t_before_train = time.time()
        qout = self.nn.forward_prop(batch['state'], self.batch_size)
        qout_next_state = self.target_q_model.predict(batch['new_state'],
                                                      self.batch_size)

        # calc rewards
        for i in range(len(qout)):
            action_name = int(batch['action'][i][0])
            action_index = self.action_indexes[action_name]

            # TODO: Write a test for this codes
            qout[i][int(action_index)] = batch['reward'][i][0]
            if not batch['done'][i]:
                decayed_r = self.next_reward_decay*max(qout_next_state[i])
                qout[i][int(action_index)] += decayed_r

        # train
        if self.learn_style == 'all_experience':
            self.nn.train(batch['state'], qout, self.batch_size,
                          epochs=epochs, verbose=2, vsplit=0.2)
            self.target_train()
        elif self.learn_style == 'mini_batch':
            self.nn.train(batch['state'], qout,
                          self.batch_size, self.fit_iterations)
            if self.step_counter % self.target_network_update_freq == 0:
                self.target_train()

        parent_sim.times['train'] += time.time() - t_before_train

    # just copy the current qnn_weights to the target_qnn_weights
    def target_train(self):
        self.target_q_model.set_weights(self.nn.nn.get_weights())
