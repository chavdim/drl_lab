import random

import keras
import numpy as np

from drl_lab.models import (
    QCNN,
    state2data,
)


class QNetworkAgent:
    def __init__(self, env, nn_hparams, agent_hprams):
        self.actions = [*range(len(env.actions))]

        # Neural network output q value for each action given state as input
        self.q_network = QCNN(env.obs_shape, len(self.actions), nn_hparams)
        self.target_q_network = keras.models.clone_model(self.q_network.model)
        self.target_q_network.set_weights(self.q_network.model.get_weights())

        self.reward_decay = agent_hprams['reward_decay']
        self.epsilon = agent_hprams['initial_epsilon']
        self.final_epsilon = agent_hprams['final_epsilon']
        self.batch_size = agent_hprams['batch_size']
        self.target_q_network_update_freq = \
            agent_hprams['target_q_network_update_freq']

        self.steps = 0

    def get_next_action(self, state):
        # Select next action (explore/exploit)
        if random.random() < self.epsilon:
            return self.get_random_action()
        else:
            return self.get_best_action(state)

    def get_random_action(self):
        return random.choice(self.actions)

    def get_best_action(self, state):
        data = state2data(state)
        qout = self.q_network.forward_prop(data, 1)
        action = np.argmax(qout[0])
        return int(action)

    # Just copy the current qnn_weights to the target_qnn_weights
    def target_train(self):
        self.target_q_network.set_weights(self.q_network.model.get_weights())

    def learn(self, parent_sim):
        # Get a batch
        batch = parent_sim.experience_data.get_batch(self.batch_size)

        # Get a q-value
        qout = self.q_network.forward_prop(batch['states'], self.batch_size)
        qout_next_state = self.target_q_network.predict(batch['new_states'],
                                                        self.batch_size)

        # Calc rewards
        for i in range(len(qout)):
            action = int(batch['actions'][i][0])
            # TODO: Write a test for this codes
            qout[i][action] = batch['rewards'][i][0]
            if not batch['dones'][i]:
                decayed_r = self.reward_decay * max(qout_next_state[i])
                qout[i][action] += decayed_r

        # Train
        self.q_network.train(batch['states'], qout, self.batch_size)

        if self.steps % self.target_q_network_update_freq == 0:
            self.target_train()

        self.steps += 1
