#!/usr/bin/env python3

import time
import random

import keras
import numpy as np

import nn_wrapper


class Qnetwork_agent:
    def __init__(self, stateSize, actions, hidden):
        # self.TAU = 0.01
        self.stateVectorSize = stateSize

        self.action_size = len(actions)
        self.actions = actions
        self.actionsNameList = []
        self.action_indexes = {}
        for i in range(len(self.actions)):
            self.actionsNameList.append(self.actions[i].action_name)
            self.actions[i].nn_index = i
            self.action_indexes[self.actions[i].action_name] = i
        # learning-related parameters
        # self.nextRewardDecay = 0.9
        self.nextRewardDecay = 0.99
        self.updateRate = 0.0002  # learn rate
        self.fitIterations = 1
        self.exploreChance = 1.0
        # learn style
        # self.learnStyle = "online"
        # self.learnStyle = "allExperience"
        self.learnStyle = "mini_batch"
        self.learnInterval = 3
        self.learnTimer = -1000
        self.batchSize = 32
        self.target_network_update_freq = 500
        self.exploration_final_eps = 0.05
        self.step_counter = 0

        # neural network output q value for each action given state as input

        # network structure
        # self.nn = new Network(self.stateVectorSize, [20,self.action_size],
        #                       self.updateRate)
        # self.nn = ConvNetWrapper(self.width, self.height, self.batchSize,
        #                          self.updateRate)
        # self.nn = nn_wrapper.QNN(self.stateVectorSize, self.action_size,
        #                          self.updateRate)
        self.nn = nn_wrapper.QCNN_keras(stateSize, self.action_size,
                                        self.updateRate, hidden)

        self.target_q_model = keras.models.clone_model(self.nn.nn)

        q_weights = self.nn.nn.get_weights()
        q_target_weights = self.target_q_model.get_weights()
        for i in range(len(q_weights)):
            q_target_weights[i] = q_weights[i]
        self.target_q_model.set_weights(q_target_weights)

        self.nextAction = self.getRandomAction()

    def update(self, parent_sim):
        # updating Q nn
        self.step_counter += 1
        if parent_sim.lastAction is None:
            if self.learnStyle == "online":
                """
                self.learnTimer += 1
                if (self.learnTimer == self.learnInterval) {
                    self.learnTimer = 0
                    self.learnFromInstance(parent_sim.lastState,parent_sim.lastAction,parent_sim.state,parent_sim.score)
                };
                """
            if self.learnStyle == "allExperience":
                if self.learnTimer == self.learnInterval:
                    self.learnTimer = 0
                    self.learnFromAllExperience(parent_sim)
                self.learnTimer += 1

            if self.learnStyle == "mini_batch":
                if self.learnTimer == self.learnInterval:
                    self.learnTimer = 0
                    self.learnFromBatchExperience(parent_sim)
                self.learnTimer += 1

    def getNextAction(self, state):
        # select next action (explore/exploit)
        r = random.random()
        if r < self.exploreChance:
            self.nextAction = self.getRandomAction()

        else:
            self.nextAction = self.getBestAction(state)
        return self.nextAction

    def getRandomAction(self):
        return random.choice(self.actionsNameList)

    # self.nextAction = self.getRandomAction()

    def learnFromAllExperience(self, parent_sim, iter_num=10):

        batch = parent_sim.experienceData.getBatch(
            parent_sim.experienceData.maxSize-1)
        # prepare training set
        # dataset = []
        t_before_train = time.time()

        qout = self.nn.forwardProp2(batch["state"], self.batchSize)

        qout_nextState = self.target_q_model.predict(batch["new_state"],
                                                     self.batchSize)
        # print(qout_nextState)
        for i in range(len(qout)):
            action_name = int(batch["action"][i][0])
            action_index = self.action_indexes[action_name]
            action_index = batch["action"][i][0]

            qout[i][int(action_index)] = batch["reward"][i][0]
            if batch["done"][i] != 1:
                # TODO: Shorten the code below
                qout[i][int(action_index)] += self.nextRewardDecay*max(qout_nextState[i])

        self.nn.train2(batch["state"], qout, self.batchSize, iter_num,
                       verbose=2, vsplit=0.2)
        # if self.step_counter % self.target_network_update_freq == 0:
        self.target_train()

        parent_sim.times["train"] += time.time() - t_before_train
        # print("training t")
        # print(qout)

    def learnFromBatchExperience(self, parent_sim):
        # if self.batchSize > parent_sim.experienceData.currentRow:
        #    return
        batch = parent_sim.experienceData.getBatch(self.batchSize)
        # batch *= self.learnInterval
        # prepare training set
        # dataset = []
        t_before_train = time.time()

        qout = self.nn.forwardProp2(batch["state"], self.batchSize)
        # print("#######################")
        # print(batch["action"])
        # print(batch["reward"])
        # print(qout)
        # print(batch["new_state"])
        qout_nextState = self.target_q_model.predict(batch["new_state"],
                                                     self.batchSize)
        # print(qout_nextState)
        for i in range(len(qout)):
            action_name = int(batch["action"][i][0])
            action_index = self.action_indexes[action_name]
            action_index = batch["action"][i][0]

            qout[i][int(action_index)] = batch["reward"][i][0]
            if batch["done"][i] != 1:
                # TODO: Shorten the code below
                qout[i][int(action_index)] += self.nextRewardDecay*max(qout_nextState[i])

        self.nn.train2(batch["state"], qout,
                       self.batchSize, self.fitIterations)
        if self.step_counter % self.target_network_update_freq == 0:
            self.target_train()

        parent_sim.times["train"] += time.time() - t_before_train
        # print("training t")
        # print(qout)

    def target_train(self):
        """
        actor_weights = self.nn.nn.get_weights()
        actor_target_weights = self.target_q_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = (self.TAU*actor_weights[i]
                                       + (1 - self.TAU)*actor_target_weights[i]
        self.target_q_model.set_weights(actor_target_weights)
        """
        # print("target q updated")
        # self.target_q_model = keras.models.clone_model(self.nn.nn)
        q_weights = self.nn.nn.get_weights()
        q_target_weights = self.target_q_model.get_weights()
        for i in range(len(q_weights)):
            q_target_weights[i] = q_weights[i]
        self.target_q_model.set_weights(q_target_weights)

    def learnFromBatchExperience2(self, parent_sim):
        if self.batchSize > len(parent_sim.experienceData):
            return
        randomSample = []
        # randomSampleValuable = []
        for i in range(self.batchSize):
            ind = int(random.random()*len(parent_sim.experienceData))
            randomSample.append(ind)
        """
        for i in range(self.batchSize):
            ind = int(random.random()*len(parent_sim.valuableExperienceData))
            randomSampleValuable.append(ind)
        """
        # prepare training set
        dataset = []
        for i in range(len(randomSample)):
            state = parent_sim.experienceData[randomSample[i]]["state"]
            action = parent_sim.experienceData[randomSample[i]]["action"]
            new_state = parent_sim.experienceData[randomSample[i]]["new_state"]
            reward = parent_sim.experienceData[randomSample[i]]["reward"]
            done = parent_sim.experienceData[randomSample[i]]["done"]

            qout = self.nn.forwardProp(state)
            qout_nextState = self.nn.forwardProp(new_state)

            action_index = self.actionsNameList.index(action)
            next_max = max(qout_nextState[0])
            # check for unchanged state
            """
            num_same = 0
            for ii in range(len(state)):
                if state[ii] == new_state[ii]:
                    num_same += 1
            same_states = 0
            if num_same == len(state):
                same_states = 1
            #end check
            """
            if done:
                qout[0][action_index] = reward

            else:
                # if same_states == 0:
                qout[0][action_index] = reward + self.nextRewardDecay*next_max

                # else:
                #    no reward if state is unchanged
                #    qout[0][action_index] = 0

            dataset.append({"input": state, "output": qout[0]})

        self.nn.train(dataset, 1)

    def getBestAction(self, state):
        qout = self.nn.forwardProp(state)
        # max_value = max(qout[0])
        action_index = np.argmax(qout[0])

        # return self.actionsNameList[action_index]
        return int(action_index)
