#!/usr/bin/env python3


# import random
import time

import cv2
import gym
# from gym import wrappers
# import imageio
import keras
from keras import backend as K
# from keras.layers import Activation
# from keras.layers import Dense
# from keras.layers import Input
from keras.layers.core import Lambda
from keras.models import Sequential
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
import gym_ple
# from mpl_toolkits.mplot3d import Axes3D
# import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
# from scipy import ndimage
from skimage.transform import rescale
import tensorflow as tf

import qnn_agent


# np.dot(rs[...,:3], [0.299, 0.587, 0.114])
# random.seed(1)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def drawShadow(im, flow, thresh=0.6):
    shadows = np.ones_like(im)/2
    fy = flow[0:, 0:, 1]
    shadows[fy > thresh] = 1.0
    shadows[fy < -thresh] = 0
    return shadows


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def grad_cam(input_model, image, category_index, layer_name, nb_classes):
    model = Sequential()
    model.add(input_model)

    nb_classes = nb_classes

    def target_layer(x):
        target_category_loss(x, category_index, nb_classes)

    model.add(Lambda(target_layer,
                     output_shape=target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)

    # print(model.layers[0].layers)
    conv_output = [l for l in model.layers[0].layers if l.name == layer_name]
    conv_output = conv_output[0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input],
                                   [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # cam = cv2.resize(cam, (224, 224))
    # cam_sm = np.copy(cam)
    # cam_sm = np.maximum(cam_sm, 0)
    # cam_sm = cam_sm / np.max(cam_sm)

    # cam = ndimage.interpolation.zoom(cam, (zm, zm), order=1)#order=3
    # cam = cv2.resize(cam,(zm*cam.shape[0],zm*cam.shape[1]))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    return heatmap


# TODO: "prev_state" => "prev_obs"?
def observation_preprocess(observation, zoom=[1, 1], opt_flow=0, prev_obs=None,
                           toGrayScale=False):
    # rescale if needed
    if zoom[0] == 1 and zoom[1] == 1:
        obs_preprocessed = np.copy(observation/255.0)
    else:
        obs_preprocessed = np.copy(rescale(observation, zoom))

    # add optical flow
    if opt_flow:
        if prev_state != "reseting":
            gray = rgb2gray(obs_preprocessed)
            of = cv2.calcOpticalFlowFarneback(rgb2gray(prev_state)*255,
                                              gray*255, None,
                                              0.5, 3, 5, 3, 5, 1.2, 0)
            if toGrayScale:
                obs_preprocessed[0:, 0:, 0] = gray
                obs_preprocessed[0:, 0:, 1] = of[0:, 0:, 0]/10
                obs_preprocessed[0:, 0:, 2] = of[0:, 0:, 1]/10
            else:
                for i in opt_flow:
                    # add x,y flow one or both depending
                    # on opt flow parameter ([0, 1] for both)
                    obs_preprocessed[0:, 0:, 3+i] = of[0:, 0:, i]/10

        elif prev_state == "reseting":
            # ray = rgb2gray(obs_preprocessed)
            if toGrayScale:
                obs_preprocessed[0:, 0:, 0] = gray
                obs_preprocessed[0:, 0:, 1] = obs_preprocessed[0:, 0:, 1]*0
                obs_preprocessed[0:, 0:, 2] = obs_preprocessed[0:, 0:, 2]*0


class Memory:
    def __init__(self, s_shape, a_size, r_size, maxSize=100000):

        s_shape.insert(0, maxSize)
        self.colSize = (a_size+r_size+1)  # +1 for done boolean
        # state storages
        self.stateStorage = np.empty(s_shape, dtype='float32')
        self.newStateStorage = np.empty(s_shape, dtype='float32')
        #
        self.storage = np.empty([maxSize, self.colSize], dtype='float32')
        self.currentRow = 0
        self.maxSize = maxSize
        self.s_size = s_shape
        self.a_size = a_size
        self.filledOnce = False

    def addData(self, s, a, s_new, r, done):
        # all_data = np.append(s,a)
        # all_data = np.append(all_data,s_new)
        self.stateStorage[self.currentRow][0:, 0:, 0:] = np.copy(s)
        self.newStateStorage[self.currentRow][0:, 0:, 0:] = np.copy(s_new)

        # print(all_data)
        all_data = np.append(a, r)
        all_data = np.append(all_data, done)

        self.storage[self.currentRow] = all_data
        self.currentRow += 1

        if self.currentRow == self.maxSize:  # reset when full
            self.full()

    def full(self):
        self.currentRow = 0
        self.filledOnce = True
        print("memory full yo")

    def getBatch(self, batchSize=10):
        if not self.filledOnce:
            choices = np.random.randint(0, self.currentRow, size=batchSize)
        else:
            choices = np.random.randint(0, self.maxSize, size=batchSize)

        return {"state": self.stateStorage[choices],
                "action": self.storage[choices][0:, 0:self.a_size],
                "new_state": self.newStateStorage[choices],
                "reward": self.storage[choices][0:, -2:-1],
                "done": self.storage[choices][0:, -1:]
                }


class Action:
    def __init__(self, name, action_range, isDiscrete):
        self.action_name = name
        self.action_range = action_range
        self.isDiscrete = isDiscrete


class Sim:
    def __init__(self, env_name, nn_params, run_params,
                 max_iterations=100000, interval=20):
        # self.env = gym.make('PixelCopter-v0')
        self.env = gym.make(env_name)
        self.skip_frames = 1

        # actions repeated skip_frames -1 times
        self.skip_frame_timer = self.skip_frames
        self.episode_maxLength = 10000
        self.run_params = run_params

        # this is for custom actions, other wise remove upcoming loop
        self.actions = [Action(0, [0, 1], True),  # left
                        Action(1, [0, 1], True),  # right
                        Action(2, [0, 1], True),
                        Action(3, [0, 1], True),
                        Action(4, [0, 1], True)
                        ]

        # use actions for all avaible actions in env
        self.actions = []
        for act in range(self.env.action_space.n):
            self.actions.append(Action(act, [0, 1], True))
        self.lastAction = None

        # cropped, rescaled
        # self.obsShape = list(self.env.observation_space.shape)

        self.obsShape = [60, 60, 3]  # cropped ,rescaled
        self.zoom = [1, 1]
        self.originalDim = list(self.env.observation_space.shape)

        t = 0
        for i in self.originalDim[0:-1]:
            self.zoom[t] = self.obsShape[t]/i
            t += 1
        zc = self.zoom[1]
        self.zoom[1] = self.zoom[0]
        self.zoom[0] = zc

        self.agent = qnn_agent.Qnetwork_agent(self.obsShape[:],
                                              self.actions, nn_params)

        self.max_iterations = max_iterations
        self.interval = interval
        self.rewards = []
        self.temp_rews = 0
        self.done = 0

        self.maxExperienceSize = 50000  # Memory size
        reward_size = 1  # reward vector size
        action_size = 1  # actions take during one step

        self.experienceData = Memory(self.obsShape[:],
                                     action_size,
                                     reward_size,
                                     self.maxExperienceSize
                                     )
        self.times = {"total": 0.0, "get_action": 0.0,
                      "train": 0.0, "create_batch": 0.0}

        self.cams0STD = []
        self.cams0MEAN = []
        self.cams1STD = []
        self.cams1MEAN = []
        self.collectCams = 100
        self.camT = 0

    def loadModel(self, name):
        self.agent.nn.nn = keras.models.load_model(name)
        self.agent.target_train()
        self.agent.exploreChance = self.agent.exploration_final_eps

    def runEpisode(self, testAgent=False, doUpdate=False):
        self.env.seed()
        observation = self.env.reset()

        # obs2 = observation
        obs2 = np.copy(observation/255.0)

        episodeReward = 0
        all_rewards = []
        # episodeData = []
        for t in range(self.episode_maxLength):
            if self.skip_frame_timer == self.skip_frames:
                if not testAgent:
                    if doUpdate:
                        self.agent.update(self)
                    t_before_action = time.time()
                    action = self.agent.getNextAction(obs2)
                    self.times["get_action"] += time.time() - t_before_action
                else:
                    self.env.render()
                    action = self.agent.getBestAction(obs2)
                self.skip_frame_timer = 0
            self.skip_frame_timer += 1
            self.lastAction = action
            prev_state = np.copy(obs2)
            observation, reward, done, info = self.env.step(action)
            episodeReward += reward
            all_rewards.append(reward)
            # print(reward)
            obs2 = np.copy(observation/255.0)

            r = reward
            # log rewards
            self.temp_rews += r
            if self.agent.step_counter % self.interval == 0:
                self.rewards.append(np.mean(self.temp_rews))
                self.temp_rews = 0
            if self.agent.step_counter > self.max_iterations:
                self.done = 1

            r = np.clip(reward, -1, 1)

            if not testAgent:
                if self.skip_frame_timer == 1:
                    self.experienceData.addData(prev_state, action,
                                                obs2, r, done)
            if done:
                self.skip_frame_timer = self.skip_frames

                break

            if t == self.episode_maxLength:  # never
                print("episode max length reached")

        # self.env.close()
        if not testAgent:
            if self.agent.exploreChance > self.agent.exploration_final_eps:
                if doUpdate:
                    self.agent.exploreChance *= 0.9
            return episodeReward
        else:
            return episodeReward

    def runIterations(self, testAgent=False, doUpdate=False, iterations=1000):
        self.env.seed()
        observation = self.env.reset()
        observation = rescale(observation, self.zoom)
        obs2 = np.copy(observation)

        all_rewards = []
        reseting = 0
        opt_flow = self.run_params["opt_flow"]

        if opt_flow:
            obs2[0:, 0:, 0] = rgb2gray(obs2)
            obs2[0:, 0:, 1] = obs2[0:, 0:, 1]*0
            obs2[0:, 0:, 2] = obs2[0:, 0:, 2]*0
            # pass

        for t in range(iterations):
            self.camT += 1
            if self.camT >= self.collectCams and t != 0:
                cam_shape = np.reshape(np.copy(obs2),
                                       (1, obs2.shape[1], obs2.shape[1], 3))
                cam = grad_cam(self.agent.nn.nn, cam_shape, 0,
                               self.agent.nn.nn.layers[-5].name,
                               len(self.actions))
                cam1 = grad_cam(self.agent.nn.nn, cam_shape, 1,
                                self.agent.nn.nn.layers[-5].name,
                                len(self.actions))
                self.camT = 0
                self.cams0STD.append(np.std(cam))
                self.cams0MEAN.append(np.mean(cam))
                self.cams1STD.append(np.std(cam1))
                self.cams1MEAN.append(np.mean(cam1))

            if self.skip_frame_timer == self.skip_frames:
                if not testAgent:
                    if doUpdate:
                        self.agent.update(self)
                    t_before_action = time.time()
                    action = self.agent.getNextAction(obs2)
                    self.times["get_action"] += time.time() - t_before_action
                else:
                    self.env.render()
                    action = self.agent.getBestAction(obs2)
                self.skip_frame_timer = 0
            self.skip_frame_timer += 1
            self.lastAction = action
            prev_state = np.copy(obs2)
            observation, reward, done, info = self.env.step(action)

            all_rewards.append(reward)
            # print(reward)
            observation = rescale(observation, self.zoom)
            obs2 = np.copy(observation)
            # plt.imshow(obs2)
            # plt.show()

            # OPTICAL FLOW
            if opt_flow:
                if reseting == 0:
                    gray = rgb2gray(obs2)
                    of = cv2.calcOpticalFlowFarneback(
                        prev_state[0:, 0:, 0]*255,
                        gray*255, None, 0.5, 3, 5, 3, 5, 1.2, 0)
                    # of = cv2.calcOpticalFlowFarneback(
                    #   rgb2gray(prev_state)*255,
                    #   gray*255,None,0.5, 3, 5, 3, 5, 1.2, 0)
                    obs2[0:, 0:, 0] = gray
                    obs2[0:, 0:, 1] = of[0:, 0:, 0]/10
                    obs2[0:, 0:, 2] = of[0:, 0:, 1]/10
                    # obs2[0:, 0:, 2] = drawShadow(gray, of)
                    # obs2[0:, 0:, 0] = obs2[0:, 0:, 0]+ of_y*500
                    # obs2[0:, 0:, 1] = obs2[0:, 0:, 1]+ of_y*500
                    # obs2[0:, 0:, 2] = obs2[0:, 0:, 2]+ of_y*500
                else:
                    obs2[0:, 0:, 0] = rgb2gray(obs2)
                    obs2[0:, 0:, 1] = obs2[0:, 0:, 1]*0
                    obs2[0:, 0:, 2] = obs2[0:, 0:, 2]*0
                    # pass
            reseting = 0
            r = reward
            # log rewards
            all_rewards.append(r)

            r = np.clip(reward, -1, 1)

            if not testAgent:
                if self.skip_frame_timer == 1:
                    self.experienceData.addData(prev_state,
                                                action, obs2,
                                                r, done)

            if t == self.episode_maxLength:  # never
                done = 1

            if done:
                self.skip_frame_timer = self.skip_frames

                self.env.seed()
                observation = self.env.reset()
                observation = rescale(observation, self.zoom)
                obs2 = np.copy(observation)
                if opt_flow:
                    obs2[0:, 0:, 0] = rgb2gray(obs2)
                    obs2[0:, 0:, 1] = obs2[0:, 0:, 1]*0
                    obs2[0:, 0:, 2] = obs2[0:, 0:, 2]*0
                reseting = 1

        # self.env.close()
        if not testAgent:
            if self.agent.exploreChance > self.agent.exploration_final_eps:
                if doUpdate:
                    self.agent.exploreChance *= 0.8
            return all_rewards
        else:
            return all_rewards

    def run(self, iterations=1000, doUpdate=True):
        # print("running...")
        results = []
        for i in range(iterations):
            # print("episode: ",i)
            results.append(self.runEpisode(testAgent=False, doUpdate=doUpdate))
        return results

    def run_iterations(self, iterations=1000, doUpdate=True):
        r = self.runIterations(testAgent=False,
                               doUpdate=doUpdate, iterations=iterations)
        return r

    def testAgent(self, iterations=5):
        results = []
        for i in range(iterations):
            results.append(self.runEpisode(testAgent=True))
        return results
