#!/usr/bin/env python3

import gym,random,time
from gym import wrappers
#
import qnn_agent

import random
import numpy as np
from skimage.transform import rescale
import gym_ple
import cv2 
import keras
#np.dot(rs[...,:3], [0.299, 0.587, 0.114])
#random.seed(1)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def drawShadow(im,flow,thresh=0.6):
    shadows = np.ones_like(im)/2
    fy = flow[0:,0:,1]
    shadows[fy>thresh] =1.0
    shadows[fy<-thresh] =0
    return shadows
    
    
######
from keras import backend as K
import keras
from keras.layers import Input
from scipy import ndimage
import tensorflow as tf
import cv2,imageio

from keras.layers.core import Lambda
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np
#import matplotlib.pyplot as plt

#####    
    
def observation_preprocess(observation,zoom=[1,1],opt_flow=0,prev_obs=None,toGrayScale=False):
    # rescale if needed
    if zoom[0]==1 and zoom[1]==1:
        obs_preprocessed = np.copy(observation/255.0)
    else:
        obs_preprocessed = np.copy(rescale(observation,zoom))
    # add optical flow
    if opt_flow:
        if prev_state!="reseting":
            gray=rgb2gray(obs_preprocessed)
            of = cv2.calcOpticalFlowFarneback(rgb2gray(prev_state)*255,
                                        gray*255,None,0.5, 3, 5, 3, 5, 1.2, 0)
            if toGrayScale:
                obs_preprocessed[0:,0:,0] =  gray
                obs_preprocessed[0:,0:,1] =of[0:,0:,0]/10
                obs_preprocessed[0:,0:,2] =of[0:,0:,1]/10
            elif not toGrayScale:
                for i in opt_flow:
                    #add x,y flow one or both depending on opt flow parameter ([0,1] for both)
                    obs_preprocessed[0:,0:,3+i] =of[0:,0:,i]/10 
                    o
                    
        elif prev_state=="reseting":
            ray=rgb2gray(obs_preprocessed)
            if toGrayScale:
                obs_preprocessed[0:,0:,0] =  gray
                obs_preprocessed[0:,0:,1] =obs_preprocessed[0:,0:,1]*0
                obs_preprocessed[0:,0:,2] =obs_preprocessed[0:,0:,2]*0
            
from memory import *  

                
###
class Action:
    def __init__(self,name,action_range,isDiscrete):
        self.action_name = name
        self.action_range = action_range
        self.isDiscrete = isDiscrete
class Sim:
    def __init__(self,env_name,nn_params,run_params,max_iterations=100000,interval=20):
        #self.env = gym.make('PixelCopter-v0')
        self.env = gym.make(env_name)
        self.skip_frames = 1
        self.skip_frame_timer = self.skip_frames #actions repeated skip_frames -1 times
        self.episode_maxLength = 10000
        self.run_params = run_params
        # this is for custom actions, other wise remove upcoming loop 
        self.actions = [Action(0,[0,1],True), #left
                        Action(1,[0,1],True)  #right
                        #Action(2,[0,1],True),
                        #Action(3,[0,1],True),
                        #Action(4,[0,1],True)
                        ]
        # use actions for all avaible actions in env
        self.actions = []
        for act in range(self.env.action_space.n):
            self.actions.append(Action(act,[0,1],True))
        self.lastAction = None
        ##self.obsShape = list(self.env.observation_space.shape)# cropped , rescaled 
        
        #
        self.obsShape = [60,60,3]# cropped , rescaled 
        self.zoom = [1,1]
        self.originalDim = list(self.env.observation_space.shape)
        #
        t=0
        for i in self.originalDim[0:-1]:
            self.zoom[t] = self.obsShape[t]/i
            t+=1
        zc = self.zoom[1]
        self.zoom[1]=self.zoom[0]
        self.zoom[0]=zc

        #
        
        self.agent = qnn_agent.Qnetwork_agent(self.obsShape[:],self.actions,nn_params)
        ####
        self.max_iterations = max_iterations
        self.interval = interval
        self.rewards = []
        self.temp_rews = 0
        self.done = 0
        ####
        self.maxExperienceSize = 50000 # Memory size
        reward_size = 1  # reward vector size
        action_size = 1  # actions take during one step
        
        self.experienceData = Memory(self.obsShape[:],
                                     action_size,
                                     reward_size,
                                     self.maxExperienceSize
                                     )
        self.times={"total":0.0,"get_action":0.0,"train":0.0,"create_batch":0.0}


        ###
        self.cams0STD=[]
        self.cams0MEAN=[]
        self.cams1STD=[]
        self.cams1MEAN=[]
        self.collectCams=100
        self.camT=0
    def loadModel(self,name):
        self.agent.nn.nn = keras.models.load_model(name)
        self.agent.target_train()
        self.agent.exploreChance =  self.agent.exploration_final_eps
    
    def runIterations(self,testAgent=False, doUpdate=False,iterations=1000):
        self.env.seed()
        observation = self.env.reset()
        observation = rescale(observation,self.zoom)
        obs2 = np.copy(observation)
        #
        all_rewards = []  
        reseting = 0
        opt_flow=self.run_params["opt_flow"]
        if opt_flow:
            obs2[0:,0:,0] = rgb2gray(obs2)
            obs2[0:,0:,1] = obs2[0:,0:,1]*0
            obs2[0:,0:,2] = obs2[0:,0:,2]*0
            #pass
        for t in range(iterations):
            """
            self.camT +=1
            if self.camT >= self.collectCams and t!=0:
                cam = grad_cam(self.agent.nn.nn,np.reshape(np.copy(obs2 ),(1,obs2.shape[1],obs2.shape[1],3)),
                           0,self.agent.nn.nn.layers[-5].name,len(self.actions))
                cam1 = grad_cam(self.agent.nn.nn,np.reshape(np.copy(obs2 ),(1,obs2.shape[1],obs2.shape[1],3)),
                           1,self.agent.nn.nn.layers[-5].name,len(self.actions))
                self.camT=0
                self.cams0STD.append(np.std(cam))
                self.cams0MEAN.append(np.mean(cam))
                self.cams1STD.append(np.std(cam1))
                self.cams1MEAN.append(np.mean(cam1))
            """
            if self.skip_frame_timer == self.skip_frames:
                if testAgent==False:
                    if doUpdate:
                        self.agent.update(self)
                    t_before_action = time.time()
                    action = self.agent.getNextAction(obs2)
                    self.times["get_action"] += time.time() - t_before_action
                elif testAgent==True:
                    self.env.render()
                    action  = self.agent.getBestAction(obs2)
                self.skip_frame_timer = 0
            self.skip_frame_timer += 1
            self.lastAction = action
            prev_state = np.copy(obs2)
            observation, reward, done, info = self.env.step(action)
            
            all_rewards.append(reward)
            #print(reward)
            observation = rescale(observation,self.zoom)
            obs2 = np.copy(observation)
            #plt.imshow(obs2)
            #plt.show()
            ####OPTICAL FLOW
            if opt_flow:
                if reseting==0:
                    gray=rgb2gray(obs2)
                    of = cv2.calcOpticalFlowFarneback(prev_state[0:,0:,0]*255,
                            gray*255,None,0.5, 3, 5, 3, 5, 1.2, 0)
                    #of = cv2.calcOpticalFlowFarneback(rgb2gray(prev_state)*255,
                    #        gray*255,None,0.5, 3, 5, 3, 5, 1.2, 0)
                    obs2[0:,0:,0] =  gray
                    obs2[0:,0:,1] =of[0:,0:,0]/10
                    obs2[0:,0:,2] =of[0:,0:,1]/10
                    #obs2[0:,0:,2] = drawShadow(gray,of)
                    #obs2[0:,0:,0] = obs2[0:,0:,0]+ of_y*500
                    #obs2[0:,0:,1] = obs2[0:,0:,1]+ of_y*500
                    #obs2[0:,0:,2] = obs2[0:,0:,2]+ of_y*500
                else:
                    obs2[0:,0:,0] = rgb2gray(obs2)
                    obs2[0:,0:,1] = obs2[0:,0:,1]*0
                    obs2[0:,0:,2] = obs2[0:,0:,2]*0
                    #pass
            reseting=0
            r = reward
            #### log rewards
            all_rewards.append(r)
            ####
            r = np.clip(reward, -1, 1)
            
            if testAgent == False:
                if self.skip_frame_timer == 1:
                    self.experienceData.addData(prev_state,action,obs2,r,done)
                    
            if t == self.episode_maxLength:# never
                done=1
                    
            if done:
                self.skip_frame_timer = self.skip_frames

                self.env.seed()
                observation = self.env.reset()
                observation = rescale(observation,self.zoom)
                obs2 = np.copy(observation)
                if opt_flow:
                    obs2[0:,0:,0] = rgb2gray(obs2)
                    obs2[0:,0:,1] = obs2[0:,0:,1]*0
                    obs2[0:,0:,2] = obs2[0:,0:,2]*0
                reseting=1
            
           
                
        #self.env.close()
        if testAgent == False:
            if self.agent.exploreChance > self.agent.exploration_final_eps:
                if doUpdate:
                    self.agent.exploreChance *= 0.8
            return all_rewards
        if testAgent == True:
            return all_rewards
    def run(self,iterations=1000,doUpdate=True):
        #print("running...")
        results = []
        for i in range(iterations):
            #print("episode: ",i)
            results.append(self.runEpisode(testAgent=False,doUpdate=doUpdate))
        return results
    def run_iterations(self,iterations=1000,doUpdate=True):
        results=[]
        
        r = self.runIterations(testAgent=False,doUpdate=doUpdate,iterations=iterations)
        return r
            
    def testAgent(self,iterations=5):
        results = []
        for i in range(iterations):
            results.append(self.runEpisode(testAgent=True))
        return results
        