import numpy as np
import keras
import gym
import gym_ple
#
from skimage.transform import rescale

env_name = 'PixelCopter-v0'
run_for = 400

obsShape = [60,60,3]# cropped , rescaled 
zoom = [1,1]
originalDim = list(env.observation_space.shape)
#
t=0
for i in originalDim[0:-1]:
    zoom[t] = obsShape[t]/i
    t+=1
zc = zoom[1]
zoom[1]=zoom[0]
zoom[0]=zc

env = gym.make(env_name)
env.seed()
observation = env.reset()
observation = rescale(observation,zoom)
action = 0

for i in range(run_for):
	observation, reward, done, info = env.step(action)
	action = qnn.getBest