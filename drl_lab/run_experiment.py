#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 12:07:14 2018

@author: chavdar
"""


# import os

# import matplotlib.pyplot as plt
# import numpy as np

import experimenter


env = 'Breakout_pygame-v0'
run_params = {"run_interval": 2000,
              "max_steps": 400000,
              "opt_flow": 1,
              "log": 1,
              "save": [0.0, 0.3, 0.6, 1.0],
              "num_runs": 1,
              "load_model": 0  # "./results/breakout_cam/models/model1"
              }

nnetwork_params = {"architecture": [
                            ["conv", 30, 8, 4],
                            ["conv", 40, 4, 3],
                            ["conv", 60, 3, 1],
                            ["gap"],
                            ["fc", 512]],
                   "learn_rate": 0.00005,
                   "optimizer": "RMSprop",
                   }


if __name__ == "__main__":
    exp = experimenter.Experiment("breakout_cam2", env)
    exp.run(run_params, nn_params=nnetwork_params)
