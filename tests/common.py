from copy import deepcopy  # NOQA
import numpy as np
import os

from drl_lab.env import Action
from drl_lab.expt import array2images, save_images  # NOQA


# dependent variables
actions = [Action(0, [0, 1], True),
           Action(1, [0, 1], True),
           Action(2, [0, 1], True),
           Action(3, [0, 1], True)]
dataset_num = 100
obs_shape = [48, 48, 3]
state_shape = obs_shape

actions_name_list = [0, 1, 2, 3]
action_indexes = {0: 0, 1: 1, 2: 2, 3: 3}
action_size = 1
batch_size = 32
dataset = [{
            'input': np.random.randn(*obs_shape),
            'output': np.random.permutation([1, 0, 0, 0])
           } for i in range(dataset_num)]
env_hparams = {
    'env_id': 'Pixelcopter-v0',
    'observation': {
        'normalize': False,
        'rescaled_shape': [],
        'opt_flow': False,
    },
    'action': {
        'excluded_actions': [],
    },
}
max_experience_size = 100
nn_hparams = {
    'layers': [
        ['conv', 30, 8, 4],
        ['conv', 40, 4, 3],
        ['conv', 60, 3, 1],
        ['gap'],
        ['fc', 512]],
    'learn_rate': 0.00005,
    'optimizer': 'RMSprop',
    'saved_model': None,
}
num_actions = len(actions)
reward_size = 1
run_hparams = {
    'test': False,
    'verbose': False,
    'save_at': None,  # None or [...] or int
    'interval': 2000,
    'max_steps': 400000,
    'num_runs': 1,
}
state = np.random.randn(*state_shape)
states = np.random.randn(32, *state_shape)


def get_resources_dir():
    here = os.path.dirname(os.path.realpath(__file__))
    resources_root = here+'/resources'
    if not os.path.exists(resources_root):
        raise FileNotFoundError(resources_root)
    return resources_root


def get_results_dir():
    here = os.path.dirname(os.path.realpath(__file__))
    results_root = here+'/results/test_'+str(os.getpid())
    if not os.path.exists(results_root):
        os.makedirs(results_root)
    return results_root


def weights_equal(source_model, target_model):
    source_weights = source_model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(source_weights)):
        if not np.array_equal(source_weights[i], target_weights[i]):
            return False
    return True
