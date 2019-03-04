import numpy as np
import os

from PIL import Image

from drl_lab.models import load_model


"""
Variables
---------
"""


actions = [0, 1, 2, 3]
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
    'env_id': 'Breakout_pygame-v0',
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


"""
Resource management methods
-------- ---------- -------
"""


def get_resources_dir():
    here = os.path.dirname(os.path.realpath(__file__))
    resources_root = here+'/resources'
    if not os.path.exists(resources_root):
        raise FileNotFoundError(resources_root)
    return resources_root


def get_test_model_path(model_name='model01'):
    resource_dir = get_resources_dir()
    model_path = resource_dir+'/models/'+model_name
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    return model_path


def get_test_model(model_name='model01'):
    model_path = get_test_model_path(model_name)
    return load_model(model_path)


def get_test_images_as_list(images_name='images1'):
    resource_dir = get_resources_dir()
    images_path = resource_dir+'/images/'+images_name
    if not os.path.exists(images_path):
        raise FileNotFoundError(images_path)
    images = []
    for something in os.listdir(images_path):
        if something[0] == '.':
            continue
        image = Image.open(something)
        images.append(image)
    return images


def get_test_images_as_array(images_name='images1'):
    images = get_test_images_as_list(images_name)
    images_array = []
    for image in images:
        image_array = np.array(image)
        images_array.append(image_array)
    return np.array(images_array)


"""
Results management methods
------- ---------- -------
"""


def get_results_root():
    here = os.path.dirname(os.path.realpath(__file__))
    results_root = here+'/results/test_'+str(os.getpid())
    if not os.path.exists(results_root):
        os.makedirs(results_root)
    return results_root


def get_results_dir(dir_name):
    results_root = get_results_root()
    results_dir = results_root+'/'+dir_name
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


"""
Utils
-----
"""


def weights_equal(source_model, target_model):
    source_weights = source_model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(source_weights)):
        if not np.array_equal(source_weights[i], target_weights[i]):
            return False
    return True
