from importlib import machinery
import os
from pprint import pprint
import sys

import numpy as np
from PIL import Image
from skimage.transform import rescale as sk_rescale

from drl_lab.agents import QnetworkAgent
from drl_lab.env import create_env
from drl_lab.expt import deprocess, arrays2images, save_gif
from drl_lab.gcam import GradCam, colorize_cam


def check_existence(path):
    if not os.path.exists(path):
        raise FileNotFoundError("not found: {}".format(path))


def get_hprams(parent_path):
    hparams_file = "{}/hparams.py".format(parent_path)
    check_existence(hparams_file)
    hparams = machinery.SourceFileLoader('hparams', hparams_file).load_module()
    env_hparams = hparams.env_hparams
    nn_hparams = hparams.nn_hparams
    return env_hparams, nn_hparams


def get_model_files(parent_path):
    models_directory = "{}/models".format(parent_path)
    check_existence(models_directory)
    model_files = {}  # {name: path}
    for model_file in os.scandir(models_directory):
        if model_file.name.startswith('model'):
            model_files[model_file.name] = \
                "{}/{}".format(models_directory, model_file.name)
    return model_files


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def initialize_play_results_directory(parent_path, num_actions):
    steps_results_directory = \
        "{}/steps".format(parent_path)
    create_directory(steps_results_directory)

    observation_results_directory = \
        "{}/observations".format(parent_path)
    create_directory(observation_results_directory)

    raw_observation_results_directory = \
        "{}/raw_observations".format(parent_path)
    create_directory(raw_observation_results_directory)

    for action_index in range(num_actions):
        grad_cam_results_directory = \
            "{}/grad_cam_action_{}".format(parent_path, action_index)
        create_directory(grad_cam_results_directory)


def save_ndarray_as_image(save_path, ndarray):
    _ndarray = np.copy(ndarray)
    if _ndarray.min() < 0.0:
        _ndarray = _ndarray - _ndarray.min()
    if _ndarray.max() > 255:
        _ndarray = _ndarray / _ndarray.max()
    if _ndarray.max() <= 1.0:
        _ndarray = _ndarray * 255
    if _ndarray.dtype != np.uint8:
        _ndarray = np.uint8(_ndarray)
    image = Image.fromarray(_ndarray)
    image.save(save_path)


def save_play_results_images(
        parent_path, step_index, observation, raw_observation, cams):
    observation_result_path = \
        "{}/observations/{:04d}.png".format(parent_path, step_index)
    save_ndarray_as_image(observation_result_path, observation)

    raw_observation_result_path = \
        "{}/raw_observations/{:04d}.png".format(parent_path, step_index)
    save_ndarray_as_image(raw_observation_result_path, raw_observation)

    for index in range(len(cams)):
        grad_cam_result_path = \
            "{}/grad_cam_action_{}/{:04d}.png".format(
                            parent_path, index, step_index)
        save_ndarray_as_image(grad_cam_result_path, cams[index])


def save_play_results_step(parent_path, step_index, action,
                           reward, observation, raw_observation, cams):
    step_results_path = "{}/steps/{:04d}.npy".format(parent_path, step_index)
    all_in_1 = np.array(
        [[action], [reward], observation, raw_observation, cams])
    np.save(step_results_path, all_in_1)
    save_play_results_images(
        parent_path, step_index, observation, raw_observation, cams)


def draw_action_as_array(action, num_actions, shape):
    width = shape[0]
    linspace = int(width / (num_actions + 1))
    action_position = (action + 1) * linspace
    action_array = np.zeros(shape, dtype=np.uint8)  # initialize
    action_array[:, action_position, :] = 255
    return action_array


def generate_gif_from_results_directory(results_directory):
    results_array = []
    for step_index in range(1000):
        step_results_path = \
            "{}/steps/{:04d}.npy".format(results_directory, step_index)
        action, _, observation, raw_observation, cams = \
            np.load(step_results_path)

        action_array = draw_action_as_array(
            action[0], len(cams), observation.shape)
        observation = deprocess(observation, False)

        results = np.concatenate(
            (action_array, observation, raw_observation), axis=1)
        for cam in cams:
            results = np.concatenate((results, cam), axis=1)

        results_array.append(deprocess(results, False))

    images = arrays2images(results_array)
    save_path = "{}/all.gif".format(results_directory)
    save_gif(save_path, images)


if __name__ == '__main__':
    # target results
    target_results_directory = sys.argv[1]
    check_existence(target_results_directory)
    print("[* Play target directory *]: {}".format(target_results_directory))

    # hprams
    env_hparams, nn_hparams = get_hprams(target_results_directory)
    print('[* Hparams loaded *]:')
    pprint({
        'env_hparams': env_hparams,
        'nn_hparams': nn_hparams,
    })

    # models
    model_files = get_model_files(target_results_directory)
    print('[* Model files *]')
    pprint(model_files)

    # play results
    play_results_root_directory = "{}/play".format(target_results_directory)
    create_directory(target_results_directory)
    print("[* Save to *]: {}".format(play_results_root_directory))

    # play
    print('[* Start playing *]')
    env = create_env(env_hparams)
    num_actions = len(env.actions)

    for model_file in model_files:
        print("[* Model file *]: {}".format(model_file))

        play_results_model_directory = \
            "{}/{}".format(play_results_root_directory, model_file)
        create_directory(play_results_model_directory)
        print("[* Save to *]: {}".format(play_results_model_directory))

        # load the model
        nn_hparams['saved_model'] = model_files[model_file]
        agent = QnetworkAgent(env, nn_hparams)

        # grad-cams
        grad_cams = []
        for action_index in range(num_actions):
            grad_cams.append(
                GradCam(agent.nn.nn, action_index, 4, num_actions))

        initialize_play_results_directory(
                        play_results_model_directory, num_actions)

        observation = env.reset()

        print("[* Initialized *]")

        for step_index in range(1000):  # until 1000 steps or done
            print('>', end='', flush=True)  # progress

            action = agent.get_best_action(observation)
            observation, reward, _, _ = env.step(action)
            raw_observation = env.last_obs_raw
            if env.rescale:
                raw_observation = sk_rescale(
                    raw_observation, env.zoom, preserve_range=True)

            cams = []
            _observation = observation.reshape(1, *observation.shape)
            for grad_cam in grad_cams:
                cam = grad_cam.do(_observation)
                cam = colorize_cam(cam, raw_observation)
                cams.append(cam)

            save_play_results_step(play_results_model_directory,
                                   step_index, action, reward,
                                   observation, raw_observation, cams)

        generate_gif_from_results_directory(play_results_model_directory)
