# https://github.com/vense/keras-grad-cam

import cv2
import keras
from keras import backend as K
from keras.layers.core import Lambda
import numpy as np
import tensorflow as tf


def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def grad_cam(input_model, image, category_index, layer_index, nb_classes):
    """
    Parameters
    ----------
    input_model   : keras.model.Model
    image         : numpy.ndarray (shape=(None, H, W, C))
    category_index: int
    layer_index   : int
    nb_classes    : int

    Returns
    -------
    cam : numpy.ndarray (shape=(H, W, 3))
    """
    def target_layer(x):
        return target_category_loss(x, category_index, nb_classes)

    x = input_model.layers[-1].output
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(x)
    model = keras.models.Model(input_model.layers[0].input, x)

    loss = K.sum(model.layers[-1].output)
    conv_output = model.layers[layer_index].output

    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function(
        [model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))

    cam = np.ones(output.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    cam = cv2.resize(cam, (image.shape[1:-1]))
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    return cam


def colorize_cam(cam, image):
    """
    Parameters
    ----------
    cam           : numpy.ndarray (shape=(H, W, 3))
    image         : numpy.ndarray (shape=(H, W, C))

    Returns
    -------
    colorized_cam : numpy.ndarray (shape=(H, W, 3))
    """
    cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    return cam


if __name__ == '__main__':
    from datetime import datetime, timedelta, timezone
    import importlib
    import os
    import sys

    from drl_lab.agents import QnetworkAgent
    from drl_lab.env import create_env
    from drl_lab.expt import array2images, save_images

    # target results
    target_dir = sys.argv[1]
    if not os.path.exists(target_dir):
        raise FileNotFoundError("not found: {}".format(target_dir))
    # hprams
    hparams_file = "{}/hprams.py".format(target_dir)
    if not os.path.exists(hparams_file):
        raise FileNotFoundError("not found: {}".format(hparams_file))
    sys.path.append(hparams_file)
    hparams = importlib.import_module('hparams')
    env_hparams = hparams.env_hparams
    nn_hparams = hparams.nn_hparams
    # models
    models_dir = "{}/models".format(target_dir)
    if not os.path.exists(models_dir):
        raise FileNotFoundError("not found: {}".format(models_dir))
    model_files = {}  # {name: path}
    for f in os.scandir():
        if f.name.startswith('model'):
            model_files[f.name] = "{}/{}".format(models_dir, f.name)

    # results
    JST = timezone(timedelta(hours=+9), 'JST')
    name = 'gcam_'+datetime.now(JST).strftime('%Y%m%d%H%M%S')
    here = os.path.dirname(os.path.realpath(__file__))
    results_root = here+'/results/'+name
    if not os.path.exists(results_root):
        os.makedirs(results_root)

    # run
    env = create_env(env_hparams)
    num_actions = len(env.actions)
    for name in model_files:
        save_dir_model = "{}/{}".format(results_root, name)
        if not os.path.exists(save_dir_model):
            os.makedirs(save_dir_model)
        nn_hparams['saved_model'] = model_files[name]
        agent = QnetworkAgent(env, nn_hparams)
        for ei in range(10):  # 10 episodes
            save_dir_episode = "{}/episode_{}".format(save_dir_model, ei)
            if not os.path.exists(save_dir_episode):
                os.makedirs(save_dir_episode)
            images = []
            observation, done = env.reset(), False
            while not done:
                action = agent.get_best_action(observation)
                observation, reward, done, info = env.step(action)
                raw_observation = env.last_obs_raw
                image = np.concatnate([raw_observation, observation], axis=1)
                for ai in range(num_actions):
                    _observation = observation.reshape(1, *observation.shape)
                    cam = grad_cam(agent.nn.nn, _observation, ai, 4)
                    cam = colorize_cam(cam, observation)
                    image = np.concatenate([image, cam], axis=1)
                images.append(image)
            images = array2images(images, False)
            save_images(save_dir_episode, images)
