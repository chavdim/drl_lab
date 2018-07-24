# https://github.com/vense/keras-grad-cam

import os
import tempfile

import cv2
import keras
from keras import activations
from keras import backend as K
from keras.layers.core import Lambda
from keras.models import load_model
import numpy as np
import tensorflow as tf


def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def apply_modifications(model):
    model_path = os.path.join(
        tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path)
    finally:
        os.remove(model_path)


class GradCam:
    def __init__(self, input_model, category_index, layer_index, nb_classes):
        """
        Parameters
        ----------
        input_model   : keras.model.Model
        category_index: int
        layer_index   : int
        nb_classes    : int
        """
        def target_layer(x):
            return target_category_loss(x, category_index, nb_classes)

        # modify last layer activation to softmax
        input_model.layers[-1].activation = activations.softmax
        input_model = apply_modifications(input_model)

        x = input_model.layers[-1].output
        x = Lambda(
            target_layer, output_shape=target_category_loss_output_shape)(x)
        model = keras.models.Model(input_model.layers[0].input, x)

        loss = K.sum(model.layers[-1].output)
        conv_output = model.layers[layer_index].output

        grads = normalize(K.gradients(loss, conv_output)[0])
        gradient_function = K.function(
            [model.layers[0].input], [conv_output, grads])

        self.gradient_function = gradient_function

    def do(self, image):
        """
        Parameters
        ----------
        image         : numpy.ndarray (shape=(None, H, W, C))

        Returns
        -------
        cam : numpy.ndarray (shape=(H, W, C))
        """

        output, grads_val = self.gradient_function([image])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        weights = np.mean(grads_val, axis=(0, 1))

        cam = np.ones(output.shape[0:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]
        cam = cv2.resize(cam, (image.shape[1:-1]))
        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-5)

        return cam


def colorize_cam(cam, image):
    """
    Parameters
    ----------
    cam           : numpy.ndarray (shape=(H, W, C))
    image         : numpy.ndarray (shape=(H, W, C))

    Returns
    -------
    colorized_cam : numpy.ndarray (shape=(H, W, C))
    """
    cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / (np.max(cam) + 1e-5)
    cam = np.uint8(cam)

    return cam
