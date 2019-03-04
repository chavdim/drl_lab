"""
Several Grad-CAM implementations.

GradCAM: Main and latest implementation.
GradCAMExperimental: Experimental implementation.
GradCAMOld: Old version implementation.

References:
- https://arxiv.org/pdf/1610.02391.pdf
- https://github.com/ramprs/grad-cam
- https://github.com/jacobgil/keras-grad-cam
"""

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
from tensorflow.python.framework import ops


def normalize(x):
    """
    Utility function to normalize a tensor by its L2 norm.

    Parameters
    ----------
    x : Tensor

    Returns
    -------
    out : Tensor
    """
    return (x + 1e-10) / (K.sqrt(K.sum(K.square(x))) + 1e-10)


class GradCAM:
    def __init__(self,
                 model,
                 logits_layer_index,
                 target_conv_layer_index,
                 class_index,
                 number_of_classes,
                 ):
        """
        Parameters
        ----------
        model : keras.model.Model
            CNN model.
        logits_layer_index : int
            Index of layer outputting logits.
        target_conv_layer_index : int
            Index of last convolutional layer.
        class_index : int
            Targeted class index to visualize.
        number_of_classes : int
            Number of all classes output by model.

        Note
        ----
        Grad-CAM では、勾配を求める時に出力層の 1-hot な勾配から畳み込み層の勾配
        を後方伝播を使って求めるが、以下の実装でも同様のことが行える。
        これは連鎖律から考えれば求まる。 (TODO: Translate this paragraph.)
        """
        y = model.layers[logits_layer_index].output[0, class_index]
        target_conv_layer_output = \
            model.layers[target_conv_layer_index].output
        grads = normalize(K.gradients(y, target_conv_layer_output)[0])
        gradient_function = K.function(
            [model.input, K.learning_phase()],
            [target_conv_layer_output, grads],
        )

        """
        The function that returns the gradients of target convolutional
        layer for y.

        Parameters
        ----------
        in : list of numpy.ndarray and keras.backend.learning_phase()
            numpy.ndarray is input of model (shape=(-1, H, W, C)).

        Returns
        -------
        out : list of two numpy.ndarrays (shape=(-1, H, W, N))
            One array represents target convolutional layer output,
            H, W is kernel_size, and N is number of filters.
            The other is gradients, H, W and N means same as another one.
        """
        self.gradient_function = gradient_function

    def do(self, image):
        """
        Execute Grad-CAM.

        Parameters
        ----------
        image : numpy.ndarray (shape=(-1, H, W, C))

        Returns
        -------
        cam : numpy.ndarray (shape=(H, W, C))
        """
        feature_maps, gradients = self.gradient_function([image, 0])
        feature_maps, gradients = feature_maps[0], gradients[0]

        weights = np.mean(gradients, axis=(0, 1))
        cam = np.dot(feature_maps, weights)
        cam = np.maximum(cam, 0)

        cam = cv2.resize(cam, (image.shape[1:-1]))
        cam = (cam + 1e-10) / (np.max(cam) + 1e-10)

        return cam


def colorize_cam(cam, image):
    """
    Utility function to colorize and overlay a cam.

    Parameters
    ----------
    cam : numpy.ndarray (shape=(H, W, C))
    image : numpy.ndarray (shape=(H, W, C))
        Image to be overlaying.

    Returns
    -------
    cam : numpy.ndarray (shape=(H, W, C))
        Colorized and overlaid cam.

    Note
    ----
    cam and image must be same size.
    """
    cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam = cam[:, :, ::-1]
    cam = np.float32(cam) + np.float32(image)
    cam = (255 * cam + 1e-10) / (np.max(cam) + 1e-10)
    cam = np.uint8(cam)

    return cam


# Experimental implementations


def apply_modifications(model):
    """
    Save the model temporarily and load to apply changes.

    Parameters
    ----------
    model : keras.models.Sequential

    Returns
    -------
    out : keras.models.Sequential
    """
    model_path = os.path.join(
        tempfile.gettempdir(),
        next(tempfile._get_candidate_names()) + '.h5'
    )
    try:
        model.save(model_path)
        return load_model(model_path)
    finally:
        os.remove(model_path)


def register_one_hot_gradient(class_index, number_of_classes):
    """
    Register new gradient function to tensorflow gradient registry.

    Parameters
    ----------
    class_index : int
        Targeted class index to visualize.
    number_of_classes : int
        Number of all classes output by model.
    """
    if "OneHotGrad" not in ops._gradient_registry._registry:
        @ops.RegisterGradient('OneHotGrad')
        def _OneHotGrad(op, grad):
            """
            Custom gradient function which returns one hot gradient.
            """
            dtype = op.inputs[0].dtype
            return tf.one_hot([class_index], number_of_classes, dtype=dtype)


def modify_gradient_function(model):
    """
    Replace the gradient function of the model using gradient_override_map.

    Parameters
    ----------
    model : keras.models.Sequential

    Returns
    -------
    model : keras.models.Sequential

    Note
    ----
    The gradient function's name of tensorflow.identity and
    keras.activation.linear are same.
    """
    g = K.get_session().graph
    with g.gradient_override_map({'Identity': 'OneHotGrad'}):
        model.layers[-1].activation = activations.linear
        model = apply_modifications(model)
    return model


def normalize_mean(x):
    """
    Old version.
    Utility function to normalize a tensor by its RMS.

    Parameters
    ----------
    x : Tensor

    Returns
    -------
    out : Tensor

    See Also
    --------
    normalize: Latest version.
    """
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


class GradCAMExperimental:
    """
    Experimental implementation of GradCAM.

    See Also
    --------
    GradCAM : Main and latest implementation.
    """
    def __init__(self,
                 model,
                 logits_layer_index,
                 target_conv_layer_index,
                 class_index,
                 number_of_classes,
                 override_gradient_function=False,
                 use_1by1_relu=False,
                 use_old_normalization=False,
                 without_normalization=False,
                 ):
        """
        Parameters
        ----------
        override_gradient_function : bool
            Use OneHotGrad.
        use_1by1_relu : bool
            Use ReLU every weight * feature-map calculation.
        use_old_normalization : bool
            Use old version normalize function.
        without_normalization : bool
            Do not use normalize function.
        """
        if use_old_normalization:
            _normalize = normalize_mean
        else:
            _normalize = normalize

        if without_normalization:
            def normalize_none(x):
                return x
            _normalize = normalize_none

        if override_gradient_function:
            register_one_hot_gradient(class_index, number_of_classes)
            model = modify_gradient_function(model)
            y = model.output
            target_conv_layer_output = \
                model.layers[target_conv_layer_index].output
            grads = _normalize(K.gradients(y, target_conv_layer_output)[0])
            gradient_function = K.function(
                [model.input, K.learning_phase()],
                [target_conv_layer_output, grads],
            )
        else:
            y = model.layers[logits_layer_index].output[0, class_index]
            target_conv_layer_output = \
                model.layers[target_conv_layer_index].output
            grads = _normalize(K.gradients(y, target_conv_layer_output)[0])
            gradient_function = K.function(
                [model.input, K.learning_phase()],
                [target_conv_layer_output, grads],
            )

        self.gradient_function = gradient_function
        self.use_1by1_relu = use_1by1_relu

    def do(self, image):
        feature_maps, gradients = self.gradient_function([image, 0])
        feature_maps, gradients = feature_maps[0], gradients[0]

        weights = np.mean(gradients, axis=(0, 1))

        if self.use_1by1_relu:
            cam = np.zeros(feature_maps.shape[0:2], dtype=np.float32)
            for i, weight in enumerate(weights):
                _relu = np.maximum(weight * feature_maps[:, :, i], 0)
                cam += _relu
        else:
            cam = np.dot(feature_maps, weights)

        cam = cv2.resize(cam, (image.shape[1:-1]))
        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-5)

        return cam


# Visualize negative contribution


class NegativeGradCAM(GradCAM):
    """
    Visualize negative contribution of Grad-CAM

    See Also
    --------
    GradCAM : Main and latest implementation.
    """
    def do(self, image):
        feature_maps, gradients = self.gradient_function([image, 0])
        feature_maps, gradients = feature_maps[0], gradients[0]

        weights = np.mean(gradients, axis=(0, 1)) * -1  # get negative grad
        cam = np.dot(feature_maps, weights)

        cam = cv2.resize(cam, (image.shape[1:-1]))
        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-5)

        return cam


# Old implementations


def target_class_loss(x, class_index, number_of_classes):
    """
    Returns only the loss of the target class.

    Parameters
    ----------
    x                  : Tensor
    class_index        : int
    number_of_classes  : int

    Returns
    -------
    ret : Tensor (shape=(None, number_of_classes))
    """
    return tf.multiply(x, K.one_hot([class_index], number_of_classes))


def target_class_loss_output_shape(input_shape):
    return input_shape


class GradCAMOld:
    """
    Old implementation of GradCAM.

    See Also
    --------
    GradCAM : Main and latest implementation.
    """
    def __init__(self,
                 input_model,
                 class_index,
                 layer_index,
                 number_of_classes,
                 ):
        """
        Parameters
        ----------
        input_model : keras.model.Model
        class_index : int
        layer_index : int
        number_of_classes : int
        """
        y = input_model.layers[-1].output
        y = Lambda(
            lambda x: target_class_loss(x, class_index, number_of_classes),
            output_shape=target_class_loss_output_shape
        )(y)

        model = keras.models.Model(input_model.layers[0].input, y)

        loss = K.sum(model.layers[-1].output)
        conv_output = model.layers[layer_index].output

        grads = normalize(K.gradients(loss, conv_output)[0])
        gradient_function = K.function(
            [model.layers[0].input], [conv_output, grads])

        self.gradient_function = gradient_function

    def do(self, image):
        output, grads_val = self.gradient_function([image])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        weights = np.mean(grads_val, axis=(0, 1))

        cam = np.zeros(output.shape[0:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]
        cam = cv2.resize(cam, (image.shape[1:-1]))
        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-5)

        return cam
