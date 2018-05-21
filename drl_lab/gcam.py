from keras import backend as K
from keras.layers.core import Lambda
from keras.models import Sequential
import numpy as np
import tensorflow as tf


def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def grad_cam(input_model, image, category_index, layer_name, nb_classes):
    model = Sequential()
    model.add(input_model)

    def target_layer(x):
        return target_category_loss(x, category_index, nb_classes)

    model.add(Lambda(target_layer,
                     output_shape=target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)

    conv_output = [l for l in model.layers[0].layers if l.name is layer_name]
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

    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    return heatmap


class MockSim:
    def __init__(self):
        self.cams_0_std = []
        self.cams_0_mean = []
        self.cams_1_std = []
        self.cams_1_mean = []
        self.collect_cams = 100
        self.cam_t = 0

    def run_iterations(self):
        observation = [1, 2, 3]
        iterations = 3
        for t in range(iterations):
            self.cam_t += 1
            if self.cam_t >= self.collect_cams and t != 0:
                cam_shape = np.reshape(np.copy(observation),
                                       (1, observation.shape[1],
                                        observation.shape[1], 3))
                cam = grad_cam(self.agent.nn.nn, cam_shape, 0,
                               self.agent.nn.nn.layers[-5].name,
                               len(self.actions))
                cam1 = grad_cam(self.agent.nn.nn, cam_shape, 1,
                                self.agent.nn.nn.layers[-5].name,
                                len(self.actions))
                self.cam_t = 0
                self.cams_0_std.append(np.std(cam))
                self.cams_0_mean.append(np.mean(cam))
                self.cams_1_std.append(np.std(cam1))
                self.cams_1_mean.append(np.mean(cam1))
