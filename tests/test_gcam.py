import unittest

import keras
from keras import backend as K
from keras.layers (
    Activation,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
)
from keras.models import Sequential
import numpy as np
import tensorflow as tf


from drl_lab.expt import (
    array2images,
    save_images,
)
from drl_lab.gcam import (
    normalize,
    GradCAM,
    colorize_cam,
    apply_modifications,
    register_one_hot_gradient,
    modify_gradient_function,
    GradCAMExperimental,
    NegativeGradCAM,
    target_class_loss,
    target_class_loss_output_shape,
    GradCAMOld,
)
from tests.common import (
    get_test_model,
    get_test_images_as_array,
    get_results_dir,
)


logits_layer_index = -1
target_conv_layer_index = -6
class_index = 1
number_of_classes = 10
input_shape = (60, 60, 3)
test_images = np.random.randint(0, 255, (1, 60, 60, 3), np.uint8)
test_image = test_images[0]


def build_test_model():
    model = Sequential()
    model.add(Conv2D(30, (8, 8), strides=(4, 4),
              padding="same", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(40, (4, 4), strides=(4, 4), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(60, (6, 6), strides=(1, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('linear'))
    opt = keras.optimizers.RMSprop(lr=0.00005)
    model.compile(loss=keras.losses.mean_squared_error, optimizer=opt)
    return model


class TestGCAM(unittest.TestCase):
    def test_normalize(self):
        x = np.array([1, 2, 3, 4, 5, 6, 3], dtype=np.float32)
        normalized_x = K.eval(normalize(K.variable(x)))
        expected = x.shape
        self.assertEqual(expected, normalized_x.shape)
        expected = x/10
        self.assertEqual(expected, normalized_x)

    def test_colorize_cam(self):
        model = build_test_model()
        gcam = GradCAM(model, logits_layer_index, target_conv_layer_index,
                       class_index, number_of_classes)
        cam = gcam.do(test_images)
        colorized_cam = colorize_cam(cam, test_image)
        expected = cam.shape
        self.assertEqual(expected, colorized_cam.shape)

    def test_apply_modifications(self):
        model = build_test_model()
        model.layers[-1].activation = keras.layers.activations.softmax
        model = apply_modifications(model)
        pass  # TODO:)

    def test_register_one_hot_gradient(self):
        register_one_hot_gradient(class_index, number_of_classes)
        pass  # TODO:)

    def test_modify_gradient_function(self):
        model = build_test_model()
        modify_gradient_function(model)
        pass  # TODO:)

    def test_normalize_mean(self):
        normalized_state = K.eval(normalize(K.variable(test_image)))
        expected = test_image.shape
        self.assertEqual(expected, normalized_state.shape)
        np_normalized_state = \
            test_image / (np.mean(np.square(test_image)) + 1e-5)
        self.assertTrue(
            normalized_state.min() - np_normalized_state.min() < 0.1)
        self.assertTrue(
            normalized_state.max() - np_normalized_state.max() < 0.1)
        self.assertTrue(
            normalized_state.mean() - np_normalized_state.mean() < 0.1)

    def test_target_class_loss(self):
        mock_output_tensor = K.variable(np.random.randn(32, 1))
        loss = target_class_loss(mock_output_tensor, 0, number_of_classes)
        expected = (32, 4)
        self.assertEqual(expected, loss.get_shape())
        expected = np.abs(K.eval(loss)).max(axis=1)
        self.assertTrue(
            np.array_equal(expected, np.abs(np.sum(K.eval(loss), axis=1))))
        expected = tf.multiply(
            mock_output_tensor, K.one_hot([0], number_of_classes))
        self.assertTrue(
            np.array_equal(K.eval(expected), K.eval(loss)))

    def test_target_class_loss_output_shape(self):
        expected = test_image.shape
        self.assertEqual(
            expected, target_class_loss_output_shape(test_image.shape))


class TestGradCAM(unittest.TestCase):
    def test_1hot_grad(self):
        x = keras.layers.Input(shape=(3,))
        pow_x = keras.layers.multiply([x, x])
        y = keras.layers.activations.linear(pow_x)
        model = keras.models.Model(inputs=[x], outputs=y)
        predicted = model.predict(np.array([[1, 2, 3]]))
        expected = np.array([[1, 4, 9]], dtype=np.float32)
        self.assertEqual(expected, predicted)
        grads = K.gradients(y, x)[0]
        gradient_function = K.function(
            [model.input, K.learning_phase()],
            [grads],
        )
        gradients = gradient_function([np.array([[1, 2, 3]]), 0])[0]
        expected = np.array([[2, 4, 6]], dtype=np.float32)
        self.assertEqual(expected, predicted)
        y_2 = model.layers[-1].output[0, 2]
        grads = K.gradients(y_2, x)[0]
        gradient_function = K.function(
            [model.input, K.learning_phase()],
            [grads],
        )
        gradients = gradient_function([np.array([[1, 2, 3]]), 0])[0]
        expected = np.array([[0, 0, 6]], dtype=np.float32)
        self.assertEqual(expected, gradients)

    def test_GradCAM(self):
        model = build_test_model()
        gcam = GradCAM(model, logits_layer_index, target_conv_layer_index,
                       class_index, number_of_classes)
        cam = gcam.do(test_images)
        expected = (60, 60, 3)
        self.assertEqual(expected, cam.shape)


class TestGradCAMExperimental(unittest.TestCase):
    def test_GradCAMExperimental(self):
        model = build_test_model()
        gcam = GradCAMExperimental(model, logits_layer_index,
                                   target_conv_layer_index,
                                   class_index, number_of_classes)
        cam = gcam.do(test_images)
        expected = (60, 60, 3)
        self.assertEqual(expected, cam.shape)


class TestNegativeGradCAM(unittest.TestCase):
    def test_GradCAMExperimental(self):
        model = build_test_model()
        gcam = NegativeGradCAM(model, logits_layer_index,
                               target_conv_layer_index,
                               class_index, number_of_classes)
        cam = gcam.do(test_images)
        expected = (60, 60, 3)
        self.assertEqual(expected, cam.shape)


class TestGradCAMOld(unittest.TestCase):
    def test_GradCAMOld(self):
        test_model = get_test_model()
        test_images = get_test_images_as_array()
        results_dir = get_results_dir('test_gradcam/test_GradCAMOld')

        processors = []
        for i in range(number_of_classes):
            processors.append(GradCAMOld(test_model, 4, number_of_classes))

        cams = []
        for test_image in test_images:
            cam = np.array([], dtype=np.uint8).reshape(*test_image.shape)
            for processor in processors:
                _cam = processor.do(test_image)
                _cam = colorize_cam(cam, test_image)
                cams.append(np.concatenate([cam, _cam], axis=1))

        images = array2images(cams)
        save_images(results_dir, images)
