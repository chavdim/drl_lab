from copy import deepcopy
import unittest

import keras
import numpy as np

from drl_lab.models import (
    build_QCNN,
    QCNN,
    dataset2XY,
    state2data,
    load_model,
)
from tests.common import (
    nn_hparams,
    get_test_model_path,
    weights_equal,
)

batch_size = 10
obs_shape = (5, 5, 3)
dataset_num = 100
dataset = [{
    'input': np.random.randn(*obs_shape),
    'output': np.random.permutation([1, 0, 0]),
} for i in range(dataset_num)]
state_shape = obs_shape
state = np.random.randn(*state_shape)
num_actions = 3


class TestQCNN(unittest.TestCase):
    def setUp(self):
        self.qcnn = QCNN(state_shape, num_actions, nn_hparams)

    def test_init(self):
        qcnn = self.qcnn
        nn = qcnn.nn

        expected = 9
        self.assertEqual(expected, len(nn.layers))
        expected = [None, *state_shape]
        self.assertEqual(expected, nn.input.get_shape().as_list())
        expected = keras.layers.Conv2D
        self.assertEqual(expected, type(nn.layers[0]))
        expected = nn_hparams['layers'][0][1]
        self.assertEqual(expected, nn.layers[0].filters)
        expected = (nn_hparams['layers'][0][2],)*2
        self.assertEqual(expected, nn.layers[0].kernel_size)
        expected = (nn_hparams['layers'][0][3],)*2
        self.assertEqual(expected, nn.layers[0].strides)
        expected = 'same'
        self.assertEqual(expected, nn.layers[0].padding)
        expected = 'relu'
        self.assertEqual(expected, nn.layers[1].get_config()['activation'])
        expected = keras.layers.Conv2D
        self.assertEqual(expected, type(nn.layers[2]))
        expected = nn_hparams['layers'][1][1]
        self.assertEqual(expected, nn.layers[2].filters)
        expected = (nn_hparams['layers'][1][2],)*2
        self.assertEqual(expected, nn.layers[2].kernel_size)
        expected = (nn_hparams['layers'][1][3],)*2
        self.assertEqual(expected, nn.layers[2].strides)
        expected = 'same'
        self.assertEqual(expected, nn.layers[2].padding)
        expected = 'relu'
        self.assertEqual(expected, nn.layers[3].get_config()['activation'])
        expected = keras.layers.Conv2D
        self.assertEqual(expected, type(nn.layers[4]))
        expected = nn_hparams['layers'][2][1]
        self.assertEqual(expected, nn.layers[4].filters)
        expected = (nn_hparams['layers'][2][2],)*2
        self.assertEqual(expected, nn.layers[4].kernel_size)
        expected = (nn_hparams['layers'][2][3],)*2
        self.assertEqual(expected, nn.layers[4].strides)
        expected = 'same'
        self.assertEqual(expected, nn.layers[4].padding)
        expected = 'relu'
        self.assertEqual(expected, nn.layers[5].get_config()['activation'])
        expected = keras.layers.GlobalAveragePooling2D
        self.assertEqual(expected, type(nn.layers[6]))
        expected = 'dens'
        self.assertEqual(expected, nn.layers[7].name[:4])
        expected = nn_hparams['layers'][4][1]
        self.assertEqual(expected, nn.layers[7].units)
        expected = 'relu'
        self.assertEqual(expected, nn.layers[7].get_config()['activation'])
        expected = 'dens'
        self.assertEqual(expected, nn.layers[8].name[:4])
        expected = num_actions
        self.assertEqual(expected, nn.layers[8].units)
        expected = 'linear'
        self.assertEqual(expected, nn.layers[8].get_config()['activation'])
        expected = keras.optimizers.RMSprop
        self.assertEqual(expected, type(nn.optimizer))
        expected = keras.losses.mean_squared_error
        self.assertEqual(expected, nn.loss)
        expected = [None, num_actions]
        self.assertEqual(expected, nn.output.get_shape().as_list())

    def test_train(self):
        qcnn = self.qcnn
        train_X, train_Y = dataset2XY(dataset)

        nn_before = keras.models.clone_model(qcnn.nn)
        nn_before.set_weights(qcnn.nn.get_weights())
        qcnn.train(train_X, train_Y, batch_size, epochs=10, shuffle=True)
        nn_after = qcnn.nn
        self.assertFalse(weights_equal(nn_before, nn_after))

    def test_forward_prop(self):
        qcnn = self.qcnn
        train_X, _ = dataset2XY(dataset)

        retval = qcnn.forward_prop(train_X, batch_size)
        expected = len(train_X)
        self.assertEqual(expected, len(retval))

        X = train_X[:batch_size]
        retval = qcnn.forward_prop(X, batch_size)
        expected = batch_size

        self.assertEqual(expected, len(retval))
        expected = num_actions

        self.assertEqual(expected, len(retval[0]))

    def test_copy_model(self):
        qcnn = self.qcnn

        target = keras.models.clone_model(qcnn.nn)
        target.set_weights(qcnn.nn.get_weights())
        self.assertTrue(weights_equal(qcnn.nn, target))

        qcnn_weights = qcnn.nn.get_weights()
        qcnn_weights[0] = qcnn_weights[0]*0.99
        qcnn.nn.set_weights(qcnn_weights)
        self.assertFalse(weights_equal(qcnn.nn, target))


class TestModel(unittest.TestCase):
    def test_build_QCNN(self):
        model = build_QCNN(obs_shape, num_actions, nn_hparams['layers'],
                           nn_hparams['learn_rate'], nn_hparams['optimizer'])

        expected = 9
        self.assertEqual(expected, len(model.layers))

    def test_dataset2XY(self):
        X, Y = dataset2XY(dataset)
        self.assertEqual(len(X), dataset_num)
        self.assertEqual(len(Y), dataset_num)
        self.assertEqual(str(type(X)), "<class 'numpy.ndarray'>")
        self.assertEqual(str(type(Y)), "<class 'numpy.ndarray'>")
        self.assertEqual(X.shape[1:], dataset[0]['input'].shape)
        self.assertEqual(Y.shape[1:], dataset[0]['output'].shape)

    def test_state2data(self):
        data = state2data(state)
        self.assertEqual(data.shape, (1, *state_shape))

    def test_load_model(self):
        test_model_path = get_test_model_path()

        nn = load_model(test_model_path)
        expected = "<class 'keras.models.Sequential'>"
        self.assertTrue(expected, str(type(nn)))

        _nn_hparams = deepcopy(nn_hparams)
        _nn_hparams['saved_model'] = test_model_path
        qcnn = QCNN(state_shape, num_actions, _nn_hparams)
        expected = "<class 'keras.models.Sequential'>"
        self.assertTrue(expected, str(type(qcnn.nn)))
        # TODO: more a more detailed test
