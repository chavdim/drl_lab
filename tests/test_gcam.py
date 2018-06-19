import unittest

import tensorflow as tf

from drl_lab.env import create_env
from drl_lab.gcam import (
    colorize_cam,
    K,
    grad_cam,
    normalize,
    target_category_loss,
    target_category_loss_output_shape,
)
from drl_lab.models import load_model
from tests.common import (
    array2images,
    env_hparams,
    get_resources_dir,
    get_results_dir,
    num_actions,
    np,
    os,
    state,
    state_shape,
    save_images,
)


class TestGCAM(unittest.TestCase):
    def test_grad_cam(self):
        resources_dir = get_resources_dir()
        test_model = resources_dir+'/test_model'
        if not os.path.exists(test_model):
            raise FileNotFoundError(test_model)
        model = load_model(test_model)
        cams = []
        for i in range(2):
            cam = grad_cam(model, state.reshape(1, *state.shape), i, 4, 2)
            cam = colorize_cam(cam, state)
            cams.append(cam)
        images = array2images(cams)
        results_root = get_results_dir()
        results_dir = "{}/{}".format(results_root, 'test_grad_cam')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        save_images(results_dir, images)

        cams = []
        env = create_env(env_hparams)
        env.reset()
        for i in range(10):
            observation, _, _, _ = env.step(1)
            _observation = observation.reshape(1, *observation.shape)
            cam0 = grad_cam(model, _observation, 0, 4, 2)
            cam0 = colorize_cam(cam0, observation)
            cam1 = grad_cam(model, _observation, 1, 4, 2)
            cam1 = colorize_cam(cam1, observation)
            cams.append(np.concatenate([cam0, cam1], axis=1))
        images = array2images(cams)
        results_dir = "{}/{}".format(results_root, 'test_grad_cam2')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        save_images(results_dir, images)

    def test_normalize(self):
        normalized_state = K.eval(normalize(K.variable(state)))
        expected = state.shape
        self.assertEqual(expected, normalized_state.shape)
        np_normalized_state = state / (np.mean(np.square(state)) + 1e-5)
        self.assertTrue(
            normalized_state.min() - np_normalized_state.min() < 0.1)
        self.assertTrue(
            normalized_state.max() - np_normalized_state.max() < 0.1)
        self.assertTrue(
            normalized_state.mean() - np_normalized_state.mean() < 0.1)

    def test_target_category_loss(self):
        mock_output_tensor = K.variable(np.random.randn(32, 1))
        loss = target_category_loss(mock_output_tensor, 0, num_actions)
        expected = (32, 4)
        self.assertEqual(expected, loss.get_shape())
        expected = np.abs(K.eval(loss)).max(axis=1)
        self.assertTrue(
            np.array_equal(expected, np.abs(np.sum(K.eval(loss), axis=1))))
        expected = tf.multiply(
            mock_output_tensor, K.one_hot([0], num_actions))
        self.assertTrue(
            np.array_equal(K.eval(expected), K.eval(loss)))

    def test_target_category_loss_output_shape(self):
        expected = state_shape
        self.assertEqual(
            expected, target_category_loss_output_shape(state_shape))

    # TODO: this
    def test_colorize_cam(self):
        pass
