import os
import shutil
import unittest

import numpy as np

from drl_lab.utils import (  # noqa
    Saver,
    deprocess,
    bulk_deprocess,
    array2image,
    arrays2images,
    save_image,
    save_images,
    save_array_as_image,
    save_arrays_as_images,
    save_gif,
)
from tests.common import (
    env_hparams,
    run_hparams,
    nn_hparams,
    get_results_dir,
    get_test_model,
)


class TestSaver(unittest.TestCase):
    def _tearDown(self, saver):
        results_dir = get_results_dir('test_utils/TestSaver')
        shutil.move(saver.results_root, results_dir+'/'+saver.name)

    def test___init__(self):
        saver = Saver()
        expected = 'saver_'+str(os.getpid())
        self.assertEqual(expected, saver.name)
        saver = Saver('test')
        expected = 'test'
        self.assertEqual(expected, saver.name)

    def test_init(self):
        saver = Saver('test_init')
        saver.init()
        self.assertTrue(os.path.exists(saver.results_root))
        self.assertTrue(os.path.exists(saver.reward_results))
        self.assertTrue(os.path.exists(saver.model_results))
        self.assertTrue(os.path.exists(saver.image_results))
        self._tearDown(saver)

    def test_save_hparams(self):
        saver = Saver('test_save_hparams')
        saver.init(False, False, False)
        saver.save_hparams(env_hparams, run_hparams, nn_hparams)
        self.assertTrue(os.path.exists(saver.results_root+'/hparams.py'))
        self._tearDown(saver)

    def test_save_run_rewards(self):
        saver = Saver('test_save_run_rewards')
        saver.init(True, False, False)
        rewards = np.random.randn(10)
        saver.save_run_rewards(rewards, 1)
        self.assertTrue(os.path.exists(saver.reward_results+'/rewards_1.npy'))
        self._tearDown(saver)

    def test_save_plot_all_n_average_rewards(self):
        saver = Saver('test_save_plot_all_n_average_rewards')
        saver.init(True, False, False)
        for i in range(1, 4):
            rewards = np.random.randn(10+(i*3))
            saver.save_run_rewards(rewards, i)
        saver.save_plot_all_n_average_rewards(3, 5)
        self.assertTrue(
            os.path.exists(saver.reward_results+'/all_n_average.png'))
        self._tearDown(saver)

    def test_save_model(self):
        saver = Saver('test_save_model')
        saver.init(False, True, False)
        model = get_test_model()
        saver.save_model(model, 1, 100)
        self.assertTrue(os.path.exists(saver.model_results+'/model_1_100'))
        self._tearDown(saver)

    def test_save_arrays_as_images(self):
        saver = Saver('test_save_arrays_as_images')
        saver.init(False, False, True)
        arrays = np.random.randint(0, 255, [10, 48, 48, 3], np.uint8)
        saver.save_arrays_as_images(arrays, 'test_arrays')
        self.assertTrue(os.path.exists(saver.image_results+'/test_arrays'))
        self._tearDown(saver)


class TestUtils(unittest.TestCase):
    def test_test_deprocess(self):
        pass

    def test_bulk_deprocess(self):
        pass

    def test_array2image(self):
        pass

    def test_arrays2images(self):
        pass

    def test_save_image(self):
        # results_dir = get_results_dir('test_utils/TestUtils')
        pass

    def test_save_images(self):
        # results_dir = get_results_dir('test_utils/TestUtils')
        pass

    def test_save_array_as_image(self):
        # results_dir = get_results_dir('test_utils/TestUtils')
        pass

    def test_save_arrays_as_images(self):
        # results_dir = get_results_dir('test_utils/TestUtils')
        pass

    def test_save_gif(self):
        # results_dir = get_results_dir('test_utils/TestUtils')
        pass


"""
    def test_array2images(self):
        image = array2images(state)
        expected = list
        self.assertEqual(expected, type(image))
        expected = 1
        self.assertEqual(expected, len(image))
        expected = "<class 'PIL.Image.Image'>"
        self.assertEqual(expected, str(type(image[0])))
        images = array2images(states)
        expected = len(states)
        self.assertEqual(expected, len(images))

    def test_save_images(self):
        results_root = get_results_dir()
        results_dir = "{}/{}".format(results_root, 'test_save_images')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        observations = []
        env = create_env(env_hparams)
        env.reset()
        for i in range(10):
            observation, _, _, _ = env.step(1)
            observations.append(observation)
        images = array2images(observations)
        save_images(results_dir, images)
"""
