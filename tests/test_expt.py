import re
import shutil
import unittest

from drl_lab.env import create_env
from drl_lab.expt import (
    Experiment,
    array2images,
    save_images,
)
from tests.common import (
    deepcopy,
    env_hparams,
    get_results_dir,
    nn_hparams,
    os,
    state,
    states,
    run_hparams,
)


class TestExperiment(unittest.TestCase):
    def setUp(self):
        run_hparams['interval'] = 10
        run_hparams['max_steps'] = 50
        self.expt = Experiment('test')
        self.results_dir_test = get_results_dir()

    def test_init(self):
        expt = self.expt
        result = re.match('test_[0-9]{14}', expt.name)
        self.assertIsNotNone(result)

    def test_run(self):
        expt = self.expt
        results_dir_test = self.results_dir_test
        results_dir_expt = (results_dir_test +
                            '/../../../drl_lab/results/' + expt.name)

        expt.run(env_hparams, run_hparams, nn_hparams)
        expected = run_hparams['interval']
        self.assertEqual(expected, expt.interval)
        expected = run_hparams['num_runs']
        self.assertEqual(expected, expt.num_runs)

        save_at = run_hparams['save_at']
        save = save_at is not None
        self.assertFalse(save)
        save_at = 10
        save = save_at is not None
        self.assertTrue(save)
        max_steps = run_hparams['max_steps']
        _save_at = [(max_steps // save_at)*i for i in range(save_at+1)]
        expected = int
        self.assertEqual(expected, type(_save_at[0]))
        expected = save_at+1
        self.assertEqual(expected, len(_save_at))
        save_at = [0.0, 1.0]
        save = save_at is not None
        self.assertTrue(save)
        max_steps = run_hparams['max_steps']
        _save_at = [int(ratio*max_steps) for ratio in save_at]
        expected = int
        self.assertEqual(expected, type(_save_at[0]))
        expected = len(save_at)
        self.assertEqual(expected, len(_save_at))

        # Test save mode
        _run_hparams = deepcopy(run_hparams)
        _run_hparams['save_at'] = [0.0, 1.0]
        expt.run(env_hparams, _run_hparams, nn_hparams)
        self.assertTrue(os.path.exists(results_dir_expt))
        self.assertTrue(
            os.path.exists(results_dir_expt+'/rewards'))
        self.assertTrue(
            os.path.exists(results_dir_expt+'/models'))
        self.assertTrue(
            os.path.exists(results_dir_expt+'/hparams.py'))
        self.assertTrue(
            os.path.exists(results_dir_expt+'/models/model_init'))
        self.assertTrue(
            os.path.exists(results_dir_expt+'/rewards/rewards_1.npy'))
        self.assertTrue(
            os.path.exists(results_dir_expt+'/results.png'))

        if os.path.exists(results_dir_expt):
            shutil.move(results_dir_expt, results_dir_test+'/test_expt_run')

    def test__run(self):
        # TODO: implement this
        pass

    def test_init_save(self):
        # combined to test_run
        pass

    def test_save_hparams(self):
        # combined to test_run
        pass

    def test_save_rewards(self):
        # combined to test_run
        pass

    def test_save_current_model(self):
        # combined to test_run
        pass

    # TODO: this
    def test_save_array_as_image(self):
        pass

    def test_plot_results(self):
        # combined to test_run
        pass


class TestExpt(unittest.TestCase):
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
