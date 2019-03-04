from copy import deepcopy
import os
import re
import shutil
import unittest

from drl_lab.expt import Experiment
from tests.common import (
    env_hparams,
    nn_hparams,
    run_hparams,
    get_results_root,
)


class TestExperiment(unittest.TestCase):
    def setUp(self):
        run_hparams['interval'] = 10
        run_hparams['max_steps'] = 50
        run_hparams['save_at'] = None
        self.expt = Experiment('test')
        self.results_dir_test = get_results_root()

    def test___init__(self):
        expt = self.expt
        result = re.match('test_[0-9]{14}', expt.name)
        self.assertIsNotNone(result)

    def test__convert_save_at(self):
        expt = self.expt
        max_steps = 100
        save_at = 1
        expected = [0, 100]
        save_at = expt._convert_save_at(save_at, max_steps)
        self.assertEqual(expected, save_at)
        save_at = 2
        expected = [0, 50, 100]
        save_at = expt._convert_save_at(save_at, max_steps)
        self.assertEqual(expected, save_at)
        save_at = 5
        expected = [0, 20, 40, 60, 80, 100]
        save_at = expt._convert_save_at(save_at, max_steps)
        self.assertEqual(expected, save_at)
        save_at = [50]
        expected = [50]
        save_at = expt._convert_save_at(save_at, max_steps)
        self.assertEqual(expected, save_at)
        save_at = [0, 100]
        expected = [0, 100]
        save_at = expt._convert_save_at(save_at, max_steps)
        self.assertEqual(expected, save_at)
        save_at = [0.0]
        expected = [0]
        save_at = expt._convert_save_at(save_at, max_steps)
        self.assertEqual(expected, save_at)
        save_at = [0.5]
        expected = [50]
        save_at = expt._convert_save_at(save_at, max_steps)
        self.assertEqual(expected, save_at)
        save_at = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        expected = [0, 20, 40, 60, 80, 100]
        save_at = expt._convert_save_at(save_at, max_steps)
        self.assertEqual(expected, save_at)

    def test_run(self):
        expt = self.expt
        results_dir_test = self.results_dir_test
        results_dir_expt = (results_dir_test +
                            '/../../../drl_lab/results/' + expt.name)

        expt.run(env_hparams, run_hparams, nn_hparams)

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
            os.path.exists(results_dir_expt+'/images'))
        self.assertTrue(
            os.path.exists(results_dir_expt+'/hparams.py'))
        self.assertTrue(
            os.path.exists(results_dir_expt+'/models/model_init'))
        self.assertTrue(
            os.path.exists(results_dir_expt+'/rewards/rewards_1.npy'))
        self.assertTrue(
            os.path.exists(results_dir_expt+'/rewards/all_n_average.png'))

        if os.path.exists(results_dir_expt):
            shutil.move(results_dir_expt, results_dir_test+'/test_expt_run')

    def test__run(self):
        pass
