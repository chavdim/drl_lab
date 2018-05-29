import unittest

import gym
import gym.spaces  # NOQA
import gym_ple  # NOQA

from drl_lab.env import (
    Action,
    create_env,
    # draw_shadow,
    np,
    PreprocessedEnv,
    rgb2gray,
)
from tests.common import (
    array2images,
    deepcopy,
    env_hparams,
    get_results_dir,
)


class TestEnv(unittest.TestCase):
    def test_create_env(self):
        env = create_env(env_hparams)
        # check initial and default attributes
        self.assertEqual(env.original_dim, env.obs_shape)
        self.assertFalse(env.normalize)
        self.assertFalse(env.opt_flow)
        self.assertFalse(env.rescale)
        expected = 2
        self.assertEqual(expected, len(env.actions))
        expected = Action
        self.assertEqual(expected, type(env.actions[0]))
        self.assertIsNone(env.last_obs)
        self.assertIsNone(env.last_obs_raw)
        self.assertIsNone(env.last_action)
        self.assertIsNone(env.last_action_raw)

    def test_rgb2gray(self):
        array_rgb = np.random.randint(0, 255, [32, 32, 3])
        array_gray = rgb2gray(array_rgb)
        result_dir = get_results_dir()
        image_rgb = array2images(array_rgb)[0]
        image_rgb.save(result_dir+'/test_rgb2gray_rgb.png')
        image_gray = array2images(array_gray)[0]
        image_gray.save(result_dir+'/test_rgb2gray_gray.png')

    def test_draw_shadow(self):
        pass


class TestAction(unittest.TestCase):
    def test_init(self):
        action = Action(1, [0, 1], True)
        expected = 1
        self.assertEqual(expected, action.name)
        expected = [0, 1]
        self.assertEqual(expected, action.range)
        self.assertTrue(action.discrete)


class TestPreprocessedEnv(unittest.TestCase):
    def setUp(self):
        env = gym.make(env_hparams['env_id'])
        self.env = PreprocessedEnv(env, **env_hparams)
        self.result_dir = get_results_dir()

    def test_init(self):
        env = self.env
        # check initial and default attributes
        self.assertEqual(env.original_dim, env.obs_shape)
        self.assertFalse(env.normalize)
        self.assertFalse(env.opt_flow)
        self.assertFalse(env.rescale)
        expected = 2
        self.assertEqual(expected, len(env.actions))
        expected = Action
        self.assertEqual(expected, type(env.actions[0]))
        self.assertIsNone(env.last_obs)
        self.assertIsNone(env.last_obs_raw)
        self.assertIsNone(env.last_action)
        self.assertIsNone(env.last_action_raw)

        # enable rescale
        _env_hparams = deepcopy(env_hparams)
        _env_hparams['observation']['rescaled_shape'] = [60, 60, 3]
        env = gym.make(_env_hparams['env_id'])
        env = PreprocessedEnv(env, **_env_hparams)
        self.assertTrue(env.rescale)
        expected = [60, 60, 3]
        self.assertEqual(expected, env.rescaled_shape)
        expected = 2
        self.assertEqual(expected, len(env.zoom))
        expected = [60, 60, 3]
        self.assertEqual(expected, env.obs_shape)

        # add excluded action
        _env_hparams = deepcopy(env_hparams)
        _env_hparams['action']['excluded_actions'] = [0]
        env = gym.make(_env_hparams['env_id'])
        env = PreprocessedEnv(env, **_env_hparams)
        unexpected = env.env.action_space.n
        self.assertNotEqual(unexpected, len(env.actions))

    def test_reset(self):
        env = self.env
        result_dir = self.result_dir

        # check default settings
        observation = env.reset()
        # save image
        observation_image = array2images(observation)[0]
        observation_image.save(result_dir+'/test_reset_obs.png')
        self.assertTrue(np.array_equal(env.last_obs, env.last_obs_raw))
        expected = (48, 48, 3)
        self.assertEqual(expected, observation.shape)
        expected = 255
        self.assertTrue(expected >= observation.max())
        expected = 0
        self.assertTrue(expected <= observation.min())

        # enable normalization
        observation = env.reset(normalize=True)
        # save image
        observation_image = array2images(observation)[0]
        observation_image.save(result_dir+'/test_reset_normalize.png')
        self.assertTrue(env.normalize)
        expected = 1.0
        self.assertTrue(expected >= observation.max())
        expected = 0.0
        self.assertTrue(expected <= observation.min())

        # enable opt_flow
        observation = env.reset(opt_flow=True)
        # save image
        observation_image = array2images(observation)[0]
        observation_image.save(result_dir+'/test_reset_opt_flow.png')
        self.assertTrue(env.opt_flow)

    def test_step(self):
        env = self.env
        result_dir = self.result_dir

        # check default settings
        observation = env.reset()
        observation, reward, done, info = env.step(0)
        # save image
        observation_image = array2images(observation)[0]
        observation_image.save(result_dir+'/test_step_obs.png')
        expected = np.ndarray
        self.assertEqual(expected, type(observation))
        expected = (48, 48, 3)
        self.assertEqual(expected, observation.shape)
        expected = 255
        self.assertTrue(expected >= observation.max())
        expected = 0
        self.assertTrue(expected <= observation.min())
        expected = float
        self.assertEqual(expected, type(reward))
        self.assertTrue(-1.0 <= reward and reward <= 1.0)
        expected = bool
        self.assertEqual(expected, type(done))
        self.assertFalse(done)
        expected = dict
        self.assertEqual(expected, type(info))

        # enable normalization
        observation = env.reset(normalize=True)
        observation, reward, done, info = env.step(0)
        # save image
        observation_image = array2images(observation)[0]
        observation_image.save(result_dir+'/test_step_normalize.png')
        expected = np.ndarray
        self.assertEqual(expected, type(observation))
        expected = (48, 48, 3)
        self.assertEqual(expected, observation.shape)
        expected = 1.0
        self.assertTrue(expected >= observation.max())
        expected = 0.0
        self.assertTrue(expected <= observation.min())

        # enable opt_flow
        observation = env.reset(opt_flow=True)
        observation, reward, done, info = env.step(0)
        # save image
        observation_image = array2images(observation)[0]
        observation_image.save(result_dir+'/test_step_opt_flow.png')
        expected = np.ndarray
        self.assertEqual(expected, type(observation))
        expected = (48, 48, 3)
        self.assertEqual(expected, observation.shape)
        expected = 1.0
        self.assertTrue(expected >= observation.max())
        expected = 0.0
        self.assertTrue(expected <= observation.min())

    def test_rescale(self):
        result_dir = self.result_dir

        # enable rescale
        _env_hparams = deepcopy(env_hparams)
        _env_hparams['observation']['rescaled_shape'] = [60, 60, 3]
        env = gym.make(_env_hparams['env_id'])
        env = PreprocessedEnv(env, **_env_hparams)
        observation = env.reset()
        observation, reward, done, info = env.step(0)
        expected = (60, 60, 3)
        self.assertEqual(expected, observation.shape)
        # save image
        observation_img = array2images(observation)[0]
        observation_img.save(result_dir+'/test_rescale.png')

    def test_action(self):
        env = self.env

        # check default settings
        action = env.action(0)
        expected = int
        self.assertEqual(expected, type(action))
        expected = 0
        self.assertEqual(expected, action)
        expected = 0
        self.assertEqual(expected, env.last_action_raw)
        expected = 0
        self.assertEqual(expected, env.last_action)

        # add exclude action
        _env_hparams = deepcopy(env_hparams)
        _env_hparams['action']['excluded_actions'] = [0]
        env = gym.make(_env_hparams['env_id'])
        env = PreprocessedEnv(env, **_env_hparams)
        unexpected = env.env.action_space.n
        self.assertNotEqual(unexpected, len(env.actions))
        action = env.action(0)
        expected = 1
        self.assertEqual(expected, action)
        expected = 0
        self.assertEqual(expected, env.last_action_raw)
        expected = 1
        self.assertEqual(expected, env.last_action)

    def test_observation(self):
        env = self.env
        result_dir = self.result_dir

        # check default settings
        observation = env.env.reset()
        observation = env.observation(observation)
        # save image
        observation_image = array2images(observation)[0]
        observation_image.save(result_dir+'/test_observation.png')
        expected = np.ndarray
        self.assertEqual(expected, type(observation))
        expected = 'uint8'
        self.assertEqual(expected, str(observation.dtype))
        observation[0, 0, 0] = 300.0
        self.assertFalse(np.array_equal(observation, env.last_obs_raw))
        self.assertFalse(np.array_equal(observation, env.last_obs))

        # enable rescale
        _env_hparams = deepcopy(env_hparams)
        _env_hparams['observation']['rescaled_shape'] = [60, 60, 3]
        env = gym.make(_env_hparams['env_id'])
        env = PreprocessedEnv(env, **_env_hparams)
        observation = env.env.reset()
        observation = env.observation(observation)
        # save image
        observation_image = array2images(observation)[0]
        observation_image.save(result_dir+'/test_observation_rescale.png')
        expected = (60, 60, 3)
        self.assertEqual(expected, observation.shape)
        env.rescale = False

        # enable normalize
        env.normalize = True
        observation = env.observation(observation)
        # save image
        observation_image = array2images(observation)[0]
        observation_image.save(result_dir+'/test_observation_normalize.png')
        expected = 1.0
        self.assertTrue(expected >= observation.max())
        expected = 0.0
        self.assertTrue(expected <= observation.min())
        env.normalize = False

        # enable opt_flow
        env.opt_flow = True
        observation = env.observation(observation)
        # save image
        observation_image = array2images(observation)[0]
        observation_image.save(result_dir+'/test_observation_opt_flow.png')
        expected = (60, 60, 3)
        self.assertEqual(expected, observation.shape)
        env.opt_flow = False
