import cv2
import gym
import gym_ple  # NOQA
import numpy as np
from skimage.transform import rescale as sk_rescale


def create_env(env_hparams):
    env = gym.make(env_hparams['env_id'])
    env = PreprocessedEnv(env, **env_hparams)
    return env


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def draw_shadow(self, image, of, thresh=0.6):
    shadows = np.ones_like(image)/2
    of_y = of[0:, 0:, 1]
    shadows[of_y > thresh] = 1.0
    shadows[of_y <= thresh] = 0
    return shadows


class Action:
    def __init__(self, name, _range, discrete):
        self.name = name
        self.range = _range
        self.discrete = discrete


class PreprocessedEnv(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super(PreprocessedEnv, self).__init__(env)

        # observation settings
        self.original_dim = list(self.env.observation_space.shape)
        self.obs_shape = self.original_dim

        self.normalize = False
        self.opt_flow = False

        # observation rescale settings
        rescaled_shape = kwargs['observation']['rescaled_shape']
        self.rescale = rescaled_shape is not None and len(rescaled_shape) > 0
        if self.rescale:
            zoom = [1, 1]
            for i, dim in enumerate(self.original_dim[0:-1]):
                zoom[i] = rescaled_shape[i]/dim
            zoom[0], zoom[1] = zoom[1], zoom[0]
            self.rescaled_shape = rescaled_shape
            self.zoom = zoom
            self.obs_shape = self.rescaled_shape

        # action settings
        self.actions = []
        excluded_actions = kwargs['action']['excluded_actions']
        if excluded_actions is None:
            excluded_actions = []
        for action in range(self.env.action_space.n):
            if action not in excluded_actions:
                self.actions.append(Action(action, [0, 1], True))

        # pre-activation
        self._reset(**kwargs['observation'])

    def _reset(self, **kwargs):
        for kw in ['normalize', 'opt_flow', 'rescale']:
            if kw in kwargs:
                setattr(self, kw, kwargs[kw])
                del kwargs[kw]

        self.last_obs = None
        self.last_obs_raw = None

        self.last_action = None
        self.last_action_raw = None

        return kwargs

    def reset(self, **kwargs):
        kwargs = self._reset(**kwargs)
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        action = self.action(action)
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def action(self, action):
        self.last_action_raw = action
        self.last_action = self.actions[action].name
        return self.actions[action].name

    def observation(self, observation):
        self.last_obs_raw = np.copy(observation)
        if self.rescale:
            observation = sk_rescale(
                observation, self.zoom, preserve_range=True)
        if self.opt_flow:
            observation = self.optical_flow(observation)
        if self.normalize:
            # TODO: /255 to (/127.5)-1
            observation = observation/255.0
        self.last_obs = np.copy(observation)
        return observation

    def optical_flow(self, observation):
        gray = rgb2gray(observation)
        if self.last_obs is not None:
            of = cv2.calcOpticalFlowFarneback(
                    self.last_obs[0:, 0:, 0]*255,
                    gray*255, None, 0.5, 3, 5, 3, 5, 1.2, 0)
            observation[0:, 0:, 0] = gray
            observation[0:, 0:, 1] = of[0:, 0:, 0]/10
            observation[0:, 0:, 2] = of[0:, 0:, 1]/10
        else:
            observation[0:, 0:, 0] = gray
            observation[0:, 0:, 1] = observation[0:, 0:, 1]*0
            observation[0:, 0:, 2] = observation[0:, 0:, 2]*0
        return observation
