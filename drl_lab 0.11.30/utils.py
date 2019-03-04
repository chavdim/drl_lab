import os
from datetime import datetime, timedelta, timezone
from pprint import pprint
import time

import matplotlib as mpl
mpl.use('SVG')  # NOQA
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _convert_s2hms(seconds):
    hours = int(seconds / (60 * 60))
    seconds = seconds - (60 * 60 * hours)
    minutes = int(seconds / 60)
    seconds = int(seconds - (60 * minutes))
    return hours, minutes, seconds


class Watcher:
    def __init__(self, name, max_steps):
        self.name = name
        self.max_steps = max_steps

        self.last = time.time()
        self.num_run = 0
        self.steps = 0
        self._best_score = -1

        print("[Name] {}".format(self.name))

    def start(self, num_run):
        self.num_run = num_run
        self.steps = 0
        self._best_score = -1
        self.last = time.time()

        print("[Run] {}".format(self.num_run))
        JST = timezone(timedelta(hours=+9), 'JST')
        print("[Start] {}.".format(str(datetime.now(JST))))

    def watch(self, steps, reward, epsilon):
        self.steps += steps
        remaining_steps = self.max_steps - self.steps

        now = time.time()
        time_per_step = (now - self.last) / steps
        remaining_time = remaining_steps * time_per_step
        hours, minutes, seconds = _convert_s2hms(remaining_time)
        self.last = now

        print("[Step] {} ({}/{}), {:.4f} sec/step".format(
            steps, self.steps, self.max_steps, time_per_step))
        print("[Remainig] {:02d}:{:02d}:{:02d}".format(
            hours, minutes, seconds))
        print("[Reward] {} (Best {})".format(reward, self._best_score))
        print("[Epsilon] {}".format(epsilon))

    def best_score(self, best_score):
        print("[New Best Score] {}".format(best_score))
        self._best_score = best_score

    def fin(self):
        print("[Name] {}".format(self.name))
        JST = timezone(timedelta(hours=+9), 'JST')
        print("[End] {}.".format(str(datetime.now(JST))))


def _convert_save_at(save_at, max_steps):
    if type(save_at) is int:
        # save_at as number of save times
        return [(max_steps // save_at)*i for i in range(save_at+1)]
    elif type(save_at) is list:
        # save_at as timing to save
        if type(save_at[0]) is int:
            return save_at
        elif type(save_at[0]) is float:
            # As ratio
            return [int(ratio*max_steps) for ratio in save_at]
        else:
            raise TypeError(
                "{} expected {} as list or int, but {} given".format(
                    '_convert_save_at()', 'save_at[0]', type(save_at)))
    else:
        raise TypeError(
            "{} expected {} as list or int, but {} given".format(
                '_convert_save_at()', 'save_at', type(save_at)))


class Saver:
    def __init__(self, name, max_steps, save_at):
        self.name = str(name)

        if save_at in [0, None, [], '']:
            save_at = 1
        self._save_at = _convert_save_at(save_at, max_steps)

    def init(self, save_reward=True, save_model=True, save_image=True):
        here = os.path.dirname(os.path.realpath(__file__))
        results_root = here+'/results/'+self.name
        results_dirs = [['results_root', results_root]]

        if save_reward:
            results_dirs.append(['reward_results', results_root+'/rewards'])
        if save_model:
            results_dirs.append(['model_results', results_root+'/models'])
        if save_image:
            results_dirs.append(['image_results', results_root+'/images'])

        for results_dir in results_dirs:
            name, path = results_dir
            if not os.path.exists(path):
                os.makedirs(path)
            setattr(self, name, path)

    def save_hparams(self, env_hparams, run_hparams,
                     nn_hparams, agent_hparams):
        with open(self.results_root+'/hparams.py', 'w') as f:
            f.write('env_hparams = ')
            pprint(env_hparams, stream=f)
            f.write('run_hparams = ')
            pprint(run_hparams, stream=f)
            f.write('nn_hparams = ')
            pprint(nn_hparams, stream=f)
            f.write('agent_hparams = ')
            pprint(agent_hparams, stream=f)

    def _save_rewards(self, rewards, path):
        np.save(path, np.array(rewards))

    def save_steps_rewards(self, num_run, rewards, steps):
        array = np.array([steps, rewards])
        path = self.reward_results+'/steps_rewards_'+str(num_run)+'.npy'
        self._save_rewards(array, path)

    def save_plot_all_steps_rewards(self, num_runs):
        plt.plot()

        for i in range(1, num_runs+1):
            steps_rewards = np.load(
                self.reward_results+'/steps_rewards_'+str(i)+'.npy')
            steps, rewards = steps_rewards

            sum_steps = steps.copy()
            for i in range(len(sum_steps)):
                if i > 0:
                    sum_steps[i] += sum_steps[i-1]

            plt.plot(sum_steps, rewards, alpha=0.75)

        plt.savefig(self.reward_results+'/all_steps_rewards.png')

    def save_model(self, model, num_run=None, step_num=None):
        path = self.model_results+'/model'
        if num_run is not None:
            path += '_'+str(num_run)
        if step_num is not None:
            path += '_'+str(step_num)
        model.save(path, include_optimizer=True)

    def save_arrays_as_images(self, arrays, name, prefix='image'):
        path = self.image_results+'/'+name
        if not os.path.exists(path):
            os.makedirs(path)
        save_arrays_as_images(arrays, path, prefix)

    def save_at(self, steps):
        if len(self._save_at) < 1:
            return False
        if self._save_at[0] > steps:
            return False
        self._save_at = self._save_at[1:]
        return True


def deprocess(array, warning=True):
    if type(array) != np.ndarray:
        raise TypeError('deprocess: np.ndarray is required.')

    deprocessed_array = np.copy(array)
    deprocessed = '|'

    if deprocessed_array.min() < 0.0:
        deprocessed_array = deprocessed_array - deprocessed_array.min()
        deprocessed += '| -min |'
    if deprocessed_array.max() > 255:
        deprocessed_array = deprocessed_array / deprocessed_array.max()
        deprocessed += '| /max |'
    if deprocessed_array.max() <= 1.0:
        deprocessed_array = deprocessed_array * 255
        deprocessed += '| *255 |'
    if deprocessed_array.dtype != np.uint8:
        deprocessed_array = np.uint8(deprocessed_array)
        deprocessed += '| uint() ||'

    if warning and deprocessed is not '|':
        print("***** Waring *****: array deprocessed: "+deprocessed)

    return deprocessed_array


def bulk_deprocess(arrays, warning=True):
    deprocessed_arrays = []
    for array in arrays:
        deprocessed_arrays.append(deprocess(array, warning))
    return deprocessed_arrays


def array2image(array):
    return Image.fromarray(array)


def arrays2images(arrays):
    """
    Notes
    -----
    uint8 is recommended for array.
    """
    images = [array2image(array) for array in arrays]

    return images


def save_image(image, save_path):
    image.save(save_path)


def save_images(images, save_dir, prefix='image'):
    for i, image in enumerate(images):
        save_path = "{}/{}_{:04d}.png".format(save_dir, prefix, i)
        save_image(image, save_path)


def save_array_as_image(array, save_path):
    image = array2image(array)
    save_image(image, save_path)


def save_arrays_as_images(arrays, save_dir, prefix='image'):
    images = arrays2images(arrays)
    save_images(images, save_dir, prefix)


def save_gif(images, save_path):
    images[0].save(save_path, save_all=True, append_images=images[1:],
                   optimize=False, duration=50, loop=0)
