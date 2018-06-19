from datetime import datetime, timedelta, timezone
import os
from pprint import pprint
import time

# import keras
import matplotlib as mpl
mpl.use('SVG')  # NOQA
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from drl_lab.sim import Simulator


class Experiment():
    def __init__(self, name):
        JST = timezone(timedelta(hours=+9), 'JST')
        self.name = name+'_'+datetime.now(JST).strftime('%Y%m%d%H%M%S')

    def run(self, env_hparams, run_hparams, nn_hparams):
        simulator = Simulator(env_hparams, nn_hparams)

        interval = run_hparams['interval']
        num_runs = run_hparams['num_runs']
        verbose = run_hparams['verbose']
        max_steps = run_hparams['max_steps']

        self.interval = interval
        self.num_runs = num_runs

        # save settings
        save_at = run_hparams['save_at']
        save = save_at is not None
        if save:
            if type(save_at) is int:
                save_at = [(max_steps // save_at)*i for i in range(save_at+1)]
            elif type(save_at) is list:
                save_at = [int(ratio*max_steps) for ratio in save_at]
            self.init_save()
            self.save_hparams(env_hparams, run_hparams, nn_hparams)
            self.save_current_model(simulator.agent.nn.nn, step_num='init')

        for num_run in range(1, num_runs+1):
            self._run(simulator, interval, max_steps,
                      num_run, save, save_at, verbose)

        # fin
        if save:
            self.plot_results()

    def _run(self, simulator, interval, max_steps,
             num_runs, save, save_at, verbose):
        average_rewards = []
        best_score = -100000

        r = simulator.run(iterations=interval, update=False)
        r = np.mean(r)  # random agent baseline ( random dqn weights )
        average_rewards.append(r)

        while True:
            steps = simulator.agent.step_counter
            t = time.time()
            r = simulator.run(iterations=interval, update=True)
            steps = steps - simulator.agent.step_counter  # steps done
            step_per_sec = steps / (t - time.time())
            mr = np.mean(r)

            if verbose:
                remaining_min = max_steps - simulator.agent.step_counter
                remaining_min /= step_per_sec
                remaining_min /= 60
                remaining_min = np.round((remaining_min, 2))
                print('steps: ', simulator.agent.step_counter,
                      'mean reward: ', mr,
                      'epsilon: ',
                      np.round(simulator.agent.explore_chance, 2),
                      'steps/sec: ', np.round(step_per_sec, 2),
                      'remaining min: ', remaining_min)

            average_rewards.append(mr)

            # save best
            if mr > best_score and save:
                if verbose:
                    print('new best score, saving model...')
                self.save_current_model(simulator.agent.nn.nn,
                                        num_runs, 'best')
                best_score = mr

            # save en route
            if save:
                if simulator.agent.step_counter in save_at:
                    self.save_current_model(simulator.agent.nn.nn, num_runs,
                                            simulator.agent.step_counter)

            if simulator.agent.step_counter >= max_steps:
                if save:
                    self.save_rewards(average_rewards, num_runs)
                break

    def init_save(self):
        here = os.path.dirname(os.path.realpath(__file__))
        results_root = here+'/results/'+self.name
        reward_results = results_root+'/rewards'
        model_results = results_root+'/models'
        # image_results = results_root+'/images'
        # for d in [results_root, reward_results, model_results, image_results]:
        for d in [results_root, reward_results, model_results]:
            if not os.path.exists(d):
                os.makedirs(d)
        self.results_root = results_root
        self.reward_results = reward_results
        self.model_results = model_results
        # self.image_results = image_results

    def save_hparams(self, env_hparams, run_hparams, nn_hparams):
        with open(self.results_root+'/hparams.py', 'w') as f:
            f.write('env_hparams = ')
            pprint(env_hparams, stream=f)
            f.write('run_hparams = ')
            pprint(run_hparams, stream=f)
            f.write('nn_hparams = ')
            pprint(nn_hparams, stream=f)

    def save_rewards(self, average_rewards, num_run):
        path = self.reward_results+'/rewards_'+str(num_run)
        np.save(path, np.array(average_rewards))

    def save_current_model(self, model, num_run=None, step_num=None):
        path = self.model_results+'/model'
        if num_run is not None:
            path += '_'+str(num_run)
        if step_num is not None:
            path += '_'+str(step_num)
        model.save(path, include_optimizer=True)

    def save_array_as_images(self, array, name,
                             prefix='image', save_gif=True):
        save_path = "{}/{}".format(self.image_results, name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        images = arrays2images(array)
        save_images(save_path, images, prefix, save_gif)

    def plot_results(self):
        for i in range(1, self.num_runs+1):
            res = np.load(self.reward_results+'/rewards_'+str(i)+'.npy')
            if i == 1:
                average_res = np.zeros((len(res), self.num_runs))
            average_res[0:, i-1] = res[0:average_res.shape[0]]
            plt.plot(range(0, average_res.shape[0]*self.interval,
                           self.interval),
                     res[0:average_res.shape[0]], alpha=0.225)
        plt.plot(range(0, average_res.shape[0]*self.interval,
                       self.interval),
                 np.mean(average_res, axis=1))
        plt.savefig(self.results_root+'/results.png')


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


def save_images(save_dir, images, prefix='image'):
    for i, image in enumerate(images):
        save_name = "{}/{}_{:04d}.png".format(save_dir, prefix, i)
        image.save(save_name)


def save_gif(save_path, images):
    images[0].save(save_path, save_all=True, append_images=images[1:],
                   optimize=False, duration=50, loop=0)
