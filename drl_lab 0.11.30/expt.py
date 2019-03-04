from datetime import datetime, timedelta, timezone

from drl_lab.sim import Simulator
from drl_lab.utils import Watcher, Saver


class Experiment():
    def __init__(self, name):
        JST = timezone(timedelta(hours=+9), 'JST')
        self.name = name+'_'+datetime.now(JST).strftime('%Y%m%d%H%M%S')

    def init(self, env_hparams, run_hparams, nn_hparams, agent_hparams):
        self.num_runs = run_hparams['num_runs']
        self.max_steps = run_hparams['max_steps']

        self.env_hparams = env_hparams
        self.nn_hparams = nn_hparams
        self.agent_hparams = agent_hparams

        # Log setting
        verbose = run_hparams['verbose']
        self.watcher = Watcher(self.name, self.max_steps) #if verbose else None

        # Save settings
        save_at = run_hparams['save_at']
        self.saver = Saver(self.name, self.max_steps, save_at)
        self.saver.init()
        self.saver.save_hparams(env_hparams, run_hparams,
                                nn_hparams, agent_hparams)

    def run(self):
        for num_run in range(1, self.num_runs+1):
            simulator = Simulator(self.env_hparams, self.nn_hparams,
                                  self.agent_hparams, self.watcher,
                                  self.saver, self.max_steps)
            simulator.run(num_run)

        self.saver.save_plot_all_steps_rewards(self.num_runs)
