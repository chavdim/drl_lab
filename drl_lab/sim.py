import time

import numpy as np

from drl_lab.env import create_env
from drl_lab.memory import Memory
from drl_lab.agents import QnetworkAgent


class Simulator:
    def __init__(self, env_hparams, nn_hparams,
                 max_iterations=100000, interval=20):
        self.env = create_env(env_hparams)
        self.agent = QnetworkAgent(self.env, nn_hparams)

        action_size = 1  # actions take during one step
        reward_size = 1  # reward vector size
        max_experience_size = 50000  # Memory size

        self.experience_data = Memory(self.env.obs_shape,
                                      action_size,
                                      reward_size,
                                      max_experience_size
                                      )

        # actions repeated skip_frames -1 times
        self.skip_frames = 1
        self.skip_frame_timer = self.skip_frames
        self.episode_max_length = 10000

        self.max_iterations = max_iterations
        self.interval = interval
        self.rewards = []
        self.temp_rews = 0
        self.done = False
        self.last_action = None

        self.times = {"total": 0.0, "get_action": 0.0,
                      "train": 0.0, "create_batch": 0.0}

    def run(self, test_agent=False, update=False, iterations=1000):
        self.env.seed()
        observation = self.env.reset()

        # episode mode
        if iterations < 0:
            iterations = self.episode_max_length
            episode_reward = 0
        episode_mode = 'episode_reward' in locals()

        all_rewards = []

        for t in range(iterations):
            if self.skip_frame_timer == self.skip_frames:
                if not test_agent:
                    if update:
                        self.agent.update(self)
                    t_before_action = time.time()
                    action = self.agent.get_next_action(observation)
                    self.times["get_action"] += time.time() - t_before_action
                else:
                    # self.env.render()
                    action = self.agent.get_best_action(observation)
                self.skip_frame_timer = 0
            self.skip_frame_timer += 1
            self.last_action = action
            prev_state = np.copy(observation)
            observation, reward, done, info = self.env.step(action)
            all_rewards.append(reward)

            if episode_mode:
                episode_reward += reward
                self.temp_rews += reward
                if self.agent.step_counter % self.interval == 0:
                    self.rewards.append(np.mean(self.temp_rews))
                    self.temp_rews = 0
                if self.agent.step_counter > self.max_iterations:
                    self.done = True

            r = np.clip(reward, -1, 1)
            if not test_agent:
                if self.skip_frame_timer == 1:
                    self.experience_data.add_data(prev_state, action,
                                                  observation, r, done)

            if t == self.episode_max_length:  # never
                print("episode max length reached")
                if not episode_mode:
                    done = True

            if done:
                self.skip_frame_timer = self.skip_frames
                if episode_mode:
                    break
                else:
                    self.env.seed()
                    observation = self.env.reset()

        if not test_agent:
            if self.agent.explore_chance > self.agent.exploration_final_eps:
                if update:
                    rate = 0.9 if episode_mode else 0.8
                    self.agent.explore_chance *= rate

        if episode_mode:
            return episode_reward
        else:
            return all_rewards

    def run_repeatedly(self, num_runs=10, iterations=1000, update=True):
        results = []
        for i in range(num_runs):
            results.append(
                self.run(
                    test_agent=False, update=update, iterations=iterations))
        return results
