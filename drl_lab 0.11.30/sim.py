import numpy as np

from drl_lab.env import create_env
from drl_lab.memory import Memory
from drl_lab.agents import QNetworkAgent


class Simulator:
    def __init__(self, env_hparams, nn_hparams, agent_hparams,
                 watcher, saver, max_steps=100000):
        self.env = create_env(env_hparams)
        self.agent = QNetworkAgent(self.env, nn_hparams, agent_hparams)

        self.watcher = watcher
        self.saver = saver

        self.max_steps = max_steps  # Max steps in this simulation
        self.initial_steps = int(max_steps*0.001)  # Initial explorations
        # max_experience_size = int(max_steps*0.02)  # Memory size
        max_experience_size = int(max_steps*0.002)  # Memory size
        self.experience_data = Memory(self.env.obs_shape, max_experience_size)
        self.epsilon_discount = 1.0/(max_steps/50)

        self.steps = 0  # Total steps count

    def run(self, num_run):
        if self.watcher:
            self.watcher.start(num_run)

        # Save initial model
        self.saver.save_model(self.agent.q_network.model, num_run, 'init')

        # Initialize
        best_reward = -10000
        rewards_per_episode = []
        steps_per_episode = []

        episode_reward = 0
        episode_steps = 0

        self.env.seed()
        observation = self.env.reset()

        while True:
            prev_state = np.copy(observation)
            action = self.agent.get_next_action(observation)
            observation, reward, done, info = self.env.step(action)
            cliped_reward = np.clip(reward, -1, 1)

            self.experience_data.add(
                prev_state, action, observation, cliped_reward, done)

            if self.steps > self.initial_steps:
                self.agent.learn(self)
                if self.agent.epsilon >= self.agent.final_epsilon:
                    self.agent.epsilon -= self.epsilon_discount

            episode_reward += cliped_reward
            episode_steps += 1
            self.steps += 1

            # save en route
            if self.saver.save_at(self.steps):
                self.saver.save_model(self.agent.q_network.model,
                                      num_run,
                                      self.steps)

            if done:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    self.saver.save_model(
                        self.agent.q_network.model, num_run, 'best')
                    if self.watcher:
                        self.watcher.best_score(best_reward)

                if self.watcher:
                    epsilon = np.round(self.agent.epsilon, 2)
                    self.watcher.watch(episode_steps, episode_reward, epsilon)

                rewards_per_episode.append(episode_reward)
                steps_per_episode.append(episode_steps)

                episode_reward = 0
                episode_steps = 0

                self.env.seed()
                observation = self.env.reset()

            if self.steps >= self.max_steps:
                break

        self.saver.save_steps_rewards(
            num_run, rewards_per_episode, steps_per_episode)

        if self.watcher:
            self.watcher.start(num_run)
