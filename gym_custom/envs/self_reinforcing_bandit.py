import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class POMDPBandit(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
            self, 
            num_arms=2,
            action_buffer_len=10,
            reward_weights=None,
            episode_len=np.inf):
        """
        args:
            num_arms: number of bandit arms
            action_buffer_len: size of state that keeps track of 
                past actions
            reward_weights: multiply rewards by these weights.
            episode_len: if passed, cap episodes at the given
                length. Otherwise infinite time horizon.
        """
        super().__init__()
        self.action_buffer_len = action_buffer_len
        self.action_space = spaces.Discrete(num_arms)
        self.observation_space = spaces.Discrete(1)
        self.dummy_ob = 0  # can add 1 to DQN input if I want
        self.episode_len = episode_len
        if reward_weights is None:
            self.reward_weights = np.ones(num_arms)
        else:
            assert len(reward_weights) == num_arms
            self.reward_weights = reward_weights

        self.reset()

    def _reward(self, action: int):
        freq_a = np.bincount(self.prev_actions).argmax()
        rew = self.reward_weights[action]
        return rew if action == freq_a else 0 

    def step(self, action: int):
        """
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (integer) :
                Timestep
            reward (float) :
                Equal to number of times the lever `action` has been pushed
                so far.
            done (bool) :
                After self.max_steps steps (default 40).
            info (dict) :
                includes environment state (lever pull counts).
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid.")

        reward = self._reward(action)
        self.current_step += 1
        self.prev_actions.append(action)
        if len(self.prev_actions) > self.action_buffer_len:
            del self.prev_actions[0]

        ob = self.dummy_ob
        done = self.current_step >= self.episode_len
        info = {"pull_counts": self.prev_actions.copy()}
        return ob, reward, done, info

    def reset(self):
        self.prev_actions = [np.random.choice(self.action_space.n)]
        self.current_step = 0
        ob = self.dummy_ob
        return ob

    def render(self, mode='human'):
        # google histogram in bash
        print("Lever pull counts:")
        print("index:", *range(self.action_space.n))
        print("count:", *self.pull_counts)

    def close(self):
        pass



class POMDPBanditEpisodic(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
            self, 
            num_arms=2,
            action_buffer_len=10,
            reward_weights=None):
        """
        args:
            num_arms: number of bandit arms
            action_buffer_len: size of state that keeps track of 
                past actions
            reward_weights: multiply rewards by these weights.
        """
        super().__init__()
        self.action_buffer_len = action_buffer_len
        self.action_space = spaces.Discrete(num_arms)
        self.observation_space = spaces.Discrete(1)
        self.dummy_ob = 0  # can add 1 to DQN input if I want
        if reward_weights is None:
            self.reward_weights = np.ones(num_arms)
        else:
            assert len(reward_weights) == num_arms
            self.reward_weights = reward_weights
        self.reset()

    def _reward(self, action: int):
        freq_a = np.bincount(self.prev_actions).argmax()
        rew = self.reward_weights[action]
        return rew if action == freq_a else 0 

    def step(self, action: int):
        """
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (integer) :
                Timestep
            reward (float) :
                Equal to number of times the lever `action` has been pushed
                so far.
            done (bool) :
                After self.max_steps steps (default 40).
            info (dict) :
                includes environment state (lever pull counts).
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid.")

        reward = self._reward(action)
        self.current_step += 1
        self.prev_actions.append(action)
        if len(self.prev_actions) > self.action_buffer_len:
            del self.prev_actions[0]

        ob = self.dummy_ob
        done = False
        info = {"pull_counts": self.prev_actions.copy()}
        return ob, reward, done, info

    def reset(self):
        self.prev_actions = [np.random.choice(self.action_space.n)]
        self.current_step = 0
        ob = self.dummy_ob
        return ob

    def render(self, mode='human'):
        # google histogram in bash
        print("Lever pull counts:")
        print("index:", *range(self.action_space.n))
        print("count:", *self.pull_counts)

    def close(self):
        pass









class SelfReinforcingBandit(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
            self, 
            episode_len=40,
            num_arms=5,
            observe_timestep=False):
        """
        args:
            observe_timestep: whether to include the current timestep
                in observations.
        """
        super().__init__()
        self.episode_len = episode_len
        self.observe_timestep = observe_timestep
        self.action_space = spaces.Discrete(num_arms)
        if self.observe_timestep:
            self.observation_space = spaces.Discrete(self.episode_len)
        else:
            self.observation_space = spaces.Discrete(1)
            self.dummy_ob = 0  # can add 1 to DQN input if I want
        self.reset()

    def step(self, action: int):
        """
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (integer) :
                Timestep
            reward (float) :
                Equal to number of times the lever `action` has been pushed
                so far.
            done (bool) :
                After self.max_steps steps (default 40).
            info (dict) :
                includes environment state (lever pull counts).
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid.")

        self.current_step += 1
        self.pull_counts[action] += 1
        if all([a == action for a in self.prev_actions[-3:]]):
            self.rew[action] += 1
        self.prev_actions.append(action)
        #reward = self.rew[action]
        #reward = 1 if action == self.prev_actions[-2] else 0
        reward = self.pull_counts[action]
        #reward = 1 if self.pull_counts[action] > 3 else 0
        #reward = self.pull_counts[action]*action
        #reward = 10 if action == 0 else 0

        ob = self.current_step if self.observe_timestep else self.dummy_ob
        done = self.current_step >= self.episode_len - 1
        info = {"pull_counts": self.pull_counts.copy()}
        return ob, reward, done, info

    def reset(self):
        self.pull_counts = [0]*self.action_space.n
        self.rew = [0]*self.action_space.n
        self.prev_actions = [np.random.choice(self.action_space.n)]
        self.current_step = 0
        ob = 0
        return ob

    def render(self, mode='human'):
        # google histogram in bash
        print("Lever pull counts:")
        print("index:", *range(self.action_space.n))
        print("count:", *self.pull_counts)

    def close(self):
        pass
