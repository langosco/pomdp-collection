import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class TwoFixedPoints(gym.Env):
    """
    Simple environment with two fixed points.
    Two states: a and b.
    Two actions: 0 moves to a, 1 moves to b
    No reward for moving. a -> a gives reward 1, b -> b gives reward 2
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
            episode_len=40,
            reset_state='random'):
        """
        args
            episode_len: integer
            reset_state: one of 'random', 'a', 'b'
        """
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)  # actually just 1 but wanna pass 1 to DQN instead of 0

        self.episode_len = episode_len
        self.reset_state = reset_state
        self.reset()

    def step(self, action):
        """
        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                Always None (no observations)
            reward (float) :
            episode_over (bool) :
            info (dict) :
        """
        self.current_step += 1
        previous_state = self.state
        self.state = ["a", "b"][action]

        if self.state == previous_state == "a":
            reward = 1
        if self.state == previous_state == "b":
            reward = 2
        else:
            reward = 0

        ob = 1
        episode_over = self.current_step >= self.episode_len
        return ob, reward, episode_over, info

    def reset(self):
        if self.reset_state == "random":
            self.state = np.random.choice(["a", "b"])
        elif self.reset_state in ["a", "b"]:
            self.state = self.reset_state
        else:
            raise ValueError('reset_state must be one of "random", "a", or "b".')
        self.current_step = 0
        ob = 1
        return ob

    def render(self, mode='human'):
        raise NotImplementedError()

    def close(self):
        pass
