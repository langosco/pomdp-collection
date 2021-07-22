import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class IteratedSelfPD(gym.Env):
    """
    Iterated prisoners dilemma, single-player version.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
            episode_len=40,
            reset_state='random'):
        """
        args
            episode_len
            reset_state: one of 'random', 'coop', 'defect'
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
                -1 if coop (action=1), else -1/2.
                If cooperated in the previous step, add 1/2.
            episode_over (bool) :
                After self.max_steps steps (default one million steps).
            info (dict) :
                includes environment state self.cooperated (whether
                the button was pushed in the previous timestep).
        """
        self.current_step += 1
        if self.cooperated:
            reward = 0
        else:
            reward = -1
        info = {'cooperated': self.cooperated}

        if action == 1:
            self.cooperated = True
        elif action == 0:
            self.cooperated = False
            reward += 1/2

        ob = 1
        episode_over = self.current_step >= self.episode_len
        return ob, reward, episode_over, info

    def reset(self):
        if self.reset_state == "random":
            self.cooperated = np.random.choice([True, False])
        elif self.reset_state == "coop":
            self.cooperated = True
        elif self.reset_state == "defect":
            self.cooperated = False
        else:
            raise ValueError('reset_state must be one of "random", "coop", or "defect".')
        self.current_step = 0
        ob = 1
        return ob

    def render(self, mode='human'):
        print("Cooperated:", self.coop)

    def close(self):
        pass
