import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class ButtonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
            episode_len=40,
            reset_state='random'):
        """
        args
            episode_len
            reset_state: one of 'random', 'push', 'pass'
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
                -1 if button is pushed (action=1), else 0.
                If button was pushed in the previous step, add 10.
            episode_over (bool) :
                After self.max_steps steps (default one million steps).
            info (dict) :
                includes environment state self.button_pushed (whether
                the button was pushed in the previous timestep).
        """
        self.current_step += 1
        if self.button_pushed:
            reward = 10
        else:
            reward = 0
        info = {'button_pushed': self.button_pushed}

        if action == 1:
            self.button_pushed = True
            reward = reward - 1
        elif action == 0:
            self.button_pushed = False

        ob = 1
        episode_over = self.current_step >= self.episode_len
        return ob, reward, episode_over, info

    def reset(self):
        if self.reset_state == "random":
            self.button_pushed = np.random.choice([True, False])
        elif self.reset_state == "push":
            self.button_pushed = True
        elif self.reset_state == "pass":
            self.button_pushed = False
        else:
            raise ValueError('reset_state must be one of "random", "push", or "pass".')
        self.current_step = 0
        ob = 1
        return ob

    def render(self, mode='human'):
        print("Button pushed:", self.button_pushed)

    def close(self):
        pass
