import gym
from gym import spaces
import numpy as np
from numpy.random import default_rng

rng = default_rng()

class GuessBit(gym.Env):
    def __init__(self, episode_len=40):
        self.action_space = spaces.MultiDiscrete([2, 2])
        self.observation_space = spaces.Discrete(2)
        self.episode_len = episode_len
        self.reset()
        
    def step(self, action):
        self.current_step += 1
        flip, guess = action
        rew = 1 if guess == self.hidden else 0
        self.update_hidden()
        ob = int(not self.hidden) if flip else self.hidden
        done = self.current_step >= self.episode_len
        info = {"step": self.current_step}
        return ob, rew, done, info
    
    def update_hidden(self):
        self.hidden = rng.integers(2)
    
    def reset(self):
        self.update_hidden()
        self.current_step = 0
        dummy_flip = rng.integers(2)
        ob = int(not self.hidden) if dummy_flip else self.hidden
        return ob
