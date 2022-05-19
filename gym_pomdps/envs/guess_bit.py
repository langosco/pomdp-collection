import gym
from gym import spaces
import numpy as np
from numpy.random import default_rng

rng = default_rng()

class GuessBit(gym.Env):
    def __init__(self, episode_len=10, num_bits=1):
        self.action_space = spaces.MultiDiscrete([2]*num_bits)
        self.observation_space = spaces.MultiDiscrete([2]*num_bits)
        self.num_bits = num_bits
        self.episode_len = episode_len
        self.reset()
    
    def step(self, action):
        self.current_step += 1
        rew = np.sum([a == o for a, o in zip(action, self.target)]) if self.current_step > 1 else 0
        self.target = np.random.randint(0, 2, self.num_bits) # TODO use rng seed
        done = self.current_step >= self.episode_len
        info = {}
        return self.target, rew, done, info

    def reset(self):
        self.target = np.random.randint(0, 2, self.num_bits) # TODO use rng seed
        self.current_step = 0


class FlipBit(gym.Env):
    def __init__(self, episode_len=40, reward_weights=[1, 1]):
        """
        args:
            extra_rews: terms to add to reward, depending on whether 
            agent chooses to flip or not. This way we can make flipping
            (or not flipping) an optimal policy.
        """
        self.action_space = spaces.MultiDiscrete([2, 2])
        self.observation_space = spaces.Discrete(2)
        self.episode_len = episode_len
        self.reward_weights = reward_weights
        self.reset()
        
    def step(self, action):
        self.current_step += 1
        flip, guess = action
        rew = 1 if guess == self.hidden else 0
        rew *= self.reward_weights[guess == self.hidden]
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
