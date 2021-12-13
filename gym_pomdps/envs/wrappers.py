import gym
from gym import spaces
import numpy as np


def flat_size(tuple_env):
    """return size of flattened env"""
    return 


class MakeActionSpaceDiscrete(gym.Wrapper):
    """
    Wrapper that changes the shape of an environment
    action space from a tuple of discrete spaces to a 
    1D discrete space.
    """
    def __init__(self, env):
        super().__init__(env)
        self._action_tuple_shape = [s.n for s in env.action_space]
        self.action_space = spaces.Discrete(np.prod(self._action_tuple_shape))

    def flat_to_tuple(self, action):
        return np.unravel_index(action, self._action_tuple_shape)

    def step(self, action):
        action = self.flat_to_tuple(action)
        return self.env.step(action)


class MakeObservationSpaceDiscrete(gym.Wrapper):
    """
    Wrapper that changes the shape of an environment
    observation space from a tuple of discrete spaces to a 
    1D discrete space.
    """
    def __init__(self, env):
        super().__init__(env)
        self._obs_tuple_shape = tuple([s.n for s in env.observation_space])
        self.observation_space = spaces.Discrete(np.prod(self._obs_tuple_shape))

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return np.ravel_multi_index(ob, self._obs_tuple_shape), reward, done, info
