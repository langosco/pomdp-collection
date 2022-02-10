import gym
from gym import spaces
import numpy as np
from gym.spaces import Box, Discrete, Tuple, MultiBinary, MultiDiscrete


def space_shape(space):
    """return the shape of the ndarray containing all elements of the
    space. NOT the shape of a single element"""
    if isinstance(space, Tuple):
        try:
            return tuple([s.n for s in space])
        except:
            raise ValueError
    elif isinstance(space, MultiDiscrete):
        return space.nvec
    elif isinstance(space, Discrete):
        return space.n
    else:
        raise ValueError


class ActionSpaceToDiscrete(gym.Wrapper):
    """tranform action space from multidiscrete to discrete"""
    def __init__(self, env):
        super().__init__(env)
        # this is the shape of the action space (NOT shape of one action)
        self._action_space_shape = space_shape(env.action_space)
        self.action_space = spaces.Discrete(np.prod(self._action_space_shape))

    def flat_to_tuple(self, action):
        return np.unravel_index(action, self._action_space_shape)

    def step(self, action):
        action = self.flat_to_tuple(action)
        return self.env.step(action)


class ObservationSpaceToDiscrete(gym.Wrapper):
    """tranform observation space from multidiscrete to discrete"""
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, (MultiDiscrete, Tuple)):
            raise ValueError

        # this is the shape of the obs space (NOT shape of one obs)
        self._space_shape = space_shape(env.observation_space)
        self.observation_space = spaces.Discrete(np.prod(self._space_shape))

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return np.ravel_multi_index(ob, self._space_shape), reward, done, info
