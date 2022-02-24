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
        self.do_nothing = False
        if isinstance(env.action_space, Discrete):
            self.do_nothing = True
        elif not isinstance(env.action_space, (MultiDiscrete, Tuple)):
            raise ValueError("Action space needs to me MultiDiscrete or Tuple")

        # this is the shape of the action space (NOT shape of one action)
        self._action_space_shape = space_shape(env.action_space)
        self.action_space = spaces.Discrete(np.prod(self._action_space_shape))

    def flat_to_tuple(self, action):
        """map integer action to tuple"""
        return action if self.do_nothing else np.unravel_index(action, self._action_space_shape)

    def tuple_to_flat(self, action):
        """map tuple action to integer"""
        return action if self.do_nothing else np.ravel_multi_index(action, self._action_space_shape)

    def step(self, action):
        action = self.flat_to_tuple(action)
        return self.env.step(action)


class ObservationSpaceToDiscrete(gym.Wrapper):
    """tranform observation space from multidiscrete to discrete"""
    def __init__(self, env):
        super().__init__(env)
        self.do_nothing = False
        if isinstance(env.observation_space, Discrete):
            self.do_nothing = True
        elif not isinstance(env.observation_space, (MultiDiscrete, Tuple)):
            raise ValueError("Observation space needs to me MultiDiscrete or Tuple")

        # this is the shape of the obs space (NOT shape of one obs)
        self._space_shape = space_shape(env.observation_space)
        self.observation_space = spaces.Discrete(np.prod(self._space_shape))

    def tuple_to_flat(self, ob):
        """map tuple observation to integer"""
        return ob if self.do_nothing else np.ravel_multi_index(ob, self._space_shape)

    def flat_to_tuple(self, ob):
        """map integer observation to tuple"""
        return ob if self.do_nothing else np.unravel_index(ob, self._space_shape)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return self.tuple_to_flat(ob), reward, done, info
