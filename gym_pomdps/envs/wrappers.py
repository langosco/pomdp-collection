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


# Memory

def concatenate(components):
    """concatenate all lists in components. If an element is 0-d, treat it as a single element list"""
    components = [c if np.ndim(c) > 0 else [c] for c in components]
    return np.concatenate(components)


def combine_spaces(space_list):
    """combine list of Discrete or MultiDiscrete spaces into 
    a single MultiDiscrete space"""
    if not all([isinstance(s, (spaces.Discrete, spaces.MultiDiscrete)) for s in space_list]):
        raise ValueError("All spaces need to be Discrete or MultiDiscrete") 
    combined_shape = concatenate([space_shape(s) for s in space_list])
    return spaces.MultiDiscrete(list(combined_shape))


class Memory(gym.Env):
    def __init__(self, num_bits=1):
        self.num_bits = num_bits
        self.action_space = spaces.MultiDiscrete([2]*num_bits)
        self.observation_space = spaces.MultiDiscrete([2]*num_bits)
        self.reset()
    
    def step(self, action):
        ob = self.memory = (self.memory + action) % 2
        return ob
    
    def reset(self):
        self.memory = np.array([0]*self.num_bits)


class CombinedSpace(spaces.MultiDiscrete):
    def __init__(self, space_list):
        if not all([isinstance(s, (spaces.Discrete, spaces.MultiDiscrete)) for s in space_list]):
            raise ValueError("All spaces need to be Discrete or MultiDiscrete")

        self.space_shapes = [space_shape(s) for s in space_list]
        self.combined_shape = concatenate(self.space_shapes)
        super().__init__(self.combined_shape) # init MultiDiscrete
        
    def unwrap_element(self, element):
        """take element and unwrap it into tuple of elements of component spaces"""
        def dim(s):
            try:
                return len(s)
            except TypeError:
                return 1

        component_dims = [dim(s) for s in self.space_shapes]  # dim of each tuple element
        components = []
        idx = 0
        for dim in component_dims:
            next_idx = idx + dim
            components.append(np.squeeze(element[idx:next_idx]))
            idx = next_idx
        return components
    
    def wrap_components(self, components):
        """take component elements and return element of combined space"""
        return concatenate(components)
    
    
class AddExternalMemory(gym.Wrapper):
    """Add read/write memory"""
    def __init__(self, env, num_bits=5):
        super().__init__(env)
        self.memory = Memory(num_bits=num_bits)

        self.action_space = CombinedSpace([self.env.action_space, self.memory.action_space])
        self.observation_space = CombinedSpace([self.env.observation_space, self.memory.observation_space])
    
    def step(self, action):
        base_action, mem_action = self.action_space.unwrap_element(action)
        base_ob, rew, done, info = super().step(base_action)
        mem_ob = self.memory.step(mem_action)
        return self.observation_space.wrap_components([base_ob, mem_ob]), rew, done, info
