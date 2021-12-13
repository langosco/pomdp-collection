import gym
from gym import error, spaces, utils
from gym.utils import seeding
import itertools
import numpy as np

###############
# DESCRIPTION 
###############
# A 1-armed bandit game with a twist.
#
# The reward from pulling the lever alternates between 
# 5 and -10. That is, it returns reward 
#       r(t, pull) = 5 if t even, else -10,
# where t is the (unobserved) timestep. Note this is a
# POMDP.
# Given these rewards, a memoryless agent should simply
# learn to always abstain from pulling the lever.
#
# But: the environment also includes a sign, painted green
# on one side and red on the other. At each timestep, in
# addition to pulling the arm, the agent may decide to 
# flip the sign. In the next timestep, the agent will
# observe the color of the sign. 
# An optimal policy will use the sign as external memory:
# it will flip the sign at each turn, and use the alternating
# observations to only pull the lever every other turn.
# This is possible only if the initial sign state is
# always the same, which is the case in this implementation.
#
# Formally: the action space is
#       A = {don't pull, pull} x {don't flip, flip}
#         ~= {0, 1} x {0, 1},
#
# the state space is
#       S = {red, green} x {dangerous, safe}
#         ~= {0, 1} x {0, 1}
#
# and the observation space is
#       O = {red, green} ~= {0, 1}.


class AlternatingBandit(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
            self, 
            episode_len=10,
            rewards=[5, -5]):
        """
        args:
            episode_len: if passed, cap episodes at the given
                length. Otherwise infinite time horizon.
                Warning: infinite episode length makes
                learning an optimal policy even more impossible.
            rewards: tuple of two scalars: reward for pulling lever
                on an even vs. odd timestep (first timestep is 1, ie odd).
        """
        super().__init__()
        arm = spaces.Discrete(2)  # {don't pull, pull}
        sign = spaces.Discrete(2)  # {don't flip, flip}
        self.action_space = spaces.Tuple((arm, sign))
        self.observation_space = spaces.Discrete(2)
        self.episode_len = episode_len
        self.rewards = rewards
        self.reset()

    def _reward(self, action: tuple):
        """
        args:
            action is a tuple (pull, flip) \in {0, 1}^2
        """
        pull, _ = action
        _, safe = self.state
        if pull == 0:
            return 0
        elif pull == 1:
            return self.rewards[1]*(1-safe) + self.rewards[0]*safe

    def _update_state(self, action):
        _, flip_sign = action
        color, safe = self.state
        if flip_sign:
            color = 1 - color
        safe = 1 - safe
        self.state = (color, safe)
        self.current_step += 1
        return

    def step(self, action: tuple):
        """
        Args
        -------
            action :
                tuple (pull, flip). Action space
                is {0, 1} x {0, 1}, corresponding
                to {don't pull, pull} x {don't flip, flip}
                
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (integer) :
                0 (for red) or 1 (for green).
            reward (float) :
                0 if agent doesn't pull, else 
                -10 or 5, depending on state
            done (bool) :
                After self.max_steps steps (default 40).
            info (dict) :
                includes full environment state.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid.")

        reward = self._reward(action)
        self._update_state(action)

        color, safe = self.state
        ob = color  # only observe color
        done = self.current_step >= self.episode_len
        info = {"sign_color": color, "safe": safe}
        return ob, reward, done, info

    def reset(self):
        self.state = (0, 0)  # (color_idx, safe)
        self.current_step = 0
        ob = self.state[0]
        return ob

    def render(self, mode='human'):
        color, safe = self.state
        print("Sign color:", ["red", "green"][color])
        print("Arm two:", ["danger", "safe"][safe])

    def close(self):
        pass


def flat_size(tuple_env):
    """return size of flattened env"""
    return 

