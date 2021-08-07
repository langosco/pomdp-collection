import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import itertools

###############
# DESCRIPTION 
###############
# A 2-bandit game with a twist.
#
# Arm b1 always gives reward 1
# Arm b2 alternates between reward 5 and -10. That is, it
# returns reward 
#       r(t, b2) = 5 if t odd, else -10,
# where t is the (unobserved) timestep. Note this is a
# POMDP.
# Given these rewards, a memoryless agent should simply
# learn to always pull b1.
#
# But: the environment also includes a rock, painted green
# on one side and red on the other. At each timestep, in
# addition to pulling an arm, the agent may decide to 
# flip the rock. In the next timestep, the agent will
# observe the color of the rock. 
# An optimal policy will use the rock as external memory:
# it will flip the rock at each turn, and use the alternating
# observations to only pull arm b2 every other turn.
# This is possible only if the initial rock state is
# always the same, which is the case in this implementation.
#
# Formally: the action space is
#       A = {b1, b2} x {flip, don't flip},
# the state space is
#       S = {red, green} x {0, 1},
# and the observation space is
#       O = {red, green}.


class AlternatingBandit(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
            self, 
            episode_len=np.inf):
        """
        args:
            episode_len: if passed, cap episodes at the given
                length. Otherwise infinite time horizon.
        """
        super().__init__()
        arms = spaces.Discrete(2)  # {b1, b2}
        rock = spaces.Discrete(2)  # {flip, don't flip}
        self.action_space = spaces.Tuple((arms, rock))
        self.observation_space = spaces.Discrete(2)
        self.episode_len = episode_len
        self.reset()

    def _reward(self, action: tuple):
        """
        side 0: danger, arm 2 gives -10
        side 1: safe, arm 2 gives +5
        """
        arm, _ = action
        _, side = self.state
        if arm == 0:
            return 1
        elif arm == 1:
            return -10*(1-side) + 5*side

    def _update_state(self, action):
        _, flip_rock = action
        color, side = self.state
        if flip_rock:
            color = 1 - color
        side = 1 - side
        self.state = (color, side)
        self.current_step += 1
        return

    def step(self, action: tuple):
        """
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (integer) :
                0 (for red) or 1 (for green).
            reward (float) :
                pull arm 0: r = 1
                pull arm 1: r = -10 or 5, depending on state
            done (bool) :
                After self.max_steps steps (default 40).
            info (dict) :
                includes full environment state.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid.")

        reward = self._reward(action)
        self._update_state(action)

        color, side = self.state
        ob = color  # only observe color
        done = self.current_step >= self.episode_len
        info = {"rock_color": color, "side": side}
        return ob, reward, done, info

    def reset(self):
        self.state = (0, 0)  # (color_idx, side)
        self.current_step = 0
        ob = self.state[0]
        return ob

    def render(self, mode='human'):
        color, side = self.state
        print("Rock color:", ["red", "green"][color])
        print("Arm two:", ["danger", "safe"][side])

    def close(self):
        pass
