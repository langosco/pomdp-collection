import gym
from gym import spaces
import numpy as np
from numpy.random import default_rng

rng = default_rng(0)

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


class AlternatingBanditWithResultObs(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
            self, 
            episode_len=20,
            rewards=[5, -5],
            init_color=None,
            init_bandit_state=None):
        """
        In this version, agent observes result (reward / loss) after every
        step and then has the option of flipping the sign, conditioning on
        the result in previous step.
        So we alternate between bandit step and sign step.

        args:
            episode_len: if passed, cap episodes at the given
                length. Otherwise infinite time horizon.
                Warning: infinite episode length makes
                learning an optimal policy impossible.
            rewards: tuple of two scalars: reward for pulling lever
                on an safe vs. unsafe timestep.
            init_color: first sign state (0 = red, 1 = green)
            init_bandit_state: first bandit state (0 = danger, 1 = safe)
        """
        super().__init__()
        # action space: {don't pull, pull} x {don't flip, flip}
        self.action_space = spaces.MultiDiscrete([2, 2])

        # obs space: {red, green} x {prev_dangerous, prev_safe, None} x {sign_round, bandit_round}
        self.observation_space = spaces.MultiDiscrete([2, 3, 2])

        if init_color is None:
            self.init_color = "random"
        else:
            self.init_color = init_color

        if init_bandit_state is None:
            self.init_bandit_state = "random"
        else:
            self.init_bandit_state = init_bandit_state

        self.dummy_obs = 2
        self.episode_len = episode_len
        self.rewards = rewards
        self.reset()

    def _reward(self, action: tuple):
        """
        args:
            action is a tuple (pull, flip) \in {0, 1}^2
        """
        pull, _ = action
        arm_active = pull and self.bandit_round
        _, safe = self.state
        return (self.rewards[1]*(1-safe) + self.rewards[0]*safe) * arm_active

    def _update_state(self, action):
        _, flip_sign = action
        color, safe = self.state

        if self.bandit_round:
            self.bandit_round = False
            safe = 1 - safe
        else:
            self.bandit_round = True

        if flip_sign: # can always flip sign
            color = 1 - color
        
        self.state = (color, safe)
        self.current_step += 1
        return

    def _get_obs(self):
        color, safe = self.state
        if self.bandit_round:
            ob = [color, self.dummy_obs, int(self.bandit_round)]  # only observe color
        else:
            ob = [color, 1-safe, int(self.bandit_round)]  # observe color and prev bandit state
        return ob

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
        ob = self._get_obs()

        done = self.current_step >= self.episode_len
        color, safe = self.state
        info = {
            "sign_color": color, 
            "safe": safe, 
            "bandit_round": self.bandit_round,
        }
        return ob, reward, done, info

    def reset(self):
        self.bandit_round = True  # first step is a bandit round
        color = rng.integers(2) if self.init_color == "random" else self.init_color
        bandit_state = rng.integers(2) if self.init_bandit_state == "random" else self.init_bandit_state
        self.state = [color, bandit_state]  
        self.current_step = 0
        return self._get_obs()

    def render(self, mode='human'):
        color, safe = self.state
        if self.bandit_round:
            print("Bandit Round")
            print("Sign color:", ["red", "green"][color])
        else:
            print("Sign Round")
            prev_result = ["danger", "safe"][1-safe]
            print("Sign color:", ["red", "green"][color])
            print("Prev Result:", f"{prev_result}")

        print()
        print("Next Bandit State (unobserved):")
        print(["danger", "safe"][safe])

    def close(self):
        pass
