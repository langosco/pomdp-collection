import gym
from gym import error, spaces, utils
import numpy as np

# multiple fixed points. normal discount rates.
# pushing button only a little hurts, but
# if agent pushes often enough it gets high return

# edited to add: 
# I don't think I've done any experiments with this one.
# Also it seems like this one should just have one
# optimal policy that is easily reachable? cause it's just
# about pushing the one button more often (you start getting
# reward when you've pushed the button > 4 times).

# Though a variant with multiple buttons might be interesting,
# I think that must be what I was orignally going for.

class GeneralButtonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
            episode_len=40,
            max_push_count=5,
            reward_fn: callable = None):
        """
        reward_fn maps push_count to reward
        """
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)  # actually just 1 but wanna pass 1 to DQN instead of 0

        self.episode_len = episode_len
        self.reset_state = reset_state

        if reward_fn:
            self.reward_fn = reward_fn
        else:
            def self.reward_fn(push_count):
                push_count = np.clip(push_count, max_push_count)
                return np.abs(push_count - 4)

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
                Always 1 (no observations)
            reward (float) :
                depends only on nr of button presses in the current episode
            episode_over (bool) :
                After self.episode_len steps (default 40)
            info (dict) :
                includes nr of button presses so far
        """
        self.current_step += 1

        if action == 1:
            self.push_count += 1

        ob = 1
        reward = self.reward_fn(self.push_count)
        done = self.current_step >= self.episode_len
        info = {'push_count': self.push_count}
        return ob, reward, done, info

    def reset(self):
        self.push_count = 0
        self.current_step = 0
        ob = 1
        return ob

    def render(self, mode='human'):
        print("Push count:", self.push_count)

    def close(self):
        pass
