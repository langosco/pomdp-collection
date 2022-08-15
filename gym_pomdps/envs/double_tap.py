import gym
from gym import error, spaces, utils
import numpy as np
from numpy.random import default_rng

rng = default_rng()


class DoubleTap(gym.Env):
    """
    Simple environment with two fixed points.
    Two states: a and b.
    Two actions: 0 moves to a, 1 moves to b
    No reward for switching state. a -> a gives reward 1, b -> b gives reward 2
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
            episode_len=40,
            reset_state='random',
            dqn=False):
        """
        args
            episode_len: integer
            reset_state: one of 'random', 'a', 'b'
        """
        super().__init__()
        self.action_space = spaces.Discrete(2)
        if dqn:
            self.observation_space = spaces.Discrete(2)  # actually just 1 but wanna pass 1 to DQN instead of 0
            self.dummy_ob = 1
        else:
            self.observation_space = spaces.Discrete(1)
            self.dummy_ob = 0

        self.episode_len = episode_len
        self.reset_state = reset_state
        self.opt_avg_timestep_rew = 2
        self.subopt_avg_timestep_rew = 1
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
                Constant (no observations)
            reward (float) :
            episode_over (bool) :
            info (dict) :
        """
        self.current_step += 1
        previous_state = self.state
        self.state = ["a", "b"][action]

        if self.state == previous_state == "a":
            reward = 1
        elif self.state == previous_state == "b":
            reward = 2
        else:
            reward = 0

        ob = self.dummy_ob
        episode_over = self.current_step >= self.episode_len
        info = {}
        return ob, reward, episode_over, info

    def reset(self):
        if self.reset_state == "random":
            self.state = np.random.choice(["a", "b"])
        elif self.reset_state in ["a", "b"]:
            self.state = self.reset_state
        else:
            raise ValueError('reset_state must be one of "random", "a", or "b".')
        self.current_step = 0
        return self.dummy_ob

    def render(self, mode='human'):
        raise NotImplementedError()

    def close(self):
        pass



class MultiArmedDoubleTap(gym.Env):
    """
    Bandit problem with k arms. Every arm returns reward only when 
    activated the second time in a row.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
            episode_len=40,
            reset_state='random',
            num_arms: int = 5,
            dummy_obs: bool = True,
            rewards = None):
        """
        args
            episode_len: integer
            reset_state: either 'random', or an integer between 0 and k.
        """
        super().__init__()
        self.action_space = spaces.Discrete(num_arms)
        if dummy_obs:
            self.observation_space = spaces.Discrete(2)  # actually just 1 but wanna pass 1 to DQN instead of 0
            self.dummy_ob = 1
        else:
            self.observation_space = spaces.Discrete(1)
            self.dummy_ob = 0

        self.num_arms = num_arms
        self.episode_len = episode_len
        self.reset_state = reset_state
        if rewards is None:
            self.rewards = np.arange(num_arms) + 1
        else:
            self.rewards = rewards
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
                Constant (no observations)
            reward (float) :
            episode_over (bool) :
            info (dict) :
        """
        self.current_step += 1

        if action == self.previous_action:
            reward = self.rewards[action]
        else:
            reward = 0

        self.previous_action = action
        ob = self.dummy_ob
        episode_over = self.current_step >= self.episode_len
        info = {}
        return ob, reward, episode_over, info

    def reset(self):
        if self.reset_state == "random":
            self.previous_action = np.random.randint(self.num_arms)
        elif self.reset_state in range(self.num_arms):
            self.previous_action = self.reset_state
        else:
            raise ValueError('reset_state must be either "random", or '
                             f'an integer smaller than {self.num_arms}')
        self.current_step = 0
        return self.dummy_ob

    def render(self, mode='human'):
        raise NotImplementedError()

    def close(self):
        pass



class MultiArmedMTap(gym.Env):
    """
    Bandit problem with k arms. Every arm returns reward only when 
    activated the m-th time in a row, where m depends on the arm.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
            episode_len=40,
            num_arms: int = 5,
            dummy_obs: bool = True,
            m_power: float = 1.5,
            m_list = None):
        """
        args
            episode_len: integer
            reset_state: either 'random', or an integer between 0 and k.
        """
        super().__init__()
        self.action_space = spaces.Discrete(num_arms)
        if dummy_obs:
            self.observation_space = spaces.Discrete(2)  # actually just 1 but wanna pass 1 to DQN instead of 0
            self.dummy_ob = 1
        else:
            self.observation_space = spaces.Discrete(1)
            self.dummy_ob = 0

        self.num_arms = num_arms
        self.episode_len = episode_len
        self.m_power = m_power
        
        if m_list is None:
            self.m_list = 1 + np.arange(num_arms)
        else:
            assert len(m_list) == num_arms
            self.m_list = m_list
        self.reset()

    def _update_pull_count(self, action):
        if self.count[0] == action:
            self.count[1] += 1
        else:
            self.count = [action, 1]

    def _get_reward(self, action):
        """
        check if count is large enough. If yes, give reward and reset count.
        """
        if self.count[1] == self.m_list[action]:
            self.count[1] = 0
            return self.m_list[action]**self.m_power
        else:
            return 0

    def step(self, action):
        """
        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                Constant (no observations)
            rew (float) :
            done (bool) :
            info (dict) :
        """
        self.current_step += 1
        self._update_pull_count(action)
        rew = self._get_reward(action)

        ob = self.dummy_ob
        done = self.current_step >= self.episode_len
        info = {"count": self.count, "step": self.current_step}
        return ob, rew, done, info

    def reset(self):
        self.count = [0, 0]  # (arm, pull_count)
        self.current_step = 0
        return self.dummy_ob

    def render(self, mode='human'):
        raise NotImplementedError()

    def close(self):
        pass
