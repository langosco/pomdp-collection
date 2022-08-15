import gym
from gym import spaces
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class SimpleSequentialBandit(gym.Env):
    """
    A simple POMDP bandit task: take n actions in sequence. 

    A memoryless policy struggles with this task, because
    it cannot remember which action it has already taken, 
    and thus cannot do better than a random policy.

    The sequence is always the same: 0 to num_arms-1.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            num_arms: int = 5,
            episode_len: int = 5,
    ):
        """
        args:
            episode_len: max episode length (might terminate earlier)
            num_arms: how many actions to take
        """
        super().__init__()
        self.action_space = spaces.Discrete(num_arms)
        self.observation_space = spaces.Discrete(0)
        self.episode_len = episode_len
        self.num_arms = num_arms
        self.reset()

    def reset(self):
        self.current_loc = 0
        self.current_step = 0
        self.next_arm = 0
        ob = 0
        return ob

    def step(self, action: tuple):
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid.")
        
        if action == self.next_arm:
            self.next_arm += 1
            reward = 1
        else:
            reward = 0

        self.current_step += 1
        ob = 0
        done = self.current_step >= self.episode_len \
                or self.next_arm >= self.num_arms # all coins collected
        info = {"timestep": self.current_step,
                "next_arm": self.next_arm}
        return ob, reward, done, info

    def render(self, mode='human'):
        g = nx.Graph()
        for arm in range(self.num_arms):
            g.add_node(arm)
            
        node_colors = ["red"] * self.next_arm + ["white"] * (len(g) - self.next_arm)
        options = {
            "node_color": node_colors,
            "edgecolors": "black",
        }
        pos = nx.spring_layout(g, seed=1)  # Seed layout for reproducibility
        nx.draw_networkx(g, pos, **options)
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")

        print(f"next arm: {self.next_arm}")
        
    def close(self):
        pass


class CoinSequentialNav(gym.Env):
    """
    A simple navigation task: collect coins from n states. All
    n states are reachable from one single central state.

    A memoryless policy struggles with this task, because
    it cannot remember which states it has already
    visited, and thus cannot do better than a random policy.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            num_goals: int = 5,
            episode_len: int = 20,
    ):
        """
        args:
            episode_len: if passed, cap episodes at the given
                length. Otherwise infinite time horizon.
            num_goals: how many coins to collect
        """
        super().__init__()
        num_locations = num_goals + 1
        self.locations = list(range(num_locations))
        # actions are (move, ), and
        # obs are (current_loc, coin yes/no)
        self.action_space = spaces.Discrete(num_locations)
        self.observation_space = spaces.MultiDiscrete([num_locations, 2])
        self.episode_len = episode_len
        self.num_goals = num_goals
        self.reset()

    def _previous_coins_collected(self):
        """check if coin can be picked up from current_loc"""
        current_goal = np.nonzero(self.coins)[0][0]
        return current_goal == self.current_loc - 1
    
    def _coin_present(self):
        """returns True if a coin is present in current_loc"""
        if self.current_loc == 0:
            return False
        else:
            return bool(self.coins[self.current_loc-1])
    
    def _try_collect_coin(self):
        """
        attempts to collect a coin.
        returns True if successful, else False
        """
        if not self._coin_present():
            return False
        elif self._previous_coins_collected():
            self.coins[self.current_loc-1] = 0
            return True
        else:
            return False

    def _update_state(self, action):
        if self.current_loc == 0 \
        or action == 0 \
        or action == self.current_loc:
            self.current_loc = action
        else:
            pass
        self.current_step += 1
        return

    def step(self, action: tuple):
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid.")
        
        success = self._try_collect_coin()
        self._update_state(action)
        
        ob = (self.current_loc, int(self._coin_present()))
        reward = int(success)
        done = self.current_step >= self.episode_len \
                or np.sum(self.coins) == 0 # all coins collected
        info = {"timestep": self.current_step,
                "coins": self.coins}
        return ob, reward, done, info

    def reset(self):
        self.current_loc = 0
        self.current_step = 0
        self.coins = [1] * self.num_goals  # 1 means coin is still there, 0 means collected
        ob = (self.current_loc, 0)
        return ob

    def render(self, mode='human'):
        g = nx.Graph()
        for loc in self.locations[1:]:
            g.add_edge(0, loc)
            
        def node_idx(node):
            """return index of node in nx.Graph ordering"""
            return np.where(np.array(g.nodes) == node)[0].squeeze()
        
        node_colors = ["white"] * len(g)
        node_colors[node_idx(self.current_loc)] = "red"
        options = {
            "node_color": node_colors,
            "edgecolors": "black",
        }
        pos = nx.spring_layout(g, seed=1)  # Seed layout for reproducibility
        nx.draw_networkx(g, pos, **options)
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        
        print("Coins collected:", self.num_goals - np.sum(self.coins))
        if self._coin_present():
            print("Coin Present")

    def close(self):
        pass
