import gym
from gym import spaces
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class SimpleSequentialNav(gym.Env):
    """
    A simple navigation task: visit n states in sequence. All
    n states are reachable from one single central state.

    A memoryless policy struggles with this task, because
    it cannot remember which states it has already
    visited, and thus cannot do better than a random policy.

    The environment includes 'external' memory that the policy
    can write to via a dedicated 'write' action.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            num_goals: int = 5,
            episode_len: int = 20,
            mem_size: int = 10,
            write_method: str = "flexible",
    ):
        """
        args:
            episode_len: if passed, cap episodes at the given
                length. Otherwise infinite time horizon.
            num_goals: how many coins to collect
            write_method: one of 'flexible': one action per integer 
                the agent can write to memory, or 'plus one': just
                two actions, do nothing or add +1 to the int in memory
                (modulo mem_size).
        """
        super().__init__()
        num_locations = num_goals + 1
        self.locations = list(range(num_locations))
        # actions are (move, write), and
        # obs are (current_loc, read, coin yes/no)
        if write_method == "flexible":
            self.action_space = spaces.MultiDiscrete([num_locations, mem_size])
        elif write_method == "plus one":
            self.action_space = spaces.MultiDiscrete([num_locations, 2])
        else:
            raise ValueError(f"write_method invalid: '{write_method}'")
        self.observation_space = spaces.MultiDiscrete([num_locations, mem_size, 2])
        self.episode_len = episode_len
        self.num_goals = num_goals
        self.write_method = write_method
        self.mem_size = mem_size
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
        move, write = action
        if self.write_method == "flexible":
            self.memory = write
        elif self.write_method == "plus one":
            if write:
                self.memory = (self.memory + 1) % self.mem_size
            else:
                pass
        else:
            raise


        if self.current_loc == 0 \
        or move == 0 \
        or move == self.current_loc:
            self.current_loc = move
        else:
            pass
        self.current_step += 1
        return

    def step(self, action: tuple):
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid.")
        
        success = self._try_collect_coin()
        self._update_state(action)
        
        ob = (self.current_loc, self.memory, int(self._coin_present()))
        reward = int(success)
        done = self.current_step >= self.episode_len \
                or np.sum(self.coins) == 0 # all coins collected
        info = {"timestep": self.current_step,
                "coins": self.coins}
        return ob, reward, done, info

    def reset(self):
        self.current_loc = 0
        self.current_step = 0
        self.memory = 0
        self.coins = [1] * self.num_goals  # 1 means coin is still there, 0 means collected
        ob = (self.current_loc, self.memory, 0)
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
        print("Memory:", self.memory)
        print("-----")
        print()
        if self._coin_present():
            print("Coin Present")

    def close(self):
        pass
