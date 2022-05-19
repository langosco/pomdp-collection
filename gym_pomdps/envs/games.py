import gym
from gym import spaces
import numpy as np
from numpy.random import default_rng

rng = default_rng()

class GameAgent:
    """simple agent that tries to maximize payoff in
    a two-player game"""
    def __init__(self, payoff_matrix, epsilon_noise=0.05):
        """
        args:
            payoff_matrix: a square array of rewards depending
        on enemy_action x self_action. So payoff_matrix[k] 
        gives vector of rewards if the other player plays action k.
            epsilon_noise: agent takes a random action with probability
        epsilon_noise.
        """
        self.payoff_matrix = np.array(payoff_matrix)
        self.epsilon_noise = epsilon_noise
        self.memory = []

    def record_action(self, player_action):
        """add action of opposing player to memory"""
        self.memory.append(player_action)

    def reset(self):
        self.memory = []

    def act(self):
        """take action that works best on average given history"""
        if rng.binomial(1, p=self.epsilon_noise):
            return self.act_randomly()
        else: 
            mean_payoffs = self.payoff_matrix[np.array(self.memory)].mean(axis=0)
            return np.argmax(mean_payoffs)

    def act_randomly(self):
        return rng.integers(0, len(self.payoff_matrix))


class TwoPlayerGame(gym.Env):
    """A basic 'player vs game' environment for any 
    two-player game with a finite payoff matrix. Note this
    is not a multi-agent environment, as the second player
    is considered part of the environment."""
    def __init__(self, burn_in, game_payoff_matrix, player_payoff_matrix, 
            episode_len=40, epsilon_noise=0.05):
        self.game_payoff_matrix = np.array(game_payoff_matrix)
        self.player_payoff_matrix = np.array(player_payoff_matrix)
        self.game_agent = GameAgent(game_payoff_matrix, epsilon_noise)
        self.burn_in = burn_in

        self.action_space = spaces.Discrete(len(player_payoff_matrix))
        self.observation_space = spaces.Discrete(1)
        self.dummy_ob = 0
        self.episode_len = episode_len
        self.reset()

    def game_act(self):
        """return game agent action"""

        if self.current_step <= self.burn_in:
            return self.game_agent.act_randomly()
        else:
            return self.game_agent.act()

    def step(self, action):
        self.current_step += 1
        game_action = self.game_act()
        rew = self.player_payoff_matrix[action, game_action]
        done = self.current_step >= self.episode_len
        info = {"step": self.current_step}
        self.game_agent.record_action(action)
        return self.dummy_ob, rew, done, info
    
    def reset(self):
        self.current_step = 0
        self.game_agent.reset()
        return self.dummy_ob


class BattleOfSexes(TwoPlayerGame):
    def __init__(self, episode_len=40, burn_in=1, num_actions=2):
        if num_actions == 2:
            game_payoff_matrix = [[0, 1],
                                  [2, 0]]
            player_payoff_matrix = [[0, 2],
                                    [1, 0]]
        elif num_actions > 2:
            returns = np.arange(num_actions) + 1
            game_payoff_matrix = np.rot90(np.flip(np.diag(returns)))
            player_payoff_matrix = np.rot90(np.diag(returns))
        super().__init__(burn_in, game_payoff_matrix, player_payoff_matrix, episode_len)

