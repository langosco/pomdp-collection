# Environments

* *Alternating bandit* aka *bandit-and-sign*
  * A non-markovian one-armed bandit environment. Bandit return depends of whether timestep is even or odd.
  * Agent can in principle keep track of this by flipping a red/green sign (which it has the option to do every turn). But this may be hard to learn.
* *Double-tap*
  * This env consists of two (or $n$) buttons (or arms, whatever). At every timestep, the agent may push one of them. If pushed twice in a row, the agent gets reward $r_k$ depending on which button $k$ was pushed twice. If pushed just once, every arm returns zero reward.
* *Iterated prisoner's dilemma*
  * At every timestep, agent can choose whether to cooperate or defect against itself. Payout is according to a PD payoff table wrt the agents decision in the previous timestep.
  * Usually a policy would learn to always cooperate. However when the discount factor is $\gamma = 0$ it should always defect.
  * Under some circumstances, $\varepsilon$-greedy Q-learning ends up cooperating even when $\gamma = 0$.
* *Self-reinforcing bandit*
  * A non-markovian multi-armed bandit. Return of arm $k$ grows proportional to nr of times that arm has been pushed in this episode.
* *Sequential navigation*
  * Navigation task: visit n states in sequence. All states are reachable from on single central state.
  * A memoryless policy struggles, because it cannot remember which states it has already visited.
  * The env includes 'external' memory that a policy can use by writing / reading from it via actions / observations.

