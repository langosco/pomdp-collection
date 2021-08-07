from gym.envs.registration import register
from .envs import SelfReinforcingBandit, POMDPBandit, IteratedSelfPD, TwoFixedPoints, AlternatingBandit

register(
    id='mirror-bandit-v0',
    entry_point='gym_pomdps.envs:SelfReinforcingBandit',
)

register(
    id='ipd-v0',
    entry_point='gym_pomdps.envs:IteratedSelfPD',
)


register(
    id='fixed-points-v0',
    entry_point='gym_pomdps.envs:TwoFixedPoints',
)

register(
    id='alternating-bandit-v0',
    entry_point='gym_pomdps.envs:AlternatingBandit',
)
