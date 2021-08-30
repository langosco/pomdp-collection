from gym.envs.registration import register
from .envs import SelfReinforcingBandit, POMDPBandit, IteratedSelfPD, DoubleTap, AlternatingBandit

register(
    id='mirror-bandit-v0',
    entry_point='gym_pomdps.envs:SelfReinforcingBandit',
)

register(
    id='ipd-v0',
    entry_point='gym_pomdps.envs:IteratedSelfPD',
)


register(
    id='double-tap-v0',
    entry_point='gym_pomdps.envs:DoubleTap',
)

register(
    id='alternating-bandit-v0',
    entry_point='gym_pomdps.envs:AlternatingBandit',
)
