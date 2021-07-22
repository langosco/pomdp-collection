from gym.envs.registration import register
from .envs import SelfReinforcingBandit, POMDPBandit, IteratedSelfPD, TwoFixedPoints

register(
    id='mirror-bandit-v0',
    entry_point='gym_custom.envs:SelfReinforcingBandit',
)

register(
    id='ipd-v0',
    entry_point='gym_custom.envs:IteratedSelfPD',
)


register(
    id='fixed-points-v0',
    entry_point='gym_custom.envs:TwoFixedPoints',
)

