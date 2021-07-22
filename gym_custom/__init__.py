from gym.envs.registration import register
from .envs import ButtonEnv, SelfReinforcingBandit, POMDPBandit, IteratedSelfPD

register(
    id='button-v0',
    entry_point='gym_custom.envs:ButtonEnv',
)

register(
    id='mirror-bandit-v0',
    entry_point='gym_custom.envs:SelfReinforcingBandit',
)

register(
    id='ipd-v0',
    entry_point='gym_custom.envs:IteratedSelfPD',
)
