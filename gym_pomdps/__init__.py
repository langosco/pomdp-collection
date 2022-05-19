from gym.envs.registration import register
from .envs import SelfReinforcingBandit, POMDPBandit, IteratedSelfPD, DoubleTap, AlternatingBandit, SimpleSequentialNav, MultiArmedDoubleTap, ActionSpaceToDiscrete, ObservationSpaceToDiscrete, MultiArmedMTap, AlternatingBanditWithResultObs, GuessBit, FlipBit, TwoPlayerGame, BattleOfSexes, AddExternalMemory

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

register(
    id='sequential-nav-v0',
    entry_point='gym_pomdps.envs:SimpleSequentialNav',
)
