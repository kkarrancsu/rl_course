from gym.envs.registration import register

register(
    id='sorty-v0',
    entry_point='sorty.envs:Sorty',
)
