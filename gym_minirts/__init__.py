from gym.envs.registration import register
from gym_minirts.main import MiniRTSEnv

register(id='minirts-v0', entry_point='gym_minirts:MiniRTSEnv')
