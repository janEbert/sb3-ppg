# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPG
from stable_baselines3.common.policies import register_policy

from .aux_ac_policy import AuxActorCriticPolicy

AuxMlpPolicy = AuxActorCriticPolicy

register_policy("AuxMlpPolicy", AuxActorCriticPolicy)
