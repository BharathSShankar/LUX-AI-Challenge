import jax.numpy as jnp

from gym import ObservationWrapper, Env, RewardWrapper, ActionWrapper
from typing import Dict, Union
from lux.kit import GameState

from agentHelpers.agent_obs_processing import fact_2_vec, global_2_vec, img_2_vec, unit_2_vec

class LuxObsWrapper(ObservationWrapper):

    def __init__(self, env: Env, player: str) -> None:
        super().__init__(env)
        self.env = env
        self.player = player
    
    def observation(self, obs: GameState) -> Dict[str, Union[jnp.array, Dict]]:
        obs_dict = {}
        obs_dict["GIV"] = global_2_vec(obs, self.player)
        obs_dict["IMG"] = img_2_vec(obs, self.player)

        obs_dict["FACT"] = {}
        for fact in obs.factories[self.player]:
            obs_dict["FACT"][fact] = fact_2_vec(obs, self.player, fact)
        
        obs_dict["UNIT"] = {}
        for unit in obs.units[self.player]:
            obs_dict["UNIT"][unit] = unit_2_vec(obs, self.player, fact) 
        
        return obs_dict

class LuxRewardWrapper(RewardWrapper):

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.env = env
    
    def reward(self, reward: float) -> float:
        pass