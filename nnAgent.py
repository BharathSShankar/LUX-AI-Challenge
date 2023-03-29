from agentHelpers.agent_controller import OverallController
from agentHelpers.agent_obs_processing import map_2_vec
from agentHelpers.agent_wrappers import JuxWrapperEnv
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys
import os.path as osp

MODEL_FACTORY = "./trainedModels/FactoryPlacementModel"
MODEL_GAME = "./trainedModels/GamePlacementModel"

class nnAgent:

    def __init__(self, player: str, env_cfg: EnvConfig, controller: OverallController) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.controller = controller
        #TODO: Load Models
        directory = osp.dirname(__file__)
        self.factory_policy = PPO.load(osp.join(directory, MODEL_FACTORY))

        directory = osp.dirname(__file__)
        self.game_policy = PPO.load(osp.join(directory, MODEL_GAME))

    
    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        return dict(faction="AlphaStrike", bid=0)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        if my_turn_to_place_factory(self.player == "player_0", step): 
            gameState = obs_to_game_state(step, self.env_cfg, obs)
            map_vec = map_2_vec(gameState)
            map_val = self.factory_policy.apply(self.factory_weights, map_vec)
            map_val[~gameState.board.valid_spawns_mask] = -np.inf
            idx = np.argmax(map_val)
            pos = idx // gameState.env_cfg.map_size, idx % gameState.env_cfg.map_size
            return dict(spawn = pos, metal = 150, water = 150)
        return {}


    def act(self, step: int, obs, remainingOverageTime: int = 60):

        gameState = obs_to_game_state(step, self.env_cfg, obs)
        obs_proc = JuxWrapperEnv.convert_obs(gameState, self.player)

        to_change, unit_actions, fact_actions = self.game_policy.apply(
            self.game_weights, 
            obs_proc["GIV"],
            obs_proc["IMG"],
            obs_proc["FACT"],
            obs_proc["UNIT"]    
        )
        
        actions = self.controller.convert_output_to_actions(
            self.player, gameState, to_change, unit_actions, fact_actions
        )

        return actions