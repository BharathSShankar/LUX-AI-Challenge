import jax.numpy as jnp
import jax
import jux
import gym
from jux.env import JuxEnv, JuxAction
from gym import ObservationWrapper, Env, RewardWrapper, ActionWrapper
from typing import Dict, Union
from lux.kit import GameState, obs_to_game_state

from agentHelpers.agent_obs_processing import fact_2_vec, global_2_vec, img_2_vec, unit_2_vec

class JuxWrapperEnv(gym.Wrapper):
    def __init__(self, env:JuxEnv, rew_weights):
        super.__init__(env)
        self.reward_weights = rew_weights
    
    def step(self, step, action, state):
        if step == 0:
            bid, faction = jux.actions.bid_action_from_lux(action)
            new_state, (obs, rewards, dones, infos) = self.env.step_bid(state, bid, faction)
        elif step < 0:
            spawn, water, metal = jux.actions.factory_placement_action_from_lux(action)
            new_state, (obs, rewards, dones, infos) = self.env.step_factory_placement(state, spawn, water, metal)
        else:
            jux_act = JuxAction.from_lux(state, action)
            new_state, (obs, rewards, dones, infos) = self.env.step_late_game(state, jux_act)

        new_state_mod = new_state.to_lux()

        reward_p0 = rewards + JuxWrapperEnv.get_dense_rewards(new_state_mod, "player_0", "player_1", self.reward_weights)
        reward_p1 = rewards + JuxWrapperEnv.get_dense_rewards(new_state_mod, "player_1", "player_0", self.reward_weights)

        return new_state, reward_p0, reward_p1, new_state_mod
        
    @staticmethod
    def convert_obs(obs:GameState, player) -> Dict[str, jnp.array]:
        obs_dict = {}
        obs_dict["GIV"] = global_2_vec(obs, player)
        obs_dict["IMG"] = img_2_vec(obs, player)

        fact_list = []
        for fact in sorted(obs.factories[player]):
            fact_list.append(fact_2_vec(obs, player, fact))
        obs_dict["FACT"] = jax.lax.concatenate(fact_list, 0)

        unit_list = []
        for unit in sorted(obs.units[player]):
            unit_list.append(unit_2_vec(obs, player, unit))
        obs_dict["UNIT"] = jax.lax.concatenate(unit_list, 0)
        return obs_dict 
    
    @staticmethod
    def get_dense_rewards(gameState, player, opposition, reward_weights):
        reward_mat = jnp.zeros([26])
        for factory in gameState.factories[player].values():
            reward_mat[0] += 1
            reward_mat[1] += factory.cargo.metal
            reward_mat[2] += factory.cargo.water
            reward_mat[3] += factory.cargo.ore
            reward_mat[4] += factory.cargo.ice
            reward_mat[5] += factory.power

        for factory in gameState.factories[opposition].values():
            reward_mat[6] += 1
            reward_mat[7] += factory.cargo.metal
            reward_mat[8] += factory.cargo.water
            reward_mat[9] += factory.cargo.ore
            reward_mat[10] += factory.cargo.ice
            reward_mat[11] += factory.power

        for unit in gameState.units[player].values():
            if unit.unit_type == "LIGHT":
                reward_mat[12] += 1
            else:
                reward_mat[13] += 1
            reward_mat[14] += unit.cargo.metal 
            reward_mat[15] += unit.cargo.water 
            reward_mat[16] += unit.cargo.ore 
            reward_mat[17] += unit.cargo.ice 
            reward_mat[18] += unit.power 
    
        for unit in gameState.units[opposition].values():
            if unit.unit_type == "LIGHT":
                reward_mat[21] += 1
            else:
                reward_mat[22] += 1
            reward_mat[28] += unit.cargo.metal
            reward_mat[29] += unit.cargo.water
            reward_mat[30] += unit.cargo.ore
            reward_mat[31] += unit.cargo.ice
            reward_mat[32] += unit.power
        return reward_mat.dot(reward_weights) 
