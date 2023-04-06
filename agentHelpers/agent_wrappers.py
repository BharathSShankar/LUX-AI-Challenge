import jax.numpy as jnp
import jax
import jux
import gym
from jux.env import JuxEnv, JuxAction
from gym import ObservationWrapper, Env, RewardWrapper, ActionWrapper
from typing import Dict, Union

import numpy as np
from agentHelpers.agent_constants import LIC_WT
from lux.kit import GameState, obs_to_game_state

from agentHelpers.agent_obs_processing import fact_2_vec, global_2_vec, img_2_vec, unit_2_vec

#TODO: add Support for vetorized envs

class JuxWrapperEnv:
    def __init__(self, env, rew_weights):
        self.env = env
        self.reward_weights = rew_weights

    def step(self, step, action, state):
        if step == 0:
            bid, faction = jux.actions.bid_action_from_lux(action)
            new_state, (obs, rewards, dones, infos) = self.env.step_bid(
                state, bid, faction)
        elif step < 0:
            spawn, water, metal = jux.actions.factory_placement_action_from_lux(
                action)
            new_state, (obs, rewards, dones, infos) = self.env.step_factory_placement(
                state, spawn, water, metal)
        else:
            jux_act = JuxAction.from_lux(state, action)
            new_state, (obs, rewards, dones, infos) = self.env.step_late_game(
                state, jux_act)

        new_state_mod = new_state.to_lux()

        reward_p0 = LIC_WT * (rewards[0] - rewards[1]) + JuxWrapperEnv.get_dense_rewards(
            new_state_mod, "player_0", "player_1", self.reward_weights)
        reward_p0 -= JuxWrapperEnv.get_dense_rewards(
            state.to_lux(), "player_0", "player_1", self.reward_weights)

        reward_p1 = LIC_WT * (rewards[1] - rewards[0]) + JuxWrapperEnv.get_dense_rewards(
            new_state_mod, "player_1", "player_0", self.reward_weights)
        reward_p1 -= JuxWrapperEnv.get_dense_rewards(
            state.to_lux(), "player_1", "player_0", self.reward_weights)
        return new_state, obs, [reward_p0, reward_p1], dones, infos

    @staticmethod
    def convert_obs(obs: GameState, player) -> Dict[str, jnp.array]:
        obs_dict = {}
        obs_dict["GIV"] = global_2_vec(obs, player)
        obs_dict["IMG"] = img_2_vec(obs, player)

        fact_list = np.zeros((40, 8))
        facts_exist = np.zeros((40,))

        for fact in obs.factories[player]:
            num = int(fact.split("_")[-1])
            fact_list[num, :] = fact_2_vec(obs, player, fact)
            facts_exist[num] = 1

        obs_dict["FACT"] = jnp.array(fact_list)
        obs_dict["ACT_FACTS"] = jnp.array(facts_exist)

        unit_list = np.zeros((300, 7))
        units_exist = np.zeros((300,))

        for unit in obs.units[player]:
            num = int(unit.split("_")[-1])
            unit_list[num, :] = unit_2_vec(obs, player, unit)

        obs_dict["UNIT"] = jnp.array(unit_list)
        obs_dict["ACT_UNITS"] = jnp.array(units_exist)

        return obs_dict

    @staticmethod
    def get_dense_rewards(gameState, player, opposition, reward_weights):
        reward_mat = np.zeros([26])
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
                reward_mat[19] += 1
            else:
                reward_mat[20] += 1
            reward_mat[21] += unit.cargo.metal
            reward_mat[22] += unit.cargo.water
            reward_mat[23] += unit.cargo.ore
            reward_mat[24] += unit.cargo.ice
            reward_mat[25] += unit.power

        return jnp.array(reward_mat).dot(reward_weights)
    
    def reset(self, seed):
        new_state = self.env.reset(seed)
        return new_state
