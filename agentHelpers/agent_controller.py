from typing import Any, Dict, Union
import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
from agentHelpers.agent_act_space import unit_action_space, fact_action_space
from gym import spaces, Space
import flax.linen as nn
from lux.kit import GameState
import jax

KEY = jax.random.PRNGKey(0)

class Controller:
    def __init__(self, action_space) -> None:
        self.action_space = action_space

class OverallController:
    def __init__(self, env_cfg) -> None:
        self.env_cfg = env_cfg

    def convert_output_to_actions(
            self, 
            player,
            gameState,
            to_change : jnp.array,
            unit_actions_logits : jnp.array,
            fact_actions_logits : jnp.array,
            unit_actions_disc_params_R : jnp.array,
            unit_actions_disc_params_Rep : jnp.array,
            unit_actions_disc_params_N : jnp.array,
            facts_exist,
            units_exist
        ) -> Dict[str, Union[int, npt.NDArray]]:

        to_change = to_change > 0.5
        actions = {}
        act_probs= {}

        for i in range(len(facts_exist)):
            if facts_exist[i]:
                fact_id = "factory_" + str(i) 
                actions[fact_id] = jax.random.categorical(KEY, logits=fact_actions_logits[i])
                act_probs[fact_id] = fact_actions_logits[i, actions[fact_id]]

        for i in range(len(units_exist)):
            if units_exist[i]:
                unit_id = "unit_" + str(i) 
                if to_change[i]:
                    actions[unit_id], act_probs[unit_id] = self.get_action_queue(unit_actions_logits[i], unit_actions_disc_params_R[i], unit_actions_disc_params_N[i], unit_actions_disc_params_Rep[i])

        return actions, act_probs

    def get_action_queue(self, unit_action:jnp.array, R_val:jnp.array, N_val:jnp.array, Rep_val: jnp.array) -> npt.NDArray:
        output = np.zeros((20, 6))

        action_list = jax.random.categorical(KEY, unit_action)
        R_val_list = jax.random.categorical(KEY, R_val)
        N_val_list = jax.random.categorical(KEY, N_val)
        Rep_val_list = jax.random.categorical(KEY, Rep_val)
        prob_lists = []
        output[:, 3] = 100
        for i, action in enumerate(action_list):
            if action < 5:
                output[i, 0] = 0
                output[i, 1] = action
            elif action < 9:
                output[i, 0] = 1
                output[i, 1] = action - 4
            else:
                output[i, 0] = action - 7
            output[i, 2] = R_val_list[i]
            output[i, 4] = N_val_list[i] + 1
            output[i, 5] = Rep_val_list[i] + 1
            prob_lists.append([unit_action[i, action], R_val[i, R_val_list[i]], N_val[i, N_val_list[i]], Rep_val[i, Rep_val_list[i]]])
        return output, jnp.array(prob_lists)
