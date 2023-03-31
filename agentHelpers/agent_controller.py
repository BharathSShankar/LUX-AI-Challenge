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
    def __init__(self, action_space: Space.Space) -> None:
        self.action_space = action_space

class OverallController(Controller):
    def __init__(self, env_cfg) -> None:
        super(spaces.Tuple(unit_action_space, fact_action_space))
        self.env_cfg = env_cfg

    def convert_output_to_actions(
            self, 
            player:str,
            gameState: GameState,
            to_change:jnp.array, 
            unit_actions:jnp.array, 
            fact_actions:jnp.array,

        ) -> Dict[str, Union[int, npt.NDArray]]:

        to_change = to_change > 0.5
        actions = {}

        for i, fact_id in enumerate(sorted(gameState.factories[player])):
            actions[fact_id] = jnp.argmax(fact_actions[i])

        for i, unit_id in enumerate(sorted(gameState.units[player])):
            if to_change[i]:
                actions[unit_id] = self.get_action_queue(unit_actions[i])

        return actions

    def get_action_queue(self, unit_action:jnp.array, R_val, cont_mean:jnp.array, cont_std) -> npt.NDArray:
        output = np.zeros((20, 6))

        action_list = jax.random.categorical(KEY, unit_action)
        R_val_list = jax.random.categorical(KEY, R_val)
        cont_vals = cont_mean + jax.random.normal(KEY, shape = (20,)) * cont_std
        cont_vals = jax.clip(cont_vals, 0, 0.9)
        n_r_vals = (cont_vals * jnp.array([10, 4])).astype(int)
        output[:, 4:] = n_r_vals
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
            
        return output
