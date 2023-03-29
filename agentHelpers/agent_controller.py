from typing import Any, Dict, Union
import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
from agentHelpers.agent_act_space import unit_action_space, fact_action_space
from gym import spaces, Space
import flax.linen as nn
from lux.kit import GameState


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

    def get_action_queue(self, unit_action:jnp.array) -> npt.NDArray:
        output = np.zeros((20, 6))

        output[:, 0] = jnp.argmax(unit_action[:, :6], axis=1)
        output[:, 1] = jnp.argmax(unit_action[:, 6:11], axis = 1)
        output[:, 2] = jnp.argmax(unit_action[:, 11:16], axis = 1)
        output[:, 3] = self.env_cfg.max_transfer_amount * (nn.sigmoid(unit_action[:, 16]) > 0.5)
        output[:, 4] = jnp.round(unit_action[:, 17])
        output[:, 5] = jnp.round(unit_action[:, 18])

        return output
