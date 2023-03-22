from typing import Any, Dict
import numpy.typing as npt
from agentHelpers.agent_act_space import unit_action_space, fact_action_space
from gym import spaces, Space


class Controller:
    def __init__(self, action_space: Space.Space) -> None:
        self.action_space = action_space

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Generates a boolean action mask indicating in each discrete dimension whether it would be valid or not
        """
        raise NotImplementedError()

class OverallController(Controller):
    def __init__(self, env_cfg) -> None:
        super(spaces.Tuple(unit_action_space, fact_action_space))
        self.env_cfg = env_cfg
        
    def action_to_lux_action(self, agent: str, obs: Dict[str, Any], action: npt.NDArray):
        pass



