from gym import ObservationWrapper, Env, RewardWrapper, ActionWrapper
from typing import Dict, Union

class LuxObsWrapper(ObservationWrapper):

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.env = env
    
    def observation(self, observation: Dict[str, Union[str, Dict]]) -> Dict[str, Union[str, Dict]]:
        return super().observation(observation)

    @staticmethod
    def process_obs(self, observation):
        procObs = {}

class LuxRewardWrapper(RewardWrapper):

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.env = env
    
    def reward(self, reward: float) -> float:
        return super().reward(reward)

class LuxActWrapper(ActionWrapper):

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.env = env
    
    def action(self, action):
        return super().action(action)