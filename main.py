from openenv import Environment, Action as BaseAction, Observation as BaseObservation
from pydantic import Field
from typing import Tuple, Dict, Any

class Observation(BaseObservation):
    message: str = Field(..., description="Message text to classify")

class Action(BaseAction):
    label: int = Field(..., description="0 = not important, 1 = important")

class MessageEnv(Environment):
    def __init__(self, messages):
        super().__init__()
        self.messages = messages
        self.index = 0

    def reset(self) -> Observation:
        self.index = 0
        return self._get_observation()

    def state(self) -> Observation:
        return self._get_observation()

    def _get_observation(self) -> Observation:
        msg, _ = self.messages[self.index]
        return Observation(message=msg)

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if action.label not in [0, 1]:
            raise ValueError("Invalid action")
        msg, true_label = self.messages[self.index]
        reward_value = 1.0 if action.label == true_label else -1.0
        self.index += 1
        done = self.index >= len(self.messages)
        next_obs = Observation(message=msg) if done else self._get_observation()
        return next_obs, reward_value, done, {}
