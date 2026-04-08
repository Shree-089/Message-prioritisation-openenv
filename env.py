from typing import Tuple, Dict, Any
from pydantic import BaseModel, Field

# OBSERVATION MODEL
class Observation(BaseModel):
    message: str = Field(..., description="Message text to classify")

# ACTION MODEL
class Action(BaseModel):
    label: int = Field(..., description="0 = not important, 1 = important")

# REWARD MODEL
class Reward(BaseModel):
    value: float = Field(..., description="Reward score")


class MessageEnv:
    def __init__(self, messages):
        self.messages = messages
        self.index = 0

    # RESET
    def reset(self) -> Observation:
        self.index = 0
        return self._get_observation()

    # STATE
    def state(self) -> Observation:
        return self._get_observation()

    # INTERNAL OBSERVATION
    def _get_observation(self) -> Observation:
        msg, _ = self.messages[self.index]
        return Observation(message=msg)

    # STEP FUNCTION (FIXED)
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:

        if action.label not in [0, 1]:
            raise ValueError("Invalid action")

        msg, true_label = self.messages[self.index]

        # reward logic
        reward_value = 1.0 if action.label == true_label else -1.0

        self.index += 1
        done = self.index >= len(self.messages)

        # ✅ FIX: NEVER return None
        if done:
            # return last valid observation again
            next_obs = Observation(message=msg)
        else:
            next_obs = self._get_observation()

        return next_obs, reward_value, done, {}
