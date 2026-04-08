from fastapi import FastAPI
from pydantic import BaseModel
from env import MessageEnv, Action
from data import messages

app = FastAPI()

env = MessageEnv(messages)

class ActionRequest(BaseModel):
    label: int


@app.post("/reset")
def reset():
    state = env.reset()

    return {
        "observation": {
            "message": state.message
        }
    }


@app.post("/step")
def step(action: ActionRequest):
    act = Action(label=action.label)
    next_state, reward, done, info = env.step(act)

    return {
        "observation": {
            "message": next_state.message
        },
        "reward": reward,
        "done": done,
        "info": info
    }
