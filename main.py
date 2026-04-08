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
    return {"message": state.message}

@app.post("/step")
def step(action: ActionRequest):
    act = Action(label=action.label)
    next_state, reward, done, _ = env.step(act)

    return {
        "message": next_state.message,
        "reward": reward,
        "done": done
    }
