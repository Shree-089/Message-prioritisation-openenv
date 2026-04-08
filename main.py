from fastapi import FastAPI
from fastapi.responses import JSONResponse
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
    return JSONResponse(content={
        "observation": {"message": state.message}
    })

@app.get("/reset")
def reset_get():
    state = env.reset()
    return JSONResponse(content={
        "observation": {"message": state.message}
    })

@app.get("/state")
def state():
    obs = env.state()
    return JSONResponse(content={
        "observation": {"message": obs.message}
    })

@app.post("/step")
def step(action: ActionRequest):
    act = Action(label=action.label)
    next_state, reward, done, info = env.step(act)
    return JSONResponse(content={
        "observation": {"message": next_state.message},
        "reward": reward,
        "done": done,
        "info": info
    })

@app.get("/health")
def health():
    return {"status": "ok"}
