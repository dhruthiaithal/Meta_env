from fastapi import FastAPI, Body
from models import Observation, Action
from env import MedSchedProEnv

app = FastAPI(title="MedSched Pro v2 OpenEnv")

# Instantiate a global environment for state tracking during evaluation
env_instance = MedSchedProEnv()

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/reset", response_model=Observation)
def reset_env(payload: dict = Body({})):
    """Reset the environment state and return initial observation."""
    obs = env_instance.reset()
    return obs

@app.post("/step")
def step_env(action: Action):
    """Execute action and return transition state."""
    obs, reward, done, info = env_instance.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }
