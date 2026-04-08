"""
FormFillEnv - FastAPI server for OpenEnv compliance.
Exposes /reset, /step, /state endpoints required by the validator.
Also serves GET / for the HF Space ping check (must return 200).
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from env.environment import FormEnv, Action, Observation
from env.tasks import TASKS

app = FastAPI(
    title="FormFillEnv",
    description="OpenEnv-compatible form-filling RL environment",
    version="1.0.0",
)

# In-memory session store: task_id -> FormEnv instance
_sessions: Dict[str, FormEnv] = {}


# ── Request / Response models ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_1_easy"


class StepRequest(BaseModel):
    task_id: str
    action: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check - must return 200 for HF Space ping."""
    return {
        "status": "ok",
        "env": "FormFillEnv",
        "tasks": [t.task_id for t in TASKS],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    """Reset the environment for a given task. Returns initial observation."""
    try:
        env = FormEnv(task_id=req.task_id)
        obs = env.reset()
        _sessions[req.task_id] = env
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/step")
def step(req: StepRequest) -> StepResponse:
    """Take one step in the environment."""
    env = _sessions.get(req.task_id)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active session for task_id='{req.task_id}'. Call /reset first.",
        )
    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    obs, reward, done, info = env.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def state(task_id: str = "task_1_easy") -> Dict[str, Any]:
    """Return current state of the environment without stepping."""
    env = _sessions.get(task_id)
    if env is None:
        # Auto-init if not started
        env = FormEnv(task_id=task_id)
        env.reset()
        _sessions[task_id] = env
    return env.state().model_dump()


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    return [
        {
            "task_id": t.task_id,
            "form_type": t.form_type,
            "difficulty": t.difficulty,
            "required_fields": t.required_fields,
            "max_steps": t.max_steps,
        }
        for t in TASKS
    ]


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
