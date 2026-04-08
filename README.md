---
title: FormFillEnv
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# FormFillEnv

An **OpenEnv-compatible** reinforcement learning environment for AI-driven form filling — targeting India-specific bureaucratic use cases.

## Tasks

| Task ID | Form Type | Difficulty |
|---|---|---|
| `task_1_easy` | College Admission | Easy |
| `task_2_medium` | Bank KYC | Medium |
| `task_3_hard` | Hospital Registration | Hard |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check (returns 200) |
| POST | `/reset` | Reset environment for a task |
| POST | `/step` | Take one action step |
| GET | `/state` | Get current state |
| GET | `/tasks` | List all tasks |

## Quick Start

### Reset
```json
POST /reset
{ "task_id": "task_1_easy" }
```

### Step
```json
POST /step
{
  "task_id": "task_1_easy",
  "action": {
    "action_type": "infer_field",
    "field_name": "full_name"
  }
}
```

## Run Inference Locally

```bash
pip install -r requirements.txt
python inference.py
```

## Run Server Locally

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | (none) | HuggingFace / API key |
