#!/usr/bin/env python3
"""
validate.py - Pre-submission validator for FormFillEnv
Run this before submitting to catch compliance issues early.

Usage: python validate.py
"""
import os
import sys
import json
import subprocess
import importlib
import yaml

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors = 0


def check(label: str, condition: bool, fail_msg: str = "", warn: bool = False):
    global errors
    tag = WARN if warn else (PASS if condition else FAIL)
    if not condition:
        if not warn:
            errors += 1
        print(f"{tag} {label}" + (f": {fail_msg}" if fail_msg else ""))
    else:
        print(f"{tag} {label}")


# ── 1. File structure ────────────────────────────────────────────────────────
print("\n=== File Structure ===")
required_files = [
    "inference.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "openenv.yaml",
    "README.md",
    "env/__init__.py",
    "env/environment.py",
    "env/tasks.py",
    "env/grader.py",
]
for f in required_files:
    check(f"File exists: {f}", os.path.exists(f))


# ── 2. openenv.yaml ──────────────────────────────────────────────────────────
print("\n=== openenv.yaml ===")
try:
    with open("openenv.yaml") as fh:
        spec = yaml.safe_load(fh)
    check("openenv.yaml is valid YAML", True)
    check("Has 'name' field", "name" in spec)
    check("Has 'tasks' field", "tasks" in spec)
    check("Has 3+ tasks", len(spec.get("tasks", [])) >= 3)
    check("Has 'grader' field", "grader" in spec)
    check("Has 'inference' field", "inference" in spec)
    check("inference.script = inference.py", spec.get("inference", {}).get("script") == "inference.py")
    for var in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]:
        check(f"Env var '{var}' declared in openenv.yaml",
              var in spec.get("inference", {}).get("required_env_vars", []))
except Exception as e:
    check("openenv.yaml parseable", False, str(e))


# ── 3. inference.py structure ────────────────────────────────────────────────
print("\n=== inference.py ===")
try:
    with open("inference.py") as fh:
        src = fh.read()
    check("Imports OpenAI client", "from openai import OpenAI" in src)
    check("Defines API_BASE_URL", 'API_BASE_URL = os.getenv("API_BASE_URL"' in src)
    check("Defines MODEL_NAME", 'MODEL_NAME = os.getenv("MODEL_NAME"' in src)
    check("Defines HF_TOKEN", 'HF_TOKEN = os.getenv("HF_TOKEN")' in src)
    check("Prints [START]", '"[START]"' in src or "'[START]'" in src)
    check("Prints [STEP]", '"[STEP]"' in src or "'[STEP]'" in src or "[STEP]" in src)
    check("Prints [END]", '"[END]"' in src or "'[END]'" in src)
    check("Uses grade_submission", "grade_submission" in src)
    check("Iterates over TASKS", "TASKS" in src)
except Exception as e:
    check("inference.py readable", False, str(e))


# ── 4. Run inference.py ──────────────────────────────────────────────────────
print("\n=== Inference Execution ===")
try:
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True, text=True, timeout=60
    )
    output = result.stdout.strip()
    check("inference.py exits with code 0", result.returncode == 0,
          result.stderr[:200] if result.stderr else "")

    lines = output.splitlines()
    check("Output starts with [START]", lines[0] == "[START]" if lines else False)
    check("Output ends with [END]", lines[-1] == "[END]" if lines else False)

    step_lines = [l for l in lines if l.startswith("[STEP]")]
    check("Has 3 [STEP] lines (one per task)", len(step_lines) == 3,
          f"Found {len(step_lines)}")

    for line in step_lines:
        has_task_id = "task_id=" in line
        has_reward = "reward=" in line
        has_score = "score=" in line
        check(f"Line has task_id, reward, score: {line[:60]}",
              has_task_id and has_reward and has_score)
        # Check score in [0,1]
        try:
            score_str = line.split("score=")[1].strip()
            score = float(score_str)
            check(f"Score in [0.0, 1.0] for line", 0.0 <= score <= 1.0,
                  f"score={score}")
        except Exception:
            check("Score parseable as float", False, line)

except subprocess.TimeoutExpired:
    check("inference.py completes in time", False, "Timed out after 60s")
except Exception as e:
    check("inference.py runnable", False, str(e))


# ── 5. FastAPI app ───────────────────────────────────────────────────────────
print("\n=== app.py (API endpoints) ===")
try:
    with open("app.py") as fh:
        app_src = fh.read()
    check("GET / endpoint defined", '@app.get("/")' in app_src)
    check("POST /reset endpoint defined", '@app.post("/reset")' in app_src)
    check("POST /step endpoint defined", '@app.post("/step")' in app_src)
    check("GET /state endpoint defined", '@app.get("/state")' in app_src)
except Exception as e:
    check("app.py readable", False, str(e))


# ── 6. Dockerfile ────────────────────────────────────────────────────────────
print("\n=== Dockerfile ===")
try:
    with open("Dockerfile") as fh:
        df = fh.read()
    check("Exposes port 7860", "7860" in df)
    check("Copies env/ directory", "COPY env/" in df)
    check("Copies inference.py", "COPY inference.py" in df or "COPY . ." in df)
    check("Has CMD or ENTRYPOINT", "CMD" in df or "ENTRYPOINT" in df)
except Exception as e:
    check("Dockerfile readable", False, str(e))


# ── 7. README HF Space header ────────────────────────────────────────────────
print("\n=== README.md ===")
try:
    with open("README.md") as fh:
        readme = fh.read()
    check("Has HF Space YAML header (---)", readme.startswith("---"))
    check("Has sdk: docker", "sdk: docker" in readme)
    check("Has title field", "title:" in readme)
except Exception as e:
    check("README.md readable", False, str(e))


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
if errors == 0:
    print("ALL CHECKS PASSED. Ready to submit.")
else:
    print(f"{errors} CHECK(S) FAILED. Fix before submitting.")
sys.exit(0 if errors == 0 else 1)
