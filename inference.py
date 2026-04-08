import os
import json

from openai import OpenAI

from env.environment import FormEnv, Action
from env.grader import grade_submission
from env.tasks import TASKS

# Environment variables - exact format required
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")


def get_mock_action(obs) -> dict:
    """
    Deterministic rule-based fallback agent.
    Does not require any API call.
    """
    filled = obs.filled_fields
    missing = obs.missing_fields
    user = obs.user_profile

    # Try to fill each missing field using known inference rules
    for field in missing:
        if field == "full_name":
            val = user.get("candidate_name") or user.get("name_on_pan") or user.get("patient")
            if val:
                return {"action_type": "infer_field", "field_name": field}

        if field == "patient_name":
            if user.get("patient"):
                return {"action_type": "infer_field", "field_name": field}

        if field == "dob":
            if user.get("dob_raw") or user.get("dob_free_text"):
                return {"action_type": "infer_field", "field_name": field}

        if field == "age":
            if user.get("birth_year"):
                return {"action_type": "infer_field", "field_name": field}

        if field == "blood_group":
            if user.get("blood"):
                return {"action_type": "infer_field", "field_name": field}

        if field == "address":
            if user.get("address_line") and user.get("city") and user.get("pincode"):
                return {"action_type": "infer_field", "field_name": field}

        if field == "emergency_contact":
            if user.get("phone_primary"):
                return {"action_type": "infer_field", "field_name": field}

        if field == "consent_given":
            if user.get("consent_checkbox"):
                return {"action_type": "infer_field", "field_name": field}

        if field == "aadhaar_number":
            return {"action_type": "request_missing", "field_name": field}

        if field == "pan_number":
            val = user.get("pan")
            if val:
                return {"action_type": "map_field", "field_name": field, "value": val}

        if field == "high_school_marks":
            val = user.get("class_12_percentage")
            if val:
                pct = val if val.endswith("%") else val + "%"
                return {"action_type": "map_field", "field_name": field, "value": pct}

        if field == "course_preference":
            val = user.get("course_choice")
            if val:
                return {"action_type": "map_field", "field_name": field, "value": val}

        if field == "symptoms":
            val = user.get("issues")
            if val:
                return {"action_type": "map_field", "field_name": field, "value": val}

        # Fallback: request the field
        return {"action_type": "request_missing", "field_name": field}

    # All fields filled - submit
    return {"action_type": "submit_form"}


def try_api_action(client, obs) -> dict:
    """
    Attempt to get action from API. Returns None on any failure.
    """
    system_prompt = (
        "You are an intelligent form-filling agent.\n"
        "Your goal:\n"
        "- Fill required fields correctly\n"
        "- Decide when to infer vs request vs map\n"
        "- Avoid unnecessary actions\n"
        "- Ensure validation before submission\n\n"
        "Available actions:\n"
        "- map_field(field_name, value)\n"
        "- infer_field(field_name)\n"
        "- request_missing(field_name)\n"
        "- validate_field(field_name)\n"
        "- submit_form()\n\n"
        "IMPORTANT:\n"
        "- Output ONLY valid JSON\n"
        "- No explanation text\n"
    )
    user_prompt = f"Current State:\n{obs.model_dump_json()}\n\nChoose the best next action."

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)


def run_inference():
    # Initialize client (may fail silently if no key)
    use_api = bool(HF_TOKEN)
    client = None
    if use_api:
        try:
            client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        except Exception:
            client = None

    print("[START]")

    for task in TASKS:
        env = FormEnv(task_id=task.task_id)
        obs = env.reset()

        done = False
        total_reward = 0.0
        action_count = 0

        while not done:
            action_count += 1

            # Try API first; fall back to mock
            action_data = None
            if client is not None:
                try:
                    action_data = try_api_action(client, obs)
                except Exception:
                    action_data = None

            if action_data is None:
                action_data = get_mock_action(obs)

            try:
                action = Action(**action_data)
            except Exception:
                action = Action(action_type="submit_form")

            obs, reward, done, info = env.step(action)
            total_reward += reward

        score = grade_submission(
            task_id=task.task_id,
            final_state=obs,
            total_reward=total_reward,
            action_count=action_count,
        )

        print(f"[STEP] task_id={task.task_id} reward={round(total_reward, 4)} score={score}")

    print("[END]")


if __name__ == "__main__":
    run_inference()
