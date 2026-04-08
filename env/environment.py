# env/environment.py
from __future__ import annotations

from typing import Tuple, Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .tasks import get_task, TaskProfile
import re
from datetime import datetime


class Observation(BaseModel):
    form_type: str
    required_fields: List[str]
    filled_fields: Dict[str, str]
    missing_fields: List[str]
    errors: List[str]
    user_profile: Dict[str, str]
    step_count: int


class Action(BaseModel):
    action_type: str = Field(
        ...,
        description="One of: map_field, request_missing, infer_field, validate_field, submit_form",
    )
    field_name: Optional[str] = None
    value: Optional[str] = None


def normalize_dob(raw: str) -> Optional[str]:
    # Accept formats like "15/08/05", "15/08/2005", "15th Aug 1998"
    raw = raw.strip()
    # 15/08/05 or 15-08-05
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", raw)
    if m:
        d, mth, y = m.groups()
        if len(y) == 2:
            # Assume 19xx vs 20xx by threshold
            y_int = int(y)
            y = f"20{y}" if y_int <= 30 else f"19{y}"
        try:
            dt = datetime(int(y), int(mth), int(d))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
    # 15th Aug 1998
    try:
        dt = datetime.strptime(raw.replace("th", "").replace("st", "").replace("nd", "").replace("rd", ""), "%d %b %Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    try:
        dt = datetime.strptime(raw, "%d %B %Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def normalize_blood_group(raw: str) -> Optional[str]:
    r = raw.strip().upper()
    mapping = {
        "O POSITIVE": "O+",
        "O+": "O+",
        "A POSITIVE": "A+",
        "A+": "A+",
        "B POSITIVE": "B+",
        "B+": "B+",
        "AB POSITIVE": "AB+",
        "AB+": "AB+",
    }
    return mapping.get(r)


def normalize_address(line: str, city: str, pincode: str) -> str:
    return f"{line}, {city} - {pincode}"


def infer_age_from_birth_year(birth_year: str, current_year: int = 2026) -> Optional[str]:
    try:
        y = int(birth_year)
        if 1900 <= y <= current_year:
            return str(current_year - y)
        return None
    except ValueError:
        return None


class FormEnv:
    def __init__(self, task_id: str):
        self.task_profile: TaskProfile = get_task(task_id)
        self.max_steps: int = self.task_profile.max_steps
        self.reset()

    def reset(self) -> Observation:
        self.filled_fields: Dict[str, str] = {}
        self.errors: List[str] = []
        self.step_count = 0
        self.done = False
        self.action_history: List[str] = []
        return self.state()

    def state(self) -> Observation:
        missing = [f for f in self.task_profile.required_fields if f not in self.filled_fields]
        return Observation(
            form_type=self.task_profile.form_type,
            required_fields=self.task_profile.required_fields,
            filled_fields=self.filled_fields.copy(),
            missing_fields=missing,
            errors=self.errors.copy(),
            user_profile=self.task_profile.user_profile.copy(),
            step_count=self.step_count,
        )

    def _is_redundant_action(self, action: Action) -> bool:
        key = f"{action.action_type}:{action.field_name}:{action.value}"
        if key in self.action_history:
            return True
        self.action_history.append(key)
        return False

    def _validate_field_format(self, field_name: str, value: str) -> bool:
        if field_name == "dob":
            return normalize_dob(value) is not None
        if field_name == "pan_number":
            return bool(re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", value))
        if field_name == "aadhaar_number":
            return bool(re.match(r"^[0-9]{12}$", value))
        if field_name == "age":
            try:
                age_int = int(value)
                return 0 < age_int < 120
            except ValueError:
                return False
        if field_name == "blood_group":
            return normalize_blood_group(value) is not None
        if field_name == "emergency_contact":
            return bool(re.match(r"^[0-9]{10}$", value))
        if field_name == "consent_given":
            return value.lower() in {"yes", "no"}
        return True

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.done:
            return self.state(), 0.0, True, {"msg": "Episode already done."}

        self.step_count += 1
        reward = -0.05  # base step penalty
        info: Dict[str, Any] = {}
        self.errors.clear()

        gt = self.task_profile.ground_truth

        redundant = self._is_redundant_action(action)
        if redundant:
            reward -= 0.1  # penalty for repeating exact same action

        if action.action_type == "map_field":
            if not action.field_name:
                reward -= 0.2
                self.errors.append("map_field requires field_name.")
            else:
                if action.field_name not in self.task_profile.required_fields:
                    reward -= 0.2
                    self.errors.append(f"{action.field_name} is not a required field.")
                else:
                    if action.value is None:
                        reward -= 0.2
                        self.errors.append("map_field requires value.")
                    else:
                        if not self._validate_field_format(action.field_name, action.value):
                            reward -= 0.3
                            self.errors.append(f"Invalid format for {action.field_name}.")
                        else:
                            self.filled_fields[action.field_name] = action.value
                            if gt.get(action.field_name) == action.value:
                                reward += 0.3
                            else:
                                reward -= 0.4
                                self.errors.append(f"Incorrect mapping for {action.field_name}.")

        elif action.action_type == "infer_field":
            if not action.field_name:
                reward -= 0.2
                self.errors.append("infer_field requires field_name.")
            else:
                # infer using task-specific logic
                inferred: Optional[str] = None
                if action.field_name == "dob":
                    raw = self.task_profile.user_profile.get("dob_raw") or self.task_profile.user_profile.get(
                        "dob_free_text"
                    )
                    if raw:
                        inferred = normalize_dob(raw)
                elif action.field_name == "age":
                    birth_year = self.task_profile.user_profile.get("birth_year")
                    if birth_year:
                        inferred = infer_age_from_birth_year(birth_year)
                elif action.field_name == "blood_group":
                    raw = self.task_profile.user_profile.get("blood")
                    if raw:
                        inferred = normalize_blood_group(raw)
                elif action.field_name == "address":
                    line = self.task_profile.user_profile.get("address_line")
                    city = self.task_profile.user_profile.get("city")
                    pincode = self.task_profile.user_profile.get("pincode")
                    if line and city and pincode:
                        inferred = normalize_address(line, city, pincode)
                elif action.field_name == "full_name":
                    # choose canonical full name over alias
                    inferred = (
                        self.task_profile.user_profile.get("candidate_name")
                        or self.task_profile.user_profile.get("name_on_pan")
                        or self.task_profile.user_profile.get("patient")
                    )
                elif action.field_name == "patient_name":
                    inferred = self.task_profile.user_profile.get("patient")
                elif action.field_name == "emergency_contact":
                    inferred = self.task_profile.user_profile.get("phone_primary")
                elif action.field_name == "consent_given":
                    raw = self.task_profile.user_profile.get("consent_checkbox")
                    if isinstance(raw, str) and raw.lower() == "true":
                        inferred = "yes"

                if inferred is None:
                    reward -= 0.2
                    self.errors.append(f"Could not infer {action.field_name}. Should request instead.")
                else:
                    # validating inferred value
                    if not self._validate_field_format(action.field_name, inferred):
                        reward -= 0.3
                        self.errors.append(f"Inferred invalid format for {action.field_name}.")
                    else:
                        self.filled_fields[action.field_name] = inferred
                        if gt.get(action.field_name) == inferred:
                            reward += 0.4  # correct inference
                        else:
                            reward -= 0.4  # wrong inference
                            self.errors.append(f"Incorrect inference for {action.field_name}.")

        elif action.action_type == "request_missing":
            if not action.field_name:
                reward -= 0.2
                self.errors.append("request_missing requires field_name.")
            else:
                if action.field_name not in self.task_profile.required_fields:
                    reward -= 0.2
                    self.errors.append(f"request_missing on non-required field {action.field_name}.")
                elif action.field_name in self.filled_fields:
                    reward -= 0.2
                    self.errors.append(f"request_missing on already filled field {action.field_name}.")
                else:
                    if gt.get(action.field_name) == "REQUEST_NEEDED":
                        reward += 0.2  # correct decision to request
                        # we simulate that the user will eventually provide this later; we don't actually fill it here
                    else:
                        reward -= -0.2  # unnecessary request
                        self.errors.append(f"Unnecessary request for {action.field_name}; could be inferred or already known.")

        elif action.action_type == "validate_field":
            if not action.field_name:
                reward -= 0.2
                self.errors.append("validate_field requires field_name.")
            else:
                if action.field_name not in self.filled_fields:
                    reward -= 0.2
                    self.errors.append(f"Cannot validate empty field {action.field_name}.")
                else:
                    value = self.filled_fields[action.field_name]
                    valid_format = self._validate_field_format(action.field_name, value)
                    if not valid_format:
                        reward -= 0.2
                        self.errors.append(f"Validation failed for {action.field_name}, bad format.")
                    else:
                        reward += 0.2

        elif action.action_type == "submit_form":
            missing = [f for f in self.task_profile.required_fields if f not in self.filled_fields]
            format_errors = [
                f
                for f, v in self.filled_fields.items()
                if not self._validate_field_format(f, v)
            ]
            all_correct = all(
                f in self.filled_fields and self.filled_fields[f] == gt.get(f)
                for f in self.task_profile.required_fields
            )

            self.done = True

            if missing:
                reward -= 0.4
                info["msg"] = f"Submitted incomplete form. Missing: {missing}"
            elif format_errors:
                reward -= 0.4
                info["msg"] = f"Submitted with format errors in: {format_errors}"
            else:
                if all_correct:
                    reward += 0.6
                    # bonus for finishing early
                    if self.step_count <= len(self.task_profile.required_fields) + 2:
                        reward += 0.2
                    info["msg"] = "Successfully submitted complete and correct form."
                else:
                    reward -= 0.2
                    info["msg"] = "Submitted complete form but with incorrect values."

        else:
            reward -= 0.3
            self.errors.append(f"Unknown action type {action.action_type}.")

        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            reward -= 0.3
            info["msg"] = "Max steps reached before submission."

        return self.state(), reward, self.done, info
