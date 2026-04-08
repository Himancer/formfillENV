# env/tasks.py
from typing import Dict, List
from pydantic import BaseModel


class TaskProfile(BaseModel):
    task_id: str
    form_type: str
    required_fields: List[str]
    user_profile: Dict[str, str]
    ground_truth: Dict[str, str]
    difficulty: str
    max_steps: int = 12


TASKS: List[TaskProfile] = [
    # Task 1: College Admission (Easy with ambiguity + normalization)
    TaskProfile(
        task_id="task_1_easy",
        form_type="College Admission",
        required_fields=[
            "full_name",
            "dob",              # must be normalized to YYYY-MM-DD
            "high_school_marks",
            "course_preference",
        ],
        user_profile={
            "candidate_name": "Rahul Kumar",
            "alias": "R. Kumar",
            "birth_year": "2005",
            "dob_raw": "15/08/05",             # ambiguous short year format
            "class_12_percentage": "92",       # missing % sign
            "course_choice": "B.Tech CSE",
        },
        ground_truth={
            "full_name": "Rahul Kumar",
            "dob": "2005-08-15",
            "high_school_marks": "92%",
            "course_preference": "B.Tech CSE",
        },
        difficulty="easy",
    ),
    # Task 2: Bank KYC (Medium with inference vs request tradeoffs)
    TaskProfile(
        task_id="task_2_medium",
        form_type="Bank KYC",
        required_fields=[
            "full_name",
            "aadhaar_number",
            "pan_number",
            "address",
            "dob",  # must be consistent with KYC norms
        ],
        user_profile={
            "name_on_pan": "Priya Sharma",
            "name_on_aadhaar": "Priya S.",
            "aadhaar_masked": "XXXX-XXXX-1234",   # incomplete, should not be used directly
            "pan": "ABCDE1234F",
            "dob_free_text": "15th Aug 1998",     # needs normalization
            "address_line": "Koramangala, Bengaluru",
            "city": "Bengaluru",
            "pincode": "560034",
        },
        ground_truth={
            "full_name": "Priya Sharma",
            "aadhaar_number": "REQUEST_NEEDED",   # must request, cannot infer safely
            "pan_number": "ABCDE1234F",
            "address": "Koramangala, Bengaluru - 560034",
            "dob": "1998-08-15",
        },
        difficulty="medium",
    ),
    # Task 3: Hospital Registration (Hard: multi-step dependency + conflicts)
    TaskProfile(
        task_id="task_3_hard",
        form_type="Hospital Registration (Linked Forms)",
        required_fields=[
            # form A: identity
            "patient_name",
            "age",
            "blood_group",
            # form B: visit details
            "symptoms",
            "emergency_contact",
            "consent_given",  # must be "yes" to be valid submission
        ],
        user_profile={
            "patient": "Amit Patel",
            "alias": "A. Patel",
            "birth_year": "1980",
            "reported_age": "40",              # conflicting with 2026-1980 = 46
            "blood": "O positive",             # needs normalization to O+
            "issues": "Severe headache; fever",
            "phone_primary": "9876543210",
            "phone_secondary": "9876500000",
            "consent_checkbox": "true",        # string boolean
        },
        ground_truth={
            "patient_name": "Amit Patel",
            "age": "46",                       # must be inferred from birth_year
            "blood_group": "O+",
            "symptoms": "Severe headache; fever",
            "emergency_contact": "9876543210",  # prefer primary
            "consent_given": "yes",
        },
        difficulty="hard",
    ),
]


def get_task(task_id: str) -> TaskProfile:
    for t in TASKS:
        if t.task_id == task_id:
            return t
    raise ValueError(f"Task {task_id} not found")
