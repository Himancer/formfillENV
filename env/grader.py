# env/grader.py
from __future__ import annotations

from .tasks import get_task
from .environment import Observation


def grade_submission(task_id: str, final_state: Observation, total_reward: float, action_count: int) -> float:
    task = get_task(task_id)
    gt = task.ground_truth

    total_fields = len(task.required_fields)
    if total_fields == 0:
        return 0.0

    # Field correctness
    correct_fields = 0
    for field in task.required_fields:
        if field in final_state.filled_fields and final_state.filled_fields[field] == gt.get(field):
            correct_fields += 1
    correctness_score = correct_fields / total_fields

    # Completeness: all required fields present
    completeness_score = 1.0 if len(final_state.missing_fields) == 0 else 0.0

    # Efficiency: fewer steps is better
    # assuming task.max_steps is upper bound
    max_steps = task.max_steps
    efficiency_score = max(0.0, 1.0 - (final_state.step_count - total_fields) / max_steps)

    # Reward-based sanity: scale total_reward into [0,1] with a rough window
    reward_score = max(0.0, min(1.0, (total_reward + 5.0) / 10.0))

    # Weighted combination
    # correctness is most important, then completeness, then efficiency, then reward signal
    final_score = (
        0.4 * correctness_score
        + 0.25 * completeness_score
        + 0.2 * efficiency_score
        + 0.15 * reward_score
    )

    return round(max(0.0, min(1.0, final_score)), 3)
