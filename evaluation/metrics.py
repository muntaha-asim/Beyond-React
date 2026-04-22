"""
Computes all metrics from a result JSON file:
  - success (bool)
  - score_delta (float, positive = improvement over baseline)
  - hallucination_count (int)
  - attempts_to_success (int or None)
"""
import json
import re
from typing import Optional

BASELINE_SCORES = {
    "house-price":       30000,
    "spaceship-titanic": 0.72,
    "vectorization":     None,
    "feedback":          0.50,
}


def lower_is_better(task: str) -> bool:
    return task == "house-price"


def is_success(task: str, score: Optional[float], baseline: Optional[float]) -> bool:
    if score is None or baseline is None:
        return False
    return score < baseline if lower_is_better(task) else score > baseline


def score_delta(task: str, score: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """Positive value means improvement over baseline."""
    if score is None or baseline is None:
        return None
    return (baseline - score) if lower_is_better(task) else (score - baseline)


def count_hallucinations(history_steps: list) -> int:
    """
    Count steps where Fact Check claims a performance improvement was
    "directly confirmed" but the next Execute Script observation shows
    no real improvement (error, unchanged output, or no numeric gain).

    This is a heuristic — we look for:
      - prev step Fact Check: claims improvement as "directly confirmed"
      - current step Action: Execute Script
      - current step Observation: contains error / traceback / no score change
    """
    hallucinations = 0

    for i, step in enumerate(history_steps):
        action = step.get("action", {})
        act_name = str(action.get("Action", "")).strip().lower()
        observation = str(step.get("observation", "")).lower()

        if "execute script" not in act_name:
            continue

        if i == 0:
            continue

        prev_fact_check = str(
            history_steps[i - 1].get("action", {}).get("Fact Check", "")
        ).lower()

        claimed_improvement = (
            "directly confirmed" in prev_fact_check
            and any(w in prev_fact_check for w in ("improved", "better", "increased", "decreased", "higher", "lower"))
        )

        if not claimed_improvement:
            continue

        no_actual_improvement = (
            "error" in observation
            or "traceback" in observation
            or observation.strip() == ""
            or "no improvement" in observation
        )

        if no_actual_improvement:
            hallucinations += 1

    return hallucinations


def compute_metrics(result_json_path: str) -> dict:
    with open(result_json_path) as f:
        data = json.load(f)

    task = data["task"]
    condition = data.get("condition", "?")
    model = data.get("model", "?")
    baseline = BASELINE_SCORES.get(task)
    final_score = data.get("final_score")
    attempts = data.get("attempts", [])

    success = is_success(task, final_score, baseline)
    delta = score_delta(task, final_score, baseline)

    # Attempts-to-success from stored value or recomputed
    attempts_to_success = data.get("attempts_to_success")
    if attempts_to_success is None:
        for i, att in enumerate(attempts):
            if is_success(task, att.get("final_score"), baseline):
                attempts_to_success = i + 1
                break

    total_hallucinations = sum(
        count_hallucinations(att.get("history_steps", []))
        for att in attempts
    )

    return {
        "condition": condition,
        "task": task,
        "model": model,
        "success": success,
        "final_score": final_score,
        "baseline_score": baseline,
        "score_delta": delta,
        "hallucination_count": total_hallucinations,
        "attempts_to_success": attempts_to_success,
        "n_attempts": len(attempts),
    }
