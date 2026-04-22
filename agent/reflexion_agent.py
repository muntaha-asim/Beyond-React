"""
Cross-run Reflexion agent (Condition C).

After each failed attempt the full trajectory is sent to the LLM to produce
a structured post-mortem (strategies tried, why they failed, what to try next,
what to avoid).  That post-mortem is prepended to the *next* attempt's system
prompt so the agent starts each retry with explicit knowledge of past failures.
"""
import os
import json
from typing import Optional

from agent.base_agent import run_single_attempt, get_task_score
from agent.llm import complete
from agent.prompts import REFLECTION_PROMPT, REFLECTION_CONTEXT_TEMPLATE

# Baseline scores from the original MLAgentBench paper / task descriptions.
# Lower is better for house-price (MAE); higher is better for the rest.
BASELINE_SCORES = {
    "house-price":       30000,   # MAE in dollars
    "spaceship-titanic": 0.72,    # accuracy
    "vectorization":     None,    # measured as relative speedup — handled separately
    "feedback":          0.50,    # macro-F1
}


def run_reflexion(
    task: str,
    model: str,
    log_dir: str,
    work_dir: str,
    max_attempts: int = 3,
    agent_max_steps: int = 30,
) -> dict:
    """
    Run up to max_attempts.  After each failed attempt, generate a reflection
    and inject it into the next attempt's prompt.

    Returns a dict with all attempt results plus aggregate metrics.
    """
    all_attempts = []
    reflection_context = ""  # grows with each failed attempt

    for attempt_idx in range(max_attempts):
        print(f"\n{'='*60}")
        print(f"[Reflexion] Task={task} | Model={model} | "
              f"Attempt {attempt_idx + 1}/{max_attempts}")
        print(f"{'='*60}")

        result = run_single_attempt(
            task=task,
            model=model,
            attempt_idx=attempt_idx,
            log_dir=log_dir,
            work_dir=work_dir,
            agent_max_steps=agent_max_steps,
            system_prompt_prefix=reflection_context,
        )
        all_attempts.append(result)

        score = result["final_score"]
        baseline = BASELINE_SCORES.get(task)
        success = _did_succeed(task, score, baseline)

        print(f"Attempt {attempt_idx + 1} → score={score}, "
              f"baseline={baseline}, success={success}")

        if success:
            print(f"Task solved on attempt {attempt_idx + 1}.")
            break

        if attempt_idx < max_attempts - 1:
            print(f"Generating reflection for attempt {attempt_idx + 2}...")
            post_mortem = _generate_post_mortem(
                task=task,
                baseline_score=baseline,
                final_score=score,
                history_steps=result["history_steps"],
                model=model,
                attempt_idx=attempt_idx,
                log_dir=log_dir,
            )
            reflection_context = REFLECTION_CONTEXT_TEMPLATE.format(
                attempt_num=attempt_idx + 1,
                reflection_text=post_mortem,
            )

    return {
        "task": task,
        "model": model,
        "condition": "C",
        "attempts": all_attempts,
        "final_score": all_attempts[-1]["final_score"],
        "attempts_to_success": _attempts_to_success(all_attempts, task, BASELINE_SCORES.get(task)),
    }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _did_succeed(task: str, score, baseline) -> bool:
    if score is None or baseline is None:
        return False
    if task == "house-price":
        return score < baseline   # lower MAE is better
    return score > baseline       # higher accuracy / F1 is better


def _attempts_to_success(attempts: list, task: str, baseline) -> Optional[int]:
    for i, attempt in enumerate(attempts):
        if _did_succeed(task, attempt.get("final_score"), baseline):
            return i + 1
    return None


def _generate_post_mortem(
    task: str,
    baseline_score,
    final_score,
    history_steps: list,
    model: str,
    attempt_idx: int,
    log_dir: str,
) -> str:
    trajectory_text = _format_trajectory(history_steps)

    prompt = REFLECTION_PROMPT.format(
        task=task,
        baseline_score=baseline_score,
        final_score=final_score,
        trajectory=trajectory_text,
    )

    post_mortem = complete(
        prompt=prompt,
        model=model,
        max_tokens=1000,
        temperature=0.3,
    )

    # Save reflection to disk for inspection / reproducibility
    reflection_path = os.path.join(log_dir, f"reflection_after_attempt_{attempt_idx + 1}.txt")
    with open(reflection_path, "w") as f:
        f.write(prompt + "\n\n--- REFLECTION ---\n\n" + post_mortem)

    return post_mortem


def _format_trajectory(history_steps: list, max_steps: int = 20) -> str:
    """
    Produce a compact text summary of the trajectory.
    Only the last max_steps are included to stay within token limits.
    """
    steps = history_steps[-max_steps:] if len(history_steps) > max_steps else history_steps
    lines = []
    for i, step in enumerate(steps):
        action = step.get("action", {})
        obs = str(step.get("observation", ""))[:400]

        thought = str(action.get("Thought", "")).strip()[:200]
        fact_check = str(action.get("Fact Check", "")).strip()[:200]
        act = str(action.get("Action", "")).strip()
        act_input = str(action.get("Action Input", "")).strip()[:100]

        lines.append(f"Step {i + 1}:")
        if thought:
            lines.append(f"  Thought: {thought}")
        if fact_check:
            lines.append(f"  Fact Check: {fact_check}")
        lines.append(f"  Action: {act}  Input: {act_input}")
        lines.append(f"  Observation: {obs}")
        lines.append("")

    return "\n".join(lines)
