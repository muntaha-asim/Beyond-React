"""
Runs one condition × task × model combination and saves the result.

Usage:
    python evaluation/run_experiment.py --condition A --task house-price --model claude-3-5-sonnet-20241022

Conditions:
    A — Baseline: 1 attempt, no reflection (replicates MLAgentBench paper)
    B — Multi-attempt: 3 attempts, no cross-run reflection (ablation control)
    C — Reflexion: 3 attempts with structured post-mortem between each (our contribution)
"""
import os
import sys
import json
import argparse
from datetime import datetime

# Set up import paths
_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MLAB = os.path.join(_PROJECT, "implementation", "MLAgentBench")
for p in (_PROJECT, _MLAB):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.base_agent import run_single_attempt
from agent.reflexion_agent import run_reflexion

RESULTS_DIR = os.path.join(_PROJECT, "results")
LOGS_DIR = os.path.join(_PROJECT, "logs")

VALID_TASKS = ["house-price", "spaceship-titanic", "vectorization", "feedback"]


def _model_slug(model: str) -> str:
    return (model
            .replace("claude-3-5-sonnet-20241022", "sonnet35")
            .replace("claude-3-5-sonnet-", "sonnet35-")
            .replace("gpt-4o", "gpt4o")
            .replace("gemini-2.5-flash-lite", "gemini25flashlite")
            .replace("gemini-2.5-flash", "gemini25flash")
            .replace("gemini-2.5-pro", "gemini25pro")
            .replace("gemini-2.0-flash-lite-001", "gemini20flashlite")
            .replace("gemini-2.0-flash-lite", "gemini20flashlite")
            .replace("gemini-2.0-flash-001", "gemini20flash001")
            .replace("gemini-2.0-flash", "gemini20flash")
            .replace("gemini-1.5-pro", "gemini15pro")
            .replace("gemini-1.5-flash", "gemini15flash")
            .replace("claude-sonnet-4-6", "sonnet46")
            .replace("claude-haiku-4-5-20251001", "haiku45")
            .replace("claude-opus-4-7", "opus47"))


def run_experiment(
    condition: str,
    task: str,
    model: str,
    agent_max_steps: int = 30,
    max_attempts: int = 3,
) -> dict:
    slug = _model_slug(model)
    run_id = f"{condition}_{task}_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(LOGS_DIR, run_id)
    work_dir = os.path.join(log_dir, "workspace")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    result_path = os.path.join(RESULTS_DIR, condition, f"{task}_{slug}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    print(f"\n[run_experiment] Condition={condition}  Task={task}  Model={model}")

    if condition == "A":
        attempt = run_single_attempt(
            task=task, model=model, attempt_idx=0,
            log_dir=log_dir, work_dir=work_dir,
            agent_max_steps=agent_max_steps,
        )
        output = {
            "condition": "A",
            "task": task,
            "model": model,
            "attempts": [attempt],
            "final_score": attempt["final_score"],
            "attempts_to_success": None,
        }

    elif condition == "B":
        attempts = []
        for i in range(max_attempts):
            att = run_single_attempt(
                task=task, model=model, attempt_idx=i,
                log_dir=log_dir, work_dir=work_dir,
                agent_max_steps=agent_max_steps,
            )
            attempts.append(att)
        output = {
            "condition": "B",
            "task": task,
            "model": model,
            "attempts": attempts,
            "final_score": attempts[-1]["final_score"],
            "attempts_to_success": None,
        }

    elif condition == "C":
        output = run_reflexion(
            task=task, model=model,
            log_dir=log_dir, work_dir=work_dir,
            max_attempts=max_attempts,
            agent_max_steps=agent_max_steps,
        )

    else:
        raise ValueError(f"Unknown condition '{condition}'. Must be A, B, or C.")

    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"[run_experiment] Saved → {result_path}")
    print(f"[run_experiment] Final score: {output.get('final_score')}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one experiment condition")
    parser.add_argument("--condition", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=VALID_TASKS)
    parser.add_argument("--model", default="claude-3-5-sonnet-20241022",
                        help="LLM model ID")
    parser.add_argument("--steps", type=int, default=30,
                        help="Max agent steps per attempt")
    parser.add_argument("--attempts", type=int, default=3,
                        help="Max attempts (Conditions B and C only)")
    args = parser.parse_args()

    run_experiment(args.condition, args.task, args.model, args.steps, args.attempts)
