"""
Runs a single ResearchAgent attempt on one MLAgentBench task.
Used for Condition A (1 attempt) and Condition B (multi-attempt, no reflection).
"""
import os
import sys
import importlib
from argparse import Namespace
from typing import Optional

# Ensure MLAgentBench is importable
_MLAB_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "implementation", "MLAgentBench")
)
if _MLAB_ROOT not in sys.path:
    sys.path.insert(0, _MLAB_ROOT)


def run_single_attempt(
    task: str,
    model: str,
    attempt_idx: int,
    log_dir: str,
    work_dir: str,
    agent_max_steps: int = 30,
    system_prompt_prefix: str = "",
) -> dict:
    """
    Run one ResearchAgent attempt on a task.

    Args:
        task:                MLAgentBench task name (e.g. "house-price")
        model:               LLM model ID (e.g. "claude-3-5-sonnet-20241022")
        attempt_idx:         0-based attempt number (used for log/work sub-dirs)
        log_dir:             Base log directory for this run
        work_dir:            Base workspace directory for this run
        agent_max_steps:     Maximum agent steps per attempt
        system_prompt_prefix: Text prepended to the agent's initial prompt
                              (used by ReflexionAgent to inject post-mortems)

    Returns:
        dict with keys: task, model, attempt, log_dir, history_steps,
                        final_message, final_score
    """
    from agent.llm import patch_mlagentbench
    patch_mlagentbench(model)

    from MLAgentBench.environment import Environment
    from MLAgentBench.agents.agent_research import ResearchAgent

    attempt_log_dir = os.path.join(log_dir, f"attempt_{attempt_idx}")
    attempt_work_dir = os.path.join(work_dir, f"attempt_{attempt_idx}")
    os.makedirs(attempt_log_dir, exist_ok=True)
    os.makedirs(attempt_work_dir, exist_ok=True)

    args = Namespace(
        task=task,
        log_dir=attempt_log_dir,
        work_dir=attempt_work_dir,
        max_steps=agent_max_steps,
        max_time=3 * 60 * 60,
        device=0,
        python=sys.executable,
        interactive=False,
        resume=None,
        resume_step=0,
        agent_type="ResearchAgent",
        llm_name=model,
        fast_llm_name=model,   # patched to cheap model in llm.py patch_mlagentbench
        edit_script_llm_name=model,
        edit_script_llm_max_tokens=2000,
        agent_max_steps=agent_max_steps,
        # Keep retrieval-related actions out of prompt (we don't use retrieval)
        actions_remove_from_prompt=[
            "Retrieval from Research Log",
            "Append Summary to Research Log",
            "Reflection",
        ],
        actions_add_to_prompt=[],
        retrieval=False,
        valid_format_entires=None,
        max_steps_in_context=1,
        max_observation_steps_in_context=1,
        max_retries=3,
        langchain_agent="zero-shot-react-description",
    )

    result = {
        "task": task,
        "model": model,
        "attempt": attempt_idx,
        "log_dir": attempt_log_dir,
        "history_steps": [],
        "final_message": "",
        "final_score": None,
    }

    with Environment(args) as env:
        agent = ResearchAgent(args, env)

        # ML knowledge always goes first, then reflection context (if any), then task prompt
        from agent.ml_knowledge import ML_KNOWLEDGE_PROMPT
        prefix = ML_KNOWLEDGE_PROMPT
        if system_prompt_prefix:
            prefix = prefix + "\n\n" + system_prompt_prefix
        agent.initial_prompt = prefix + "\n\n" + agent.initial_prompt

        result["final_message"] = agent.run(env)
        result["history_steps"] = _serialize_history(agent.history_steps)

    # env.save must be called after the with-block (matches MLAgentBench runner.py)
    env.save("final")

    result["final_score"] = get_task_score(task, attempt_log_dir)
    return result


def get_task_score(task: str, log_dir: str) -> Optional[float]:
    """
    Extract the final evaluation score by calling the task's eval script.
    Returns None if the score file doesn't exist or evaluation fails.
    """
    try:
        from MLAgentBench.prepare_task import get_task_info
        benchmark_folder_name, _ = get_task_info(task)
        module = importlib.import_module(
            f"MLAgentBench.benchmarks.{benchmark_folder_name}.scripts.eval"
        )
        score_folder = os.path.join(log_dir, "env_log", "traces", "step_final_files")
        if os.path.exists(score_folder):
            return module.get_score(score_folder)
    except Exception as e:
        print(f"[score extraction] {task}: {e}")
    return None


def _serialize_history(history_steps: list) -> list:
    """Convert history_steps to a JSON-safe list (actions are already plain dicts)."""
    safe = []
    for step in history_steps:
        safe.append({
            "step_idx": step.get("step_idx"),
            "action": {k: str(v) for k, v in step.get("action", {}).items()},
            "observation": str(step.get("observation", "")),
        })
    return safe
