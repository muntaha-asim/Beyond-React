"""
Runs all conditions × tasks × models sequentially.
Skips any combination that already has a result file.
Prints a summary table when done.

Usage:
    python evaluation/run_all.py                    # full run
    python evaluation/run_all.py --dry-run          # print plan, don't execute
    python evaluation/run_all.py --task house-price # one task, all conditions/models
"""
import os
import sys
import json
import argparse

_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MLAB = os.path.join(_PROJECT, "implementation", "MLAgentBench")
for p in (_PROJECT, _MLAB):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.run_experiment import run_experiment, _model_slug, RESULTS_DIR
from evaluation.metrics import compute_metrics

ALL_TASKS = ["house-price", "spaceship-titanic", "vectorization", "feedback"]
ALL_MODELS = ["claude-3-5-sonnet-20241022", "gpt-4o"]
ALL_CONDITIONS = ["A", "B", "C"]


def run_all(
    tasks=None,
    models=None,
    conditions=None,
    steps: int = 30,
    max_attempts: int = 3,
    dry_run: bool = False,
):
    tasks = tasks or ALL_TASKS
    models = models or ALL_MODELS
    conditions = conditions or ALL_CONDITIONS

    plan = [
        (cond, task, model)
        for cond in conditions
        for task in tasks
        for model in models
    ]

    print(f"Plan: {len(plan)} runs\n")

    for cond, task, model in plan:
        slug = _model_slug(model)
        result_path = os.path.join(RESULTS_DIR, cond, f"{task}_{slug}.json")

        if os.path.exists(result_path):
            print(f"[SKIP] {cond}/{task}/{slug} — result exists")
            continue

        if dry_run:
            print(f"[DRY RUN] Condition={cond}  Task={task}  Model={model}")
            continue

        try:
            run_experiment(cond, task, model, steps, max_attempts)
        except Exception as e:
            print(f"[ERROR] {cond}/{task}/{model}: {e}")

    # Print summary table from all result files
    _print_summary(tasks, models, conditions)


def _print_summary(tasks, models, conditions):
    rows = []
    for cond in conditions:
        for task in tasks:
            for model in models:
                slug = _model_slug(model)
                path = os.path.join(RESULTS_DIR, cond, f"{task}_{slug}.json")
                if not os.path.exists(path):
                    continue
                try:
                    m = compute_metrics(path)
                    rows.append(m)
                except Exception as e:
                    print(f"Metrics error for {path}: {e}")

    if not rows:
        print("\nNo results to summarize yet.")
        return

    print("\n\n=== RESULTS SUMMARY ===")
    hdr = f"{'Cond':<6}{'Task':<22}{'Model':<12}{'Success':<10}{'Delta':<12}{'Halluc.':<10}{'Att→Win'}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        slug = _model_slug(r["model"])
        delta = f"{r['score_delta']:+.1f}" if r["score_delta"] is not None else "N/A"
        print(
            f"{r['condition']:<6}{r['task']:<22}{slug:<12}"
            f"{str(r['success']):<10}{delta:<12}{str(r['hallucination_count']):<10}"
            f"{str(r['attempts_to_success'])}"
        )

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="+", choices=ALL_TASKS,
                        help="Restrict to specific task(s)")
    parser.add_argument("--model", nargs="+",
                        help="Restrict to specific model(s)")
    parser.add_argument("--condition", nargs="+", choices=ALL_CONDITIONS,
                        help="Restrict to specific condition(s)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_all(
        tasks=args.task,
        models=args.model,
        conditions=args.condition,
        steps=args.steps,
        max_attempts=args.attempts,
        dry_run=args.dry_run,
    )
