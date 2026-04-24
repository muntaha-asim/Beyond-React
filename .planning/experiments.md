# Experiment Plan

## Research Question
Does cross-run Reflexion (structured post-mortem between attempts) reduce hallucination
and improve task success rate in an LLM-based ML experimentation agent?

## Thesis
Cross-run Reflexion reduces hallucination frequency and improves task success rates
without modifying model weights.

---

## Conditions

| Condition | Description | Attempts | Reflection | Purpose |
|---|---|---|---|---|
| A | Baseline (original MLAgentBench) | 1 | None | Replicate paper, reference point |
| B | Multi-attempt, no reflection | 3 | None | Controls for "more compute" |
| C | Reflexion (our contribution) | 3 | Structured post-mortem | Tests our hypothesis |

**Why Condition B matters:** If C > B, improvement is from Reflexion specifically.
If C ≈ B, more attempts alone explain it and Reflexion adds nothing.

---

## Tasks

| Task | Type | Metric | Direction | Baseline Score |
|---|---|---|---|---|
| `house-price` | Tabular regression | MAE | Lower is better | ~30,000 |
| `spaceship-titanic` | Tabular classification | Accuracy | Higher is better | ~0.72 |
| `vectorization` | Code optimization | Relative speedup | Higher is better | 1.0× |
| `feedback` | NLP classification | Macro-F1 | Higher is better | ~0.50 |

All tasks are CPU-feasible — no GPU required.

---

## Models

| Model | Provider | API Key Env Var |
|---|---|---|
| `claude-3-5-sonnet-20241022` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` |
| `gemini-1.5-pro` | Google | `GEMINI_API_KEY` |

**Testing / smoke-run model:** `gemini-1.5-flash` (free tier, fast)

---

## Full Experiment Matrix

3 conditions × 4 tasks × 2 primary models = **24 combinations**
(Gemini used for testing; not part of the primary paper results)

| | house-price | spaceship-titanic | vectorization | feedback |
|---|---|---|---|---|
| A — sonnet35 | | | | |
| A — gpt4o | | | | |
| B — sonnet35 | | | | |
| B — gpt4o | | | | |
| C — sonnet35 | | | | |
| C — gpt4o | | | | |

Fill with ✓ (success) / ✗ (fail) / score delta as runs complete.

---

## Metrics Collected Per Run

| Metric | Definition | Collected by |
|---|---|---|
| Success | Final score beats baseline (True/False) | metrics.py |
| Score Delta | final_score − baseline_score | metrics.py |
| Hallucination Count | Steps where Fact Check claims improvement but observation contradicts | metrics.py |
| Attempts to Success | Which attempt (1/2/3) first beat baseline | reflexion_agent.py |

---

## Per-Run Parameters

| Parameter | Value | Notes |
|---|---|---|
| `--steps` | 30 (full) / 10-15 (smoke test) | Max agent steps per attempt |
| `--attempts` | 3 | For Conditions B and C |
| `random_state` | 42 | Set in ML knowledge prompt |
| Max reflection tokens | 1000 | In _generate_post_mortem() |
| Trajectory window | Last 20 steps | In _format_trajectory() |

---

## Run Commands

### Smoke test (fast, free)
```bash
source implementation/venv2/bin/activate
python evaluation/run_experiment.py \
  --condition A --task house-price \
  --model gemini-1.5-flash --steps 10
```

### Full single run
```bash
python evaluation/run_experiment.py \
  --condition C --task spaceship-titanic \
  --model claude-3-5-sonnet-20241022 --steps 30 --attempts 3
```

### Full experiment (all 24 combinations)
```bash
python evaluation/run_all.py --steps 30 --attempts 3
```

### Dry run (preview without executing)
```bash
python evaluation/run_all.py --dry-run
```

### Resume after interruption
Re-run the same command — already-completed result files are automatically skipped.
