# Autonomous ML Experimentation Agent with Cross-Run Reflexion

**Course:** Agentic AI — Final Semester Project
**University:** FAST National University of Computer & Emerging Sciences
**Authors:** Muntaha Asim · Muhammad Ammar Kashif

> **Survey paper:** `papers/Beyond ReAct.pdf`
> **MLAgentBench paper:** `papers/Huang 24y.pdf`

---

## What This Project Does

We add **cross-run Reflexion** to MLAgentBench's ResearchAgent and measure whether it reduces hallucination and improves task success.

The ResearchAgent (from [MLAgentBench, ICML 2024](https://arxiv.org/abs/2310.03302)) autonomously writes code, runs experiments, reads results, and iterates to improve an ML model. Its built-in Fact Check reduces hallucination but doesn't eliminate it. Our contribution: after each *failed attempt*, the full trajectory is sent to the LLM which writes a structured post-mortem (what was tried, why it failed, what to try next, what to avoid). That post-mortem is prepended to the next attempt's prompt.

**Thesis:** Cross-run Reflexion reduces hallucination frequency and improves task success rates without modifying model weights.

---

## Experiment Design

### Three Conditions

| Condition | Description | Attempts | Reflection |
|-----------|-------------|----------|------------|
| **A — Baseline** | Original ResearchAgent, 1 attempt | 1 | None |
| **B — Multi-attempt** | ResearchAgent, 3 attempts, no reflection | 3 | None |
| **C — Reflexion** | Our agent, 3 attempts + post-mortem between each | 3 | Yes |

Condition B is a critical control: if C beats B, the gain is from Reflexion specifically — not just having more compute.

### Tasks

| Task | Type | Metric | Baseline | Direction |
|------|------|--------|----------|-----------|
| `house-price` | Tabular regression | MAE | 30,000 | Lower is better |
| `spaceship-titanic` | Tabular classification | Accuracy | 0.72 | Higher is better |
| `vectorization` | Code optimisation | Speedup ratio | 1.0× | Higher is better |
| `feedback` | NLP classification | Macro-F1 | 0.50 | Higher is better |

### Primary Models

| Model ID | Provider | API key env var |
|----------|----------|-----------------|
| `claude-sonnet-4-6` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` |

Full matrix: 3 conditions × 4 tasks × 2 models = **24 runs**.

---

## Completed Runs

| Condition | Task | Model | Final Score | Beat Baseline? |
|-----------|------|-------|-------------|----------------|
| A | house-price | claude-sonnet-4-6 | 16,679 MAE | ✓ (−44%) |
| A | house-price | gemini-2.5-flash | 19,919 MAE | ✓ (−34%) |
| A | vectorization | claude-sonnet-4-6 | 1.256× speedup | ✓ |
| B | house-price | gemini-2.5-flash | 20,595 MAE | ✓ |
| C | house-price | gemini-2.5-flash | 16,682 MAE | ✓ (attempt 1) |

Gemini runs were smoke tests. **Primary paper results use `claude-sonnet-4-6` and `gpt-4o`.**

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd project
```

### 2. Clone MLAgentBench inside `implementation/`

```bash
mkdir -p implementation
git clone https://github.com/snap-stanford/MLAgentBench implementation/MLAgentBench
```

### 3. Create the virtual environment

```bash
python3 -m venv implementation/venv2
source implementation/venv2/bin/activate
pip install -r requirements.txt
```

Also install MLAgentBench's own dependencies:

```bash
pip install dacite tiktoken
```

### 4. Add API keys

```bash
cp .env.example .env
# Open .env and fill in your keys
```

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...       # optional, only needed for Gemini runs
```

### 5. Activate the venv before every session

```bash
source implementation/venv2/bin/activate
```

---

## Running Experiments

### Run one condition

```bash
python evaluation/run_experiment.py \
  --condition C \
  --task house-price \
  --model claude-sonnet-4-6 \
  --steps 30 \
  --attempts 3
```

**Options:**

| Flag | Values | Default |
|------|--------|---------|
| `--condition` | `A`, `B`, `C` | required |
| `--task` | `house-price`, `spaceship-titanic`, `vectorization`, `feedback` | required |
| `--model` | `claude-sonnet-4-6`, `gpt-4o`, `gemini-2.5-flash` | `claude-sonnet-4-6` |
| `--steps` | integer | `30` |
| `--attempts` | integer (B and C only) | `3` |

Results are saved to `results/{condition}/{task}_{model_slug}.json`. **Already-completed runs are skipped automatically** — safe to re-run after interruption.

### Run everything at once

```bash
python evaluation/run_all.py --steps 30 --attempts 3
```

Preview without executing:

```bash
python evaluation/run_all.py --dry-run
```

### Compute metrics on a result file

```python
from evaluation.metrics import compute_metrics
m = compute_metrics("results/C/house-price_sonnet46.json")
print(m)
# {condition, task, model, success, final_score, baseline_score,
#  score_delta, hallucination_count, attempts_to_success, n_attempts}
```

---

## What Still Needs to Be Run (Ammar's Part)

The table below shows what's left for the primary models. Run these in order — cheaper tasks first.

```bash
# --- Condition A (baseline, 1 attempt each) ---
python evaluation/run_experiment.py --condition A --task spaceship-titanic --model claude-sonnet-4-6 --steps 30
python evaluation/run_experiment.py --condition A --task feedback         --model claude-sonnet-4-6 --steps 30
python evaluation/run_experiment.py --condition A --task house-price      --model gpt-4o            --steps 30
python evaluation/run_experiment.py --condition A --task spaceship-titanic --model gpt-4o           --steps 30
python evaluation/run_experiment.py --condition A --task vectorization    --model gpt-4o            --steps 30
python evaluation/run_experiment.py --condition A --task feedback         --model gpt-4o            --steps 30

# --- Condition B (multi-attempt, no reflection) ---
python evaluation/run_experiment.py --condition B --task house-price      --model claude-sonnet-4-6 --steps 30 --attempts 3
python evaluation/run_experiment.py --condition B --task spaceship-titanic --model claude-sonnet-4-6 --steps 30 --attempts 3
python evaluation/run_experiment.py --condition B --task vectorization    --model claude-sonnet-4-6 --steps 30 --attempts 3
python evaluation/run_experiment.py --condition B --task feedback         --model claude-sonnet-4-6 --steps 30 --attempts 3
python evaluation/run_experiment.py --condition B --task house-price      --model gpt-4o            --steps 30 --attempts 3
python evaluation/run_experiment.py --condition B --task spaceship-titanic --model gpt-4o           --steps 30 --attempts 3
python evaluation/run_experiment.py --condition B --task vectorization    --model gpt-4o            --steps 30 --attempts 3
python evaluation/run_experiment.py --condition B --task feedback         --model gpt-4o            --steps 30 --attempts 3

# --- Condition C (Reflexion — our contribution) ---
python evaluation/run_experiment.py --condition C --task house-price      --model claude-sonnet-4-6 --steps 30 --attempts 3
python evaluation/run_experiment.py --condition C --task spaceship-titanic --model claude-sonnet-4-6 --steps 30 --attempts 3
python evaluation/run_experiment.py --condition C --task vectorization    --model claude-sonnet-4-6 --steps 30 --attempts 3
python evaluation/run_experiment.py --condition C --task feedback         --model claude-sonnet-4-6 --steps 30 --attempts 3
python evaluation/run_experiment.py --condition C --task house-price      --model gpt-4o            --steps 30 --attempts 3
python evaluation/run_experiment.py --condition C --task spaceship-titanic --model gpt-4o           --steps 30 --attempts 3
python evaluation/run_experiment.py --condition C --task vectorization    --model gpt-4o            --steps 30 --attempts 3
python evaluation/run_experiment.py --condition C --task feedback         --model gpt-4o            --steps 30 --attempts 3
```

Each run takes roughly 20–60 minutes depending on task and model. Condition C runs (3 attempts) take up to 3× longer than Condition A.

---

## Project Structure

```
project/
├── agent/
│   ├── llm.py              # Claude / GPT-4o / Gemini wrapper; patches MLAgentBench
│   ├── prompts.py          # REFLECTION_PROMPT + REFLECTION_CONTEXT_TEMPLATE
│   ├── ml_knowledge.py     # ML domain knowledge injected into every run
│   ├── base_agent.py       # Single-attempt runner (Conditions A and B)
│   └── reflexion_agent.py  # Cross-run Reflexion runner (Condition C)
│
├── evaluation/
│   ├── metrics.py          # compute_metrics(): success, score_delta, hallucination_count
│   ├── run_experiment.py   # CLI: run one condition × task × model
│   └── run_all.py          # Run all 24 combinations; prints summary table
│
├── implementation/
│   └── MLAgentBench/       # Cloned from snap-stanford/MLAgentBench (do not modify)
│
├── results/                # Output JSONs — gitignored, generated locally
│   ├── A/
│   ├── B/
│   └── C/
│
├── logs/                   # Per-run trajectory logs — gitignored
│
├── papers/
│   ├── Beyond ReAct.pdf    # Our submitted Part 1 survey
│   ├── CourseProject_part1.tex  # LaTeX source
│   ├── Huang 24y.pdf       # MLAgentBench paper (Huang et al., ICML 2024)
│   └── benchmark paper.pdf # Reference
│
├── dashboard/
│   ├── server.py           # Local results viewer (Flask)
│   └── index.html
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## How the Reflexion Mechanism Works

After each failed attempt the agent's full trajectory is sent to the LLM with this prompt:

```
TASK: house-price
BASELINE SCORE: 30000
YOUR FINAL SCORE: 31500
RESULT: Failed

[trajectory of last 50 steps...]

Write a structured post-mortem:
1. STRATEGIES TRIED
2. WHY THEY FAILED
3. HYPOTHESIS FOR NEXT ATTEMPT
4. WHAT TO AVOID
```

The response is saved to `logs/.../reflection_after_attempt_N.txt` and prepended to the next attempt's system prompt. The agent starts each retry with explicit knowledge of past failures.

Prompt injection order per attempt:

```
[1] ML_KNOWLEDGE_PROMPT       ← anti-hallucination rules + task playbook
[2] Reflection post-mortem    ← only Condition C, attempt 2+
[3] MLAgentBench task prompt  ← original ResearchAgent task description
```

---

## Metrics

| Metric | Definition |
|--------|-----------|
| `success` | Final score beats the task baseline |
| `score_delta` | Absolute improvement over baseline (positive = better) |
| `hallucination_count` | Steps where Fact Check claims confirmed improvement but observation shows error/traceback |
| `attempts_to_success` | Which attempt (1/2/3) first beat the baseline; `null` if never |

---

## Key References

1. Q. Huang et al., "MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation," ICML 2024.
2. N. Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning," NeurIPS 2023.
3. S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR 2023.
