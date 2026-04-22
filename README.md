# Autonomous ML Experimentation Agent with Cross-Run Reflexion

**Course:** Agentic AI — Final Semester Project  
**University:** FAST National University of Computer & Emerging Sciences  
**Authors:** Muntaha Asim · Muhammad Ammar Kashif  

---

## Overview

This project investigates whether adding a structured **cross-run Reflexion** mechanism to an LLM-based ML experimentation agent reduces hallucination and improves task success rates.

We build on top of [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) (Huang et al., ICML 2024) — a benchmark where an LLM agent is given an ML task (e.g. improve accuracy on a Kaggle dataset) and must autonomously write code, run experiments, read results, and iterate.

### The Problem

Existing agents **hallucinate improvements** — the agent's internal Fact Check claims a score improved, but the actual execution output shows no change. MLAgentBench found that even the best model (Claude Opus) only succeeds on 37.5% of tasks, with hallucination as a primary failure mode.

### Our Contribution

The existing `ResearchAgent` includes per-step reflection (asking "what did I just observe?") but treats each complete run as independent. We introduce **cross-run Reflexion**: after each failed attempt, the full trajectory is analyzed to produce a structured post-mortem (strategies tried, why they failed, what to try next, what to avoid). This post-mortem is prepended to the next attempt's system prompt.

> **Thesis:** Cross-run Reflexion reduces hallucination frequency and improves task success rates without modifying model weights.

---

## Experiment Design

### Three Conditions

| Condition | Description | Purpose |
|-----------|-------------|---------|
| **A — Baseline** | Original ResearchAgent, 1 attempt, 30 steps | Replicates MLAgentBench paper |
| **B — Multi-attempt** | ResearchAgent, 3 attempts, no reflection | Controls for "more compute" |
| **C — Reflexion (ours)** | Our agent, 3 attempts + structured post-mortem | Tests our contribution |

Condition B is a critical control: if C outperforms B, the improvement is due to Reflexion specifically, not just having more attempts.

### Tasks

| Task | Domain | Metric |
|------|--------|--------|
| `house-price` | Tabular regression | MAE (lower is better) |
| `spaceship-titanic` | Tabular classification | Accuracy (higher is better) |
| `vectorization` | Code optimization | Relative speedup |
| `feedback` | NLP classification | Macro-F1 (higher is better) |

### Models

| Model | Provider |
|-------|----------|
| Claude 3.5 Sonnet | Anthropic API |
| GPT-4o | OpenAI API |

### Metrics

| Metric | Definition |
|--------|-----------|
| **Success Rate** | % of tasks where final score beats the baseline |
| **Score Delta** | Absolute improvement over baseline |
| **Hallucination Count** | Steps where Fact Check claims improvement but execution shows no change |
| **Attempts to Success** | Which attempt (1, 2, or 3) first beat the baseline |

---

## Project Structure

```
project/
├── agent/
│   ├── llm.py              # Modern Claude + GPT-4o wrapper; patches MLAgentBench
│   ├── prompts.py          # Post-run reflection prompt + context template
│   ├── base_agent.py       # Single-attempt runner (Conditions A & B)
│   └── reflexion_agent.py  # Cross-run Reflexion runner (Condition C)
├── evaluation/
│   ├── metrics.py          # Success rate, score delta, hallucination count
│   ├── run_experiment.py   # Run one condition × task × model
│   └── run_all.py          # Run all 24 combinations
├── results/                # Output JSONs per run (git-ignored)
│   ├── A/
│   ├── B/
│   └── C/
├── implementation/
│   └── MLAgentBench/       # Cloned benchmark (not modified)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd project
```

### 2. Clone MLAgentBench inside `implementation/`

```bash
mkdir -p implementation
git clone https://github.com/snap-stanford/MLAgentBench implementation/MLAgentBench
```

### 3. Create a virtual environment

```bash
python3 -m venv implementation/venv2
source implementation/venv2/bin/activate
pip install -r requirements.txt
```

Also install MLAgentBench dependencies:

```bash
pip install dacite tiktoken anthropic openai
```

### 4. Add your API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

---

## Running Experiments

### Run one condition

```bash
source implementation/venv2/bin/activate

python evaluation/run_experiment.py \
  --condition A \
  --task house-price \
  --model claude-3-5-sonnet-20241022 \
  --steps 30
```

Options:
- `--condition` : `A`, `B`, or `C`
- `--task` : `house-price`, `spaceship-titanic`, `vectorization`, `feedback`
- `--model` : `claude-3-5-sonnet-20241022` or `gpt-4o`
- `--steps` : max agent steps per attempt (default 30)
- `--attempts` : max attempts for Conditions B and C (default 3)

### Run everything

```bash
python evaluation/run_all.py --steps 30 --attempts 3
```

Use `--dry-run` to see the full plan without executing.  
Already-completed runs are automatically skipped (safe to resume after interruption).

### Compute metrics on existing results

```python
from evaluation.metrics import compute_metrics
m = compute_metrics("results/C/house-price_sonnet35.json")
print(m)
```

---

## How the Reflexion Mechanism Works

After each failed attempt, the full trajectory is sent to the LLM with this prompt structure:

```
TASK: house-price
BASELINE SCORE: 30000
YOUR FINAL SCORE: 31500
RESULT: Failed

[trajectory...]

Write a structured post-mortem:
1. STRATEGIES TRIED
2. WHY THEY FAILED
3. HYPOTHESIS FOR NEXT ATTEMPT
4. WHAT TO AVOID
```

The LLM's response is prepended to the next attempt's system prompt, giving the agent explicit memory of past failures before it starts.

---

## Key References

1. Q. Huang et al., "MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation," ICML 2024.
2. N. Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning," NeurIPS 2023.
3. S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR 2023.
4. J. Wang et al., "GTA: A Benchmark for General Tool Agents," NeurIPS 2024.
5. Z. Chen et al., "Agent-FLAN: Designing Data and Methods for Effective Agent Tuning," ACL Findings 2024.
