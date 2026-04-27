# Cross-Run Reflexion for Autonomous ML Experimentation

**Course:** Agentic AI — Final Semester Project
**University:** FAST National University of Computer & Emerging Sciences
**Authors:** Muntaha Asim · Muhammad Ammar Kashif

> **Research paper:** `papers/research_paper.tex` (compile in Overleaf for PDF)

---

## What This Project Does

We add **cross-run Reflexion** to MLAgentBench's ResearchAgent and measure
whether it reduces hallucination and improves task performance.

The ResearchAgent autonomously writes code, runs experiments, reads results,
and iterates to improve an ML model. Its built-in Fact Check reduces
hallucination but does not eliminate it across independent runs. Our
contribution: after each *failed attempt*, the full trajectory is sent to
the LLM which writes a structured **post-mortem** (what was tried, why it
failed, what to try next, what to avoid). That post-mortem is prepended to
the next attempt's prompt.

**Thesis:** Cross-run Reflexion improves task performance where multi-attempt
search without feedback degrades it, confirming that the gains are attributable
to structured reflection rather than additional compute.

---

## Experiment Design

### Three Conditions

| Condition | Description | Attempts | Reflection |
|-----------|-------------|----------|------------|
| **A — Baseline** | Original ResearchAgent, 1 attempt | 1 | None |
| **B — Multi-attempt** | 3 independent attempts, no reflection | 3 | None |
| **C — Reflexion** | 3 attempts + structured post-mortem between each | 3 | Yes |

Condition B is the critical control: if C beats B, the gain is from Reflexion
specifically — not from having more compute.

### Tasks

| Task | Type | Metric | Baseline | Direction |
|------|------|--------|----------|-----------|
| `house-price` | Tabular regression | MAE | 30,000 | Lower is better |
| `spaceship-titanic` | Tabular classification | Accuracy | 0.720 | Higher is better |
| `vectorization` | Code optimisation | Speedup | 1.0× | Higher is better |
| `feedback` | NLP classification | Macro-F1 | 0.500 | Higher is better |

### Models

| Model | Provider | Conditions Run |
|-------|----------|----------------|
| `gemini-2.5-flash` | Google (free tier) | A, B, C — all 4 tasks |
| `gpt-5.4-mini` | OpenAI | A, B, C — all 4 tasks |
| `claude-sonnet-4-6` | Anthropic | A only — all 4 tasks |

---

## Results Summary (28 / 36 runs completed)

### Condition A — Baseline

| Task | Gemini 2.5 Flash | GPT-5.4-mini | Claude Sonnet 4.6 |
|------|-----------------|--------------|-------------------|
| house-price (MAE ↓) | 19,919 ✓ | 23,301 ✓ | **16,679** ✓ |
| spaceship-titanic (Acc ↑) | 0.711 ✗ | 0.769 ✓ | **0.828** ✓ |
| vectorization (Speedup ↑) | **6.99×** ✓ | 0.005× ✗ | 1.26× ✓ |
| feedback (F1 ↑) | 0.562 ✓ | DNF | **0.580** ✓ |

### Condition B — Multi-attempt (no reflection)

| Task | Gemini 2.5 Flash | GPT-5.4-mini |
|------|-----------------|--------------|
| house-price (MAE ↓) | 20,595 ✓ | 17,308 ✓ |
| spaceship-titanic (Acc ↑) | 0.628 ✗ | 0.827 ✓ |
| vectorization (Speedup ↑) | 13.74× ✓ | 1.26× ✓ |
| feedback (F1 ↑) | 0.589 ✓ | 0.546 ✓ |

### Condition C — Reflexion

| Task | Gemini 2.5 Flash | GPT-5.4-mini |
|------|-----------------|--------------|
| house-price (MAE ↓) | **16,682** ✓ | 17,529 ✓ |
| spaceship-titanic (Acc ↑) | 0.822 ✓ | **0.835** ✓ |
| vectorization (Speedup ↑) | **14.31×** ✓ | 1.28× ✓ |
| feedback (F1 ↑) | **0.595** ✓ | 0.561 ✓ |

**Key finding:** Condition C achieves the best score in 7 of 8 model–task pairs.
Gemini spaceship-titanic: A=0.711 (fail) → B=0.628 (fail worse) → C=0.822 (beat).
This is direct proof that Reflexion — not extra compute — drives improvement.

---

## Repository Layout

```
project/
├── agent/
│   ├── llm.py              # Unified LLM client (Anthropic / OpenAI / Gemini)
│   ├── base_agent.py       # Single-attempt runner
│   ├── reflexion_agent.py  # Condition C: multi-attempt + post-mortem loop
│   ├── prompts.py          # Post-mortem prompt templates
│   └── ml_knowledge.py     # Anti-hallucination knowledge prompt
├── evaluation/
│   ├── run_experiment.py   # CLI: run one condition × task × model
│   ├── run_all.py          # Run all combinations
│   └── metrics.py          # Score extraction and success evaluation
├── results/
│   ├── A/                  # Baseline results (12 JSON files)
│   ├── B/                  # Multi-attempt results (8 JSON files)
│   └── C/                  # Reflexion results (8 JSON files)
├── papers/
│   └── research_paper.tex  # Final IEEE-format paper (compile in Overleaf)
├── implementation/
│   └── MLAgentBench/       # Benchmark (submodule, not modified)
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv implementation/venv2
source implementation/venv2/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install lightgbm

# 3. Set API keys
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
```

---

## Running Experiments

```bash
source implementation/venv2/bin/activate

# Single run
python evaluation/run_experiment.py \
  --condition C --task house-price \
  --model gemini-2.5-flash --steps 30 --attempts 3

# All combinations (skips already-completed runs)
python evaluation/run_all.py --steps 30 --attempts 3

# Dry run (preview without executing)
python evaluation/run_all.py --dry-run
```

**Valid tasks:** `house-price`, `spaceship-titanic`, `vectorization`, `feedback`
**Valid conditions:** `A`, `B`, `C`
**Supported models:** `gemini-2.5-flash`, `gpt-5.4-mini`, `claude-sonnet-4-6`

Results are saved to `results/{condition}/{task}_{model_slug}.json`.
Already-completed result files are automatically skipped on re-run.

---

## Hallucination Metric

An **action-level hallucination** is defined as any agent step where the
Fact Check field claims "Confirmed" improvement while the same step's
Observation contains an error, traceback, or empty output.

**Result:** Zero hallucinations detected across all 28 completed runs.
