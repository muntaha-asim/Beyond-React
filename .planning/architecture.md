# System Architecture

## High-Level Overview

```
project/
├── agent/                  ← our code: wrappers, prompts, runners
├── evaluation/             ← experiment orchestration and metrics
├── implementation/
│   └── MLAgentBench/       ← cloned benchmark (DO NOT MODIFY)
├── results/                ← output JSONs, git-ignored
├── logs/                   ← per-run trajectory logs, git-ignored
└── .planning/              ← this folder
```

---

## Module Roles

### agent/llm.py
- Unified `complete(prompt, model, ...)` function for Claude, Gemini, GPT-4o
- `patch_mlagentbench(model)` — monkey-patches MLAgentBench's internal
  `complete_text` / `complete_text_fast` to route through our wrapper
- `_stub_missing_modules()` — stubs torch, transformers, helm, vertexai so
  MLAgentBench can be imported without installing those heavy packages
- API clients: Anthropic SDK, OpenAI SDK, google-genai SDK
- Keys loaded from .env via python-dotenv

### agent/ml_knowledge.py
- `ML_KNOWLEDGE_PROMPT` — a rich, structured prompt block injected into every
  agent run before the task description
- Covers: anti-hallucination rules, task-specific playbooks (house-price,
  spaceship-titanic, vectorization, feedback), common ML failure modes,
  model selection guide, efficient execution rules

### agent/prompts.py
- `REFLECTION_PROMPT` — sent to the LLM after a failed attempt, asks it to
  produce a structured post-mortem (strategies tried / why failed /
  hypothesis / what to avoid)
- `REFLECTION_CONTEXT_TEMPLATE` — wraps the post-mortem before injecting
  it into the next attempt's initial prompt

### agent/base_agent.py
- `run_single_attempt(task, model, attempt_idx, ...)` — runs one ResearchAgent
  attempt on one task; used by Conditions A and B
- Always prepends ML_KNOWLEDGE_PROMPT to the agent's initial prompt
- If `system_prompt_prefix` is passed (from Reflexion), it goes between
  ML knowledge and the task description
- `get_task_score(task, log_dir)` — calls the task's eval script and returns
  the numeric score

### agent/reflexion_agent.py
- `run_reflexion(task, model, ...)` — Condition C runner
- Loop: run attempt → check success → if failed, generate post-mortem →
  inject into next attempt
- `_generate_post_mortem()` — calls complete() with REFLECTION_PROMPT,
  saves the raw reflection to disk
- Stops early if any attempt succeeds
- Returns dict with all attempts + attempts_to_success

### evaluation/run_experiment.py
- CLI entry point: `--condition A/B/C --task --model --steps --attempts`
- Dispatches to base_agent or reflexion_agent
- Saves result JSON to `results/{condition}/{task}_{model_slug}.json`
- Skips if result file already exists (safe to resume)

### evaluation/run_all.py
- Runs all conditions × tasks × models (24 combinations by default)
- `--dry-run` flag to preview without executing
- Prints a summary table at the end
- Saves `results/summary.json`

### evaluation/metrics.py
- `compute_metrics(result_path)` — reads a result JSON and returns:
  success (bool), score_delta, hallucination_count, attempts_to_success

---

## Prompt Injection Order (every run)

```
[1] ML_KNOWLEDGE_PROMPT          ← always first (ml_knowledge.py)
[2] Reflection post-mortem       ← only Condition C, attempt 2+
[3] MLAgentBench task prompt     ← the original ResearchAgent task description
```

---

## Data Flow for One Run

```
run_experiment.py
    → creates log_dir/workspace
    → calls run_single_attempt() or run_reflexion()
        → patch_mlagentbench() stubs bad imports, replaces complete_text
        → Environment(args) sets up the task sandbox
        → ResearchAgent.run() steps through the task
            → each step: LLM call via our complete()
            → observations written to env_log/
        → env.save("final") persists workspace
        → get_task_score() calls task eval script → numeric score
    → result dict serialized to results/{C}/{task}_{model}.json
```

---

## Models in Use

| Model ID | Provider | Role |
|---|---|---|
| `claude-3-5-sonnet-20241022` | Anthropic | Main agent (primary experiment) |
| `gpt-4o` | OpenAI | Main agent (comparison) |
| `gemini-1.5-pro` | Google | Main agent (testing / smoke runs) |
| `gemini-2.0-flash` | Google | Fast summarization when main=gemini |
| `claude-3-5-haiku-20241022` | Anthropic | Fast summarization when main=claude/gpt |

---

## Key File: MLAgentBench ResearchAgent

`implementation/MLAgentBench/MLAgentBench/agents/agent_research.py`

This is the base agent we build on. Do not modify it. Key attributes:
- `agent.initial_prompt` — the string we prepend our knowledge + reflection to
- `agent.history_steps` — list of {step_idx, action, observation} dicts
- `agent.run(env)` — main loop, returns a final message string
