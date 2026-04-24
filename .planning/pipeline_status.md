# Pipeline Status

Last updated: 2026-04-24

---

## What Is Working

| Component | Status | Notes |
|---|---|---|
| All Python imports end-to-end | PASSING | Verified with full import test |
| Anthropic SDK (`anthropic>=0.40`) | READY | Key in .env |
| OpenAI SDK (`openai>=1.50`) | READY | Key in .env |
| Google SDK (`google-genai>=1.0`) | READY | Switched from deprecated google-generativeai |
| Gemini API key | READY | Key in .env |
| MLAgentBench import | READY | torch/transformers/helm stubbed out |
| patch_mlagentbench() | READY | Replaces complete_text in all sub-modules |
| ML knowledge prompt injection | READY | Prepended to every agent run |
| Reflexion post-mortem generation | READY | prompts.py + reflexion_agent.py |
| Reflection context injection | READY | Condition C, attempt 2+ |
| Result JSON serialization | READY | results/{condition}/{task}_{model}.json |
| Resume/skip logic | READY | Skips existing result files |
| Summary table + summary.json | READY | run_all.py |
| Metrics extraction | READY | evaluation/metrics.py |
| 503 retry logic | READY | 5 retries × 15s backoff in llm.py complete() |
| Kaggle CLI + credentials | READY | ~/.kaggle/kaggle.json in place |
| lightgbm installed in venv2 | READY | Agent scripts use it successfully |

---

## Gemini Model Names (as of 2026-04-24)

| Model | Status | Use for |
|---|---|---|
| `gemini-2.5-flash` | WORKING | Main agent (free tier) |
| `gemini-2.5-flash-lite` | WORKING | Fast/summarization model |
| `gemini-2.5-pro` | WORKING | Higher quality (may cost) |
| `gemini-2.0-flash` | DEAD — 404 new users | Do not use |
| `gemini-2.0-flash-lite` | DEAD — 404 new users | Do not use |
| `gemini-2.0-flash-lite-001` | DEAD — 404 new users | Do not use |
| `gemini-1.5-flash` | DEAD — 404 new users | Do not use |

---

## Completed Runs

| Condition | Task | Model | Score | Baseline | Beat? | Date |
|---|---|---|---|---|---|---|
| A | house-price | gemini-2.5-flash | 19,918 MAE | 30,000 | YES (−34%) | 2026-04-24 |

---

## Pending / Not Yet Done (priority order)

| Item | Priority | Command |
|---|---|---|
| Condition C — house-price — gemini-2.5-flash | HIGH | `python evaluation/run_experiment.py --condition C --task house-price --model gemini-2.5-flash --steps 30 --attempts 3` |
| Condition B — house-price — gemini-2.5-flash | HIGH | `python evaluation/run_experiment.py --condition B --task house-price --model gemini-2.5-flash --steps 30 --attempts 3` |
| All conditions — other 3 tasks — gemini | MEDIUM | Run run_all.py with --model gemini-2.5-flash |
| All conditions — claude-3-5-sonnet-20241022 | HIGH | Same commands with Claude model (costs $) |
| All conditions — gpt-4o | HIGH | Same commands with GPT-4o (costs $) |
| Analysis notebook / script | MEDIUM | For paper tables and plots |
| Add lightgbm to requirements.txt | LOW | Already installed, just not in file yet |

---

## Known Issues (non-fatal)

| Issue | Severity | Notes |
|---|---|---|
| Python 3.9 EOL warnings from google-genai | LOW | Noise only |
| MLAgentBench key file warnings (crfm, etc.) | LOW | Noise only — our .env is used |
| `Response is invalid and discarded` at some steps | LOW | MLAgentBench retries internally, run continues |
| Pyright import warnings in base_agent.py | LOW | Runtime imports work fine |

---

## How to Resume (start of next session)

```bash
cd /Users/muntahaasim/Documents/University/AgenticAI/agenticai/project
source implementation/venv2/bin/activate

# Next run to do — Condition C (Reflexion) on house-price
python evaluation/run_experiment.py --condition C --task house-price --model gemini-2.5-flash --steps 30 --attempts 3
```
