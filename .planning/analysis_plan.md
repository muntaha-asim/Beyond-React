# Analysis Plan

How to turn raw result JSONs into paper tables and conclusions.

---

## Step 1 — Collect Results

After all runs complete, results live in:
```
results/
├── A/
│   ├── house-price_sonnet35.json
│   ├── house-price_gpt4o.json
│   └── ...
├── B/
└── C/
```

Run `python evaluation/run_all.py --dry-run` to see what's missing.
Run `python evaluation/run_all.py` to fill gaps (skips completed runs).

---

## Step 2 — Compute Metrics

```python
from evaluation.metrics import compute_metrics
import os, json

results_dir = "results"
rows = []
for cond in ["A", "B", "C"]:
    for f in os.listdir(f"{results_dir}/{cond}"):
        m = compute_metrics(f"{results_dir}/{cond}/{f}")
        rows.append(m)

import pandas as pd
df = pd.DataFrame(rows)
print(df)
```

Each row has: `condition, task, model, success, score_delta, hallucination_count, attempts_to_success`

---

## Step 3 — Primary Tables for the Paper

### Table 1 — Success Rate by Condition
Aggregate success (True/False) across all tasks per condition per model.

| Model | Cond A | Cond B | Cond C | C vs B |
|---|---|---|---|---|
| Claude 3.5 Sonnet | X/4 | X/4 | X/4 | +N |
| GPT-4o | X/4 | X/4 | X/4 | +N |

### Table 2 — Score Delta by Task and Condition
Show how much each condition improved over the task baseline.

| Task | Metric | A | B | C |
|---|---|---|---|---|
| house-price | MAE Δ | | | |
| spaceship-titanic | Acc Δ | | | |
| vectorization | Speedup | | | |
| feedback | F1 Δ | | | |

### Table 3 — Hallucination Count
| Model | Cond A | Cond B | Cond C | C vs A |
|---|---|---|---|---|
| Claude 3.5 Sonnet | N | N | N | −N |
| GPT-4o | N | N | N | −N |

### Table 4 — Attempts to Success (Condition C only)
How often did Reflexion help on attempt 2 vs 3?

| Task | Solved on Attempt 1 | Attempt 2 | Attempt 3 | Never |
|---|---|---|---|---|

---

## Step 4 — Key Claims to Test

**Claim 1:** C > B on success rate  
→ If true: Reflexion helps beyond just more compute  
→ If false: More attempts alone explain the improvement

**Claim 2:** C < A on hallucination count  
→ If true: Reflexion reduces hallucination  
→ If false: Hallucination is not addressable by verbal reflection alone

**Claim 3:** Score delta C > A > ... (C improves most)  
→ Supports the thesis quantitatively

**Claim 4:** Attempts to success peaks at attempt 2 (not 3)  
→ Shows reflection has immediate effect, not just random variation

---

## Step 5 — Failure Analysis

For every C run where success=False after 3 attempts:
1. Read `logs/{run_id}/reflection_after_attempt_1.txt`
2. Check: was the reflection specific? Did attempt 2 avoid the flagged strategies?
3. If attempt 2 repeated the same mistakes → reflection quality issue
4. If attempt 2 tried new things but still failed → task is too hard / not enough steps

Include 1-2 qualitative examples in the paper as case studies.

---

## Step 6 — Paper Contribution Structure

```
4. Results
  4.1 Success Rate (Table 1) — C outperforms A and B
  4.2 Score Improvement (Table 2) — task-level breakdown
  4.3 Hallucination Reduction (Table 3) — Reflexion's anti-hallucination effect
  4.4 Reflection Quality (case study) — why it worked / where it didn't
5. Discussion
  5.1 Reflexion vs More Compute (C vs B interpretation)
  5.2 Limitations: token cost, reflection quality variance, CPU-only tasks
  5.3 Future Work: fine-tuning on trajectories, multi-agent debate
```

---

## Plotting (optional, for slides)

```python
import matplotlib.pyplot as plt

# Bar chart: success rate per condition per model
fig, ax = plt.subplots()
# ... group df by condition, plot success rate

# Line chart: hallucination count A → B → C
# ... show downward trend

# Scatter: score_delta vs hallucination_count
# ... show negative correlation for Condition C
```
