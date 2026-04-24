# Risks and Mitigations

---

## R1 — API costs exceed budget

**Risk:** 24 full runs × ~$2-5 per run = $48-120 total. With $5 Anthropic credit,
only a few runs are feasible.

**Likelihood:** High for full experiment
**Impact:** Cannot run all 24 combinations

**Mitigation:**
- Use Gemini (free tier) for all smoke tests and debugging
- Run full experiments only with Claude + GPT-4o
- Reduce --steps to 15-20 if budget is tight (costs ~50% less)
- Run Condition A first (cheapest: 1 attempt) to verify pipeline before B/C

---

## R2 — get_task_score() returns None for all tasks

**Risk:** MLAgentBench's eval scripts may have changed paths or expect specific
file structures that our run doesn't produce.

**Likelihood:** Medium
**Impact:** Cannot measure success — entire experiment is invalid

**Mitigation:**
- After first smoke run, manually inspect the log_dir for eval output files
- Check `env_log/traces/step_final_files/` exists and has content
- If score extraction fails, write a fallback parser that reads the last
  "SCORE: <value>" line from the agent's final output message

---

## R3 — ResearchAgent crashes mid-run due to API changes

**Risk:** MLAgentBench was written for older Anthropic/OpenAI APIs. Our patch
covers `complete_text` and `complete_text_fast` but there may be other call sites.

**Likelihood:** Low (we patched the key functions)
**Impact:** Runs crash silently with no output

**Mitigation:**
- Run with --steps 5 first to catch crashes early
- Check logs/ directory for any LLM calls that route to the stub instead of our wrapper
- If new call sites found, add them to patch_mlagentbench()

---

## R4 — Hallucination metric is inaccurate

**Risk:** The hallucination counter in metrics.py uses a heuristic (Fact Check claims
improvement but observation doesn't). This heuristic may produce false positives or
miss hallucinations.

**Likelihood:** Medium
**Impact:** Weakens the paper's hallucination analysis

**Mitigation:**
- Manually review 2-3 trajectories to validate the heuristic
- If heuristic is unreliable, fall back to reporting only success rate and score delta
- Consider a secondary metric: % of runs where claimed score != eval score

---

## R5 — Reflection post-mortem quality is poor

**Risk:** The LLM-generated post-mortems may be generic ("try a better model") rather
than specific, making Condition C no better than B.

**Likelihood:** Medium (depends on how well REFLECTION_PROMPT guides the model)
**Impact:** C ≈ B, weakening our contribution

**Mitigation:**
- After first C run, manually read the reflection files in logs/
- If quality is poor, strengthen REFLECTION_PROMPT with few-shot examples
- The ML_KNOWLEDGE_PROMPT already gives the agent vocabulary for specific strategies

---

## R6 — `vectorization` task is hard to score reliably

**Risk:** Speedup measurement depends on machine load and run-to-run variance.
A "2×" speedup on one run might be 1.5× on another.

**Likelihood:** Medium
**Impact:** Noisy results for this specific task

**Mitigation:**
- Run vectorization multiple times and average
- Report variance alongside mean speedup
- If too noisy, exclude vectorization from primary analysis and note it in paper

---

## R7 — Python 3.9 incompatibility with google-genai

**Risk:** google-genai posts FutureWarnings about Python 3.9 EOL. May eventually
drop support.

**Likelihood:** Low (currently just warnings, not errors)
**Impact:** Gemini calls fail

**Mitigation:**
- Currently working fine despite warnings
- If it breaks: upgrade to Python 3.11 and rebuild venv
