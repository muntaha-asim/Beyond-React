# Key Design Decisions

A record of non-obvious choices made during the project and the reasoning behind them.

---

## D1 — Monkey-patch MLAgentBench instead of forking it

**Decision:** We do not modify any file inside `implementation/MLAgentBench/`.
Instead, `patch_mlagentbench()` replaces `complete_text` at runtime.

**Why:** Keeping the benchmark unmodified means our Condition A exactly replicates the
original paper's setup. Any result difference is attributable to our changes, not to
modifications in the benchmark itself. It also makes it easy to update MLAgentBench
without merge conflicts.

**Tradeoff:** The patch is fragile — if MLAgentBench refactors its internal imports,
we need to update the patch. Acceptable for a course project.

---

## D2 — Stub torch/transformers/helm instead of installing them

**Decision:** `_stub_missing_modules()` injects fake modules into `sys.modules` before
MLAgentBench is imported, preventing import errors for packages we don't use.

**Why:** `transformers` pulls in PyTorch (>2GB). `helm` is a Stanford-specific research
package not on PyPI. Neither is needed — we patch `complete_text` before those code
paths are ever reached.

**Tradeoff:** If a future MLAgentBench update actually calls transformer-based inference
before our patch runs, the stubs will silently return None and crash in a confusing way.
The fix would be to install the real packages at that point.

---

## D3 — ML knowledge prompt always injected, not optional

**Decision:** `ML_KNOWLEDGE_PROMPT` is prepended to every agent run (Conditions A, B, C)
not just Condition C.

**Why:** We want the anti-hallucination rules and model selection guide to apply to all
conditions. This makes the comparison fair — all agents have the same domain grounding.
It also gives Condition A a better chance of succeeding, which makes our baseline more
credible (if we beat a stronger baseline, the contribution is more meaningful).

**Tradeoff:** This deviates from exact MLAgentBench paper replication. If reviewers ask,
we note that Condition A + ML knowledge is our "improved baseline" and acknowledge the
difference.

---

## D4 — Reflection post-mortem capped at 1000 tokens

**Decision:** `_generate_post_mortem()` uses `max_tokens=1000`.

**Why:** The post-mortem is prepended to the next attempt's prompt. Longer reflections
eat into the context window available for the actual task. 1000 tokens is enough to cover
all four sections (strategies tried, why failed, hypothesis, what to avoid) concisely.

**Tradeoff:** A very complex failed attempt might benefit from a longer reflection.
If hallucination rates remain high in results, increase to 1500.

---

## D5 — Trajectory window capped at last 20 steps for reflection

**Decision:** `_format_trajectory()` only sends the last 20 steps to the reflection LLM.

**Why:** Full 30-step trajectories can exceed 10k tokens when formatted. The most
informative steps for failure analysis are the recent ones (where the agent made its
final attempts and saw the final scores).

**Tradeoff:** Early strategy attempts (steps 1-10) are dropped. If an agent tried
something useful early and abandoned it, the reflection won't know. Acceptable given
token constraints.

---

## D6 — gemini-2.0-flash as fast model for Gemini runs

**Decision:** When the main model starts with "gemini", `complete_text_fast` uses
`gemini-2.0-flash` instead of Claude Haiku.

**Why:** Summarization within MLAgentBench should use the same provider to avoid
cross-provider inconsistency in the trajectory (and to avoid needing two API keys
simultaneously). `gemini-2.0-flash` is fast and free-tier eligible.

**Tradeoff:** Gemini Flash's summarization quality may differ from Haiku's. This is
acceptable for summarization (not the main agent loop).

---

## D7 — Condition B does not carry scores between attempts

**Decision:** In Condition B, each of the 3 attempts starts fresh with no information
from prior attempts. The `final_score` reported is the last attempt's score.

**Why:** This isolates the variable. B tests "does more compute help?" — if we gave B
access to scores from prior attempts, it would be a weaker control for Condition C's
structured reflection.

**Tradeoff:** B is a pessimistic control (a human would at least try different things
each time even without structured reflection). This is intentional — it makes C's
advantage more conservative.
