"""
All LLM prompts used by the agent system.
"""

# ------------------------------------------------------------------
# POST-RUN REFLECTION PROMPT
# Sent to the LLM after a failed attempt to produce a structured
# post-mortem that is prepended to the next attempt's system prompt.
# ------------------------------------------------------------------
REFLECTION_PROMPT = """\
You just completed an ML experimentation run on the following task and did not beat the baseline.

TASK: {task}
BASELINE SCORE: {baseline_score}
YOUR FINAL SCORE: {final_score}
RESULT: Failed (did not beat baseline)

Here is your trajectory from that attempt (recent steps):
{trajectory}

Write a structured post-mortem with EXACTLY these four sections:

1. STRATEGIES TRIED
   List each distinct approach you attempted (e.g., "tried XGBoost with default params", "added feature engineering").

2. WHY THEY FAILED
   For each strategy, explain concretely why it did not improve the score.
   Reference actual error messages, output values, or observations you saw.

3. HYPOTHESIS FOR NEXT ATTEMPT
   State ONE specific, concrete thing to try next and why you believe it will work.
   Be specific: name the algorithm, hyperparameter, or code change.

4. WHAT TO AVOID
   List specific actions or approaches that wasted time or made things worse.
   The next agent will read this list and skip those strategies.

Be concise but specific. Vague advice like "try a better model" is not useful.
"""


# ------------------------------------------------------------------
# REFLECTION CONTEXT HEADER
# Wraps the LLM-generated post-mortem before prepending to the
# next attempt's initial system prompt.
# ------------------------------------------------------------------
REFLECTION_CONTEXT_TEMPLATE = """\
=== REFLECTION FROM PREVIOUS ATTEMPT {attempt_num} ===
The following is a structured analysis of what was tried and why it failed.
Read this carefully before starting — do not repeat the same mistakes.

{reflection_text}

=== END REFLECTION ===

Now begin a NEW attempt. Use the reflection above to guide your strategy.
Do NOT repeat strategies listed under "WHAT TO AVOID".
"""
