"""
Modern LLM wrapper supporting Claude 3.5 Sonnet and GPT-4o.
Reads API keys from .env and patches MLAgentBench.LLM so the
existing ResearchAgent code works without modification.
"""
import os
import sys

from dotenv import load_dotenv
load_dotenv()

import anthropic as _anthropic_sdk
import openai as _openai_sdk

_anthropic_client = _anthropic_sdk.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
_openai_client = _openai_sdk.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# Fast/cheap model used for summarization within the agent loop
FAST_MODEL = "claude-3-5-haiku-20241022"


def complete(
    prompt: str,
    model: str,
    stop_sequences: list = None,
    max_tokens: int = 2000,
    temperature: float = 0.5,
    log_file: str = None,
) -> str:
    """Call Claude or GPT-4o and return the completion string."""
    stop_sequences = stop_sequences or []

    if model.startswith("claude"):
        kwargs = {}
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences
        response = _anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        completion = response.content[0].text
    else:
        response = _openai_client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            stop=stop_sequences if stop_sequences else None,
        )
        completion = response.choices[0].message.content

    # Manual stop-sequence trim (belt-and-suspenders for Claude)
    for seq in stop_sequences:
        if seq in completion:
            completion = completion[: completion.index(seq)]

    if log_file:
        _write_log(log_file, prompt, completion, model)

    return completion


def _write_log(log_file: str, prompt: str, completion: str, model: str):
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
        with open(log_file, "a") as f:
            f.write(f"\n===PROMPT===\n{prompt}\n===RESPONSE ({model})===\n{completion}\n\n")
    except Exception:
        pass


def patch_mlagentbench(model: str):
    """
    Monkey-patch MLAgentBench.LLM (and its consumers) to use our modern API
    wrapper instead of the legacy completions API + text-file key loading.

    Must be called BEFORE any MLAgentBench agent is constructed.
    """
    mlab_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "implementation", "MLAgentBench")
    )
    if mlab_root not in sys.path:
        sys.path.insert(0, mlab_root)

    import MLAgentBench.LLM as _mlab_llm
    import MLAgentBench.agents.agent as _agent_mod
    import MLAgentBench.agents.agent_research as _research_mod
    import MLAgentBench.high_level_actions as _hl_mod

    # ------------------------------------------------------------------
    # complete_text(prompt, log_file, model_name, **kwargs)
    # Called by ResearchAgent with self.args.llm_name as the third arg.
    # ------------------------------------------------------------------
    def _complete_text(prompt, log_file, model_name=model, **kwargs):
        return complete(
            prompt=prompt,
            model=model_name,
            stop_sequences=["Observation:"],
            max_tokens=kwargs.get("max_tokens_to_sample", 2000),
            temperature=kwargs.get("temperature", 0.5),
            log_file=log_file,
        )

    # ------------------------------------------------------------------
    # complete_text_fast(prompt, log_file=..., **kwargs)
    # Used for summarisation — always uses the cheap fast model.
    # ------------------------------------------------------------------
    def _complete_text_fast(prompt, log_file=None, **kwargs):
        return complete(
            prompt=prompt,
            model=FAST_MODEL,
            stop_sequences=[],
            max_tokens=2000,
            temperature=0.01,
            log_file=log_file,
        )

    # Patch module-level names (affects future imports)
    _mlab_llm.complete_text = _complete_text
    _mlab_llm.complete_text_fast = _complete_text_fast
    _mlab_llm.FAST_MODEL = FAST_MODEL

    # Patch already-imported references in sub-modules
    _agent_mod.complete_text = _complete_text
    _research_mod.complete_text = _complete_text
    _research_mod.complete_text_fast = _complete_text_fast
    _hl_mod.complete_text = _complete_text
    _hl_mod.complete_text_fast = _complete_text_fast
