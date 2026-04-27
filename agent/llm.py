"""
Modern LLM wrapper supporting Claude 3.5 Sonnet, GPT-4o, and Gemini.
Reads API keys from .env and patches MLAgentBench.LLM so the
existing ResearchAgent code works without modification.
"""
import os
import sys
import time
import types

from dotenv import load_dotenv
load_dotenv()

import anthropic as _anthropic_sdk
import openai as _openai_sdk
from google import genai as _gemini_sdk
from google.genai import types as _gemini_types

_anthropic_client = _anthropic_sdk.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
_openai_client = _openai_sdk.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
_gemini_client = _gemini_sdk.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Fast/cheap model used for summarization within the agent loop
FAST_MODEL = "claude-haiku-4-5-20251001"


def complete(
    prompt: str,
    model: str,
    stop_sequences: list = None,
    max_tokens: int = 2000,
    temperature: float = 0.5,
    log_file: str = None,
) -> str:
    """Call Claude, Gemini, or GPT-4o and return the completion string."""
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
    elif model.startswith("gemini"):
        # Retry up to 10 times on 503 (server overload) with exponential backoff
        for attempt in range(10):
            try:
                response = _gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=_gemini_types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        stop_sequences=stop_sequences if stop_sequences else None,
                    ),
                )
                completion = response.text or ""
                break
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = min(300, 30 * (attempt + 1))
                    print(f"[llm] Gemini overload — retrying in {wait}s (attempt {attempt+1}/10)")
                    time.sleep(wait)
                    if attempt == 9:
                        raise
                else:
                    raise
    else:
        for attempt in range(10):
            try:
                new_model = any(m in model for m in ["o1", "o3", "o4", "gpt-5"])
                kwargs = dict(
                    model=model,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                if not new_model:
                    kwargs["stop"] = stop_sequences if stop_sequences else None
                if new_model:
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens
                response = _openai_client.chat.completions.create(**kwargs)
                completion = response.choices[0].message.content
                break
            except Exception as e:
                if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                    kwargs.pop("max_tokens", None)
                    kwargs["max_completion_tokens"] = max_tokens
                    response = _openai_client.chat.completions.create(**kwargs)
                    completion = response.choices[0].message.content
                    break
                elif "429" in str(e) or "rate_limit" in str(e).lower():
                    wait = min(300, 30 * (attempt + 1))
                    print(f"[llm] OpenAI rate limit — retrying in {wait}s (attempt {attempt+1}/10)")
                    time.sleep(wait)
                    if attempt == 9:
                        raise
                else:
                    raise

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


class _StubClass:
    """Generic stub base class — satisfies isinstance checks and subclassing."""
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return None


def _stub_missing_modules():
    """
    MLAgentBench's LLM.py imports transformers, torch, helm, and vertexai at
    the top level even when they are not used. Stub them so the import does not
    crash. We replace complete_text immediately after, so these stubs are never
    actually invoked.
    """
    # torch stub — needs a real 'no_grad' context manager and a non-None __spec__
    # so datasets/importlib checks don't treat the module as broken.
    if "torch" not in sys.modules:
        import contextlib
        import importlib.util
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = _StubClass
        torch_stub.tensor = lambda *a, **kw: None
        torch_stub.no_grad = contextlib.nullcontext
        torch_stub.manual_seed = lambda *a, **kw: None
        torch_stub.cuda = types.ModuleType("torch.cuda")
        torch_stub.device = _StubClass
        torch_stub.__spec__ = importlib.util.spec_from_loader("torch", loader=None)
        sys.modules["torch"] = torch_stub

    # transformers stub — StoppingCriteria must be a real base class
    if "transformers" not in sys.modules:
        tf_stub = types.ModuleType("transformers")
        tf_stub.AutoModelForCausalLM = _StubClass
        tf_stub.AutoTokenizer = _StubClass
        tf_stub.StoppingCriteria = _StubClass
        tf_stub.StoppingCriteriaList = _StubClass
        sys.modules["transformers"] = tf_stub

    # helm + vertexai — plain stubs, never subclassed
    for name, attrs in [
        ("helm", []),
        ("helm.common", []),
        ("helm.common.authentication", ["Authentication"]),
        ("helm.common.request", ["Request", "RequestResult"]),
        ("helm.proxy", []),
        ("helm.proxy.accounts", ["Account"]),
        ("helm.proxy.services", []),
        ("helm.proxy.services.remote_service", ["RemoteService"]),
        ("vertexai", []),
        ("vertexai.preview", []),
        ("vertexai.preview.generative_models", ["GenerativeModel", "Part"]),
    ]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for attr in attrs:
                setattr(mod, attr, _StubClass)
            sys.modules[name] = mod


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

    # Stub heavy/absent dependencies before MLAgentBench imports them
    _stub_missing_modules()

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
    # Used for summarisation — uses a cheap fast model matched to the provider.
    # ------------------------------------------------------------------
    fast_model = "gemini-2.5-flash-lite" if model.startswith("gemini") else FAST_MODEL

    def _complete_text_fast(prompt, log_file=None, **kwargs):
        return complete(
            prompt=prompt,
            model=fast_model,
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
