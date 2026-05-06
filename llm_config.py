import os
import time
import random
from dotenv import load_dotenv

load_dotenv()

_CONFIG_WARNINGS_EMITTED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _CONFIG_WARNINGS_EMITTED:
        return
    _CONFIG_WARNINGS_EMITTED.add(key)
    print(message)


# ── Gemini retry helper ────────────────────────────────────────────────────────

def _gemini_retry_call(fn, *args, **kwargs):
    """
    Retry a Gemini API call on transient errors (503 Unavailable, 429 Too Many
    Requests, 500 Internal Server Error) using exponential back-off with jitter.

    Configuration via environment variables:
        GEMINI_MAX_RETRIES  – maximum number of retries (default: 5)
        GEMINI_RETRY_BASE   – initial delay in seconds   (default: 2.0)
        GEMINI_RETRY_MAX    – maximum delay in seconds   (default: 60.0)
    """
    max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
    base_delay  = float(os.getenv("GEMINI_RETRY_BASE", "2.0"))
    max_delay   = float(os.getenv("GEMINI_RETRY_MAX",  "60.0"))
    _RETRYABLE = {429, 500, 503}

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            status = getattr(e, "status_code", None)
            # openai SDK wraps HTTP errors; also check message for 503 strings
            if status not in _RETRYABLE:
                msg = str(e).lower()
                if not any(str(s) in msg for s in _RETRYABLE):
                    raise
            last_exc = e
            if attempt >= max_retries:
                break
            delay = min(base_delay * (2 ** attempt) + random.uniform(0.0, 1.0), max_delay)
            print(
                f"[RETRY] Gemini API transient error"
                f" (status={status}, attempt {attempt + 1}/{max_retries})."
                f" Retrying in {delay:.1f}s … | {e}"
            )
            time.sleep(delay)
    raise last_exc

_DEFAULT_CHAT_MODEL = "deepseek-chat"

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_chat_model(default: str = _DEFAULT_CHAT_MODEL) -> str:
    """
    Returns the chat completion model name from environment variables.
    Falls back to `default` (deepseek-chat) when nothing is configured.
    """
    candidate = (
        os.getenv("LLM_CHAT_MODEL")
        or os.getenv("OPENAI_MODEL_NAME")
        or os.getenv("DEEPSEEK_MODEL_NAME")
    )
    value = candidate.strip() if candidate else ""
    return value or default


def get_agent_model(default: str = _DEFAULT_CHAT_MODEL) -> str:
    """
    Returns the model name for the agent (planner / executor).
    Priority: AGENT_MODEL_NAME → default fallback.
    """
    candidate = os.getenv("AGENT_MODEL_NAME")
    value = candidate.strip() if candidate else ""
    return value or get_chat_model(default)


def get_retrieve_model(default: str = _DEFAULT_CHAT_MODEL) -> str:
    """
    Returns the model name for retrieval toolbox functions.
    Retrieval is intentionally unified with the agent model.
    """
    return get_agent_model(default)


def get_closeqa_model(default: str = _DEFAULT_CHAT_MODEL) -> str:
    """
    Returns the model name for closeQA (open answer → option mapping).
    CloseQA is intentionally unified with the agent model.
    """
    return get_agent_model(default)


def is_deepseek_model(model_name: str | None) -> bool:
    return bool(model_name and "deepseek" in model_name.lower())


def is_gemini_model(model_name: str | None) -> bool:
    return bool(model_name and model_name.lower().startswith("gemini"))


def _resolve_deepseek_reasoning(raw: str) -> tuple[str | None, str]:
    """
    Map project-wide REASONING_EFFORT values to DeepSeek-V4 values.

    DeepSeek docs (Create Chat Completion) allow:
      - reasoning_effort: high | max
      - thinking.type: enabled | disabled

    Compatibility mapping:
      - none -> thinking disabled
      - low/medium/high/minimal -> high
      - xhigh/max -> max
    """
    effort = (raw or "").strip().lower()
    if effort in {"none", "off", "disabled", "0", "false"}:
        return None, "disabled"
    if effort in {"xhigh", "max"}:
        return "max", "enabled"
    if effort in {"minimal", "low", "medium", "high"}:
        return "high", "enabled"
    return "high", "enabled"


def _resolve_deepseek_thinking_mode(role: str, default_mode: str) -> str:
    role_key = f"DEEPSEEK_{role.upper()}_THINKING"
    raw = (os.getenv(role_key, "") or os.getenv("DEEPSEEK_THINKING_MODE", "auto")).strip().lower()
    if raw in {"enabled", "disabled"}:
        return raw
    return default_mode


def _build_deepseek_reasoning_config(raw: str, role: str) -> tuple[str | None, str]:
    mapped_effort, mapped_mode = _resolve_deepseek_reasoning(raw)
    mode = _resolve_deepseek_thinking_mode(role, mapped_mode)
    if mode == "disabled":
        return None, "disabled"
    return mapped_effort, mode


def _gemini_supports_minimal(model_name: str | None) -> bool:
    """Gemini 3.1 Pro does NOT support 'minimal' thinking_level; all others do."""
    return "gemini-3.1-pro" not in (model_name or "").lower()


def _resolve_gemini_reasoning_effort(raw: str, model_name: str | None = None) -> str:
    """Clamp REASONING_EFFORT to what the Gemini model supports.

    Mapping (OpenAI reasoning_effort → Gemini thinking_level):
        minimal → minimal  (gemini-3-flash, gemini-3.1-flash-lite, gemini-2.5-*)
                  low      (gemini-3.1-pro — does not support minimal)
        low     → low
        medium  → medium
        high    → high
        xhigh   → high    (no xhigh in Gemini)
        none    → minimal (closest to "off")
    """
    effort = raw.lower()
    if effort == "xhigh":
        effort = "high"
    elif effort == "none":
        effort = "minimal"
    if effort not in ("minimal", "low", "medium", "high"):
        effort = "medium"
    if effort == "minimal" and not _gemini_supports_minimal(model_name):
        effort = "low"
    return effort


def get_chat_client_kwargs(model_name: str | None = None) -> dict:
    model = model_name or get_chat_model()
    if is_deepseek_model(model):
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return kwargs
    if is_gemini_model(model):
        api_key = os.getenv("GEMINI_API_KEY")
        base_url = os.getenv("GEMINI_API_BASE", _GEMINI_BASE_URL)
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return kwargs
    return {}


def is_gpt5_model(model_name: str | None) -> bool:
    """Check if the model is any GPT-5 variant (gpt-5 or gpt-5.4 series)."""
    return bool(model_name and model_name.lower().startswith("gpt-5"))


def is_gpt54_model(model_name: str | None) -> bool:
    """Check if the model is a GPT-5.4 series (gpt-5.4-mini, gpt-5.4-nano, gpt-5.4, ...).
    GPT-5.4 supports 'xhigh' reasoning effort; plain gpt-5 series does not.
    """
    name = model_name.lower() if model_name else ""
    return name.startswith("gpt-5.4")


def _resolve_reasoning_effort(raw: str, supports_xhigh: bool) -> str:
    """Clamp reasoning effort to what the model actually supports.
    gpt-5 series:   minimal, low, medium, high                     (no none, no xhigh)
    gpt-5.4 series: none, minimal, low, medium, high, xhigh        (full range)
    """
    effort = raw.lower()
    if supports_xhigh:
        # gpt-5.4: none / minimal / low / medium / high / xhigh
        return effort if effort in ("none", "minimal", "low", "medium", "high", "xhigh") else "medium"
    else:
        # gpt-5: minimal / low / medium / high  (no none, no xhigh)
        if effort == "none":
            return "minimal"
        if effort == "xhigh":
            return "high"
        return effort if effort in ("minimal", "low", "medium", "high") else "medium"


def _get_reasoning_effort_raw(role: str, model_name: str | None = None) -> str:
    """
    Returns the configured reasoning-effort string for the given role.
    """
    if role == "camera_detection":
        return os.getenv(
            "CAMERA_DETECTION_VLM_REASONING_EFFORT",
            os.getenv("VLM_REASONING_EFFORT", os.getenv("REASONING_EFFORT", "medium")),
        )
    if role == "vlm":
        return os.getenv("VLM_REASONING_EFFORT", os.getenv("REASONING_EFFORT", "medium"))
    if role == "closeqa":
        return os.getenv("CLOSEQA_REASONING_EFFORT", os.getenv("REASONING_EFFORT", "medium"))
    if role == "retrieve":
        return os.getenv("RETRIEVE_REASONING_EFFORT", os.getenv("REASONING_EFFORT", "medium"))
    return os.getenv("REASONING_EFFORT", "medium")


def get_reasoning_params(model_name: str | None = None, role: str = "agent") -> dict:
    """
    Returns flat reasoning params for client.chat.completions.create().
    GPT-5 series : {"reasoning_effort": "..."}
    Gemini series: {"reasoning_effort": "..."}  (maps to thinking_level / thinking_budget)

    Environment variables:
        REASONING_EFFORT: minimal, low, medium, high, xhigh (default: medium)
        VLM_REASONING_EFFORT: override for role="vlm" (optional)
        CAMERA_DETECTION_VLM_REASONING_EFFORT: override for role="camera_detection" (optional)
        gpt-5.4 supports xhigh; Gemini / plain gpt-5 cap at high.
    """
    model = model_name or get_chat_model()
    raw = _get_reasoning_effort_raw(role, model)
    if is_gpt5_model(model):
        effort = _resolve_reasoning_effort(raw, supports_xhigh=is_gpt54_model(model))
        return {"reasoning_effort": effort}
    if is_deepseek_model(model):
        effort, thinking_mode = _build_deepseek_reasoning_config(raw, role)
        params: dict = {"extra_body": {"thinking": {"type": thinking_mode}}}
        if effort is not None:
            params["reasoning_effort"] = effort
        return params
    if is_gemini_model(model):
        effort = _resolve_gemini_reasoning_effort(raw, model)
        return {"reasoning_effort": effort}
    return {}


def get_responses_params(model_name: str | None = None, role: str = "agent") -> tuple[dict, dict]:
    """
    Returns (reasoning_dict, text_dict) for client.responses.create().

    Usage:
        reasoning, text = get_responses_params(model, role="agent")
        client.responses.create(..., reasoning=reasoning, text=text)

    Environment variables:
        REASONING_EFFORT: minimal, low, medium, high, xhigh (default: medium)
        VLM_REASONING_EFFORT: override for role="vlm" (optional)
        CAMERA_DETECTION_VLM_REASONING_EFFORT: override for role="camera_detection" (optional)
        CLOSEQA_REASONING_EFFORT: override for role="closeqa" (optional)
        RETRIEVE_REASONING_EFFORT: override for role="retrieve" (optional)
        VERBOSITY: low, medium, high (default: low)  — GPT-5 only
        CLOSEQA_VERBOSITY: low, medium, high (optional, GPT-5 CloseQA only)
        gpt-5.4 supports xhigh; Gemini / plain gpt-5 cap at high.
    """
    model = model_name or get_chat_model()
    raw = _get_reasoning_effort_raw(role, model)
    if is_gpt5_model(model):
        effort = _resolve_reasoning_effort(raw, supports_xhigh=is_gpt54_model(model))
        if role == "closeqa":
            verbosity = (os.getenv("CLOSEQA_VERBOSITY", "") or os.getenv("VERBOSITY", "low")).lower()
        else:
            verbosity = os.getenv("VERBOSITY", "low").lower()
        if verbosity not in ("low", "medium", "high"):
            verbosity = "low"
        return {"effort": effort}, {"verbosity": verbosity}
    if is_deepseek_model(model):
        effort, thinking_mode = _build_deepseek_reasoning_config(raw, role)
        reasoning = {"thinking": thinking_mode}
        if effort is not None:
            reasoning["reasoning_effort"] = effort
        return reasoning, {}
    if is_gemini_model(model):
        effort = _resolve_gemini_reasoning_effort(raw, model)
        # reasoning_effort is forwarded by _GeminiResponses.create()
        return {"reasoning_effort": effort}, {}
    return {}, {}


def get_temperature(role: str = "agent") -> float:
    """
    Returns the LLM temperature for the given role. Only applied to non-GPT-5 models
    (GPT-5 series uses reasoning_effort instead of temperature).

    Roles:
        "agent"    → AGENT_TEMPERATURE    (default: 0.7)
        "retrieve" → RETRIEVE_TEMPERATURE (default: 0.0)
        "closeqa"  → CLOSEQA_TEMPERATURE  (default: 0.0)
    """
    defaults = {"agent": 0.7, "retrieve": 0.0, "closeqa": 0.0}
    env_keys = {"agent": "AGENT_TEMPERATURE", "retrieve": "RETRIEVE_TEMPERATURE", "closeqa": "CLOSEQA_TEMPERATURE"}
    key = env_keys.get(role, "AGENT_TEMPERATURE")
    default = defaults.get(role, 0.7)
    val = os.getenv(key, "")
    try:
        return float(val) if val.strip() else default
    except ValueError:
        return default


def get_max_tokens_param(n: int, model_name: str | None = None) -> dict:
    """
    Returns the correct token-limit parameter for client.chat.completions.create().
    GPT-5 variants use 'max_completion_tokens'; all others use 'max_tokens'.
    """
    model = model_name or get_chat_model()
    key = "max_completion_tokens" if is_gpt5_model(model) else "max_tokens"
    return {key: n}


# ── Gemini: chat.completions retry wrapper ────────────────────────────────────

class _RetryCompletions:
    """Wraps chat.completions so that .create() automatically retries on
    Gemini transient errors (503, 429, 500)."""

    def __init__(self, completions):
        self._cc = completions

    def create(self, *args, **kwargs):
        return _gemini_retry_call(self._cc.create, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._cc, name)


class _RetryChat:
    """Thin wrapper around the OpenAI chat namespace that exposes a
    retry-capable .completions attribute."""

    def __init__(self, chat):
        self._chat = chat
        self.completions = _RetryCompletions(chat.completions)

    def __getattr__(self, name):
        return getattr(self._chat, name)


# ── Gemini: responses.create() compatibility wrapper ──────────────────────────
class _GeminiResponseCompat:
    """Exposes .output_text like the return value of client.responses.create()."""
    def __init__(self, text: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.output_text = text
        self.usage = type("U", (), {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        })()


class _GeminiResponses:
    """Simulation of responses.create() for the Gemini OpenAI-compatible client.
    Internally calls chat.completions.create()."""
    def __init__(self, chat_completions):
        self._cc = chat_completions

    def create(self, model, instructions="", input=None,
               reasoning=None, text=None, temperature=None, **kwargs):
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        if isinstance(input, str):
            messages.append({"role": "user", "content": input})
        elif isinstance(input, list):
            for msg in input:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })
        max_tokens = kwargs.get("max_output_tokens", 50000)
        extra = {}
        if temperature is not None:
            extra["temperature"] = temperature
        # Forward reasoning_effort (Gemini thinking_level / thinking_budget)
        if isinstance(reasoning, dict) and "reasoning_effort" in reasoning:
            extra["reasoning_effort"] = reasoning["reasoning_effort"]
        resp = self._cc.create(
            model=model, messages=messages, max_tokens=max_tokens, **extra
        )
        usage = resp.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        return _GeminiResponseCompat(
            resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
        )


class _DeepSeekResponseCompat:
    """client.responses.create() compatible object for DeepSeek."""
    def __init__(self, text: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.output_text = text
        self.usage = type("U", (), {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        })()


class _DeepSeekResponses:
    """DeepSeek OpenAI-compat client responses adapter.
    Internally calls chat.completions.create().
    """
    def __init__(self, chat_completions):
        self._cc = chat_completions

    def create(self, model, instructions="", input=None,
               reasoning=None, text=None, temperature=None, **kwargs):
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})

        if isinstance(input, str):
            messages.append({"role": "user", "content": input})
        elif isinstance(input, list):
            for msg in input:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        req = {"model": model, "messages": messages}
        max_tokens = kwargs.pop("max_output_tokens", None)
        if max_tokens is not None:
            req["max_tokens"] = max_tokens
        if temperature is not None:
            req["temperature"] = temperature

        extra_body = kwargs.pop("extra_body", None) or {}
        thinking_payload = None
        if isinstance(reasoning, dict):
            effort = reasoning.get("reasoning_effort") or reasoning.get("effort")
            if effort:
                req["reasoning_effort"] = effort
            thinking = reasoning.get("thinking")
            if isinstance(thinking, str) and thinking in {"enabled", "disabled"}:
                thinking_payload = {"type": thinking}
            elif isinstance(thinking, dict):
                t = thinking.get("type")
                if t in {"enabled", "disabled"}:
                    thinking_payload = {"type": t}

        if thinking_payload is not None:
            extra_body = {**extra_body, "thinking": thinking_payload}
        if extra_body:
            req["extra_body"] = extra_body

        resp = self._cc.create(**req, **kwargs)
        usage = resp.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        return _DeepSeekResponseCompat(
            resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
        )


class _DeepSeekClient:
    """OpenAI-like client exposing .responses and .chat for DeepSeek."""
    def __init__(self, **kwargs):
        from openai import OpenAI as _OpenAI
        self._inner = _OpenAI(**kwargs)
        self.chat = self._inner.chat
        self.responses = _DeepSeekResponses(self.chat.completions)


class _GeminiClient:
    """Gemini wrapper that exposes both .responses and .chat like an OpenAI client.
    chat.completions.create() is automatically retried via _RetryCompletions."""
    def __init__(self, **kwargs):
        from openai import OpenAI as _OpenAI
        self._inner = _OpenAI(**kwargs)
        self.chat = _RetryChat(self._inner.chat)
        self.responses = _GeminiResponses(self.chat.completions)


def make_client(model_name: str | None = None):
    """Return an OpenAI-compatible client for the given model.
    Returns _GeminiClient for Gemini models; otherwise returns a standard OpenAI client."""
    model = model_name or get_chat_model()
    kwargs = get_chat_client_kwargs(model)
    if is_gemini_model(model):
        return _GeminiClient(**kwargs)
    if is_deepseek_model(model):
        return _DeepSeekClient(**kwargs)
    from openai import OpenAI as _OpenAI
    return _OpenAI(**kwargs)


CHAT_MODEL = get_chat_model()
