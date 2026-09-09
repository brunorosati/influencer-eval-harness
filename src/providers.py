"""
One function, call_model(cfg, system, messages), for every provider.

Returns dict(text, tokens_in, tokens_out, latency_ms, attempts, error).
Transport failures (rate limit, 5xx, network) retry with backoff up to
config.API_MAX_ATTEMPTS. Format failures are handled by the caller, not here.

Determinism, per provider (this is why "temperature=0 everywhere" is not
possible in 2026 and why the harness supports --repeats):
  - Claude Haiku 4.5, open models: temperature=0 (+ seed where accepted)
  - Claude Sonnet 5 / Opus 5: sampling params return HTTP 400; we disable
    adaptive thinking instead. Outputs are near-deterministic but not guaranteed.
  - GPT-5.x: temperature must stay at the default (1). reasoning_effort="none"
    plus seed=SEED gives best-effort reproducibility.
"""
import hashlib
import json
import os
import random
import time

import config


def _sleep_backoff(attempt):
    time.sleep(min(2 ** attempt, 30) + random.random())


# ----------------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------------
_anthropic_client = None


def _anthropic(cfg, system, messages):
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic()  # reads ANTHROPIC_API_KEY
    kwargs = dict(model=cfg["model_id"], max_tokens=config.MAX_TOKENS,
                  system=system, messages=messages)
    if cfg["sampling"] == "temperature":
        # SDK >= 1.x dropped temperature as a named argument; pass it through
        # extra_body. The API still accepts it on Claude 4.6 and earlier.
        kwargs["extra_body"] = {"temperature": 0}
    elif cfg["sampling"] == "claude5":
        if cfg.get("thinking", "disabled") == "disabled":
            kwargs["thinking"] = {"type": "disabled"}
        else:
            # adaptive thinking stays on; effort controls depth
            kwargs["output_config"] = {"effort": cfg.get("effort", "medium")}
    resp = _anthropic_client.messages.create(**kwargs)
    # Select text blocks by type: with thinking on, thinking blocks come first.
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


# ----------------------------------------------------------------------------
# OpenAI and OpenAI-compatible (Groq, Together, Mistral, OpenRouter ...)
# ----------------------------------------------------------------------------
_openai_clients = {}


def _openai_like(cfg, system, messages):
    from openai import OpenAI
    key = (cfg.get("base_url"), cfg.get("api_key_env"))
    if key not in _openai_clients:
        if cfg["provider"] == "openai":
            _openai_clients[key] = OpenAI()  # reads OPENAI_API_KEY
        else:
            _openai_clients[key] = OpenAI(
                base_url=cfg["base_url"],
                api_key=os.environ[cfg["api_key_env"]])
    client = _openai_clients[key]
    chat = [{"role": "system", "content": system}] + messages
    kwargs = dict(model=cfg["model_id"], messages=chat,
                  max_completion_tokens=config.MAX_TOKENS, seed=config.SEED)
    if cfg["sampling"] == "temperature":
        kwargs["temperature"] = 0
    elif cfg["sampling"] == "gpt5":
        kwargs["reasoning_effort"] = "none"
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:  # some hosts reject seed or max_completion_tokens
        msg = str(e).lower()
        if "seed" in msg:
            kwargs.pop("seed", None)
        if "max_completion_tokens" in msg:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    return text, resp.usage.prompt_tokens, resp.usage.completion_tokens


# ----------------------------------------------------------------------------
# Mock: deterministic fake responses for testing the pipeline end to end.
# Never use mock output in the README.
# ----------------------------------------------------------------------------
_MOCK_ACC = {"frontier": 0.90, "mid": 0.84, "budget": 0.74, "open": 0.66}


def _mock(cfg, system, messages, case):
    user = messages[-1]["content"]
    seed_src = f"{cfg['model_id']}|{system}|{user}"
    h = int(hashlib.md5(seed_src.encode()).hexdigest(), 16)
    rng = random.Random(h)
    truth = case["label"]
    truth_reason = case["key_factor"]
    acc = _MOCK_ACC.get(cfg["tier"], 0.7)
    acc += 0.04 if "WORKED EXAMPLES" in system else 0
    acc += 0.06 if "Apply these criteria" in system else 0
    is_retry = any("did not match the required schema" in m["content"] for m in messages)
    if rng.random() < (0.30 if is_retry else 0.06):   # simulate malformed output
        return "Sure! Here is my analysis: the creator seems fine.", 900, 40
    correct = rng.random() < acc
    decision = truth if correct else ("disqualify" if truth == "qualify" else "qualify")
    reason = truth_reason if (correct and rng.random() < 0.8) else rng.choice(config.VALID_REASONS)
    conf = rng.choices(["high", "medium", "low"], weights=[0.6, 0.3, 0.1])[0] if correct \
        else rng.choices(["high", "medium", "low"], weights=[0.3, 0.4, 0.3])[0]
    obj = {"decision": decision, "confidence": conf, "primary_reason": reason,
           "niche": "Content centres on beauty and fragrance routines.",
           "engagement": "Engagement rate sits above the 2.5 percent threshold with real comments.",
           "audience": "Comments and city tags point to urban Indian viewers in their twenties.",
           "aesthetic": "Neutral palettes and natural light match the brand look.",
           "justification": "The profile matches niche, audience and look, so the decision follows."}
    time.sleep(0.001)
    return json.dumps(obj), 950 + rng.randint(-80, 80), 160 + rng.randint(-30, 30)


# ----------------------------------------------------------------------------
def call_model(cfg, system, messages, mock_case=None):
    """mock_case: pass the case dict to fake the call (pipeline tests only)."""
    if mock_case is not None:
        fn = lambda c, s, m: _mock(c, s, m, mock_case)
    else:
        fn = {"anthropic": _anthropic,
              "openai": _openai_like,
              "openai_compatible": _openai_like}[cfg["provider"]]
    last_err = None
    for attempt in range(config.API_MAX_ATTEMPTS):
        t0 = time.perf_counter()
        try:
            text, tin, tout = fn(cfg, system, messages)
            return dict(text=text, tokens_in=tin, tokens_out=tout,
                        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                        attempts=attempt + 1, error=None)
        except Exception as e:  # rate limit, timeout, 5xx
            last_err = f"{type(e).__name__}: {e}"[:300]
            _sleep_backoff(attempt)
    return dict(text="", tokens_in=0, tokens_out=0, latency_ms=None,
                attempts=config.API_MAX_ATTEMPTS, error=last_err)
