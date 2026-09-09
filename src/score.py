"""
Parse model output and score it.

Dimension 1 (decision): exact match against ground-truth label.
Dimension 2 (justification), automated part:
  - checklist_coverage: how many of the 4 required elements (niche, engagement,
    audience, aesthetic) contain a substantive sentence (0..4)
  - reason_match: model primary_reason == ground-truth key_factor (bool)
LLM-as-judge scoring lives in judge.py and is optional.

Malformed handling (decided, see README):
  1. one format retry with FORMAT_RETRY_MESSAGE appended to the conversation
  2. still malformed -> decision=None, counted WRONG in strict accuracy,
     and reported separately as format_failure_rate.
"""
import json
import re

import config

REQUIRED_KEYS = ["decision", "confidence", "primary_reason",
                 "niche", "engagement", "audience", "aesthetic", "justification"]
MIN_WORDS_PER_ELEMENT = 4


def extract_json(text):
    """Return (obj, error). Tolerates code fences and leading prose."""
    if not text:
        return None, "empty"
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, flags=re.S)  # first {...} block
    if not m:
        return None, "no_json_object"
    try:
        return json.loads(m.group(0)), None
    except json.JSONDecodeError as e:
        return None, f"json_error: {e.msg}"


def validate(obj):
    """Return (normalized_dict, error). error=None means the schema is satisfied."""
    if not isinstance(obj, dict):
        return None, "not_an_object"
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        return None, f"missing_keys: {','.join(missing)}"
    decision = str(obj["decision"]).strip().lower()
    if decision not in config.VALID_DECISIONS:
        return None, f"invalid_decision: {decision}"
    confidence = str(obj["confidence"]).strip().lower()
    if confidence not in config.VALID_CONFIDENCE:
        confidence = "medium"  # tolerate, do not fail the row
    reason = str(obj["primary_reason"]).strip().lower()
    norm = dict(
        decision=decision,
        confidence=confidence,
        primary_reason=reason,
        reason_valid=reason in config.VALID_REASONS,
        niche=str(obj["niche"]),
        engagement=str(obj["engagement"]),
        audience=str(obj["audience"]),
        aesthetic=str(obj["aesthetic"]),
        justification=str(obj["justification"]),
    )
    return norm, None


def parse_and_validate(text):
    obj, err = extract_json(text)
    if err:
        return None, err
    return validate(obj)


def checklist_coverage(norm):
    """Count elements with at least MIN_WORDS_PER_ELEMENT words."""
    n = 0
    for k in ("niche", "engagement", "audience", "aesthetic"):
        if len(re.findall(r"\w+", norm.get(k, ""))) >= MIN_WORDS_PER_ELEMENT:
            n += 1
    return n


def reason_match(norm, case):
    return norm["primary_reason"] == case["key_factor"]
