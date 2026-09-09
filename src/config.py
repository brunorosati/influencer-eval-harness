"""
Central configuration for the influencer-qualification eval.

Prices are USD per 1M tokens, standard (non-batch, non-cached) API rates.
PRICING_VERIFIED_ON is the date the numbers below were checked. Re-verify
before every run. Sources:
  Anthropic  https://platform.claude.com/docs/en/about-claude/pricing
  OpenAI     https://developers.openai.com/api/docs/pricing
  Groq       https://groq.com/pricing
"""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
PROMPT_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"

CASES_FILE = DATA_DIR / "cases.jsonl"
RAW_RESULTS = RESULTS_DIR / "raw_results.csv"
SUMMARY_MD = RESULTS_DIR / "summary.md"
DIVERGENT_MD = RESULTS_DIR / "divergent_cases.md"
JUDGE_CSV = RESULTS_DIR / "judge_scores.csv"

PRICING_VERIFIED_ON = "2026-09-01"

# sampling modes
#   "temperature": model accepts temperature=0 (Haiku 4.5, most open models)
#   "claude5":     Claude Sonnet 5 / Opus 5. Non-default temperature returns 400.
#                  We disable adaptive thinking so every model runs in the same
#                  no-reasoning configuration and cost stays comparable.
#   "gpt5":        GPT-5.x. Only temperature=1 is accepted. reasoning_effort="none"
#                  gives the no-reasoning baseline.
MODELS = {
    # ---- frontier -------------------------------------------------------
    "claude-opus-5": dict(
        provider="anthropic", model_id="claude-opus-5", tier="frontier",
        price_in=5.00, price_out=25.00, sampling="claude5"),
    "gpt-5.6-sol": dict(
        provider="openai", model_id="gpt-5.6-sol", tier="frontier",
        price_in=4.00, price_out=20.00, sampling="gpt5",
        note="promo rate through at least 2026-11-21; list rate 5.00 / 30.00"),
    # ---- mid ------------------------------------------------------------
    "claude-sonnet-5": dict(
        provider="anthropic", model_id="claude-sonnet-5", tier="mid",
        price_in=2.00, price_out=10.00, sampling="claude5"),
    "gpt-5.6-terra": dict(
        provider="openai", model_id="gpt-5.6-terra", tier="mid",
        price_in=2.00, price_out=12.00, sampling="gpt5"),
    # ---- budget ---------------------------------------------------------
    "claude-haiku-4.5": dict(
        provider="anthropic", model_id="claude-haiku-4-5-20251001", tier="budget",
        price_in=1.00, price_out=5.00, sampling="temperature"),
    "gpt-5.6-luna": dict(
        provider="openai", model_id="gpt-5.6-luna", tier="budget",
        price_in=0.20, price_out=1.20, sampling="gpt5"),
    # ---- open weights (optional; any OpenAI-compatible host works) -------
    "llama-4-scout": dict(
        provider="openai_compatible", tier="open",
        model_id="meta-llama/llama-4-scout-17b-16e-instruct",
        base_url="https://api.groq.com/openai/v1", api_key_env="GROQ_API_KEY",
        price_in=0.11, price_out=0.34, sampling="temperature",
        note="price from third-party trackers of groq.com/pricing; confirm model id in Groq console"),
}

# Optional extra config: Sonnet 5 with adaptive thinking left ON.
# Uncomment to measure whether reasoning tokens buy accuracy on edge cases.
# MODELS["claude-sonnet-5-think"] = dict(
#     provider="anthropic", model_id="claude-sonnet-5", tier="mid",
#     price_in=2.00, price_out=10.00, sampling="claude5", thinking="adaptive", effort="medium")

PROMPT_VARIANTS = ["minimal", "detailed", "rubric"]

# The config you run in production today. analyze.py uses it as the baseline
# for McNemar tests and monthly-savings math.
PRODUCTION_BASELINE = ("claude-sonnet-5", "detailed")

# Profiles the n8n pipeline scores per month. Used for the savings estimate.
DEFAULT_MONTHLY_VOLUME = 600

# Business cost of a wrong decision, USD. Used by analyze.py to compute total
# cost (API + errors). Defaults: a false qualify burns one personalised
# outreach (about 3 minutes of founder time plus a wasted DM slot); a false
# disqualify forfeits the expected value of a fit creator. Change to your numbers.
COST_FALSE_QUALIFY_USD = 0.50
COST_FALSE_DISQUALIFY_USD = 2.00

MAX_TOKENS = 700
SEED = 42            # passed where the API accepts it (OpenAI, Groq)
API_MAX_ATTEMPTS = 5  # transport retries (rate limits, 5xx). Format retries are separate.

VALID_DECISIONS = {"qualify", "disqualify"}
VALID_CONFIDENCE = {"high", "medium", "low"}

# Decisive-criterion vocabulary. Ground truth uses the same list (key_factor),
# which is what makes reason-match scoring objective.
VALID_REASONS = [
    "niche_fit", "niche_adjacent", "niche_mismatch",
    "followers_in_range", "followers_below_range", "followers_above_range",
    "engagement_fit", "engagement_low", "engagement_suspicious",
    "audience_fit", "audience_age_mismatch", "audience_geo_mismatch",
    "aesthetic_fit", "aesthetic_mismatch",
    "content_mixed", "language_mismatch",
    "hard_disqualifier",
]

# Identical across all prompt variants. Variants differ in ICP depth, few-shot
# examples and explicit criteria, never in output format, so the eval measures
# reasoning, not format compliance.
OUTPUT_SCHEMA = """
OUTPUT FORMAT (strict). Respond with one JSON object and nothing else. No prose
before or after, no markdown fences.
{
  "decision": "qualify" | "disqualify",
  "confidence": "high" | "medium" | "low",
  "primary_reason": one of %s,
  "niche": "<one sentence on niche fit>",
  "engagement": "<one sentence on engagement quality>",
  "audience": "<one sentence on audience fit>",
  "aesthetic": "<one sentence on aesthetic fit>",
  "justification": "<2-3 sentences tying the above to the decision>"
}
""" % ", ".join(f'"{r}"' for r in VALID_REASONS)

FORMAT_RETRY_MESSAGE = (
    "Your previous response did not match the required schema. "
    "Reply with only the JSON object described in the instructions, "
    "with all eight keys present and valid values."
)
