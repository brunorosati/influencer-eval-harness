"""
Run the eval grid and write results/raw_results.csv (one row per
case x model x prompt x repeat). Resumable: rerunning skips rows already
present, so a rate-limit crash costs nothing.

Usage
  python src/run_eval.py                       # full grid, real APIs
  python src/run_eval.py --models claude-sonnet-5 --prompts minimal detailed rubric
  python src/run_eval.py --repeats 3 --models claude-haiku-4.5
  python src/run_eval.py --mock                # fake responses, tests the pipeline
  python src/run_eval.py --limit 5             # first 5 cases only (smoke test)
"""
import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from providers import call_model
from score import parse_and_validate, checklist_coverage, reason_match

CSV_COLUMNS = [
    "run_id", "timestamp", "case_id", "stratum", "edge_subtype", "modelo",
    "prompt_variant", "run_idx", "decisao_modelo", "decisao_ground_truth", "match",
    "confidence", "primary_reason", "key_factor_ground_truth", "reason_match",
    "reason_valid", "checklist_coverage", "justificativa_raw", "format_failed",
    "format_retries", "api_attempts", "latencia_ms", "tokens_input",
    "tokens_output", "custo_estimado_usd", "error",
]


def load_cases(path=config.CASES_FILE):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_prompt(variant):
    body = (config.PROMPT_DIR / f"{variant}.txt").read_text(encoding="utf-8").strip()
    return body + "\n\n" + config.OUTPUT_SCHEMA.strip()


def render_profile(case):
    """What the model sees. Mirrors the fields the n8n pipeline sends."""
    caps = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(case["recent_captions"]))
    return (
        "INFLUENCER PROFILE\n"
        f"Handle: {case['handle_id']}\n"
        f"Bio: {case['bio']}\n"
        f"Followers: {case['followers']:,}\n"
        f"Following: {case['following']:,}\n"
        f"Posts: {case['posts_count']}\n"
        f"Engagement rate (last 12 posts): {case['engagement_rate']}%\n"
        f"Avg comments per post: {case['avg_comments']}\n"
        f"Location signal: {case['location']}\n"
        f"Primary language: {case['primary_language']}\n"
        f"Days since last post: {case['days_since_last_post']}\n"
        f"Recent captions:\n{caps}\n\n"
        "Decide whether this creator qualifies for the ICP. Reply with the JSON object only."
    )


def cost_usd(cfg, tin, tout):
    return round(tin * cfg["price_in"] / 1e6 + tout * cfg["price_out"] / 1e6, 6)


def evaluate_case(model_key, cfg, variant, system, case, run_idx, run_id, mock):
    messages = [{"role": "user", "content": render_profile(case)}]
    mock_case = case if mock else None
    total_in = total_out = 0
    latency = 0.0
    api_attempts = 0
    format_retries = 0
    norm, err, raw = None, None, ""

    for fmt_attempt in range(2):  # first try + one format retry
        r = call_model(cfg, system, messages, mock_case=mock_case)
        api_attempts += r["attempts"]
        if r["error"]:
            err = r["error"]
            break
        total_in += r["tokens_in"]
        total_out += r["tokens_out"]
        latency += r["latency_ms"]
        raw = r["text"]
        norm, err = parse_and_validate(raw)
        if norm:
            break
        if fmt_attempt == 0:
            format_retries = 1
            messages = messages + [
                {"role": "assistant", "content": raw or "(empty)"},
                {"role": "user", "content": config.FORMAT_RETRY_MESSAGE},
            ]

    truth = case["label"]
    decision = norm["decision"] if norm else None
    return {
        "run_id": run_id,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "case_id": case["case_id"],
        "stratum": case["stratum"],
        "edge_subtype": case.get("edge_subtype") or "",
        "modelo": model_key,
        "prompt_variant": variant,
        "run_idx": run_idx,
        "decisao_modelo": decision or "",
        "decisao_ground_truth": truth,
        "match": int(decision == truth),
        "confidence": norm["confidence"] if norm else "",
        "primary_reason": norm["primary_reason"] if norm else "",
        "key_factor_ground_truth": case["key_factor"],
        "reason_match": int(reason_match(norm, case)) if norm else 0,
        "reason_valid": int(norm["reason_valid"]) if norm else 0,
        "checklist_coverage": checklist_coverage(norm) if norm else 0,
        "justificativa_raw": raw.replace("\n", " ")[:2000],
        "format_failed": int(norm is None),
        "format_retries": format_retries,
        "api_attempts": api_attempts,
        "latencia_ms": round(latency, 1) if latency else "",
        "tokens_input": total_in,
        "tokens_output": total_out,
        "custo_estimado_usd": cost_usd(cfg, total_in, total_out),
        "error": err or "",
    }


def existing_keys(path):
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["modelo"], r["prompt_variant"], r["case_id"], r["run_idx"])
                for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(config.MODELS))
    ap.add_argument("--prompts", nargs="*", default=config.PROMPT_VARIANTS)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N cases only")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default=str(config.RAW_RESULTS))
    args = ap.parse_args()

    # Mock rows must never share a CSV with real rows: the resume key would
    # skip real calls for any (model, prompt, case, run) a mock run already wrote.
    if args.mock and args.out == str(config.RAW_RESULTS):
        args.out = str(config.RESULTS_DIR / "raw_results_mock.csv")

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(out)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ("-mock" if args.mock else "")
    write_header = not out.exists()

    total = len(args.models) * len(args.prompts) * len(cases) * args.repeats
    n_done = n_new = 0
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        for model_key in args.models:
            cfg = config.MODELS[model_key]
            for variant in args.prompts:
                system = load_prompt(variant)
                for run_idx in range(args.repeats):
                    for case in cases:
                        key = (model_key, variant, case["case_id"], str(run_idx))
                        if key in done:
                            n_done += 1
                            continue
                        row = evaluate_case(model_key, cfg, variant, system, case,
                                            run_idx, run_id, args.mock)
                        w.writerow(row)
                        f.flush()
                        n_new += 1
                        if n_new % 20 == 0:
                            print(f"[{n_done + n_new}/{total}] {model_key} x {variant}")
    print(f"done. new rows: {n_new}, skipped (already present): {n_done}. -> {out}")


if __name__ == "__main__":
    main()
