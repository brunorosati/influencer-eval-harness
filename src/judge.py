"""
Optional: score justification quality 1-5 with an LLM judge.

Why optional: the automated reason_match + checklist metrics are objective and
free. The judge adds a "does the reasoning actually hold" read at the cost of
judge bias (a Claude judge tends to favour Claude outputs). Run two judges from
different families and validate against your own manual scores on 20 cases.

Usage
  python src/judge.py --judge claude-opus-5 --sample 30
  python src/judge.py --judge gpt-5.6-sol   --sample 30
  python src/judge.py --agreement            # judge vs judge vs manual
Manual scores file (optional): data/manual_justification_scores.csv
  columns: case_id, modelo, prompt_variant, manual_score (1-5)
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from providers import call_model
from run_eval import load_cases, render_profile
from score import extract_json

JUDGE_SYSTEM = """You are auditing an influencer-qualification classifier for a D2C fragrance
brand. You will see the creator profile, the ground-truth decision and the
decisive criterion chosen by a human annotator, and the classifier's decision
plus justification. Score the JUSTIFICATION only, 1-5, using these anchors:

5  Names the decisive criterion, cites the specific profile evidence for it,
   and the other elements are accurate. Nothing invented.
4  Names the decisive criterion with evidence; one minor inaccuracy or
   one generic element.
3  Reaches a defensible conclusion but leans on the wrong criterion or on
   generic statements ("good engagement") without evidence.
2  Contains a factual error about the profile, or the justification
   contradicts the decision.
1  Invented facts, or no reasoning that connects to the profile.

Score the reasoning quality regardless of whether the final decision matched
the ground truth. Respond with JSON only: {"score": <1-5>, "note": "<one sentence>"}"""


def judge_prompt(case, row):
    return (
        render_profile(case).replace("Decide whether this creator qualifies for the ICP. Reply with the JSON object only.", "")
        + f"\nGROUND TRUTH: {case['label']} (decisive criterion: {case['key_factor']})\n"
        + f"CLASSIFIER DECISION: {row['decisao_modelo'] or '(malformed)'}\n"
        + f"CLASSIFIER OUTPUT:\n{row['justificativa_raw']}\n\nScore the justification."
    )


def run_judge(judge_key, sample, seed, mock):
    cfg = config.MODELS[judge_key]
    cases = {c["case_id"]: c for c in load_cases()}
    df = pd.read_csv(config.RAW_RESULTS)
    df = df[df["run_idx"].astype(str) == df["run_idx"].astype(str).min()]
    rng = random.Random(seed)
    # same sampled case ids for every config so scores are comparable
    edge_ids = sorted(df[df["stratum"] == "edge"]["case_id"].unique())
    other_ids = sorted(df[df["stratum"] != "edge"]["case_id"].unique())
    ids = rng.sample(edge_ids, min(sample // 2, len(edge_ids))) + \
          rng.sample(other_ids, min(sample - sample // 2, len(other_ids)))
    sub = df[df["case_id"].isin(ids) & (df["format_failed"] == 0)]

    out = config.JUDGE_CSV
    existing = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            existing = {(r["judge"], r["case_id"], r["modelo"], r["prompt_variant"]) for r in csv.DictReader(f)}
    write_header = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["judge", "case_id", "modelo", "prompt_variant", "judge_score", "judge_note"])
        if write_header:
            w.writeheader()
        for _, row in sub.iterrows():
            key = (judge_key, row["case_id"], row["modelo"], row["prompt_variant"])
            if key in existing:
                continue
            case = cases[row["case_id"]]
            r = call_model(cfg, JUDGE_SYSTEM, [{"role": "user", "content": judge_prompt(case, row)}],
                           mock_case=(case if mock else None))
            obj, _ = extract_json(r["text"]) if r["text"] else (None, "err")
            score = obj.get("score") if isinstance(obj, dict) else None
            if mock:
                score = random.Random(hash(key)).choice([3, 4, 4, 5, 5])
            w.writerow(dict(judge=judge_key, case_id=row["case_id"], modelo=row["modelo"],
                            prompt_variant=row["prompt_variant"], judge_score=score,
                            judge_note=(obj or {}).get("note", "") if isinstance(obj, dict) else r["error"] or ""))
            f.flush()
    print(f"judge {judge_key} done -> {out}")


def report_agreement():
    df = pd.read_csv(config.JUDGE_CSV)
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    print("\nMean judge score per config:")
    print(df.pivot_table(index=["modelo", "prompt_variant"], columns="judge", values="judge_score", aggfunc="mean").round(2))
    judges = df["judge"].unique()
    if len(judges) >= 2:
        a, b = judges[:2]
        m = df[df["judge"] == a].merge(df[df["judge"] == b], on=["case_id", "modelo", "prompt_variant"], suffixes=("_a", "_b"))
        d = (m["judge_score_a"] - m["judge_score_b"]).abs()
        print(f"\n{a} vs {b}: exact agreement {(d == 0).mean():.0%}, within 1 point {(d <= 1).mean():.0%}, n={len(m)}")
    manual = config.DATA_DIR / "manual_justification_scores.csv"
    if manual.exists():
        mdf = pd.read_csv(manual)
        for j in judges:
            m = df[df["judge"] == j].merge(mdf, on=["case_id", "modelo", "prompt_variant"])
            d = (m["judge_score"] - m["manual_score"]).abs()
            print(f"{j} vs manual: exact {(d == 0).mean():.0%}, within 1 {(d <= 1).mean():.0%}, "
                  f"mean abs diff {d.mean():.2f}, n={len(m)}")
    else:
        print("\nNo data/manual_justification_scores.csv yet. Score 20 cases by hand to validate the judge.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="claude-opus-5")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--agreement", action="store_true")
    args = ap.parse_args()
    if args.agreement:
        report_agreement()
    else:
        run_judge(args.judge, args.sample, args.seed, args.mock)
