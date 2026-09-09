"""
Turn the private raw profile export into the public cases.jsonl.

Input  data/private/raw_profiles.csv  (gitignored) with columns:
  handle, bio, followers, following, posts_count, engagement_rate, avg_comments,
  location, primary_language, days_since_last_post,
  caption_1 ... caption_5, label, key_factor, stratum, edge_subtype, annotator_confidence
Output data/cases.jsonl (public) and data/private/id_map.csv (gitignored)

What changes
  handle            -> creator_001 ... (mapping kept private)
  followers/following/posts/avg_comments -> jittered by a seeded +/-15%
  engagement_rate   -> kept (it is a ratio, jittering it would change the label)
  bio, captions     -> paraphrased by Claude to keep niche, tone and aesthetic
                       signals while dropping names, brand names, cities beyond
                       tier, handles, and anything searchable. You review every
                       paraphrase by hand before commit.
Case ids are shuffled with a fixed seed so file order carries no label signal.

Usage
  python tools/anonymize.py               # paraphrase with Claude Sonnet 5
  python tools/anonymize.py --no-llm      # numbers + ids only, keep text as is
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from providers import call_model

PARAPHRASE_SYSTEM = """Rewrite Instagram bios and captions for a public research dataset.
Keep: niche, tone, aesthetic cues (minimal / loud / discount-heavy / editorial),
audience cues (age, city tier, language mix, Hinglish), product mentions at
category level, emoji density. Remove or generalise: person names, handles,
brand names (say "a mass beauty brand", "a premium skincare label"), specific
neighbourhoods, event names, dates, discount codes, anything searchable.
Keep length within 20 percent of the original. Return JSON: {"text": "..."}"""

FIELDS_JITTER = ["followers", "following", "posts_count", "avg_comments"]


def jitter(value, rng, pct=0.15):
    return int(round(float(value) * (1 + rng.uniform(-pct, pct))))


def paraphrase(text, use_llm):
    if not use_llm or not text.strip():
        return text
    cfg = config.MODELS["claude-sonnet-5"]
    r = call_model(cfg, PARAPHRASE_SYSTEM, [{"role": "user", "content": text}])
    try:
        return json.loads(r["text"])["text"]
    except Exception:
        return text  # keep original, flag for manual review


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(config.DATA_DIR / "private" / "raw_profiles.csv"))
    ap.add_argument("--out", default=str(config.CASES_FILE))
    ap.add_argument("--map", default=str(config.DATA_DIR / "private" / "id_map.csv"))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.src, newline="", encoding="utf-8-sig") as f:  # -sig: Excel/Airtable exports carry a BOM
        rows = list(csv.DictReader(f))
    order = list(range(len(rows)))
    rng.shuffle(order)

    cases, id_map = [], []
    for new_idx, old_idx in enumerate(order, start=1):
        r = rows[old_idx]
        case_id = f"IQ-{new_idx:03d}"
        id_map.append(dict(case_id=case_id, handle=r["handle"]))
        captions = [paraphrase(r[f"caption_{i}"], not args.no_llm) for i in range(1, 6) if r.get(f"caption_{i}", "").strip()]
        cases.append(dict(
            case_id=case_id, handle_id=f"creator_{new_idx:03d}",
            bio=paraphrase(r["bio"], not args.no_llm),
            followers=jitter(r["followers"], rng), following=jitter(r["following"], rng),
            posts_count=jitter(r["posts_count"], rng),
            engagement_rate=round(float(r["engagement_rate"]), 1),
            avg_comments=jitter(r["avg_comments"], rng),
            location=r["location"], primary_language=r["primary_language"],
            days_since_last_post=int(r["days_since_last_post"]),
            recent_captions=captions,
            label=r["label"].strip().lower(), key_factor=r["key_factor"].strip(),
            stratum=r["stratum"].strip(), edge_subtype=r.get("edge_subtype", "").strip() or None,
            annotator_confidence=int(r.get("annotator_confidence", 3)),
        ))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    Path(args.map).parent.mkdir(parents=True, exist_ok=True)
    with open(args.map, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "handle"])
        w.writeheader()
        w.writerows(id_map)
    print(f"wrote {len(cases)} cases -> {args.out}; private map -> {args.map}")
    print("Now open cases.jsonl and read every bio and caption. If you can still identify the person, rewrite it.")


if __name__ == "__main__":
    main()
