"""
Generate a SYNTHETIC 80-case dataset with the right strata so the pipeline can
be tested end to end without real data. Never publish results from it.

  python tests/make_mock_cases.py   -> data/cases.jsonl (only if absent)
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config

rng = random.Random(1)
CITIES = ["Mumbai", "Delhi", "Bangalore", "Pune", "Hyderabad", "Gurgaon"]
CAPS = {
    "beauty": ["morning skin + scent routine", "layering two florals for summer", "minimal vanity restock",
               "what I wear to the office", "cafe hop in the city"],
    "adjacent": ["gym progress week 8", "outfit repeat challenge", "protein pancakes", "sunday reset", "run club"],
    "unrelated": ["street food crawl", "goa road trip", "reacting to memes", "crypto update", "bike review"],
    "loud": ["USE CODE SAVE40 🔥🔥", "GIVEAWAY tag 3 friends", "MEGA SALE link in bio", "DEAL OF THE DAY", "WIN NOW"],
}
SUBTYPES = ["E1_niche_adjacent", "E2_wrong_audience_high_er", "E3_follower_boundary",
            "E4_mixed_content", "E5_suspicious_engagement", "E6_aesthetic_or_language"]


def case(i, stratum, subtype=None):
    label, kf = "qualify", "niche_fit"
    caps = CAPS["beauty"]; foll = rng.randint(15000, 60000); er = round(rng.uniform(3.5, 7), 1)
    loc, lang, days = rng.choice(CITIES), "English/Hinglish", rng.randint(1, 20)
    if stratum == "clear_disqualify":
        label = "disqualify"; caps = CAPS["unrelated"]; kf = "niche_mismatch"
        if rng.random() < 0.4:
            days = 120; kf = "hard_disqualifier"
    elif stratum == "edge":
        if subtype == "E1_niche_adjacent":
            caps = CAPS["adjacent"]; label = rng.choice(["qualify", "disqualify"]); kf = "niche_adjacent" if label == "disqualify" else "audience_fit"
        elif subtype == "E2_wrong_audience_high_er":
            er = round(rng.uniform(6, 9), 1); loc = rng.choice(["Toronto", "Dubai", "London"]); label = "disqualify"; kf = "audience_geo_mismatch"
        elif subtype == "E3_follower_boundary":
            foll = rng.choice([rng.randint(4000, 9000), rng.randint(81000, 130000)])
            label = "qualify" if (foll < 10000 and er >= 6) or (foll > 80000 and er >= 3) else "disqualify"
            kf = "followers_in_range" if label == "qualify" else ("followers_below_range" if foll < 10000 else "followers_above_range")
        elif subtype == "E4_mixed_content":
            caps = CAPS["beauty"][:3] + CAPS["unrelated"][:2]; label = rng.choice(["qualify", "disqualify"]); kf = "content_mixed" if label == "disqualify" else "niche_fit"
        elif subtype == "E5_suspicious_engagement":
            er = round(rng.uniform(13, 22), 1); label = "disqualify"; kf = "engagement_suspicious"
        else:
            caps = CAPS["loud"]; label = "disqualify"; kf = "aesthetic_mismatch"
    return dict(case_id=f"IQ-{i:03d}", handle_id=f"creator_{i:03d}",
                bio=f"{rng.choice(['beauty', 'scent', 'skin', 'daily'])} notes | {loc.lower()} | synthetic profile",
                followers=foll, following=rng.randint(300, 1500), posts_count=rng.randint(80, 900),
                engagement_rate=er, avg_comments=int(foll * er / 100 * 0.08), location=loc,
                primary_language=lang, days_since_last_post=days, recent_captions=caps,
                label=label, key_factor=kf, stratum=stratum, edge_subtype=subtype, annotator_confidence=3)


def main():
    out = config.CASES_FILE
    if out.exists():
        print(f"{out} exists, not overwriting"); return
    cases = [case(i, "clear_qualify") for i in range(1, 26)]
    cases += [case(i, "clear_disqualify") for i in range(26, 51)]
    cases += [case(51 + j, "edge", SUBTYPES[j % 6]) for j in range(30)]
    rng.shuffle(cases)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cases)} synthetic cases -> {out}")


if __name__ == "__main__":
    main()
