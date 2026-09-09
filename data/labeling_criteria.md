# Labeling criteria (frozen before any profile was labeled)

Freeze rule: commit this file, note the commit hash below, then label. If a
criterion changes after labeling starts, every case is relabeled and the eval
is rerun from scratch. Never edit a label after seeing model results.

Frozen at commit: `<paste hash>`   Labeler: Bruno Rosati   Date: `<date>`

## Decision rule
Qualify only if every criterion below passes. Record the single decisive
criterion as `key_factor`. For a disqualify, key_factor is the FIRST failing
criterion in the order below. For a qualify, key_factor is the strongest
positive driver (default `niche_fit`).

| # | Criterion | Pass | Fail -> key_factor |
|---|---|---|---|
| 1 | Hard stops | active in last 60 days, identifiable creator, no adult content, no exclusive deal with a competing fragrance brand | `hard_disqualifier` |
| 2 | Followers | 10,000 to 80,000. Soft bands: 5,000 to 9,999 if ER >= 6.0%; 80,001 to 120,000 if ER >= 3.0% and creator-led | `followers_below_range` / `followers_above_range` |
| 3 | Niche | fragrance, beauty, self-care or lifestyle with product or routine content in >= 2 of 5 captions. Adjacent niches (skincare-only, fashion, fitness, wellness) pass only if audience AND aesthetic pass | `niche_mismatch` / `niche_adjacent` / `content_mixed` |
| 4 | Audience | urban India, 18 to 28, English or Hinglish | `audience_age_mismatch` / `audience_geo_mismatch` / `language_mismatch` |
| 5 | Engagement | ER >= 2.5% with real comments; 1.5 to 2.5% passes only with substantive comments | `engagement_low` / `engagement_suspicious` |
| 6 | Aesthetic | clean, minimal, natural light, restrained captions | `aesthetic_mismatch` |

Positive key_factors for qualify: `niche_fit` (default), `engagement_fit`,
`audience_fit`, `aesthetic_fit`, `followers_in_range`.

## Strata (80 cases)
- `clear_qualify` 25: every criterion passes with margin (followers 15K to 60K, ER >= 3.5%, niche core).
- `clear_disqualify` 25: at least two criteria fail, or a hard stop.
- `edge` 30, five per subtype:
  - `E1_niche_adjacent` skincare / fashion / wellness creators with fit audience and look
  - `E2_wrong_audience_high_er` strong numbers, audience 35+, diaspora, or male tech
  - `E3_follower_boundary` right niche and look, 4K to 9K or 80K to 130K followers
  - `E4_mixed_content` beauty split roughly half with travel, food or comedy
  - `E5_suspicious_engagement` ER > 12% with generic or contest comments, or comment counts out of proportion
  - `E6_aesthetic_or_language` right niche and numbers, loud discount aesthetic or regional-language-only feed

## Blind labeling procedure
1. Write and freeze this file BEFORE collecting or reading profiles.
2. Write the three prompts BEFORE labeling, from this file only. (If prompts
   are written after you know which edge cases exist, the rubric prompt will
   quietly overfit to them.)
3. Export the 80 profiles into `data/private/raw_profiles.csv`, then shuffle
   rows with a fixed seed. Label in that order, one pass, no skipping.
4. For each case record: `label`, `key_factor`, `annotator_confidence` (1 to 3).
5. Do not reopen labels after `run_eval.py` has run. If a label is wrong, log
   it in `CHANGELOG.md`, fix it, delete `results/`, rerun everything.

## Second annotator check
Give a second person this file plus 24 profiles (30%): 12 edge, 6 clear
qualify, 6 clear disqualify, shuffled, labels hidden. Run
`python tools/kappa.py data/labels_bruno.csv data/labels_annotator2.csv`.
Kappa >= 0.70: proceed. Below 0.70: discuss every disagreement, tighten the
criterion that caused it, relabel all 80, rerun kappa.
