# Influencer qualification eval

Which model, and which prompt, should decide whether an Instagram creator gets a personalised outreach message from a D2C fragrance brand? This repo measures that on 80 hand-labeled profiles, seven models and three prompts, then turns the numbers into a production routing rule.

> Lines starting with `>>>` are filled in from `results/summary.md` after the run. Everything else is final.

## What this is and why it matters

This is a domain-specific eval harness for one classification task inside the Montrega influencer outreach pipeline (n8n + Airtable + Claude API, [pipeline repo](https://github.com/brunorosati)): given a creator profile, decide qualify or disqualify against the brand's ICP and justify the call. In production, a classifier that is 85% accurate on easy profiles and 60% accurate on the ambiguous ones wastes outreach on the wrong creators and silently drops the right ones, and nobody notices without a labeled set to check against. `>>> The non-obvious finding: <one sentence, e.g. "the cheapest model with the rubric prompt matched the production model on edge cases at one fifth of the cost, and the prompt mattered more than the model">`.

## Results at a glance

`>>> paste section 1 and section 2 tables from results/summary.md`

![Cost vs edge-case accuracy](results/cost_vs_edge_accuracy.png)

![Edge-case agreement](results/edge_case_agreement.png)

## The task

Input: bio, follower and following counts, post count, engagement rate over the last 12 posts, average comments, location signal, primary language, days since last post, five recent captions. Output: a JSON object with a decision, a confidence level, a `primary_reason` drawn from a fixed 17-term vocabulary, one sentence each on niche, engagement, audience and aesthetic, and a short justification.

ICP: fragrance, beauty or lifestyle creators, roughly 10K to 80K followers, an engaged urban Indian audience aged 18 to 28, a clean minimal aesthetic. The full criteria with thresholds live in [`data/labeling_criteria.md`](data/labeling_criteria.md).

## Method

**Dataset.** 80 anonymised profiles from the pipeline's own sourcing: 25 clear qualifiers, 25 clear disqualifiers, 30 edge cases split evenly across six failure patterns (adjacent niche, wrong audience with strong numbers, follower-count boundary, mixed content, suspicious engagement, aesthetic or language mismatch). The edge cases are the point of the exercise. Clear cases confirm the models can read; edge cases show whether they can judge. *Status: the `data/cases.jsonl` in the repo right now is the synthetic stand-in from `tests/make_mock_cases.py` (same strata, every bio marked "synthetic profile") so the pipeline runs end to end; it is replaced by the real anonymised set, produced with `tools/anonymize.py`, in the same release that fills the results below.*

**Ground truth.** I labeled every profile myself because I know these creators from running outreach to them, and I wrote and froze the criteria before labeling, shuffled the profiles with a fixed seed, labeled in one pass, and did not touch a label after the first model run. A second annotator labeled 24 profiles (12 edge) from the same criteria doc. `>>> Cohen's kappa = X.XX (n=24, SE about 0.15)`. Each case carries a `key_factor`, the single decisive criterion, which makes justification scoring objective.

**Scoring.** Decision: accuracy with a 95% Wilson interval, precision and recall for both classes, overall and edge-only. Justification: `reason_match` (the model's `primary_reason` equals the annotator's `key_factor`) and a 0 to 4 checklist of substantive elements. An optional LLM judge (Claude Opus 5 and GPT-5.6 Sol, scored against my own manual scores on 20 cases) reads for factual errors and invented evidence; see `src/judge.py`. Malformed output gets one format retry; a second failure counts as a wrong decision and is also reported as `format_failure_rate`, because a malformed response in production is a failed qualification, not a missing data point.

**Models.** Frontier: Claude Opus 5, GPT-5.6 Sol. Mid: Claude Sonnet 5 (production baseline), GPT-5.6 Terra. Budget: Claude Haiku 4.5, GPT-5.6 Luna. Open weights: Llama 4 Scout on Groq. All run without reasoning (adaptive thinking disabled on Claude 5, `reasoning_effort="none"` on GPT-5.6) so cost per profile is comparable. Prices in `src/config.py`, verified 2026-09-01.

**Prompts.** Three system prompts on every model: `minimal` (instruction plus a two-line ICP), `detailed` (expanded ICP plus two worked examples that are not in the eval set), `rubric` (explicit criteria with thresholds and a priority order). The output schema is identical across variants, so differences reflect reasoning, not format compliance. The prompt comparison on the production model is reported first because it answers the cheaper question: should I spend the afternoon on the prompt or on switching models?

**Reproducibility.** `temperature=0` and a fixed seed wherever the API accepts them (Haiku 4.5, Groq, GPT-5.6 seed). Claude Sonnet 5 and Opus 5 reject sampling parameters, and GPT-5.6 only accepts the default temperature, so for those the harness records repeat consistency instead (`--repeats 3`): `>>> stable decisions across 3 repeats on the production config: XX%`. Every row carries a timestamp, token counts and the estimated cost.

## Results

### Prompt variation on the production model

`>>> table + two sentences: which prompt won on edge cases, by how much, and whether the McNemar test could tell them apart`

### Model comparison

`>>> table + three sentences. Name the Pareto frontier configs. State which differences are inside the confidence intervals.`

### Edge cases

`>>> accuracy by edge subtype. Which subtype broke every model (a labeling or task-definition problem) vs which subtype separated the tiers (a capability problem). Point to results/divergent_cases.md.`

### Justification quality

`>>> reason_match by config. Note any config with high accuracy but low reason_match: right answer, wrong reason, which is the pattern that fails on new data.`

## Production decision

`>>> Fill from sections 5, 8 and 9 of summary.md. Template:`

The pipeline currently runs Claude Sonnet 5 with the detailed prompt. The eval says `<config>` is the lowest total-cost option once the cost of wrong decisions is included (a false qualify burns one personalised outreach, a false disqualify forfeits a fit creator), and `<config>` sits on the cost-accuracy frontier for edge cases. At the pipeline's current volume of about 600 profiles a month, API cost is a few dollars a month for every config (under $6 even for the frontier tier at the prices in `config.py`, under $3 for everything below it), so the model decision is an accuracy decision, not a budget decision. The rule I am shipping: `<route first with X; escalate to Y when X returns confidence below "high" or the profile falls in subtype E1/E3>`. Simulated on the eval set, that cascade reaches `<acc>` on edge cases at `<$/1k>`, escalating `<pct>` of profiles. Rerun this eval when a model in the config is deprecated, when the ICP changes, or when the pipeline's volume crosses 20K profiles a month, which is where API cost starts to matter.

## Limitations

- **Sample size.** With 80 cases the 95% interval on an 85% accuracy is about ±8 points; on 30 edge cases at 70% it is about ±16. The eval separates configs that differ by 20 points on edge cases and cannot rank configs within 10 points of each other. Pairwise comparisons use an exact McNemar test on discordant cases. To rank within ±10 points you need about 100 edge cases, which is the v2 plan.
- **One labeler.** Ground truth is my judgment, checked by a second annotator on 30% of cases. The kappa estimate at n=24 has a standard error near 0.15, so 0.70 is a screening bar, not a precise gate.
- **Class balance on edge cases.** Most edge cases are disqualifies, so a model that always says disqualify scores `>>> XX%` on the edge set. Every table reports that majority-class baseline; read edge accuracy against it, not against 50%.
- **Prompt and label contamination.** I wrote the rubric prompt from the same criteria doc I labeled with, before labeling. If you write prompts after seeing which edge cases exist, the rubric prompt will overfit to them and the eval will overstate its lift.
- **Text stands in for images.** The aesthetic criterion is judged from captions and bio in this eval. In production a human looks at the grid. Any config's aesthetic score here is a proxy.
- **Anonymisation changes the input.** Paraphrased bios and captions are cleaner than real Instagram text (emoji floods, Hinglish, broken grammar). Real-world accuracy is probably lower than what you see here, for every model.
- **Judge bias.** A Claude judge scoring Claude outputs leans favourable. Two judge families and a manual check on 20 cases limit this; they do not remove it.
- **Non-determinism.** Three of the seven models cannot be pinned with `temperature=0`. Repeat consistency is reported for the production candidate; the other configs were run once.
- **Selection bias.** All 80 profiles came from the pipeline's own scraping (hashtags and lookalike lists), not a random sample of Indian beauty creators. The eval measures performance on the profiles the pipeline actually sees, which is the right population for the routing decision and the wrong one for a general claim.
- **Cost figures move.** GPT-5.6 Sol is on a promotional rate through at least 2026-11-21. Rerun `analyze.py` after updating `config.py` prices; the charts regenerate.
- **Eval accuracy is not campaign ROI.** Agreement with my label says nothing about whether the qualified creators converted. That needs outcome data joined to this dataset, which is the next artifact.

## Landscape

Braintrust, Promptfoo and DeepEval are general eval frameworks; you could run this dataset inside any of them. This repo is intentionally small and domain-specific, and its value is the ground truth: I labeled the profiles because I know the creators, and the `key_factor` field turns "was the justification good" into a measurable question. The harness is about 1,300 lines of plain Python across nine files so a non-engineer can read every decision it makes.

## Reproduce

```bash
pip install -r requirements.txt
cp .env.example .env                  # add ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY
python src/run_eval.py --limit 3 --models claude-haiku-4.5 --prompts minimal   # smoke test
python src/run_eval.py                # full grid, resumable, about 1,700 calls, under $10
python src/run_eval.py --repeats 3 --models claude-sonnet-5 --prompts rubric   # consistency
python src/analyze.py --monthly-volume 600
python src/judge.py --judge claude-opus-5 --sample 30
python src/judge.py --judge gpt-5.6-sol --sample 30
python src/judge.py --agreement
```

`python src/run_eval.py --mock` exercises the whole pipeline with fake responses and no API keys. Never publish mock numbers.

## Files

```
data/labeling_criteria.md   frozen ICP criteria, strata, blind-labeling procedure
data/cases.jsonl            80 anonymised, labeled profiles
prompts/                    minimal.txt, detailed.txt, rubric.txt
src/config.py               models, prices, output schema, business cost of errors
src/providers.py            Anthropic / OpenAI / OpenAI-compatible / mock
src/run_eval.py             the grid loop, format retry, resumable CSV
src/score.py                JSON parsing, validation, checklist, reason match
src/analyze.py              metrics, CIs, McNemar, agreement, cascade, charts, summary.md
src/judge.py                optional LLM judge with dual-judge and manual agreement
tools/anonymize.py          raw export -> cases.jsonl with private id map
tools/kappa.py              Cohen's kappa for the second-annotator check
tests/make_mock_cases.py    synthetic dataset for pipeline tests only
results/                    raw_results.csv, summary.md, divergent_cases.md, two charts
```
