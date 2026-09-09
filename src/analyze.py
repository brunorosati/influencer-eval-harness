"""
Turn results/raw_results.csv into:
  results/summary.md               all tables (paste into README)
  results/divergent_cases.md       edge cases where configs disagree
  results/cost_vs_edge_accuracy.png  scatter + Pareto frontier
  results/edge_case_agreement.png    heatmap, rows = edge cases, cols = configs

Usage
  python src/analyze.py
  python src/analyze.py --monthly-volume 1200
  python src/analyze.py --baseline claude-sonnet-5 detailed
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

CONFIG_SEP = " | "


# ----------------------------------------------------------------------------
# statistics helpers (no scipy dependency)
# ----------------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts.
    b = baseline right & candidate wrong, c = baseline wrong & candidate right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def prf(df, positive):
    """precision, recall, f1 treating `positive` as the positive class.
    Format failures (empty decision) count as wrong predictions."""
    y = df["decisao_ground_truth"] == positive
    yhat = df["decisao_modelo"] == positive
    tp = int((y & yhat).sum())
    fp = int((~y & yhat).sum())
    fn = int((y & ~yhat).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def config_label(model, prompt):
    return f"{model}{CONFIG_SEP}{prompt}"


# ----------------------------------------------------------------------------
# per-config metrics
# ----------------------------------------------------------------------------
def summarize(df):
    rows = []
    for (model, prompt), g in df.groupby(["modelo", "prompt_variant"], sort=False):
        edge = g[g["stratum"] == "edge"]
        n, k = len(g), int(g["match"].sum())
        ne, ke = len(edge), int(edge["match"].sum())
        lo, hi = wilson_ci(k, n)
        elo, ehi = wilson_ci(ke, ne)
        pq, rq, fq = prf(g, "qualify")
        pd_, rd, fd = prf(g, "disqualify")
        epq, erq, _ = prf(edge, "qualify")
        epd, erd, _ = prf(edge, "disqualify")
        cost_per_1000 = g["custo_estimado_usd"].sum() / n * 1000
        fp_rate = ((g["decisao_modelo"] == "qualify") & (g["decisao_ground_truth"] == "disqualify")).mean()
        fn_rate = ((g["decisao_modelo"] != "qualify") & (g["decisao_ground_truth"] == "qualify")).mean()
        lat = pd.to_numeric(g["latencia_ms"], errors="coerce").dropna()
        parsed = g[g["format_failed"] == 0]
        rows.append(dict(
            modelo=model, prompt_variant=prompt, n=n,
            accuracy=k / n, acc_ci_lo=lo, acc_ci_hi=hi,
            edge_n=ne, edge_accuracy=ke / ne if ne else float("nan"),
            edge_ci_lo=elo, edge_ci_hi=ehi,
            prec_qualify=pq, rec_qualify=rq, f1_qualify=fq,
            prec_disqualify=pd_, rec_disqualify=rd, f1_disqualify=fd,
            edge_prec_qualify=epq, edge_rec_qualify=erq,
            edge_prec_disqualify=epd, edge_rec_disqualify=erd,
            macro_f1=(fq + fd) / 2,
            cost_per_1000=cost_per_1000, fp_rate=fp_rate, fn_rate=fn_rate,
            latency_mean_ms=lat.mean() if len(lat) else float("nan"),
            latency_p95_ms=lat.quantile(0.95) if len(lat) else float("nan"),
            format_failure_rate=g["format_failed"].mean(),
            format_retry_rate=g["format_retries"].mean(),
            reason_match_rate=parsed["reason_match"].mean() if len(parsed) else float("nan"),
            reason_valid_rate=parsed["reason_valid"].mean() if len(parsed) else float("nan"),
            checklist_mean=parsed["checklist_coverage"].mean() if len(parsed) else float("nan"),
            tokens_in_mean=g["tokens_input"].mean(),
            tokens_out_mean=g["tokens_output"].mean(),
        ))
    return pd.DataFrame(rows)


def subtype_accuracy(df):
    edge = df[df["stratum"] == "edge"]
    if edge.empty:
        return pd.DataFrame()
    t = edge.pivot_table(index=["modelo", "prompt_variant"], columns="edge_subtype", values="match", aggfunc="mean")
    return t.reset_index()


def majority_baseline(df_cases):
    """Accuracy of always predicting the majority class, overall and edge-only."""
    out = {}
    for name, sub in (("overall", df_cases), ("edge", df_cases[df_cases["stratum"] == "edge"])):
        counts = sub["decisao_ground_truth"].value_counts()
        out[name] = (counts.idxmax(), counts.max() / len(sub)) if len(sub) else ("", float("nan"))
    return out


def mcnemar_vs_baseline(df, baseline):
    """Compare every config against the production baseline on the same cases."""
    bm, bp = baseline
    base = df[(df["modelo"] == bm) & (df["prompt_variant"] == bp)].set_index("case_id")["match"]
    if base.empty:
        return pd.DataFrame()
    rows = []
    for (model, prompt), g in df.groupby(["modelo", "prompt_variant"], sort=False):
        cand = g.set_index("case_id")["match"]
        idx = base.index.intersection(cand.index)
        b = int(((base.loc[idx] == 1) & (cand.loc[idx] == 0)).sum())
        c = int(((base.loc[idx] == 0) & (cand.loc[idx] == 1)).sum())
        rows.append(dict(modelo=model, prompt_variant=prompt, baseline_only_right=b,
                         candidate_only_right=c, p_value=mcnemar_exact(b, c)))
    return pd.DataFrame(rows)


def agreement(df):
    """Per case: how many configs got it right; list of divergent cases."""
    piv = df.pivot_table(index="case_id", columns=["modelo", "prompt_variant"],
                         values="match", aggfunc="first")
    n_cfg = piv.shape[1]
    all_right = int((piv.sum(axis=1) == n_cfg).sum())
    all_wrong = int((piv.sum(axis=1) == 0).sum())
    divergent = piv[(piv.sum(axis=1) > 0) & (piv.sum(axis=1) < n_cfg)]
    return dict(n_configs=n_cfg, n_cases=len(piv), all_right=all_right,
                all_wrong=all_wrong, divergent=len(divergent)), piv, divergent


def cascade(df, cheap, expensive):
    """Route with the cheap config; escalate to the expensive one when the cheap
    config's confidence is not 'high'. Returns accuracy, cost/1000, escalation rate."""
    a = df[(df["modelo"] == cheap[0]) & (df["prompt_variant"] == cheap[1])].set_index("case_id")
    b = df[(df["modelo"] == expensive[0]) & (df["prompt_variant"] == expensive[1])].set_index("case_id")
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return None
    a, b = a.loc[idx], b.loc[idx]
    escalate = (a["confidence"] != "high") | (a["format_failed"] == 1)
    match = np.where(escalate, b["match"], a["match"])
    cost = a["custo_estimado_usd"].values + np.where(escalate, b["custo_estimado_usd"].values, 0)
    edge_mask = (a["stratum"] == "edge").values
    return dict(cheap=config_label(*cheap), expensive=config_label(*expensive),
                accuracy=match.mean(), edge_accuracy=match[edge_mask].mean() if edge_mask.any() else float("nan"),
                cost_per_1000=cost.mean() * 1000, escalation_rate=escalate.mean())


def consistency(df):
    """If --repeats > 1 was used: share of cases where every repeat agreed."""
    if df["run_idx"].nunique() < 2:
        return None
    rows = []
    for (model, prompt), g in df.groupby(["modelo", "prompt_variant"], sort=False):
        per_case = g.groupby("case_id")["decisao_modelo"].nunique()
        rows.append(dict(modelo=model, prompt_variant=prompt, repeats=g["run_idx"].nunique(),
                         stable_share=(per_case == 1).mean()))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# charts
# ----------------------------------------------------------------------------
def pareto_front(points, cost_tolerance=0.05):
    """points: list of (x=cost, y=accuracy, label). Keep points not dominated.
    A point is dominated when another point is at least as accurate and
    cheaper, or more accurate at a cost within `cost_tolerance` (5%), so
    three prompts on the same model at near-identical cost keep only the best."""
    front = []
    for x, y, lbl in points:
        dominated = any(
            ((x2 < x and y2 >= y) or (x2 <= x * (1 + cost_tolerance) and y2 > y)) and (x2, y2) != (x, y)
            for x2, y2, _ in points)
        if not dominated:
            front.append((x, y, lbl))
    return sorted(front)


def plot_scatter(summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(r.cost_per_1000, r.edge_accuracy, config_label(r.modelo, r.prompt_variant))
           for r in summary.itertuples()]
    front = pareto_front(pts)
    fig, ax = plt.subplots(figsize=(11, 7))
    # stagger labels for points that land on (nearly) the same spot
    seen = {}
    for x, y, lbl in pts:
        key = (round(math.log10(max(x, 1e-9)), 1), round(y, 2))
        k = seen.get(key, 0)
        seen[key] = k + 1
        ax.scatter(x, y, s=70, zorder=3)
        if k % 2 == 0:
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(7, 4 - 11 * k), fontsize=7.5)
        else:
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(-7, 4 - 11 * (k - 1)),
                        fontsize=7.5, ha="right")
    if len(front) > 1:
        ax.plot([p[0] for p in front], [p[1] for p in front], linestyle="--",
                linewidth=1, zorder=2, label="Pareto frontier")
        ax.legend(loc="lower right")
    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 evaluations (USD, log scale)")
    ax.set_ylabel("Accuracy on edge cases (n=%d)" % int(summary["edge_n"].max()))
    ax.set_title("Cost vs edge-case accuracy, one point per model x prompt")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_heatmap(piv_edge, out_path, cost_by_config=None, subtype_by_case=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # rows: hardest cases first; columns: cheapest config first
    piv_edge = piv_edge.loc[piv_edge.sum(axis=1).sort_values().index]
    if cost_by_config:
        piv_edge = piv_edge[sorted(piv_edge.columns, key=lambda c: cost_by_config.get(c, 0))]
    data = piv_edge.fillna(-1).values.astype(float)
    cmap = ListedColormap(["#d9d9d9", "#c0392b", "#27ae60"])  # missing, wrong, right
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * data.shape[1] + 3), max(6, 0.28 * data.shape[0] + 2)))
    ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(data.shape[0]))
    ylabels = [f"{i} ({subtype_by_case.get(i, '')})" if subtype_by_case else i for i in piv_edge.index]
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([config_label(*c) for c in piv_edge.columns], rotation=45, ha="right", fontsize=7)
    ax.set_title("Edge cases: green = correct, red = wrong. Row-wide red = hard case; column-wide red = weak config.")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------
def fmt_pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def md_table(df, cols, headers=None, fmt=None):
    headers = headers or cols
    fmt = fmt or {}
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            cells.append(f(v) if f else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(summary, maj, mc, agree, casc_rows, cons, baseline, monthly_volume, out_path,
                  subtype_tbl=None, cost_fp=0.0, cost_fn=0.0):
    pct = fmt_pct
    money = lambda v: f"${v:,.2f}"
    ms = lambda v: "n/a" if pd.isna(v) else f"{v:,.0f}"
    L = []
    L.append("# Eval summary (auto-generated by analyze.py)\n")
    L.append(f"Pricing verified {config.PRICING_VERIFIED_ON}. Baseline: `{config_label(*baseline)}`. "
             f"Majority-class baseline: overall {pct(maj['overall'][1])} (always `{maj['overall'][0]}`), "
             f"edge {pct(maj['edge'][1])} (always `{maj['edge'][0]}`).\n")

    # 1. prompt variation on the mid-tier model, reported first
    mid = summary[summary["modelo"] == baseline[0]]
    if len(mid) > 1:
        L.append("## 1. Prompt variation (same model, three prompts)\n")
        L.append(md_table(mid, ["prompt_variant", "accuracy", "acc_ci_lo", "acc_ci_hi", "edge_accuracy",
                                "reason_match_rate", "format_failure_rate", "cost_per_1000", "tokens_in_mean"],
                          ["prompt", "acc", "ci lo", "ci hi", "edge acc", "reason match", "format fail", "$/1k", "in tok"],
                          {"accuracy": pct, "acc_ci_lo": pct, "acc_ci_hi": pct, "edge_accuracy": pct,
                           "reason_match_rate": pct, "format_failure_rate": pct, "cost_per_1000": money,
                           "tokens_in_mean": ms}))
        L.append("")

    # 2. all configs
    L.append("## 2. All model x prompt configs\n")
    L.append(md_table(summary, ["modelo", "prompt_variant", "accuracy", "acc_ci_lo", "acc_ci_hi", "macro_f1",
                                "edge_accuracy", "edge_ci_lo", "edge_ci_hi", "cost_per_1000",
                                "latency_mean_ms", "latency_p95_ms", "format_failure_rate"],
                      ["model", "prompt", "acc", "ci lo", "ci hi", "macro F1", "edge acc", "edge lo", "edge hi",
                       "$/1k", "lat mean", "lat p95", "format fail"],
                      {"accuracy": pct, "acc_ci_lo": pct, "acc_ci_hi": pct, "edge_accuracy": pct,
                       "edge_ci_lo": pct, "edge_ci_hi": pct, "cost_per_1000": money,
                       "latency_mean_ms": ms, "latency_p95_ms": ms, "format_failure_rate": pct}))
    L.append("")
    L.append("### Per-class precision / recall (overall)\n")
    L.append(md_table(summary, ["modelo", "prompt_variant", "prec_qualify", "rec_qualify", "prec_disqualify", "rec_disqualify"],
                      ["model", "prompt", "P(qualify)", "R(qualify)", "P(disqualify)", "R(disqualify)"],
                      {c: pct for c in ["prec_qualify", "rec_qualify", "prec_disqualify", "rec_disqualify"]}))
    L.append("")
    L.append("### Per-class precision / recall (edge cases only)\n")
    L.append(md_table(summary, ["modelo", "prompt_variant", "edge_prec_qualify", "edge_rec_qualify",
                                "edge_prec_disqualify", "edge_rec_disqualify"],
                      ["model", "prompt", "P(qualify)", "R(qualify)", "P(disqualify)", "R(disqualify)"],
                      {c: pct for c in ["edge_prec_qualify", "edge_rec_qualify", "edge_prec_disqualify", "edge_rec_disqualify"]}))
    L.append("")
    L.append("### Justification quality (automated)\n")
    L.append(md_table(summary, ["modelo", "prompt_variant", "reason_match_rate", "reason_valid_rate", "checklist_mean"],
                      ["model", "prompt", "reason match", "reason in vocab", "checklist (0-4)"],
                      {"reason_match_rate": pct, "reason_valid_rate": pct, "checklist_mean": lambda v: f"{v:.2f}"}))
    L.append("")

    # 3. significance
    if len(mc):
        L.append(f"## 3. McNemar exact test vs baseline `{config_label(*baseline)}`\n")
        L.append("Differences with p >= 0.05 are not distinguishable at n=80. Treat them as ties.\n")
        L.append(md_table(mc, ["modelo", "prompt_variant", "baseline_only_right", "candidate_only_right", "p_value"],
                          ["model", "prompt", "baseline right, cand wrong", "cand right, baseline wrong", "p"],
                          {"p_value": lambda v: f"{v:.3f}"}))
        L.append("")

    # 4. agreement
    L.append("## 4. Agreement across configs\n")
    L.append(f"{agree['n_configs']} configs on {agree['n_cases']} cases. All correct: {agree['all_right']}. "
             f"All wrong: {agree['all_wrong']}. Divergent: {agree['divergent']} (see divergent_cases.md).\n")

    # 5. cascade
    if casc_rows:
        L.append("## 5. Cascade simulation (cheap model, escalate when confidence != high)\n")
        cdf = pd.DataFrame(casc_rows)
        L.append(md_table(cdf, ["cheap", "expensive", "accuracy", "edge_accuracy", "cost_per_1000", "escalation_rate"],
                          ["route first", "escalate to", "acc", "edge acc", "$/1k", "escalated"],
                          {"accuracy": pct, "edge_accuracy": pct, "cost_per_1000": money, "escalation_rate": pct}))
        L.append("")

    # 6. consistency
    if cons is not None:
        L.append("## 6. Repeat consistency (share of cases with identical decision across repeats)\n")
        L.append(md_table(cons, ["modelo", "prompt_variant", "repeats", "stable_share"],
                          ["model", "prompt", "repeats", "stable"], {"stable_share": pct}))
        L.append("")

    # 7. accuracy by edge subtype
    if subtype_tbl is not None and len(subtype_tbl):
        cols = [c for c in subtype_tbl.columns if c not in ("modelo", "prompt_variant")]
        L.append("## 7. Accuracy by edge subtype (where each config breaks)\n")
        L.append(md_table(subtype_tbl, ["modelo", "prompt_variant"] + cols,
                          ["model", "prompt"] + [c.replace("_", " ") for c in cols], {c: pct for c in cols}))
        L.append("")

    # 8. total cost = API + error cost
    L.append(f"## 8. Total cost per 1,000 profiles = API cost + error cost "
             f"(false qualify = ${cost_fp:.2f}, false disqualify = ${cost_fn:.2f})\n")
    L.append("Error cost dominates API cost at almost any price. Pick the config with the lowest TOTAL, not the cheapest API.\n")
    t2 = summary.copy()
    t2["error_cost_per_1000"] = 1000 * (t2["fp_rate"] * cost_fp + t2["fn_rate"] * cost_fn)
    t2["total_per_1000"] = t2["cost_per_1000"] + t2["error_cost_per_1000"]
    L.append(md_table(t2.sort_values("total_per_1000"),
                      ["modelo", "prompt_variant", "fp_rate", "fn_rate", "cost_per_1000", "error_cost_per_1000", "total_per_1000"],
                      ["model", "prompt", "false qualify", "false disqualify", "API $/1k", "error $/1k", "TOTAL $/1k"],
                      {"fp_rate": pct, "fn_rate": pct, "cost_per_1000": money, "error_cost_per_1000": money, "total_per_1000": money}))
    L.append("")

    # 9. monthly cost
    L.append(f"## 9. Monthly API cost at {monthly_volume:,} profiles/month\n")
    tmp = summary.copy()
    tmp["monthly_usd"] = tmp["cost_per_1000"] * monthly_volume / 1000
    base_row = tmp[(tmp["modelo"] == baseline[0]) & (tmp["prompt_variant"] == baseline[1])]
    base_cost = float(base_row["monthly_usd"].iloc[0]) if len(base_row) else float("nan")
    tmp["delta_vs_baseline"] = tmp["monthly_usd"] - base_cost
    L.append(md_table(tmp.sort_values("monthly_usd"), ["modelo", "prompt_variant", "edge_accuracy", "monthly_usd", "delta_vs_baseline"],
                      ["model", "prompt", "edge acc", "USD / month", "delta vs baseline"],
                      {"edge_accuracy": pct, "monthly_usd": money, "delta_vs_baseline": lambda v: f"{v:+,.2f}"}))
    L.append("")
    Path(out_path).write_text("\n".join(L), encoding="utf-8")


def write_divergent(divergent, cases_meta, out_path):
    L = ["# Divergent edge cases (some configs right, some wrong)\n"]
    for case_id, row in divergent.iterrows():
        meta = cases_meta.get(case_id, {})
        wrong = [config_label(*c) for c, v in row.items() if v == 0]
        L.append(f"## {case_id} ({meta.get('edge_subtype', '')}) truth={meta.get('label', '')} key_factor={meta.get('key_factor', '')}")
        L.append(f"Wrong: {', '.join(wrong) if wrong else 'none'}\n")
    Path(out_path).write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(config.RAW_RESULTS))
    ap.add_argument("--baseline", nargs=2, default=list(config.PRODUCTION_BASELINE))
    ap.add_argument("--monthly-volume", type=int, default=config.DEFAULT_MONTHLY_VOLUME)
    ap.add_argument("--cost-fp", type=float, default=config.COST_FALSE_QUALIFY_USD,
                    help="USD cost of one false qualify (wasted personalised outreach)")
    ap.add_argument("--cost-fn", type=float, default=config.COST_FALSE_DISQUALIFY_USD,
                    help="USD cost of one false disqualify (lost expected creator value)")
    args = ap.parse_args()

    df = pd.read_csv(args.results, dtype={"edge_subtype": str, "decisao_modelo": str, "confidence": str})
    df["decisao_modelo"] = df["decisao_modelo"].fillna("")
    df["confidence"] = df["confidence"].fillna("")
    df["edge_subtype"] = df["edge_subtype"].fillna("")
    df["run_idx"] = df["run_idx"].astype(str)
    cons = consistency(df)
    df0 = df[df["run_idx"] == df["run_idx"].min()]  # main tables use the first repeat
    baseline = tuple(args.baseline)

    summary = summarize(df0)
    maj = majority_baseline(df0.drop_duplicates("case_id"))
    mc = mcnemar_vs_baseline(df0, baseline)
    agree_stats, piv, divergent = agreement(df0)

    # cascades: every budget/open config -> the baseline, and -> each frontier config with the same prompt
    casc_rows = []
    configs = list(summary[["modelo", "prompt_variant"]].itertuples(index=False, name=None))
    for cheap in configs:
        if config.MODELS[cheap[0]]["tier"] not in ("budget", "open"):
            continue
        for exp in configs:
            if config.MODELS[exp[0]]["tier"] in ("mid", "frontier") and exp[1] == cheap[1]:
                r = cascade(df0, cheap, exp)
                if r:
                    casc_rows.append(r)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    write_summary(summary, maj, mc, agree_stats, casc_rows, cons, baseline, args.monthly_volume,
                  config.SUMMARY_MD, subtype_accuracy(df0), args.cost_fp, args.cost_fn)

    edge_ids = df0[df0["stratum"] == "edge"]["case_id"].unique()
    piv_edge = piv.loc[[i for i in piv.index if i in set(edge_ids)]]
    cases_meta = df0.drop_duplicates("case_id").set_index("case_id")[["edge_subtype", "decisao_ground_truth", "key_factor_ground_truth"]]
    cases_meta = {i: dict(edge_subtype=r["edge_subtype"], label=r["decisao_ground_truth"], key_factor=r["key_factor_ground_truth"])
                  for i, r in cases_meta.iterrows()}
    write_divergent(divergent.loc[[i for i in divergent.index if i in set(edge_ids)]], cases_meta, config.DIVERGENT_MD)

    plot_scatter(summary, config.RESULTS_DIR / "cost_vs_edge_accuracy.png")
    if len(piv_edge):
        cost_by_config = {(r.modelo, r.prompt_variant): r.cost_per_1000 for r in summary.itertuples()}
        subtype_by_case = {i: m["edge_subtype"] for i, m in cases_meta.items()}
        plot_heatmap(piv_edge, config.RESULTS_DIR / "edge_case_agreement.png", cost_by_config, subtype_by_case)

    summary.to_csv(config.RESULTS_DIR / "summary_table.csv", index=False)
    print(f"wrote {config.SUMMARY_MD}, {config.DIVERGENT_MD}, 2 charts, summary_table.csv")
    print(summary[["modelo", "prompt_variant", "accuracy", "edge_accuracy", "cost_per_1000", "format_failure_rate"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
