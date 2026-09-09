"""
Cohen's kappa between your labels and a second annotator's labels.

Usage
  python tools/kappa.py data/labels_bruno.csv data/labels_annotator2.csv
Both CSVs: columns case_id,label   (label in {qualify, disqualify})
Only case_ids present in BOTH files are compared.

Read the result like this:
  kappa >= 0.70   ground truth is usable
  0.40 - 0.69     refine the criteria doc, relabel the disputed cases, rerun
  < 0.40          the criteria are ambiguous; do not run the eval yet
With ~24 overlap cases the standard error is about 0.15, so treat 0.70 as a
screening threshold, not a precise cutoff.
"""
import csv
import math
import sys


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:  # -sig: Excel/PowerShell CSVs carry a BOM
        return {r["case_id"].strip(): r["label"].strip().lower() for r in csv.DictReader(f)}


def cohen_kappa(a, b):
    ids = sorted(set(a) & set(b))
    n = len(ids)
    if n == 0:
        raise SystemExit("no overlapping case_ids")
    labels = sorted(set(a[i] for i in ids) | set(b[i] for i in ids))
    po = sum(a[i] == b[i] for i in ids) / n
    pe = sum((sum(a[i] == l for i in ids) / n) * (sum(b[i] == l for i in ids) / n) for l in labels)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2)) if pe < 1 else 0.0
    disagreements = [i for i in ids if a[i] != b[i]]
    return dict(n=n, observed_agreement=po, expected_agreement=pe, kappa=kappa,
                se=se, ci_lo=max(-1.0, kappa - 1.96 * se), ci_hi=min(1.0, kappa + 1.96 * se), disagreements=disagreements)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    r = cohen_kappa(load(sys.argv[1]), load(sys.argv[2]))
    print(f"n={r['n']}  observed={r['observed_agreement']:.3f}  expected={r['expected_agreement']:.3f}")
    print(f"kappa={r['kappa']:.3f}  (SE {r['se']:.3f}, 95% CI {r['ci_lo']:.2f} to {r['ci_hi']:.2f})")
    print("disagreements:", ", ".join(r["disagreements"]) or "none")
