#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical tests for reviewer response:
- DL-AG vs BERT-Base / DistilBERT (Welch t-test + 95% CI of mean diff)
- PKD-Skip vs Logit-Only (paired/unpaired t-test on seed scores)

Reads JSONs from multiseed_results_full/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def welch_and_ci(a, b, label_a, label_b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = a.mean() - b.mean()
    # CI for difference of means (Welch approx)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    # Welch-Satterthwaite df
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    tcrit = stats.t.ppf(0.975, df) if df > 0 else 1.96
    ci = (float(diff - tcrit * se), float(diff + tcrit * se))
    return {
        "A": label_a,
        "B": label_b,
        "mean_A": float(a.mean()),
        "std_A": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "mean_B": float(b.mean()),
        "std_B": float(b.std(ddof=1)) if len(b) > 1 else 0.0,
        "diff_A_minus_B": float(diff),
        "t": float(t),
        "p": float(p),
        "df": float(df),
        "ci95_diff": ci,
        "n_A": int(na),
        "n_B": int(nb),
    }


def collect_dlag_scores(results_dir: Path):
    vals = []
    for p in sorted(results_dir.glob("seed_*_sst2_*.json")):
        d = _load(p)
        s = d.get("summary", {})
        if "advglue_l1l2" in s:
            vals.append(s["advglue_l1l2"]["RA"])
        elif "mean_attack_RA" in s:
            vals.append(s["mean_attack_RA"])
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="./multiseed_results_full")
    ap.add_argument("--baselines", type=str, default="./multiseed_results_full/baselines_summary.json")
    ap.add_argument("--pkd", type=str, default="./multiseed_results_full/pkd_ablation_summary.json")
    ap.add_argument("--out", type=str, default="./multiseed_results_full/significance_tests.json")
    args = ap.parse_args()

    root = Path(args.results_dir)
    report = {"tests": []}

    dlag = collect_dlag_scores(root)
    print(f"DL-AG scores ({len(dlag)}): {dlag}")

    if Path(args.baselines).exists() and dlag:
        base = _load(args.baselines)
        for model_name in ["bert-base-uncased", "distilbert-base-uncased"]:
            if model_name in base:
                bvals = base[model_name]["values"]
                res = welch_and_ci(dlag, bvals, "DL-AG", model_name)
                report["tests"].append(res)
                print(
                    f"DL-AG vs {model_name}: "
                    f"{res['mean_A']:.2f}±{res['std_A']:.2f} vs {res['mean_B']:.2f}±{res['std_B']:.2f} | "
                    f"Δ={res['diff_A_minus_B']:.2f} | t={res['t']:.3f} p={res['p']:.4f} | "
                    f"95% CI {res['ci95_diff']}"
                )

    if Path(args.pkd).exists():
        pkd = _load(args.pkd)["strategies"]
        if "PKD-Skip" in pkd and "Logit-Only" in pkd:
            res = welch_and_ci(
                pkd["PKD-Skip"]["values"],
                pkd["Logit-Only"]["values"],
                "PKD-Skip",
                "Logit-Only",
            )
            report["tests"].append(res)
            print(
                f"PKD-Skip vs Logit-Only: "
                f"{res['mean_A']:.2f}±{res['std_A']:.2f} vs {res['mean_B']:.2f}±{res['std_B']:.2f} | "
                f"Δ={res['diff_A_minus_B']:.2f} | t={res['t']:.3f} p={res['p']:.4f} | "
                f"95% CI {res['ci95_diff']}"
            )
        if "PKD-Last" in pkd and "Logit-Only" in pkd:
            res = welch_and_ci(
                pkd["PKD-Last"]["values"],
                pkd["Logit-Only"]["values"],
                "PKD-Last",
                "Logit-Only",
            )
            report["tests"].append(res)

    # Table 9 style: if per-seed L1/L2/hybrid JSONs exist later
    report["dlag_values"] = dlag
    if dlag:
        arr = np.array(dlag, dtype=float)
        report["dlag_mean_std"] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
