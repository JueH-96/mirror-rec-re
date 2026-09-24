"""Pilot statistical analysis (RESEARCH_BRIEF §12).

- DVR/SRR/RR per model x variant with bootstrap CIs (resampling chains,
  i.e. clustered by (set_id, template_family) to respect dependence).
- Paired permutation tests: plain vs each Study-2 variant per model (paired on
  chain id), Holm-corrected.
- Sensitivity of DVR to epsilon over a grid.
Writes results/statistical_tests.json and results/aggregate_results.csv.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(42)


def load(model, variant):
    p = ROOT / "results" / f"metrics_{model}_{variant}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["chain_id"] = df["set_id"] + "|" + df["template_family"]
    return df


def boot_ci(df, col, n=2000):
    chains = df.groupby("chain_id")[col].mean()
    vals = chains.to_numpy()
    if len(vals) == 0:
        return None
    boots = [RNG.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return {"mean": float(np.mean(vals)), "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)), "n_chains": int(len(vals))}


def paired_perm(a, b, n=5000):
    """a, b: per-chain rates aligned on chain_id."""
    diff = a - b
    obs = diff.mean()
    if len(diff) == 0:
        return None
    signs = RNG.choice([-1, 1], size=(n, len(diff)))
    null = (signs * diff.to_numpy()).mean(axis=1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return {"mean_diff": float(obs), "p_value": p, "n_pairs": int(len(diff))}


def holm(pvals):
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    adj, prev = {}, 0.0
    for rank, k in enumerate(order):
        val = min(1.0, (m - rank) * pvals[k])
        prev = max(prev, val)
        adj[k] = prev
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    args = ap.parse_args()

    agg_rows, tests = [], {"paired_permutation": {}, "epsilon_sensitivity": {}}
    frames = {}
    for m, v in itertools.product(args.models, args.variants):
        df = load(m, v)
        if df is None:
            continue
        frames[(m, v)] = df
        row = {"model": m, "variant": v}
        for col, name in [("violation", "DVR"), ("strict_reversal", "SRR"),
                          ("responsive", "RR"), ("rank_violation", "rank_DVR")]:
            ci = boot_ci(df, col)
            if ci:
                row[name] = ci["mean"]
                row[f"{name}_ci"] = f"[{ci['ci_lo']:.4f},{ci['ci_hi']:.4f}]"
                row["n_chains"] = ci["n_chains"]
        for extra in ["chain_fully_consistent", "pairwise_flip_rate"]:
            if extra in df:
                row[extra] = float(df.groupby("chain_id")[extra].first().mean())
        # per-attribute and per-set-type and per-family breakdown
        for attr, g in df.groupby("target_attribute"):
            row[f"DVR_{attr}"] = float(g["violation"].mean())
        for st, g in df.groupby("set_type"):
            row[f"DVR_type_{st}"] = float(g["violation"].mean())
        for fam, g in df.groupby("template_family"):
            row[f"DVR_fam_{fam}"] = float(g["violation"].mean())
        agg_rows.append(row)

    # Study 2 paired tests: plain vs other variants, per model, Holm across variants.
    for m in args.models:
        base = frames.get((m, "plain"))
        if base is None:
            continue
        pvals, details = {}, {}
        a = base.groupby("chain_id")["violation"].mean()
        for v in args.variants:
            if v == "plain" or (m, v) not in frames:
                continue
            b = frames[(m, v)].groupby("chain_id")["violation"].mean()
            common = a.index.intersection(b.index)
            res = paired_perm(a.loc[common], b.loc[common])
            if res:
                pvals[v] = res["p_value"]
                details[v] = res
        if pvals:
            adj = holm(pvals)
            for v in details:
                details[v]["p_holm"] = adj[v]
            tests["paired_permutation"][m] = details

    # Epsilon sensitivity on the plain variant.
    for (m, v), df in frames.items():
        if v != "plain":
            continue
        sens = {}
        for eps in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]:
            sens[str(eps)] = float((df["delta_k1"] < df["delta_k"] - eps).mean())
        tests["epsilon_sensitivity"][m] = sens

    pd.DataFrame(agg_rows).to_csv(ROOT / "results" / "aggregate_results.csv", index=False)
    with open(ROOT / "results" / "statistical_tests.json", "w") as f:
        json.dump(tests, f, indent=2)
    print(json.dumps({"aggregate_rows": len(agg_rows)}, indent=2))


if __name__ == "__main__":
    main()
